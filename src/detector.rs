use std::path::PathBuf;

use image::{imageops::FilterType, Rgb, RgbImage};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use ort::{session::Session, value::Tensor};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DetectorError {
    #[error("ONNX Runtime error in detector: {0}")]
    Ort(#[from] ort::Error),
    #[error("Detector model has no input tensors")]
    MissingInput,
    #[error("Detector model has no output tensors")]
    MissingOutput,
    #[error("Unexpected detector output shape {0:?}; expected rank-3 tensor [1, C, N] or [1, N, C]")]
    InvalidOutputShape(Vec<i64>),
    #[error("Invalid image dimensions: {width}x{height}")]
    InvalidImageDimensions { width: u32, height: u32 },
}

#[derive(Debug, Clone)]
pub struct DetectorConfig {
    pub model_path: PathBuf,
    pub input_width: u32,
    pub input_height: u32,
    pub conf_threshold: f32,
    pub iou_threshold: f32,
    pub crop_padding_ratio: f32,
    pub enable_deskew: bool,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/yolo_plate.onnx"),
            input_width: 640,
            input_height: 640,
            conf_threshold: 0.35,
            iou_threshold: 0.45,
            crop_padding_ratio: 0.08,
            enable_deskew: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    #[allow(dead_code)]
    pub class_id: usize,
}

#[derive(Debug, Clone, Copy)]
struct LetterboxMeta {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    orig_w: f32,
    orig_h: f32,
}

/// YOLOv8 / YOLO11 license plate detector backed by ONNX Runtime.
pub struct PlateDetector {
    session: Session,
    input_name: String,
    config: DetectorConfig,
}

impl PlateDetector {
    pub fn new(session: Session, config: DetectorConfig) -> Result<Self, DetectorError> {
        let input_name = session
            .inputs()
            .first()
            .ok_or(DetectorError::MissingInput)?
            .name()
            .to_string();

        Ok(Self {
            session,
            input_name,
            config,
        })
    }

    /// Runs plate detection on an RGB frame and returns NMS-filtered bounding boxes
    /// in original image pixel coordinates.
    pub fn detect(&mut self, img: &RgbImage) -> Result<Vec<BoundingBox>, DetectorError> {
        let (w, h) = img.dimensions();
        if w == 0 || h == 0 {
            return Err(DetectorError::InvalidImageDimensions { width: w, height: h });
        }

        let (tensor_data, meta) =
            letterbox_rgb_nchw(img, self.config.input_width, self.config.input_height);

        let shape = [
            1usize,
            3,
            self.config.input_height as usize,
            self.config.input_width as usize,
        ];
        let input_tensor = Tensor::from_array((shape, tensor_data))?;

        let outputs = self.session.run(ort::inputs![
            self.input_name.as_str() => input_tensor
        ])?;

        if outputs.len() == 0 {
            return Err(DetectorError::MissingOutput);
        }

        let (out_shape, out_data) = outputs[0].try_extract_tensor::<f32>()?;
        let dims: Vec<i64> = out_shape.iter().copied().collect();

        let candidates = parse_yolo_output(&dims, out_data, self.config.conf_threshold, meta)?;
        Ok(non_max_suppression(candidates, self.config.iou_threshold))
    }

    /// Crops the detected license plate with configurable margin padding and
    /// optionally applies lightweight deskewing (tilt correction).
    pub fn crop_plate(&self, img: &RgbImage, bbox: &BoundingBox) -> RgbImage {
        let (img_w, img_h) = img.dimensions();
        let box_w = (bbox.x2 - bbox.x1).max(1.0);
        let box_h = (bbox.y2 - bbox.y1).max(1.0);

        let pad_x = box_w * self.config.crop_padding_ratio;
        let pad_y = box_h * self.config.crop_padding_ratio;

        let x1 = (bbox.x1 - pad_x).floor().max(0.0) as u32;
        let y1 = (bbox.y1 - pad_y).floor().max(0.0) as u32;
        let x2 = (bbox.x2 + pad_x).ceil().min(img_w as f32) as u32;
        let y2 = (bbox.y2 + pad_y).ceil().min(img_h as f32) as u32;

        let crop_w = x2.saturating_sub(x1).max(1);
        let crop_h = y2.saturating_sub(y1).max(1);

        let cropped = image::imageops::crop_imm(img, x1, y1, crop_w, crop_h).to_image();

        if self.config.enable_deskew {
            deskew_plate(&cropped)
        } else {
            cropped
        }
    }
}

/// Resizes with aspect ratio preservation (letterboxing) and converts to NCHW float32 [0, 1].
fn letterbox_rgb_nchw(img: &RgbImage, target_w: u32, target_h: u32) -> (Vec<f32>, LetterboxMeta) {
    let (orig_w, orig_h) = img.dimensions();
    let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);

    let new_w = ((orig_w as f32) * scale).round().max(1.0) as u32;
    let new_h = ((orig_h as f32) * scale).round().max(1.0) as u32;

    let pad_x = ((target_w - new_w) as f32) / 2.0;
    let pad_y = ((target_h - new_h) as f32) / 2.0;

    let resized = image::imageops::resize(img, new_w, new_h, FilterType::Triangle);

    let plane_size = (target_w * target_h) as usize;
    let pad_val = 114.0_f32 / 255.0_f32;
    let mut buffer = vec![pad_val; 3 * plane_size];

    let offset_x = pad_x.floor() as u32;
    let offset_y = pad_y.floor() as u32;

    for y in 0..new_h {
        let dst_y = y + offset_y;
        if dst_y >= target_h {
            continue;
        }
        for x in 0..new_w {
            let dst_x = x + offset_x;
            if dst_x >= target_w {
                continue;
            }
            let pixel = resized.get_pixel(x, y);
            let idx = (dst_y * target_w + dst_x) as usize;
            buffer[idx] = f32::from(pixel[0]) / 255.0;
            buffer[plane_size + idx] = f32::from(pixel[1]) / 255.0;
            buffer[2 * plane_size + idx] = f32::from(pixel[2]) / 255.0;
        }
    }

    let meta = LetterboxMeta {
        scale,
        pad_x: offset_x as f32,
        pad_y: offset_y as f32,
        orig_w: orig_w as f32,
        orig_h: orig_h as f32,
    };

    (buffer, meta)
}

/// Parses YOLOv8/YOLO11 raw output `[1, 4+C, N]` or `[1, N, 4+C]`.
fn parse_yolo_output(
    dims: &[i64],
    data: &[f32],
    conf_threshold: f32,
    meta: LetterboxMeta,
) -> Result<Vec<BoundingBox>, DetectorError> {
    if dims.len() != 3 || dims[0] != 1 || dims[1] <= 0 || dims[2] <= 0 {
        return Err(DetectorError::InvalidOutputShape(dims.to_vec()));
    }

    let d1 = dims[1] as usize;
    let d2 = dims[2] as usize;

    // Heuristic to distinguish [1, 4+C, num_anchors] from [1, num_anchors, 4+C].
    let (channels_first, num_attrs, num_anchors) = if d1 < d2 {
        (true, d1, d2)
    } else {
        (false, d2, d1)
    };

    if num_attrs < 5 {
        return Err(DetectorError::InvalidOutputShape(dims.to_vec()));
    }

    let num_classes = num_attrs - 4;
    let mut boxes = Vec::new();

    for anchor_idx in 0..num_anchors {
        let get_attr = |attr: usize| -> f32 {
            if channels_first {
                data[attr * num_anchors + anchor_idx]
            } else {
                data[anchor_idx * num_attrs + attr]
            }
        };

        let mut best_class = 0usize;
        let mut best_conf = 0.0_f32;
        for c in 0..num_classes {
            let score = get_attr(4 + c);
            if score > best_conf {
                best_conf = score;
                best_class = c;
            }
        }

        if best_conf < conf_threshold {
            continue;
        }

        let cx = get_attr(0);
        let cy = get_attr(1);
        let bw = get_attr(2);
        let bh = get_attr(3);

        let x1 = ((cx - bw * 0.5 - meta.pad_x) / meta.scale).clamp(0.0, meta.orig_w);
        let y1 = ((cy - bh * 0.5 - meta.pad_y) / meta.scale).clamp(0.0, meta.orig_h);
        let x2 = ((cx + bw * 0.5 - meta.pad_x) / meta.scale).clamp(0.0, meta.orig_w);
        let y2 = ((cy + bh * 0.5 - meta.pad_y) / meta.scale).clamp(0.0, meta.orig_h);

        if x2 > x1 + 1.0 && y2 > y1 + 1.0 {
            boxes.push(BoundingBox {
                x1,
                y1,
                x2,
                y2,
                confidence: best_conf,
                class_id: best_class,
            });
        }
    }

    Ok(boxes)
}

fn non_max_suppression(mut boxes: Vec<BoundingBox>, iou_threshold: f32) -> Vec<BoundingBox> {
    boxes.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));

    let mut kept = Vec::with_capacity(boxes.len());
    let mut suppressed = vec![false; boxes.len()];

    for i in 0..boxes.len() {
        if suppressed[i] {
            continue;
        }
        let current = boxes[i];
        kept.push(current);

        for j in (i + 1)..boxes.len() {
            if !suppressed[j] && compute_iou(&current, &boxes[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    kept
}

fn compute_iou(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let inter_x1 = a.x1.max(b.x1);
    let inter_y1 = a.y1.max(b.y1);
    let inter_x2 = a.x2.min(b.x2);
    let inter_y2 = a.y2.min(b.y2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

/// Estimates slight horizontal tilt of a plate crop using left/right vertical
/// edge-centroid moments and rotates around the center when tilt is between 1.5° and 15°.
fn deskew_plate(crop: &RgbImage) -> RgbImage {
    let (w, h) = crop.dimensions();
    if w < 16 || h < 8 {
        return crop.clone();
    }

    let mut left_y_sum = 0.0_f64;
    let mut left_w_sum = 0.0_f64;
    let mut right_y_sum = 0.0_f64;
    let mut right_w_sum = 0.0_f64;

    let mid_x = w / 2;
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let luma_left = luma(crop.get_pixel(x - 1, y));
            let luma_right = luma(crop.get_pixel(x + 1, y));
            let luma_up = luma(crop.get_pixel(x, y - 1));
            let luma_down = luma(crop.get_pixel(x, y + 1));

            let grad = (luma_right - luma_left).abs() + (luma_down - luma_up).abs();
            if grad < 30.0 {
                continue;
            }
            if x < mid_x {
                left_y_sum += f64::from(y) * grad;
                left_w_sum += grad;
            } else {
                right_y_sum += f64::from(y) * grad;
                right_w_sum += grad;
            }
        }
    }

    if left_w_sum <= 1.0 || right_w_sum <= 1.0 {
        return crop.clone();
    }

    let left_cy = left_y_sum / left_w_sum;
    let right_cy = right_y_sum / right_w_sum;
    let dx = f64::from(w) * 0.5;
    let angle_rad = ((right_cy - left_cy) / dx).atan();
    let angle_deg = angle_rad.to_degrees().abs();

    if !(1.5..=15.0).contains(&angle_deg) {
        return crop.clone();
    }

    rotate_about_center(
        crop,
        -(angle_rad as f32),
        Interpolation::Bilinear,
        Rgb([114, 114, 114]),
    )
}

#[inline]
fn luma(p: &Rgb<u8>) -> f64 {
    0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2])
}
