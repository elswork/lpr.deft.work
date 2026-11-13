use std::fs;
use std::path::{Path, PathBuf};

use image::{imageops::FilterType, Rgb, RgbImage};
use ort::{session::Session, value::Tensor};
use serde::Deserialize;
use thiserror::Error;

pub const DEFAULT_PLATE_ALPHABET: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";

#[derive(Debug, Error)]
pub enum OcrError {
    #[error("ONNX Runtime error in OCR: {0}")]
    Ort(#[from] ort::Error),
    #[error("Failed to read OCR config YAML '{path}': {source}")]
    ConfigIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Failed to parse OCR config YAML '{path}': {source}")]
    ConfigParse {
        path: PathBuf,
        #[source]
        source: serde_yaml::Error,
    },
    #[error("OCR model has no input tensors")]
    MissingInput,
    #[error("OCR model has no output tensors")]
    MissingOutput,
    #[error("Invalid OCR alphabet: cannot be empty")]
    EmptyAlphabet,
    #[error("Unexpected OCR output shape {shape:?}; expected {expected}")]
    InvalidOutputShape { shape: Vec<i64>, expected: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrDecodingMode {
    /// Multi-slot classification (Fast-Plate-OCR: `[1, max_plate_slots, vocab]` or flattened).
    SlotSoftmax,
    /// Sequence CTC greedy decoding (CRNN: `[1, T, vocab]` or `[T, 1, vocab]`).
    CtcGreedy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrInputDtype {
    Uint8,
    Float32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrLayout {
    Nhwc,
    Nchw,
}

/// Mirrors `plate_config.yaml` exported by `fast-plate-ocr`.
#[derive(Debug, Clone, Deserialize)]
pub struct PlateYamlConfig {
    pub max_plate_slots: usize,
    pub alphabet: String,
    #[serde(default = "default_pad_char")]
    pub pad_char: char,
    pub img_height: u32,
    pub img_width: u32,
    #[serde(default = "default_true")]
    pub keep_aspect_ratio: bool,
    #[serde(default = "default_color_mode")]
    pub image_color_mode: String,
}

fn default_pad_char() -> char {
    '_'
}

fn default_true() -> bool {
    true
}

fn default_color_mode() -> String {
    "grayscale".to_string()
}

#[derive(Debug, Clone)]
pub struct OcrConfig {
    pub model_path: PathBuf,
    pub input_width: u32,
    pub input_height: u32,
    pub grayscale: bool,
    pub keep_aspect_ratio: bool,
    pub max_plate_slots: usize,
    pub alphabet: Vec<char>,
    pub pad_char: char,
    pub decoding_mode: OcrDecodingMode,
    pub input_dtype: OcrInputDtype,
    pub layout: OcrLayout,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/fast_plate_ocr.onnx"),
            input_width: 160,
            input_height: 64,
            grayscale: true,
            keep_aspect_ratio: true,
            max_plate_slots: 9,
            alphabet: DEFAULT_PLATE_ALPHABET.chars().collect(),
            pad_char: '_',
            decoding_mode: OcrDecodingMode::SlotSoftmax,
            input_dtype: OcrInputDtype::Uint8,
            layout: OcrLayout::Nhwc,
        }
    }
}

impl OcrConfig {
    /// Loads dimensions, alphabet, and color mode from a `fast-plate-ocr` YAML config file.
    pub fn apply_yaml_file(&mut self, path: &Path) -> Result<(), OcrError> {
        let raw = fs::read_to_string(path).map_err(|source| OcrError::ConfigIo {
            path: path.to_path_buf(),
            source,
        })?;
        let parsed: PlateYamlConfig =
            serde_yaml::from_str(&raw).map_err(|source| OcrError::ConfigParse {
                path: path.to_path_buf(),
                source,
            })?;

        if parsed.alphabet.is_empty() {
            return Err(OcrError::EmptyAlphabet);
        }

        self.max_plate_slots = parsed.max_plate_slots;
        self.alphabet = parsed.alphabet.chars().collect();
        self.pad_char = parsed.pad_char;
        self.input_height = parsed.img_height;
        self.input_width = parsed.img_width;
        self.keep_aspect_ratio = parsed.keep_aspect_ratio;
        self.grayscale = parsed.image_color_mode.eq_ignore_ascii_case("grayscale");
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct OcrPrediction {
    pub text: String,
    pub confidence: f32,
}

/// ONNX-backed license plate OCR engine supporting Fast-Plate-OCR and CRNN-CTC models.
pub struct PlateOcr {
    session: Session,
    input_name: String,
    config: OcrConfig,
}

impl PlateOcr {
    pub fn new(session: Session, config: OcrConfig) -> Result<Self, OcrError> {
        if config.alphabet.is_empty() {
            return Err(OcrError::EmptyAlphabet);
        }
        let input_name = session
            .inputs()
            .first()
            .ok_or(OcrError::MissingInput)?
            .name()
            .to_string();

        Ok(Self {
            session,
            input_name,
            config,
        })
    }

    pub fn recognize(&mut self, crop: &RgbImage) -> Result<OcrPrediction, OcrError> {
        let prepared = prepare_crop(
            crop,
            self.config.input_width,
            self.config.input_height,
            self.config.keep_aspect_ratio,
        );

        let channels: usize = if self.config.grayscale { 1 } else { 3 };
        let h = self.config.input_height as usize;
        let w = self.config.input_width as usize;

        let outputs = match (self.config.input_dtype, self.config.layout) {
            (OcrInputDtype::Uint8, OcrLayout::Nhwc) => {
                let data = encode_nhwc_u8(&prepared, self.config.grayscale);
                let tensor = Tensor::from_array(([1usize, h, w, channels], data))?;
                self.session
                    .run(ort::inputs![self.input_name.as_str() => tensor])?
            }
            (OcrInputDtype::Uint8, OcrLayout::Nchw) => {
                let data = encode_nchw_u8(&prepared, self.config.grayscale);
                let tensor = Tensor::from_array(([1usize, channels, h, w], data))?;
                self.session
                    .run(ort::inputs![self.input_name.as_str() => tensor])?
            }
            (OcrInputDtype::Float32, OcrLayout::Nhwc) => {
                let data: Vec<f32> = encode_nhwc_u8(&prepared, self.config.grayscale)
                    .into_iter()
                    .map(|v| f32::from(v) / 255.0)
                    .collect();
                let tensor = Tensor::from_array(([1usize, h, w, channels], data))?;
                self.session
                    .run(ort::inputs![self.input_name.as_str() => tensor])?
            }
            (OcrInputDtype::Float32, OcrLayout::Nchw) => {
                let data: Vec<f32> = encode_nchw_u8(&prepared, self.config.grayscale)
                    .into_iter()
                    .map(|v| f32::from(v) / 255.0)
                    .collect();
                let tensor = Tensor::from_array(([1usize, channels, h, w], data))?;
                self.session
                    .run(ort::inputs![self.input_name.as_str() => tensor])?
            }
        };

        if outputs.len() == 0 {
            return Err(OcrError::MissingOutput);
        }

        let (out_shape, out_data) = outputs[0].try_extract_tensor::<f32>()?;
        let dims: Vec<i64> = out_shape.iter().copied().collect();

        match self.config.decoding_mode {
            OcrDecodingMode::SlotSoftmax => decode_slot_softmax(
                &dims,
                out_data,
                &self.config.alphabet,
                self.config.pad_char,
                self.config.max_plate_slots,
            ),
            OcrDecodingMode::CtcGreedy => {
                decode_ctc_greedy(&dims, out_data, &self.config.alphabet, self.config.pad_char)
            }
        }
    }
}

fn prepare_crop(crop: &RgbImage, target_w: u32, target_h: u32, keep_aspect: bool) -> RgbImage {
    if !keep_aspect {
        return image::imageops::resize(crop, target_w, target_h, FilterType::Triangle);
    }

    let (src_w, src_h) = crop.dimensions();
    if src_w == 0 || src_h == 0 {
        return RgbImage::from_pixel(target_w, target_h, Rgb([114, 114, 114]));
    }

    let scale = (target_w as f32 / src_w as f32).min(target_h as f32 / src_h as f32);
    let new_w = ((src_w as f32) * scale).round().max(1.0) as u32;
    let new_h = ((src_h as f32) * scale).round().max(1.0) as u32;

    let resized = image::imageops::resize(crop, new_w, new_h, FilterType::Triangle);
    let mut canvas = RgbImage::from_pixel(target_w, target_h, Rgb([114, 114, 114]));

    let off_x = (target_w - new_w) / 2;
    let off_y = (target_h - new_h) / 2;
    image::imageops::overlay(&mut canvas, &resized, i64::from(off_x), i64::from(off_y));
    canvas
}

fn encode_nhwc_u8(img: &RgbImage, grayscale: bool) -> Vec<u8> {
    let (w, h) = img.dimensions();
    let channels = if grayscale { 1 } else { 3 };
    let mut out = Vec::with_capacity((w * h) as usize * channels);

    for pixel in img.pixels() {
        if grayscale {
            let gray = (0.299 * f32::from(pixel[0])
                + 0.587 * f32::from(pixel[1])
                + 0.114 * f32::from(pixel[2]))
            .round()
            .clamp(0.0, 255.0) as u8;
            out.push(gray);
        } else {
            out.extend_from_slice(&pixel.0);
        }
    }
    out
}

fn encode_nchw_u8(img: &RgbImage, grayscale: bool) -> Vec<u8> {
    let (w, h) = img.dimensions();
    let plane = (w * h) as usize;
    if grayscale {
        return encode_nhwc_u8(img, true);
    }

    let mut out = vec![0u8; 3 * plane];
    for (idx, pixel) in img.pixels().enumerate() {
        out[idx] = pixel[0];
        out[plane + idx] = pixel[1];
        out[2 * plane + idx] = pixel[2];
    }
    out
}

/// Decodes Fast-Plate-OCR slot-based predictions `[1, slots, vocab]` or `[1, slots * vocab]`.
fn decode_slot_softmax(
    dims: &[i64],
    data: &[f32],
    alphabet: &[char],
    pad_char: char,
    expected_slots: usize,
) -> Result<OcrPrediction, OcrError> {
    let vocab = alphabet.len();
    let (slots, classes) = match dims {
        [1, s, c] if *s > 0 && *c > 0 => (*s as usize, *c as usize),
        [1, flat] if *flat > 0 && (*flat as usize) % vocab == 0 => {
            ((*flat as usize) / vocab, vocab)
        }
        [1, flat] if *flat > 0 && expected_slots > 0 && (*flat as usize) % expected_slots == 0 => {
            (expected_slots, (*flat as usize) / expected_slots)
        }
        _ => {
            return Err(OcrError::InvalidOutputShape {
                shape: dims.to_vec(),
                expected: "[1, slots, vocab] or [1, slots * vocab]".to_string(),
            })
        }
    };

    let usable_classes = classes.min(vocab);
    let mut text = String::with_capacity(slots);
    let mut conf_sum = 0.0_f32;
    let mut conf_count = 0usize;

    for slot in 0..slots {
        let start = slot * classes;
        let slice = &data[start..start + usable_classes];
        let (best_idx, best_prob) = argmax_prob(slice);
        let ch = alphabet[best_idx];

        if ch != pad_char {
            text.push(ch);
            conf_sum += best_prob;
            conf_count += 1;
        }
    }

    let confidence = if conf_count > 0 {
        conf_sum / conf_count as f32
    } else {
        0.0
    };

    Ok(OcrPrediction { text, confidence })
}

/// Decodes CRNN-CTC predictions `[1, T, vocab]` or `[T, 1, vocab]` using greedy CTC collapse.
fn decode_ctc_greedy(
    dims: &[i64],
    data: &[f32],
    alphabet: &[char],
    blank_char: char,
) -> Result<OcrPrediction, OcrError> {
    let (time_steps, classes) = match dims {
        [1, t, c] if *t > 0 && *c > 0 => (*t as usize, *c as usize),
        [t, 1, c] if *t > 0 && *c > 0 => (*t as usize, *c as usize),
        _ => {
            return Err(OcrError::InvalidOutputShape {
                shape: dims.to_vec(),
                expected: "[1, T, vocab] or [T, 1, vocab]".to_string(),
            })
        }
    };

    let usable_classes = classes.min(alphabet.len());
    let mut text = String::new();
    let mut prev_idx: Option<usize> = None;
    let mut conf_sum = 0.0_f32;
    let mut conf_count = 0usize;

    for t in 0..time_steps {
        let start = t * classes;
        let slice = &data[start..start + usable_classes];
        let (best_idx, best_prob) = argmax_prob(slice);
        let ch = alphabet[best_idx];

        if ch != blank_char && prev_idx != Some(best_idx) {
            text.push(ch);
            conf_sum += best_prob;
            conf_count += 1;
        }
        prev_idx = Some(best_idx);
    }

    let confidence = if conf_count > 0 {
        conf_sum / conf_count as f32
    } else {
        0.0
    };

    Ok(OcrPrediction { text, confidence })
}

/// Returns `(argmax_index, probability)`. Handles both raw logits and pre-normalized probabilities.
fn argmax_prob(values: &[f32]) -> (usize, f32) {
    let mut max_idx = 0usize;
    let mut max_val = f32::NEG_INFINITY;
    let mut sum = 0.0_f32;
    let mut all_unit_interval = true;

    for (idx, &v) in values.iter().enumerate() {
        if !(0.0..=1.0).contains(&v) {
            all_unit_interval = false;
        }
        sum += v;
        if v > max_val {
            max_val = v;
            max_idx = idx;
        }
    }

    if all_unit_interval && (sum - 1.0).abs() < 0.02 {
        return (max_idx, max_val);
    }

    let mut exp_sum = 0.0_f32;
    for &v in values {
        exp_sum += (v - max_val).exp();
    }
    let prob = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };
    (max_idx, prob)
}
