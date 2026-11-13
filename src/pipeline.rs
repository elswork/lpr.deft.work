use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use crossbeam_channel::{bounded, Receiver, Sender};
use image::RgbImage;
use ort::{
    environment::Environment,
    execution_providers::CPUExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
};
use rayon::prelude::*;
use tracing::{debug, info, warn};

use crate::detector::{DetectorConfig, PlateDetector};
use crate::ocr::{OcrConfig, PlateOcr};
use crate::postprocess::{BboxOutput, FrameOutput, PlatePostprocessor, PostprocessConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProviderKind {
    Auto,
    Cpu,
    Cuda,
    TensorRt,
    DirectMl,
    CoreMl,
}

pub struct PipelineConfig {
    pub detector: DetectorConfig,
    pub ocr: OcrConfig,
    pub postprocess: PostprocessConfig,
    pub execution_provider: ExecutionProviderKind,
    pub intra_threads: usize,
}

pub struct AlprPipeline {
    detector: PlateDetector,
    ocr: PlateOcr,
    postprocessor: PlatePostprocessor,
}

impl AlprPipeline {
    pub fn new(config: PipelineConfig) -> Result<Self> {
        let env = ort::init()
            .with_name("lpr_rs_ort")
            .build()
            .context("Failed to initialize ONNX Runtime environment")?;

        let det_session = build_ort_session(
            &env,
            &config.detector.model_path,
            config.execution_provider,
            config.intra_threads,
        )
        .with_context(|| {
            format!(
                "Failed to load detector ONNX model from '{}'",
                config.detector.model_path.display()
            )
        })?;

        let ocr_session = build_ort_session(
            &env,
            &config.ocr.model_path,
            config.execution_provider,
            config.intra_threads,
        )
        .with_context(|| {
            format!(
                "Failed to load OCR ONNX model from '{}'",
                config.ocr.model_path.display()
            )
        })?;

        let detector = PlateDetector::new(det_session, config.detector)?;
        let ocr = PlateOcr::new(ocr_session, config.ocr)?;
        let postprocessor = PlatePostprocessor::new(config.postprocess)?;

        Ok(Self {
            detector,
            ocr,
            postprocessor,
        })
    }

    /// Processes a single RGB frame through Stage 1 (YOLO), cropping/deskewing,
    /// Stage 2 (OCR), and regex/heuristic post-processing.
    pub fn process_frame(&mut self, img: &RgbImage, frame_id: u64) -> Result<FrameOutput> {
        let started = Instant::now();
        let boxes = self
            .detector
            .detect(img)
            .context("Stage 1 plate detection failed")?;

        // Crop and deskew all detected plates in parallel when multiple plates appear.
        let crops: Vec<RgbImage> = if boxes.len() > 1 {
            boxes
                .par_iter()
                .map(|bbox| self.detector.crop_plate(img, bbox))
                .collect()
        } else {
            boxes
                .iter()
                .map(|bbox| self.detector.crop_plate(img, bbox))
                .collect()
        };

        let mut detections = Vec::new();
        for (bbox, crop) in boxes.iter().zip(crops.iter()) {
            let ocr_pred = self
                .ocr
                .recognize(crop)
                .context("Stage 2 OCR inference failed")?;

            let bbox_out = BboxOutput {
                x1: bbox.x1,
                y1: bbox.y1,
                x2: bbox.x2,
                y2: bbox.y2,
            };

            if let Some(processed) = self.postprocessor.process(
                &ocr_pred.text,
                ocr_pred.confidence,
                bbox.confidence,
                bbox_out,
            ) {
                detections.push(processed);
            }
        }

        let inference_ms = started.elapsed().as_secs_f64() * 1000.0;
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Ok(FrameOutput {
            frame_id,
            timestamp_ms,
            inference_ms,
            detections,
        })
    }

    /// Processes a static image file from disk.
    pub fn process_image_file(&mut self, path: &Path) -> Result<FrameOutput> {
        let img = image::open(path)
            .with_context(|| format!("Failed to open image '{}'", path.display()))?
            .to_rgb8();
        self.process_frame(&img, 0)
    }

    /// Streams video or RTSP frames via `ffmpeg` using a bounded non-blocking channel
    /// so frame capture never stalls while inference runs.
    pub fn process_video_stream(
        &mut self,
        source: &str,
        width_override: Option<u32>,
        height_override: Option<u32>,
        only_hits: bool,
        pretty_json: bool,
        shutdown: Arc<AtomicBool>,
    ) -> Result<()> {
        let (width, height) = match (width_override, height_override) {
            (Some(w), Some(h)) => (w, h),
            _ => probe_video_dimensions(source).context(
                "Could not determine video dimensions via ffprobe; pass --width and --height explicitly",
            )?,
        };

        info!(%source, width, height, "Starting video/RTSP capture via ffmpeg");

        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-hide_banner", "-loglevel", "error"]);
        if source.starts_with("rtsp://") || source.starts_with("rtsps://") {
            cmd.args(["-rtsp_transport", "tcp"]);
        }
        cmd.args(["-i", source]);
        if width_override.is_some() || height_override.is_some() {
            cmd.args(["-vf", &format!("scale={width}:{height}")]);
        }
        cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .context("Failed to spawn ffmpeg; ensure ffmpeg is installed and on PATH")?;

        let stdout = child
            .stdout
            .take()
            .context("Failed to capture ffmpeg stdout")?;

        let frame_bytes = (width as usize) * (height as usize) * 3;
        let (tx, rx): (Sender<(u64, Vec<u8>)>, Receiver<(u64, Vec<u8>)>) = bounded(2);
        let reader_shutdown = Arc::clone(&shutdown);

        let reader_handle = std::thread::Builder::new()
            .name("ffmpeg-frame-reader".to_string())
            .spawn(move || {
                let mut reader = std::io::BufReader::new(stdout);
                let mut frame_id = 0_u64;

                while !reader_shutdown.load(Ordering::Relaxed) {
                    let mut buf = vec![0u8; frame_bytes];
                    if let Err(err) = reader.read_exact(&mut buf) {
                        debug!(%err, "FFmpeg stream ended or read failed");
                        break;
                    }

                    // Drop-oldest policy so real-time capture never blocks on slow inference.
                    if tx.try_send((frame_id, buf)).is_err() {
                        let _ = rx_drain_one_hint();
                        debug!(frame_id, "Dropped frame due to inference backpressure");
                    }
                    frame_id += 1;
                }
            })
            .context("Failed to spawn ffmpeg reader thread")?;

        while !shutdown.load(Ordering::Relaxed) {
            let Ok((frame_id, buf)) = rx.recv() else {
                break;
            };

            let Some(img) = RgbImage::from_raw(width, height, buf) else {
                warn!(frame_id, "Invalid raw frame buffer size");
                continue;
            };

            let result = self.process_frame(&img, frame_id)?;
            if !only_hits || !result.detections.is_empty() {
                let json = if pretty_json {
                    serde_json::to_string_pretty(&result)?
                } else {
                    serde_json::to_string(&result)?
                };
                println!("{json}");
            }
        }

        shutdown.store(true, Ordering::Relaxed);
        let _ = child.kill();
        let _ = child.wait();
        let _ = reader_handle.join();

        Ok(())
    }
}

#[inline]
fn rx_drain_one_hint() {}

fn build_ort_session(
    env: &Environment,
    model_path: &Path,
    ep: ExecutionProviderKind,
    intra_threads: usize,
) -> Result<Session> {
    let mut builder = Session::builder(env)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(intra_threads)?;

    let providers = collect_execution_providers(ep);
    builder = builder.with_execution_providers(providers)?;

    let session = builder.commit_from_file(model_path)?;
    Ok(session)
}

fn collect_execution_providers(
    ep: ExecutionProviderKind,
) -> Vec<ort::execution_providers::ExecutionProviderDispatch> {
    let mut providers = Vec::new();

    match ep {
        ExecutionProviderKind::Cpu => {
            providers.push(CPUExecutionProvider::default().build());
        }
        ExecutionProviderKind::Cuda => {
            #[cfg(feature = "cuda")]
            providers.push(ort::execution_providers::CUDAExecutionProvider::default().build());
            #[cfg(not(feature = "cuda"))]
            warn!("CUDA EP requested, but binary was compiled without 'cuda' feature; falling back to CPU");
            providers.push(CPUExecutionProvider::default().build());
        }
        ExecutionProviderKind::TensorRt => {
            #[cfg(feature = "tensorrt")]
            providers.push(ort::execution_providers::TensorRTExecutionProvider::default().build());
            #[cfg(not(feature = "tensorrt"))]
            warn!("TensorRT EP requested, but binary was compiled without 'tensorrt' feature; falling back to CPU");
            providers.push(CPUExecutionProvider::default().build());
        }
        ExecutionProviderKind::DirectMl => {
            #[cfg(feature = "directml")]
            providers.push(ort::execution_providers::DirectMLExecutionProvider::default().build());
            #[cfg(not(feature = "directml"))]
            warn!("DirectML EP requested, but binary was compiled without 'directml' feature; falling back to CPU");
            providers.push(CPUExecutionProvider::default().build());
        }
        ExecutionProviderKind::CoreMl => {
            #[cfg(feature = "coreml")]
            providers.push(ort::execution_providers::CoreMLExecutionProvider::default().build());
            #[cfg(not(feature = "coreml"))]
            warn!("CoreML EP requested, but binary was compiled without 'coreml' feature; falling back to CPU");
            providers.push(CPUExecutionProvider::default().build());
        }
        ExecutionProviderKind::Auto => {
            #[cfg(feature = "tensorrt")]
            providers.push(ort::execution_providers::TensorRTExecutionProvider::default().build());
            #[cfg(feature = "cuda")]
            providers.push(ort::execution_providers::CUDAExecutionProvider::default().build());
            #[cfg(feature = "directml")]
            providers.push(ort::execution_providers::DirectMLExecutionProvider::default().build());
            #[cfg(feature = "coreml")]
            providers.push(ort::execution_providers::CoreMLExecutionProvider::default().build());
            providers.push(CPUExecutionProvider::default().build());
        }
    }

    providers
}

fn probe_video_dimensions(source: &str) -> Result<(u32, u32)> {
    let mut cmd = Command::new("ffprobe");
    cmd.args(["-v", "error"]);
    if source.starts_with("rtsp://") || source.starts_with("rtsps://") {
        cmd.args(["-rtsp_transport", "tcp"]);
    }
    cmd.args([
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        source,
    ]);

    let output = cmd
        .output()
        .context("Failed to execute ffprobe; ensure ffmpeg/ffprobe is installed")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffprobe failed: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.trim();
    let (w_str, h_str) = line
        .split_once('x')
        .with_context(|| format!("Unexpected ffprobe output: '{line}'"))?;

    let w = w_str
        .trim()
        .parse::<u32>()
        .with_context(|| format!("Invalid width '{w_str}'"))?;
    let h = h_str
        .trim()
        .parse::<u32>()
        .with_context(|| format!("Invalid height '{h_str}'"))?;

    Ok((w, h))
}
