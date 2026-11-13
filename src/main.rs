mod detector;
mod ocr;
mod pipeline;
mod postprocess;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use tracing_subscriber::EnvFilter;

use crate::detector::DetectorConfig;
use crate::ocr::{OcrConfig, OcrDecodingMode, OcrInputDtype, OcrLayout};
use crate::pipeline::{AlprPipeline, ExecutionProviderKind, PipelineConfig};
use crate::postprocess::{PlatePattern, PostprocessConfig};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliEp {
    Auto,
    Cpu,
    Cuda,
    Tensorrt,
    Directml,
    Coreml,
}

impl From<CliEp> for ExecutionProviderKind {
    fn from(value: CliEp) -> Self {
        match value {
            CliEp::Auto => Self::Auto,
            CliEp::Cpu => Self::Cpu,
            CliEp::Cuda => Self::Cuda,
            CliEp::Tensorrt => Self::TensorRt,
            CliEp::Directml => Self::DirectMl,
            CliEp::Coreml => Self::CoreMl,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliOcrMode {
    /// Fast-Plate-OCR slot-based classification
    Slot,
    /// CRNN CTC greedy decoding
    Ctc,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliPattern {
    /// Spanish modern format: 0000 BBB (no vowels, Q, or Ñ)
    Es,
    /// Generic European alphanumeric (4..=10 chars)
    Eu,
    /// Custom regex provided via --custom-regex
    Custom,
}

#[derive(Debug, Parser)]
#[command(
    name = "lpr-rs",
    version,
    about = "Two-stage ALPR/ANPR system in Rust powered by ONNX Runtime (YOLO + Fast-Plate-OCR)"
)]
struct Cli {
    /// Input source: image file, video file, or rtsp:// stream URL
    #[arg(short, long)]
    input: String,

    /// Path to Stage 1 YOLO plate detector ONNX model
    #[arg(long, default_value = "models/yolo_plate.onnx")]
    det_model: PathBuf,

    /// Path to Stage 2 OCR ONNX model (Fast-Plate-OCR or CRNN)
    #[arg(long, default_value = "models/fast_plate_ocr.onnx")]
    ocr_model: PathBuf,

    /// Optional path to Fast-Plate-OCR plate_config.yaml
    #[arg(long)]
    ocr_config: Option<PathBuf>,

    /// OCR decoding mode
    #[arg(long, value_enum, default_value_t = CliOcrMode::Slot)]
    ocr_mode: CliOcrMode,

    /// Minimum detection confidence threshold (0.0 - 1.0)
    #[arg(long, default_value_t = 0.35)]
    det_conf: f32,

    /// Minimum OCR confidence threshold (0.0 - 1.0)
    #[arg(long, default_value_t = 0.50)]
    ocr_conf: f32,

    /// NMS IoU threshold (0.0 - 1.0)
    #[arg(long, default_value_t = 0.45)]
    iou_thresh: f32,

    /// Plate validation preset
    #[arg(long, value_enum, default_value_t = CliPattern::Es)]
    pattern: CliPattern,

    /// Custom regex used when --pattern=custom
    #[arg(long)]
    custom_regex: Option<String>,

    /// ONNX Runtime execution provider
    #[arg(long, value_enum, default_value_t = CliEp::Auto)]
    ep: CliEp,

    /// Number of intra-op CPU threads for ONNX Runtime
    #[arg(long, default_value_t = 4)]
    threads: usize,

    /// Disable lightweight plate deskewing before OCR
    #[arg(long, default_value_t = false)]
    no_deskew: bool,

    /// Disable positional confusable-character heuristics
    #[arg(long, default_value_t = false)]
    no_heuristics: bool,

    /// Only emit JSON lines for frames with at least one detection (video/RTSP mode)
    #[arg(long, default_value_t = false)]
    only_hits: bool,

    /// Pretty-print JSON output
    #[arg(long, default_value_t = false)]
    pretty: bool,

    /// Optional decoded video/stream width override
    #[arg(long)]
    width: Option<u32>,

    /// Optional decoded video/stream height override
    #[arg(long)]
    height: Option<u32>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    let pattern = match cli.pattern {
        CliPattern::Es => PlatePattern::Spanish,
        CliPattern::Eu => PlatePattern::EuGeneric,
        CliPattern::Custom => {
            let expr = cli
                .custom_regex
                .clone()
                .context("--custom-regex is required when --pattern=custom")?;
            PlatePattern::Custom(expr)
        }
    };

    let mut ocr_config = OcrConfig {
        model_path: cli.ocr_model.clone(),
        ..OcrConfig::default()
    };

    match cli.ocr_mode {
        CliOcrMode::Slot => {
            ocr_config.decoding_mode = OcrDecodingMode::SlotSoftmax;
            ocr_config.input_dtype = OcrInputDtype::Uint8;
            ocr_config.layout = OcrLayout::Nhwc;
        }
        CliOcrMode::Ctc => {
            ocr_config.decoding_mode = OcrDecodingMode::CtcGreedy;
            ocr_config.input_dtype = OcrInputDtype::Float32;
            ocr_config.layout = OcrLayout::Nchw;
        }
    }

    if let Some(yaml_path) = cli.ocr_config.as_deref() {
        ocr_config.apply_yaml_file(yaml_path)?;
    }

    let config = PipelineConfig {
        detector: DetectorConfig {
            model_path: cli.det_model.clone(),
            conf_threshold: cli.det_conf,
            iou_threshold: cli.iou_thresh,
            enable_deskew: !cli.no_deskew,
            ..DetectorConfig::default()
        },
        ocr: ocr_config,
        postprocess: PostprocessConfig {
            pattern,
            min_ocr_confidence: cli.ocr_conf,
            enable_heuristics: !cli.no_heuristics,
        },
        execution_provider: cli.ep.into(),
        intra_threads: cli.threads,
    };

    let is_stream = is_video_or_stream(&cli.input);
    let input = cli.input.clone();
    let pretty = cli.pretty;
    let only_hits = cli.only_hits;
    let width = cli.width;
    let height = cli.height;

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_signal = Arc::clone(&shutdown);
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            shutdown_signal.store(true, Ordering::Relaxed);
        }
    });

    tokio::task::spawn_blocking(move || -> Result<()> {
        let mut pipeline = AlprPipeline::new(config)?;

        if is_stream {
            pipeline.process_video_stream(&input, width, height, only_hits, pretty, shutdown)?;
        } else {
            let path = Path::new(&input);
            if !path.exists() {
                bail!("Input file does not exist: '{}'", path.display());
            }
            let result = pipeline.process_image_file(path)?;
            let json = if pretty {
                serde_json::to_string_pretty(&result)?
            } else {
                serde_json::to_string(&result)?
            };
            println!("{json}");
        }

        Ok(())
    })
    .await
    .context("Inference task panicked")??;

    Ok(())
}

fn is_video_or_stream(input: &str) -> bool {
    let lower = input.to_ascii_lowercase();
    if lower.starts_with("rtsp://")
        || lower.starts_with("rtsps://")
        || lower.starts_with("http://")
        || lower.starts_with("https://")
    {
        return true;
    }

    matches!(
        Path::new(&lower).extension().and_then(|e| e.to_str()),
        Some("mp4" | "mkv" | "avi" | "mov" | "webm" | "ts" | "m4v")
    )
}
