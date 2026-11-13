#!/usr/bin/env python3
"""
Helper script to download and export the two ONNX models required by lpr-rs:
  1. Stage 1 (Detector): YOLOv8n / YOLO11n license-plate detector -> models/yolo_plate.onnx
  2. Stage 2 (OCR): Fast-Plate-OCR (cct-xs-v1-global) -> models/fast_plate_ocr.onnx + models/plate_config.yaml

Requirements:
  pip install ultralytics onnx onnxslim fast-plate-ocr
"""

import argparse
import shutil
from pathlib import Path


def export_yolo(weights: str, output_path: Path, imgsz: int, opset: int) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing 'ultralytics'. Install with: pip install ultralytics onnx onnxslim"
        ) from exc

    print(f"[1/2] Exporting YOLO model '{weights}' to ONNX (imgsz={imgsz}, opset={opset})...")
    model = YOLO(weights)
    exported = Path(model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True, dynamic=False))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), str(output_path))
    print(f"      Saved detector ONNX to: {output_path}")


def download_fast_plate_ocr(model_name: str, out_dir: Path) -> None:
    try:
        from fast_plate_ocr.inference.hub import download_model
    except ImportError as exc:
        raise SystemExit(
            "Missing 'fast-plate-ocr'. Install with: pip install fast-plate-ocr"
        ) from exc

    print(f"[2/2] Downloading Fast-Plate-OCR model '{model_name}'...")
    model_path, config_path = download_model(model_name=model_name)

    out_dir.mkdir(parents=True, exist_ok=True)
    dst_onnx = out_dir / "fast_plate_ocr.onnx"
    dst_yaml = out_dir / "plate_config.yaml"

    shutil.copy2(model_path, dst_onnx)
    shutil.copy2(config_path, dst_yaml)

    print(f"      Saved OCR ONNX to:   {dst_onnx}")
    print(f"      Saved OCR config to: {dst_yaml}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export/download ONNX models for lpr-rs"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models"),
        help="Destination directory for ONNX models (default: models)",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="yolo11n.pt",
        help="Path or name of fine-tuned YOLO plate weights (.pt). "
             "Replace with your license-plate fine-tuned checkpoint (e.g. license_plate_detector.pt).",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=640,
        help="YOLO ONNX input resolution (default: 640)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--ocr-model",
        type=str,
        default="cct-xs-v1-global-model",
        help="Fast-Plate-OCR hub model name (e.g. cct-xs-v1-global-model, cct-s-v1-global-model)",
    )
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Skip YOLO export (only download Fast-Plate-OCR)",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip Fast-Plate-OCR download (only export YOLO)",
    )
    args = parser.parse_args()

    if not args.skip_yolo:
        export_yolo(
            weights=args.yolo_weights,
            output_path=args.out_dir / "yolo_plate.onnx",
            imgsz=args.yolo_imgsz,
            opset=args.opset,
        )

    if not args.skip_ocr:
        download_fast_plate_ocr(
            model_name=args.ocr_model,
            out_dir=args.out_dir,
        )

    print("\nDone. You can now run:")
    print(
        f"  cargo run --release -- --input sample.jpg "
        f"--det-model {args.out_dir / 'yolo_plate.onnx'} "
        f"--ocr-model {args.out_dir / 'fast_plate_ocr.onnx'} "
        f"--ocr-config {args.out_dir / 'plate_config.yaml'}"
    )


if __name__ == "__main__":
    main()
