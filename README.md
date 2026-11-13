# lpr-rs — Sistema ALPR / ANPR de Alto Rendimiento en Rust

Pipeline nativo en **dos etapas** para Reconocimiento Automático de Matrículas (ALPR/ANPR) construido en Rust sobre **ONNX Runtime (`ort`)**, sin dependencias pesadas de OpenCV C++ ni Tesseract.

## Arquitectura del Pipeline

```text
Imagen / Vídeo / RTSP (ffmpeg rgb24 + canal acotado drop-oldest)
        │
        ▼
[Etapa 1: src/detector.rs]
  • Letterboxing NCHW [1, 3, 640, 640] (f32 [0, 1])
  • Inferencia YOLOv8 / YOLO11 (ONNX)
  • Non-Maximum Suppression (NMS)
  • Recorte con margen + Deskewing ligero (imageproc)
        │
        ▼
[Etapa 2: src/ocr.rs]
  • Preprocesado NHWC uint8 (Fast-Plate-OCR) o NCHW f32 (CRNN)
  • Inferencia Fast-Plate-OCR (cct-xs-v1-global) / CRNN
  • Decodificación SlotSoftmax o CTC Greedy
        │
        ▼
[Post-procesamiento: src/postprocess.rs]
  • Filtrado por confianza mínima
  • Heurísticas posicionales (0↔O, 8↔B, 1↔I, 5↔S, etc.)
  • Validación Regex (por defecto España: 0000 BBB sin vocales, Q ni Ñ)
  • Salida estructurada en JSON / NDJSON
```

## Requisitos del Sistema

- **Rust** (`rustup` / `cargo` >= 1.78, edición 2021)
- **FFmpeg & FFprobe** en el `PATH` (necesario únicamente para vídeo y streams RTSP)
- **Python 3.9+** (solo para ejecutar `scripts/export_models.py` una vez)
- *(Opcional)* CUDA / TensorRT / DirectML / CoreML según la aceleración hardware deseada

## Descarga y Exportación de Modelos ONNX

Instala las dependencias Python en un entorno virtual y ejecuta el script auxiliar:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics onnx onnxslim fast-plate-ocr

python3 scripts/export_models.py \
  --yolo-weights /ruta/a/tu_modelo_matriculas.pt \
  --ocr-model cct-xs-v1-global-model \
  --out-dir models
```

Esto generará:
- `models/yolo_plate.onnx`
- `models/fast_plate_ocr.onnx`
- `models/plate_config.yaml`

> **Nota:** `yolo11n.pt` por defecto es el modelo base COCO; para detección de matrículas en producción debes pasar tus pesos fine-tuned (`.pt`) de matrícula a `--yolo-weights`.

## Compilación

### CPU (por defecto)
```bash
cargo build --release
```

### Con aceleración por hardware
```bash
# NVIDIA CUDA
cargo build --release --features cuda

# NVIDIA TensorRT
cargo build --release --features tensorrt

# Windows DirectML
cargo build --release --features directml

# macOS CoreML
cargo build --release --features coreml
```

## Ejemplos de Ejecución

### 1. Imagen estática (formato España por defecto)
```bash
./target/release/lpr-rs \
  --input coche.jpg \
  --det-model models/yolo_plate.onnx \
  --ocr-model models/fast_plate_ocr.onnx \
  --ocr-config models/plate_config.yaml \
  --pretty
```

### 2. Stream RTSP en tiempo real con CUDA y solo detecciones válidas
```bash
./target/release/lpr-rs \
  --input rtsp://usuario:clave@192.168.1.50:554/stream1 \
  --det-model models/yolo_plate.onnx \
  --ocr-model models/fast_plate_ocr.onnx \
  --ocr-config models/plate_config.yaml \
  --ep cuda \
  --det-conf 0.40 \
  --ocr-conf 0.55 \
  --only-hits
```

### 3. Vídeo local con patrón Regex personalizado
```bash
./target/release/lpr-rs \
  --input trafico.mp4 \
  --pattern custom \
  --custom-regex '^[A-Z]{2}\d{3}[A-Z]{2}$' \
  --only-hits
```

## Tests unitarios
```bash
cargo test
```
