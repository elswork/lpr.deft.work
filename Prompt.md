Actúa como un Arquitecto de Software Senior especializado en Visión por Computador (Computer Vision) de alto rendimiento y desarrollo de sistemas en Rust.

Quiero desarrollar desde cero un **sistema de Reconocimiento Automático de Matrículas (ALPR / ANPR) en Rust**, utilizando las técnicas y arquitecturas más avanzadas y eficientes del estado del arte actual.

### 1. Stack Tecnológico y Arquitectura Requerida
Diseña la solución utilizando un pipeline moderno en dos etapas (Two-Stage Detection + OCR) ejecutado de forma nativa en Rust mediante inferencia de modelos ONNX:
- **Motor de Inferencia:** `ort` (ONNX Runtime bindings para Rust) con soporte preparado para aceleración por hardware (CUDA/TensorRT/DirectML/CoreML) y fallback automático a CPU.
- **Procesamiento de Imagen:** `image` y `imageproc` (o bindings de `opencv` solo si es estrictamente necesario, priorizando Rust puro para facilitar la compilación cruzada).
- **Etapa 1 - Detección de Matrículas:** Integración de un detector de objetos moderno y ligero (como **YOLOv8 / YOLO11** exportado a ONNX) para localizar el bounding box de la matrícula y aplicar recortes (cropping) con margen y corrección de perspectiva/inclinación (deskewing) si es viable.
- **Etapa 2 - Reconocimiento de Caracteres (OCR):** Integración de un modelo OCR especializado en matrículas (como **Fast-Plate-OCR**, **PARSeq** o **CRNN** en formato ONNX) evitando motores OCR genéricos pesados como Tesseract.
- **Post-procesamiento:** Limpieza de texto, filtrado por confianza (confidence score) y validación mediante expresiones regulares (`regex`), optimizado por defecto para el formato de matrículas europeas/españolas (`0000 BBB`, excluyendo vocales y Q/Ñ), pero configurable para otros países.
- **Concurrencia / Streaming:** Arquitectura asíncrona o multihilo (`tokio` o `rayon` / canales `crossbeam`) capaz de procesar tanto imágenes estáticas como flujos de vídeo RTSP / archivos de vídeo en tiempo real sin bloquear la captura de fotogramas.

### 2. Estructura del Proyecto a Generar
Crea un workspace o crate limpio y modular con la siguiente separación de responsabilidades:
- `Cargo.toml`: Todas las dependencias necesarias con versiones actualizadas y features bien configuradas.
- `src/main.rs`: CLI funcional (usando `clap`) que permita pasar una imagen, un vídeo o un stream RTSP, además de rutas a los modelos `.onnx` y umbral de confianza.
- `src/pipeline.rs`: Orquestador del flujo completo (preprocesamiento -> detección -> recorte -> OCR -> validación).
- `src/detector.rs`: Lógica de inferencia del modelo YOLO (letterboxing/resizing, normalización de tensores, ejecución en `ort` y Non-Maximum Suppression - NMS).
- `src/ocr.rs`: Lógica de inferencia del modelo OCR (preprocesamiento del recorte, decodificación de logits/CTC o greedy decoding).
- `src/postprocess.rs`: Validación por Regex, heurísticas de corrección de caracteres confundibles (ej. `0` vs `O`, `8` vs `B` según posición en el patrón) y estructuración del resultado en JSON.
- `scripts/export_models.py`: Un script auxiliar en Python con las instrucciones exactas para descargar/exportar los modelos preentrenados recomendados a formato `.onnx` listos para ser consumidos por el binario en Rust.
- `README.md`: Guía completa de instalación, requisitos del sistema, descarga/exportación de modelos y ejemplos de ejecución.

### 3. Plan de Ejecución
1. Antes de escribir todo el código, preséntame brevemente la arquitectura propuesta y confirma qué modelos ONNX exactos vas a configurar en el pipeline.
2. Genera la estructura completa de archivos y el `Cargo.toml`.
3. Implementa cada módulo asegurando un manejo de errores robusto (usando `anyhow` / `thiserror`), cero `unwrap()` en producción y código idiomático en Rust.
4. Verifica que el proyecto compila limpiamente (`cargo check` / `cargo build`) y corrige cualquier advertencia del compilador o incompatibilidad de versiones en los crates.