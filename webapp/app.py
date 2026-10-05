from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
MODELS_DIR = REPO_ROOT / "models"
TEMPLATES_DIR = BASE_DIR / "templates"
INDEX_PATH = TEMPLATES_DIR / "index.html"

YOLO_PATH = Path(os.getenv(
    "PLANTDOC_YOLO_PATH", str(MODELS_DIR / "yolo_leaf_detector_best.pt")
))
FP16_PATH = Path(os.getenv(
    "PLANTDOC_FP16_PATH", str(MODELS_DIR / "efficientnet_b0_exp010e_fp16.onnx")
))
FP32_PATH = Path(os.getenv(
    "PLANTDOC_FP32_PATH", str(MODELS_DIR / "efficientnet_b0_exp010e_fp32.onnx")
))

CLASS_ORDER_ENV = os.getenv("PLANTDOC_CLASS_ORDER_PATH")
if CLASS_ORDER_ENV and Path(CLASS_ORDER_ENV).exists():
    CLASS_ORDER_PATH = Path(CLASS_ORDER_ENV)
elif (REPO_ROOT / "configs" / "class_order.json").exists():
    CLASS_ORDER_PATH = REPO_ROOT / "configs" / "class_order.json"
else:
    CLASS_ORDER_PATH = MODELS_DIR / "class_order.json"

IMAGE_SIZE = 640
YOLO_CONFIDENCE = 0.25
MIN_CLASSIFIER_CONFIDENCE = 0.45
MIN_TOP2_MARGIN = 0.05
CLASSIFIER_SIZE = 224
MAX_UPLOAD_MB = 15
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(TEMPLATES_DIR))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def parse_class_names(data: Any) -> list[str]:
    """Read supported class-order JSON formats without changing index order."""
    if isinstance(data, list):
        names = [str(item) for item in data]
        if names and all(name.strip() for name in names):
            return names

    if isinstance(data, dict):
        for key in ("class_names", "classes", "class_order", "class_list", "labels"):
            if isinstance(data.get(key), list):
                return [str(item) for item in data[key]]

        for key in ("class_to_idx", "class_to_index", "label_to_idx", "class_to_id"):
            mapping = data.get(key)
            if isinstance(mapping, dict):
                try:
                    return [
                        str(name) for name, _ in sorted(
                            mapping.items(), key=lambda item: int(item[1])
                        )
                    ]
                except (TypeError, ValueError):
                    pass

        if data and all(isinstance(value, int) for value in data.values()):
            return [
                str(name) for name, _ in sorted(
                    data.items(), key=lambda item: int(item[1])
                )
            ]

        if data and all(str(key).isdigit() for key in data.keys()):
            return [str(data[key]) for key in sorted(data, key=lambda key: int(key))]

    raise ValueError(
        f"Unsupported class-order JSON structure in {CLASS_ORDER_PATH}. "
        "Expected a list of labels or an indexed dictionary."
    )


def display_class_name(class_key: str) -> str:
    """Format a raw PlantDoc label for the UI without changing its identity."""
    return str(class_key).replace("_", " ").strip().title()


HEALTHY_CLASS_KEYS = {
    "Apple_leaf",
    "Bell_pepper_leaf",
    "Blueberry_leaf",
    "Cherry_leaf",
    "Peach_leaf",
    "Raspberry_leaf",
    "Soyabean_leaf",
    "Strawberry_leaf",
    "Tomato_leaf",
    "grape_leaf",
}


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    denom = float(exp_values.sum())
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("Classifier returned invalid scores.")
    return exp_values / denom


def choose_onnx_session() -> tuple[ort.InferenceSession, str, str, str, list[str]]:
    """Prefer the FP16 artifact; fall back to the exported FP32 artifact if required."""
    available = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )

    candidates: list[tuple[Path, str]] = []
    if FP16_PATH.exists():
        candidates.append((FP16_PATH, "FP16 ONNX"))
    if FP32_PATH.exists():
        candidates.append((FP32_PATH, "FP32 ONNX fallback"))

    if not candidates:
        raise FileNotFoundError(
            "No EfficientNet ONNX model was found. Expected one of:\n"
            f"  {FP16_PATH}\n  {FP32_PATH}\n"
            "Run export_models_from_drive.py in Google Colab and extract the model ZIP."
        )

    errors = []
    for model_path, precision_name in candidates:
        try:
            session = ort.InferenceSession(str(model_path), providers=providers)
            input_meta = session.get_inputs()[0]
            output_meta = session.get_outputs()[0]
            if len(output_meta.shape) != 2:
                raise ValueError(f"Unexpected ONNX output shape: {output_meta.shape}")
            if output_meta.shape[-1] not in (None, "num_classes") and int(output_meta.shape[-1]) != len(CLASS_NAMES):
                raise ValueError(
                    f"ONNX output has {output_meta.shape[-1]} classes, "
                    f"but class_order.json contains {len(CLASS_NAMES)} labels."
                )
            test_input = np.zeros((1, 3, CLASSIFIER_SIZE, CLASSIFIER_SIZE), dtype=np.float32)
            test_output = session.run(
                [output_meta.name], {input_meta.name: test_input}
            )[0]
            if test_output.shape != (1, len(CLASS_NAMES)):
                raise ValueError(f"Unexpected smoke-test output shape: {test_output.shape}")
            if not np.isfinite(test_output).all():
                raise ValueError("ONNX smoke test returned non-finite output values.")
            return session, input_meta.name, output_meta.name, precision_name, session.get_providers()
        except Exception as exc:
            errors.append(f"{model_path.name}: {exc}")

    raise RuntimeError(
        "Could not start the EfficientNet ONNX model. Tried available artifacts:\n- "
        + "\n- ".join(errors)
        + "\nIf the FP16 model is unsupported on this machine, ensure the FP32 ONNX fallback is present."
    )


yolo_model = None
classifier_session = None
classifier_input_name = ""
classifier_output_name = ""
classifier_precision = "Not Loaded"
active_providers = []
CLASS_NAMES = []

if CLASS_ORDER_PATH.exists():
    with CLASS_ORDER_PATH.open("r", encoding="utf-8") as handle:
        CLASS_NAMES = parse_class_names(json.load(handle))

if YOLO_PATH.exists() and CLASS_ORDER_PATH.exists():
    print("[PlantDoc Edge] Loading YOLO detector...")
    yolo_model = YOLO(str(YOLO_PATH))

    print("[PlantDoc Edge] Loading EfficientNet-B0 ONNX classifier...")
    classifier_session, classifier_input_name, classifier_output_name, classifier_precision, active_providers = choose_onnx_session()
    print(f"[PlantDoc Edge] Classifier: {classifier_precision}")
    print(f"[PlantDoc Edge] ONNX providers: {active_providers}")
    print(f"[PlantDoc Edge] Loaded {len(CLASS_NAMES)} class labels.")


def prepare_classifier_input(crop_rgb: np.ndarray) -> np.ndarray:
    crop_image = Image.fromarray(crop_rgb).resize(
        (CLASSIFIER_SIZE, CLASSIFIER_SIZE),
        resample=Image.Resampling.BILINEAR,
    )
    values = np.asarray(crop_image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    values = (values - mean) / std
    values = np.transpose(values, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(values, dtype=np.float32)


def encode_crop_jpeg(crop_rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(crop_rgb).save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@app.get("/")
def index():
    return send_from_directory(TEMPLATES_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "application": "PlantDoc Edge",
        "detector_loaded": yolo_model is not None,
        "classifier_loaded": classifier_session is not None,
        "classifier_precision": classifier_precision,
        "onnx_providers": active_providers,
        "num_classes": len(CLASS_NAMES),
    })


@app.get("/api/info")
def info():
    return jsonify({
        "application": "PlantDoc Edge",
        "architecture": "YOLO leaf localization + EfficientNet-B0 ONNX classification",
        "classifier_precision": classifier_precision,
        "num_classes": len(CLASS_NAMES),
        "image_size": CLASSIFIER_SIZE,
        "yolo_image_size": IMAGE_SIZE,
        "yolo_confidence_threshold": YOLO_CONFIDENCE,
        "classes": [
            {"key": key, "name": display_class_name(key)}
            for key in CLASS_NAMES
        ],
    })


@app.post("/api/predict")
def predict():
    if yolo_model is None or classifier_session is None:
        return jsonify({"error": "Model files are not loaded yet. Place model weights in models/ folder."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file was uploaded."}), 400

    uploaded = request.files["image"]
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Choose an image file before analyzing."}), 400

    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": "Unsupported file type. Upload JPG, PNG, WEBP, BMP, or TIFF."
        }), 415

    total_start = time.perf_counter()
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            tmp_path = Path(temp_file.name)
        uploaded.save(str(tmp_path))

        try:
            with Image.open(tmp_path) as check_image:
                check_image.verify()
            with Image.open(tmp_path) as loaded_image:
                original = loaded_image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            return jsonify({"error": f"The uploaded file is not a valid image: {exc}"}), 400

        original_rgb = np.asarray(original)
        height, width = original_rgb.shape[:2]

        detection_start = time.perf_counter()
        yolo_result = yolo_model.predict(
            source=str(tmp_path),
            imgsz=IMAGE_SIZE,
            conf=YOLO_CONFIDENCE,
            device=0 if _torch_cuda_available() else "cpu",
            verbose=False,
        )[0]
        detection_time_ms = (time.perf_counter() - detection_start) * 1000.0

        boxes = yolo_result.boxes
        if boxes is None or len(boxes) == 0:
            return jsonify({
                "error": "Analysis unsuccessful: We could not find a clear plant leaf. Please upload a close-up photo of a single leaf and try again.",
                "stage": "localization",
                "detection_time_ms": round(detection_time_ms, 2),
            }), 422

        confidences = boxes.conf.detach().float().cpu().numpy()
        selected_index = int(np.argmax(confidences))
        yolo_confidence = float(confidences[selected_index])
        coords = boxes.xyxy[selected_index].detach().float().cpu().numpy()

        x1, y1, x2, y2 = [int(value) for value in coords]
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return jsonify({
                "error": "Analysis unsuccessful: The leaf could not be isolated clearly. Please try a sharper photo showing one leaf.",
                "stage": "crop",
            }), 422

        crop_rgb = original_rgb[y1:y2, x1:x2]
        if crop_rgb.size == 0:
            return jsonify({"error": "Analysis unsuccessful: We could not isolate the leaf. Please upload a clear photo of a single plant leaf.", "stage": "crop"}), 422

        classification_start = time.perf_counter()
        input_tensor = prepare_classifier_input(crop_rgb)
        logits = classifier_session.run(
            [classifier_output_name],
            {classifier_input_name: input_tensor},
        )[0][0]
        probabilities = softmax(logits)
        classification_time_ms = (time.perf_counter() - classification_start) * 1000.0

        if len(probabilities) != len(CLASS_NAMES):
            return jsonify({"error": "Classifier output count does not match class labels."}), 500

        order = np.argsort(probabilities)[::-1][:3]
        predicted_index = int(order[0])
        raw_label = CLASS_NAMES[predicted_index]
        top_confidence = float(probabilities[order[0]])
        second_confidence = float(probabilities[order[1]]) if len(order) > 1 else 0.0
        top2_margin = top_confidence - second_confidence
        top_predictions = [
            {
                "class_key": CLASS_NAMES[int(index)],
                "class_name": display_class_name(CLASS_NAMES[int(index)]),
                "confidence": round(float(probabilities[index]) * 100.0, 2),
            }
            for index in order
        ]
        total_time_ms = (time.perf_counter() - total_start) * 1000.0

        if top_confidence < MIN_CLASSIFIER_CONFIDENCE or top2_margin < MIN_TOP2_MARGIN:
            return jsonify({
                "error": "Analysis unsuccessful: We could not identify the leaf confidently. Please upload a clear, close-up photo of one plant leaf, with the leaf in focus and filling most of the image.",
                "stage": "classification",
                "reason": "low_confidence",
                "confidence": round(top_confidence * 100.0, 2),
                "top_predictions": top_predictions,
                "yolo_confidence": round(yolo_confidence * 100.0, 2),
                "detection_time_ms": round(detection_time_ms, 2),
                "classification_time_ms": round(classification_time_ms, 2),
                "total_time_ms": round(total_time_ms, 2),
            }), 422

        is_healthy = raw_label in HEALTHY_CLASS_KEYS
        diagnosis = (
            f"Healthy {display_class_name(raw_label)}"
            if is_healthy else display_class_name(raw_label)
        )

        return jsonify({
            "status": "healthy" if is_healthy else "success",
            "result_type": "healthy_leaf" if is_healthy else "disease_prediction",
            "diagnosis": diagnosis,
            "class_key": raw_label,
            "confidence": round(top_confidence * 100.0, 2),
            "yolo_confidence": round(yolo_confidence * 100.0, 2),
            "box": [x1, y1, x2, y2],
            "detection_time_ms": round(detection_time_ms, 2),
            "classification_time_ms": round(classification_time_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
            "classifier_precision": classifier_precision,
            "top_predictions": top_predictions,
            "crop_preview": encode_crop_jpeg(crop_rgb),
            "image_width": width,
            "image_height": height,
            "note": (
                "Healthy status is inferred from a generic leaf class label; verify these labels against original training annotations."
                if is_healthy else
                "This is a model prediction for research/demo purposes, not an agricultural diagnosis or treatment recommendation."
            ),
        })

    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "We could not complete the analysis. Please try a clear plant-leaf image again.",
            "detail": str(exc),
        }), 500
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": f"Image is too large. Maximum upload size is {MAX_UPLOAD_MB} MB."}), 413


if __name__ == "__main__":
    print("\nPlantDoc Edge local web application")
    print("Open: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server.\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
