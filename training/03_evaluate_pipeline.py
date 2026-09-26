import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from ultralytics import YOLO

YOLO_PATH = Path("models/yolo_leaf_detector_best.pt")
ONNX_PATH = Path("models/efficientnet_b0_exp010e_fp16.onnx")
CLASS_PATH = Path("configs/class_order.json")
DATASET_PATH = Path("datasets/plantdoc_validation")
IMAGE_SIZE = 640
CONFIDENCE = 0.25
INPUT_SIZE = 224

class_order = json.loads(CLASS_PATH.read_text())
class_to_idx = {name: i for i, name in enumerate(class_order)}

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(crop):
    image = crop.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - mean) / std
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def predict_image(yolo, session, input_name, path):
    image = Image.open(path).convert("RGB")
    result = yolo.predict(
        source=np.asarray(image),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    confidences = result.boxes.conf.cpu().numpy()
    best_index = int(np.argmax(confidences))
    x1, y1, x2, y2 = result.boxes.xyxy[best_index].cpu().numpy().astype(int)
    x1 = max(0, min(x1, image.width))
    x2 = max(0, min(x2, image.width))
    y1 = max(0, min(y1, image.height))
    y2 = max(0, min(y2, image.height))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image.crop((x1, y1, x2, y2))
    logits = session.run(None, {input_name: preprocess(crop)})[0][0]
    return int(np.argmax(logits))


def main():
    if not YOLO_PATH.exists() or not ONNX_PATH.exists():
        print("Required model files missing. Ensure models are exported before evaluation.")
        return

    if not DATASET_PATH.exists():
        print(f"Validation dataset path {DATASET_PATH} not found.")
        return

    yolo = YOLO(str(YOLO_PATH))
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    y_true, y_pred = [], []
    misses = 0

    for folder in sorted(DATASET_PATH.iterdir()):
        if not folder.is_dir() or folder.name not in class_to_idx:
            continue
        label = class_to_idx[folder.name]
        for path in folder.iterdir():
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            try:
                prediction = predict_image(yolo, session, input_name, path)
            except Exception as exc:
                print(f"Could not process {path.name}: {exc}")
                prediction = None
            y_true.append(label)
            if prediction is None:
                misses += 1
                y_pred.append(-1)
            else:
                y_pred.append(prediction)

    if not y_true:
        raise RuntimeError("No labeled validation images were found.")

    coverage = 100 * (len(y_true) - misses) / len(y_true)
    accuracy = 100 * accuracy_score(y_true, y_pred)
    macro_f1 = 100 * f1_score(
        y_true,
        y_pred,
        labels=list(range(len(class_order))),
        average="macro",
        zero_division=0,
    )

    print(f"Images evaluated: {len(y_true)}")
    print(f"Detection misses: {misses}")
    print(f"Detection coverage: {coverage:.2f}%")
    print(f"End-to-end accuracy: {accuracy:.2f}%")
    print(f"Macro-F1: {macro_f1:.2f}%")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_order))),
            target_names=class_order,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
