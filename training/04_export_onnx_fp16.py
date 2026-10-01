import json
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from onnxconverter_common import float16
from torchvision import models

CHECKPOINT = Path("outputs/efficientnet_b0/efficientnet_b0_best.pth")
CLASS_ORDER = Path("configs/class_order.json")
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_CLASSES = 27
INPUT_SIZE = 224


def main():
    class_order = json.loads(CLASS_ORDER.read_text())
    if len(class_order) != NUM_CLASSES:
        raise ValueError("Expected exactly 27 class labels.")

    if not CHECKPOINT.exists():
        print(f"Checkpoint file {CHECKPOINT} does not exist. Train model first before exporting.")
        return

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    fp32_path = OUTPUT_DIR / "efficientnet_b0_fp32.onnx"
    fp16_path = OUTPUT_DIR / "efficientnet_b0_fp16.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(fp32_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )

    onnx_fp32 = onnx.load(str(fp32_path))
    onnx_fp16 = float16.convert_float_to_float16(onnx_fp32, keep_io_types=True)
    onnx.save(onnx_fp16, str(fp16_path))

    print("FP32 ONNX export succeeded:", fp32_path)
    print("FP16 ONNX export succeeded:", fp16_path)
    print("Class order configuration used:", CLASS_ORDER)


if __name__ == "__main__":
    main()
