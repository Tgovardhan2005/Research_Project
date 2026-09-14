# PlantDoc Edge

An edge-oriented plant leaf analysis system that combines YOLO leaf
localization with EfficientNet-B0 classification.

## Pipeline

Input image
    |
    v
YOLO leaf localization
    |
    v
Highest-confidence bounding box
    |
    v
Crop and resize to 224 x 224 RGB
    |
    v
ImageNet normalization
    |
    v
EfficientNet-B0 classifier
    |
    v
One of 27 class predictions

## Project structure

- `training/01_train_yolo_detector.py`: YOLO detector training
- `training/02_train_efficientnet_b0.py`: EfficientNet-B0 training
- `training/03_evaluate_pipeline.py`: End-to-end evaluation
- `training/04_export_onnx_fp16.py`: ONNX FP32 and FP16 export
- `training/05_gradcam_explainability.py`: Grad-CAM visualization
- `configs/class_order.json`: The 27-class label order
- `configs/data.yaml`: YOLO dataset configuration
- `webapp/`: Flask inference application and user interface

## Installation

```bash
python -m venv .venv
```

Windows:

```cmd
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset preparation

Prepare the annotated YOLO dataset with train, validation, and test splits.

Prepare the classification dataset with one directory per class.

Keep the classification label order consistent with `configs/class_order.json`.

Do not use the same images for training and independent testing.

## Training

Train the detector:

```bash
python training/01_train_yolo_detector.py
```

Train the classifier:

```bash
python training/02_train_efficientnet_b0.py
```

Evaluate the pipeline:

```bash
python training/03_evaluate_pipeline.py
```

Export ONNX models:

```bash
python training/04_export_onnx_fp16.py
```

Generate Grad-CAM:

```bash
python training/05_gradcam_explainability.py
```

## Evaluation configuration

The final deployment pipeline uses:

- YOLO input size: 640
- YOLO confidence threshold: 0.25
- Bounding box: highest-confidence detection
- Classifier input: 224 x 224 RGB
- Classifier: EfficientNet-B0
- Class count: 27
- Deployment classifier: FP16-weight ONNX

The project validation results previously recorded were approximately 71.76% end-to-end accuracy and 71.11% macro-F1 on 262 validation images. These are project validation results, not an independent external benchmark or a guarantee that a newly trained model will reproduce them.

## Limitations

- The classifier does not have a dedicated healthy-leaf class in its current 27-class label set.
- Confidence thresholds alone cannot guarantee rejection of unrelated images.
- Edge-device latency and power consumption require measurements on actual target hardware.
- The provided training scripts are reusable implementations and may require adjustment to reproduce the original experiment exactly.

## Model files

Large model weights and datasets are not committed to this repository. Provide separate download instructions for authorized model artifacts.

## License

Add the license appropriate for your code and datasets before publishing.
