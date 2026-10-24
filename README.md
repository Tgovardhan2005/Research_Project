
<div align="center">

# 🌿 PlantDoc Edge

*An edge-oriented plant leaf disease analysis system combining YOLO-based leaf localization, EfficientNet-B0 classification, and Grad-CAM explainability.*

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Object_Detection-111F68?style=flat-square)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![EfficientNet](https://img.shields.io/badge/EfficientNet-B0-4285F4?style=flat-square)](https://arxiv.org/abs/1905.11946)
[![Flask](https://img.shields.io/badge/Flask-Web_App-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)

</div>

---

## 🌱 Overview

**PlantDoc Edge** is a deep learning-based plant leaf disease analysis system designed to recognize diseases from leaf images, including images captured under challenging real-world conditions.

The system uses a two-stage pipeline: YOLO localizes the leaf, and EfficientNet-B0 classifies the cropped image into one of 27 plant-disease classes. Grad-CAM provides visual explanations of the classifier's predictions.

## ✨ Features

| Feature | Description |
|---|---|
| 🍃 **Leaf Localization** | Detects candidate leaf regions using YOLO |
| 🔬 **Disease Classification** | Classifies leaf images into 27 disease classes using EfficientNet-B0 |
| 🖼️ **Image Preprocessing** | Crops, resizes, and normalizes images before classification |
| 🔎 **Explainable AI** | Uses Grad-CAM to visualize image regions influencing predictions |
| ⚡ **Optimized Inference** | Exports models to ONNX and supports FP16 classifier weights |
| 🌐 **Web Interface** | Flask-based inference application and user interface |
| 📊 **Pipeline Evaluation** | Evaluates the complete localization and classification workflow |

---

## 🧠 Model Pipeline

```text
Input Leaf Image
       |
       v
YOLO Leaf Localization
       |
       v
Highest-Confidence Bounding Box
       |
       v
Crop and Resize (224 × 224 RGB)
       |
       v
ImageNet Normalization
       |
       v
EfficientNet-B0 Classifier
       |
       v
One of 27 Class Predictions
       |
       v
Grad-CAM Visualization
```

Grad-CAM is used to interpret the classifier's prediction; it does not independently determine the predicted disease.

---

## 🛠️ Tech Stack

```text
Component             Technology
────────────────────────────────────────
Programming Language  Python
Object Detection      YOLO
Image Classification  EfficientNet-B0
Deep Learning         PyTorch
Model Optimization    ONNX, FP16
Explainable AI        Grad-CAM
Web Application       Flask
Data Processing       Image preprocessing
```

---

## 📊 Evaluation Results

The recorded project validation results are shown below.

| Metric | Result |
|---|---:|
| Validation Images | 262 |
| End-to-End Accuracy | 71.76% |
| Macro-F1 Score | 71.11% |
| Disease Classes | 27 |

These results represent the project's recorded validation evaluation. They are not an independent external benchmark or a guarantee that a retrained model will reproduce the same performance.

---

## 🚀 Getting Started

### Prerequisites

- Python installed
- Git
- A compatible environment for the project's dependencies
- Dataset files and trained model weights, where required

### 1. Clone the Repository

```bash
git clone https://github.com/Tgovardhan2005/Research_Project.git
cd Research_Project
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

**Windows:**

```cmd
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure that the required dependencies and compatible versions are specified in `requirements.txt`.

---

## 🏋️ Training and Evaluation

Run the following commands from the repository root after preparing the datasets and configuring the required paths.

### 1. Train the YOLO Detector

```bash
python training/01_train_yolo_detector.py
```

Trains the detector to localize leaf regions.

### 2. Train the EfficientNet-B0 Classifier

```bash
python training/02_train_efficientnet_b0.py
```

Trains the image classifier using the prepared classification dataset.

### 3. Evaluate the Complete Pipeline

```bash
python training/03_evaluate_pipeline.py
```

Evaluates end-to-end disease recognition performance.

### 4. Export ONNX Models

```bash
python training/04_export_onnx_fp16.py
```

Exports the models in ONNX format, including the FP16 classifier variant.

### 5. Generate Grad-CAM Explanations

```bash
python training/05_gradcam_explainability.py
```

Generates visual explanations highlighting image regions relevant to classifier predictions.

---

## ⚙️ Inference Configuration

| Parameter | Configuration |
|---|---|
| YOLO Input Size | 640 |
| YOLO Confidence Threshold | 0.25 |
| Bounding Box Selection | Highest-confidence detection |
| Classifier Input Size | 224 × 224 |
| Image Format | RGB |
| Classifier | EfficientNet-B0 |
| Number of Classes | 27 |
| Deployment Classifier | FP16-weight ONNX |

---

## 📁 Project Structure

```text
Research_Project/
├── training/
│   ├── 01_train_yolo_detector.py
│   ├── 02_train_efficientnet_b0.py
│   ├── 03_evaluate_pipeline.py
│   ├── 04_export_onnx_fp16.py
│   └── 05_gradcam_explainability.py
│
├── configs/
│   ├── class_order.json
│   └── data.yaml
│
├── webapp/
│   └── Flask inference application and UI
│
├── requirements.txt
└── README.md
```

The structure above summarizes the documented project files. Additional application files, model artifacts, and datasets may be present in the repository.

---

## ⚠️ Limitations

- The current 27-class label set does not include a dedicated healthy-leaf class.
- Confidence thresholds alone cannot guarantee rejection of unrelated images.
- Real-world performance may vary with lighting, background clutter, and image quality.
- Edge-device latency and power consumption have not been established without measurements on the target hardware.
- Reproducing the recorded results may require the original datasets, model weights, configuration, and training environment.

---

## 📦 Model Weights and Datasets

Large model weights and datasets are not committed to the repository. Follow the project's authorized artifact download instructions to obtain the required files.

Before publishing, verify that the datasets and pretrained weights can be redistributed under their respective licenses.

---

## 🔮 Future Improvements

- Add healthy-leaf recognition and out-of-distribution image rejection.
- Evaluate generalization on independent field datasets.
- Measure inference latency and memory usage on edge devices.
- Improve prediction explanations and confidence calibration.
- Optimize the complete inference pipeline for deployment.

---

## 📄 License

Add an appropriate license for the source code and verify the licensing terms of the datasets and pretrained models before distribution.

---

<div align="center">

Developed by **Govardhan T**

*Making plant disease analysis more interpretable and deployment-oriented.*

</div>
