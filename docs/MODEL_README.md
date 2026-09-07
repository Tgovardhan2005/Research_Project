# 🌱 Edge-Optimized YOLOv8-CBAM for Multi-Crop Disease Detection

## 📌 Project Overview

Despite significant advancements in agricultural computer vision, deploying highly accurate plant disease diagnostics to rural environments remains an unsolved challenge due to hardware constraints and in-the-wild background clutter.

This project introduces a custom, edge-optimized object detection pipeline. By modifying the YOLOv8-Nano backbone with a Convolutional Block Attention Module (CBAM) and engineering an 18 Super-Class data structure, this system effectively isolates microscopic lesions from soil and weed backgrounds while maintaining a footprint small enough for real-time, offline smartphone execution.

---

## ✨ Key Innovations

- **Dual-Domain Attention (YOLOv8-CBAM):** A custom architectural modification injecting CBAM into Layer 10 (post-SPPF). This computes both channel ("what") and spatial ("where") attention, creating a mathematical mask that suppresses background environmental noise and focuses on leaf lesions.
- **High-Frequency Geometry Preservation:** The input pipeline processes images at a high spatial resolution of 800x800, ensuring microscopic early-stage fungal lesions survive deep convolutional pooling layers.
- **Super-Class Data Engineering:** Solved minority class data starvation inherently found in the PlantDoc dataset by mathematically aggregating visually and biologically similar strains into 18 robust crop-level categories.
- **Offline Edge Deployment:** Model weights are compressed using Post-Training INT8 Quantization via TensorFlow Lite, shrinking the memory footprint by ~75% for zero-latency execution on mobile CPUs.

---

## 📊 Dataset & Class Structure

Trained on the **PlantDoc** dataset, which captures heavily occluded leaves, unpredictable lighting, and complex soil textures. To ensure mathematical convergence, the original 30 imbalanced classes were mapped into 18 Super-Classes.

*Example Super-Classes:* `Tomato_Diseased`, `Tomato_Healthy`, `Apple_Diseased`, `Corn_Diseased`, `Bell_pepper_Healthy`, etc.

---

## 🚀 Installation & Requirements

Ensure you have Python 3.8+ installed. The primary framework used is Ultralytics (YOLOv8).

```bash
# Clone the repository
git clone https://github.com/yourusername/crop-disease-yolo-cbam.git
cd crop-disease-yolo-cbam

# Install dependencies
pip install ultralytics torch torchvision opencv-python
```

---

## 🧠 Model Architecture & Files

- `yolov8n-cbam-v2.yaml`: The custom blueprint defining the modified neural network structure.
- `weights/best.pt`: The finalized PyTorch weights after 100 epochs of training.
- `tasks.py`: Contains the Python class definition required to register the CBAM module in the Ultralytics engine.

---

## 💻 Usage & Inference

To run inference on a test image, you must first register the custom CBAM module so the Ultralytics engine can parse the custom `.yaml` blueprint.

```python
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules.conv import CBAM

# 1. Register the custom attention module
tasks.CBAM = CBAM

# 2. Load the trained model
model = YOLO('weights/best.pt')

# 3. Run Inference on a leaf image
results = model.predict(source='test_leaf.jpg', imgsz=800, save=True, conf=0.4)

# Output is saved to the runs/detect/predict directory
```

---

## 📈 Results Highlights

- **Total Parameters:** ~3.08 Million
- **GFLOPs:** 8.2 (Highly optimized for Edge)
- **Performance:** Achieved **87.7% mAP50** on the highly complex `Tomato_Diseased` class, demonstrating the effectiveness of the attention-driven Super-Class pooling strategy. Overall validation mAP50 reached **~60%** across the heavily unconstrained PlantDoc dataset.

---

## 📱 Exporting for Edge (Mobile)

To deploy this model to an Android or iOS application, convert the PyTorch weights to an INT8 Quantized TensorFlow Lite model:

```python
model.export(format='tflite', int8=True, imgsz=800)
```

This generates a `best_saved_model/best_int8.tflite` file ready for mobile deployment.

