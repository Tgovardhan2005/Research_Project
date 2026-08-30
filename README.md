# AgriVision: Two-Stage Edge AI Pipeline for Multi-Crop Disease Detection

> *Bridging the "Reality Gap" in Agricultural Edge AI through Decoupled Localization and Diagnosis.*

---

## 📌 Project Overview

Standard monolithic object detection models struggle in real-world agricultural deployment. Forcing a single architecture to handle both background filtering (weeds, soil, shadows) and fine-grained lesion classification leads to parameter bloat and semantic confusion.

This project solves the "Monolithic Bottleneck" by decoupling the tasks into a highly optimized **Two-Stage Pipeline**. By extracting clean leaf patches dynamically from in-the-wild field imagery, this system brings lab-level diagnostic accuracy (**96.7%**) to offline, low-power edge devices.

---

## 🚀 Key Innovations & Architecture

- **Dynamic Environmental Filtering (Stage 1):** Utilizes a YOLOv8-Nano locator trained on wild field data (PlantDoc/FieldPlant) to isolate target leaves and eliminate 100% of background clutter.
- **High-Fidelity Micro-Classification (Stage 2):** Passes the cropped leaf to a MobileNetV3-Small diagnostician trained on a 38-class augmented PlantVillage dataset.
- **Ultra-Lightweight Footprint:** The combined pipeline operates on just **~3.9 Million parameters**, replacing monolithic models that require 11.7M+ parameters.
- **Accessible Edge UI:** Features an interactive Flask-based web interface with offline Text-to-Speech (TTS) alerts for farmers.

---

## 📊 Intermediate Performance Metrics

- **Stage 1 (Localization):** Effectively maps target leaf coordinates, completely removing soil, weed, and shadow interference.
- **Stage 2 (Classification):** Achieved **96.7% Validation Accuracy** across 38 distinct crop-disease conditions.
- **Computational Efficiency:** Requires only **~8.9 GFLOPs**, optimized via INT8 quantization for sub-10ms edge inference.

---

## 🛠️ Tech Stack

- **Backend & Deep Learning:** Python, Flask, PyTorch, Torchvision, Ultralytics YOLOv8
- **Frontend & Accessibility:** HTML5, CSS3, Vanilla JavaScript, Web Speech API (TTS)

---

## 📖 Documentation

Please refer to the following documents in `docs/` for deep technical context:

- [`PROBLEM_STATEMENT.md`](docs/PROBLEM_STATEMENT.md) - The "Reality Gap" and architectural solutions.
- [`DOMAIN_DOCUMENT.md`](docs/DOMAIN_DOCUMENT.md) - Background research on Multimodal Fusion and Edge AI in agriculture.
- [`RESEARCH_GAP.md`](docs/RESEARCH_GAP.md) - Analysis of SOTA academic research versus rural agricultural realities.
- [`LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md) - Evolution of Agricultural Diagnostic AI and validation of core architectures.
- [`METHODOLOGY.md`](docs/METHODOLOGY.md) - System Pipeline & Edge Implementation.
