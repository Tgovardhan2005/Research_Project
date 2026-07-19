# Multimodal Plant Disease Prediction at the Edge

An offline, context-aware AI ecosystem for connectivity-constrained rural agriculture. This project deploys an edge-optimized hybrid architecture that fuses a nano-scale vision model with a meteorological logic gate to eliminate false positives in plant disease diagnosis.

## Overview

Current agricultural AI diagnostic tools fail in rural environments due to a heavy reliance on cloud connectivity and a critical lack of environmental context. While state-of-the-art vision models excel in laboratory settings, they suffer from severe domain shift in the field and ignore crucial epidemiological triggers like temperature and humidity. 

This project solves this by operating entirely offline on sub-$50 Android devices via a Progressive Web App (PWA). It uses a heavily quantized YOLOv8-Nano model for visual extraction and a Multi-Layer Perceptron (MLP) to cross-reference visual symptoms against locally cached weather data.

## Key Features

* **Offline-First Execution:** Runs entirely in the browser using TensorFlow.js and local IndexedDB caching, requiring zero active internet connection in the field.
* **Active Quality-Gating:** Utilizes real-time MediaPipe tracking to lock the camera shutter until image focus, luminance, and leaf alignment meet strict diagnostic standards.
* **Edge-Optimized Vision:** Powered by a YOLOv8-Nano INT8 quantized model, compressed to just 1.54 MB while retaining a 91.2% Top-5 accuracy.
* **Contradiction Score Network (CSN):** A late-fusion MLP that acts as a biological logic gate, preventing false positives by catching conflicts between visual AI predictions and cached micro-climate data (temperature, humidity) or soil types.
* **Accessible Voice UI:** Features a simple, icon-driven interface with native text-to-speech voice support designed for low-literacy users.

## System Architecture

The inference pipeline is segmented into three sequential gating stages to guarantee accuracy in uncontrolled field environments:

1. **Spatial Gating:** A pre-processing YOLO detection instance scans the camera feed to isolate the target leaf and mathematically mask out overlapping healthy plants and background dirt.
2. **Feature Extraction:** The cropped tensor is processed by the YOLOv8-Nano engine to output a raw probability distribution vector across a 28-class agricultural vocabulary.
3. **Meteorological Contradiction Scoring:** The YOLO prediction, user inputs (plant/soil type), and cached weather data are fed into the MLP. The MLP calculates a contradiction scalar (0.0 to 1.0). High scores suppress biologically impossible predictions and prompt the user to recapture the image.

## Technology Stack

* **Machine Learning Training:** Python, PyTorch, Ultralytics (YOLOv8)
* **Model Export & Edge Inference:** TensorFlow Lite (INT8 Quantization), TensorFlow.js
* **Frontend Application:** HTML, CSS, JavaScript (Progressive Web App)
* **Pre-processing:** MediaPipe (Web)
* **Local Storage:** IndexedDB (Service Workers for cache-first weather sync)

## Installation & Local Development

1. Clone the repository to your local machine.
2. Ensure you have a local web server installed (e.g., VS Code Live Server or Python's `http.server`) to bypass browser CORS restrictions for local module loading.
3. Serve the project directory locally.
4. Navigate to the `localhost` URL provided by your server in any modern web browser.

## Performance Metrics

* **Model Payload:** 1.54 MB (`.tflite` format)
* **Top-1 Accuracy (INT8):** 63.9% (Raw visual extraction)
* **Top-5 Accuracy (INT8):** 91.2% (Diagnostic baseline passed to the CSN)

## 📖 Documentation

Please refer to the following documents for deep technical context:

- `PROBLEM_STATEMENT.md` - The "Reality Gap" and architectural solutions.
- `DOMAIN_DOCUMENT.md` - Background research on Multimodal Fusion and Edge AI in agriculture.
- `RESEARCH_GAP.md` - Analysis of SOTA academic research versus rural agricultural realities.
- `LITERATURE_REVIEW.md` - Evolution of Agricultural Diagnostic AI and validation of core architectures (2024-2026).
- `METHODOLOGY.md` - System Pipeline & Edge Implementation.
