# 🌍 Domain Document: Precision Agriculture & Edge AI

This document outlines the scientific and technical domains this project intersects, providing necessary context for researchers and contributors.

## 1. Precision Agriculture and AI Diagnostics

Precision agriculture uses technology to ensure crops and soil receive exactly what they need for optimum health and productivity. Historically, disease diagnosis relied on visual inspection by expert agronomists. Recently, Convolutional Neural Networks (CNNs) have automated this process. However, the agricultural domain is biologically complex; symptoms of abiotic stress (like extreme heat) often physically mimic biotic stress (like bacterial infections).

**Research Direction:** Moving beyond image-only models to multimodal models that understand biological context.

## 2. Multimodal Late Fusion in Deep Learning

In machine learning, "Multimodality" refers to an AI's ability to process and relate information from multiple different data types (e.g., text, images, audio).

- **Early Fusion:** Combining raw data before processing (highly inefficient for different data types like images and tabular weather).
- **Late Fusion (Our Approach):** Processing the image through a vision model (EfficientNet-B0) to get a feature vector, simultaneously processing weather data through an MLP to get a tabular vector, and mathematically concatenating them just before the final classification layer. This allows the network to learn the synergistic correlations between climate and disease.

## 3. Edge Computing & Quantization

Edge computing refers to processing data near the source of data generation (the smartphone) rather than relying on a centralized cloud server.

- **The Constraint:** Rural farms lack reliable internet. Cloud-AI fails in these environments.
- **The Domain Solution:** We utilize TensorFlow Lite (TFLite) to apply Post-Training Quantization (converting 32-bit floats to 8-bit integers). This compresses massive neural networks (like YOLOv8 and EfficientNet) by up to 4x, allowing them to run inference on a sub-$50 Android device's CPU/NPU in milliseconds, drawing zero internet bandwidth.

## 4. Asynchronous Telemetry (The SMS Matrix)

A major bottleneck in offline agriculture AI is obtaining dynamic weather data without API calls. This project introduces a novel telemetry domain hack:

- **GSM Data Transmission:** Instead of HTTP protocols, we utilize 2G/GSM SMS protocols.
- **Data Encoding:** 7 days of weather variables (Temperature, Humidity, VPD) are hex-encoded into a tiny 50-character string by a central server and pushed to the device via SMS.
- **Local Caching:** A background service in the Progressive Web App (PWA) intercepts and decrypts this payload into an IndexedDB cache, creating a perpetually updated "offline environment" for the AI to query.

## 5. Human-Computer Interaction (HCI) in Rural Tech

AI is only as good as its user interface. This project heavily researches rural HCI:

- **Voice UI:** Circumventing the literacy barrier through text-to-speech native browser APIs.
- **Active Shutter Locking:** Preventing user error through passive UI gating (using MediaPipe to analyze real-time video frames).
- **Explainable AI (XAI):** Generating natural language logs explaining why the AI made a decision, fostering trust between rural communities and advanced technology.
