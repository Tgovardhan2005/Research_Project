# 🌍 Domain Document: Precision Agriculture & Edge AI

This document outlines the scientific and technical domains this project intersects, providing necessary context for researchers and contributors.

## 1. Precision Agriculture and AI Diagnostics

Precision agriculture uses technology to ensure crops and soil receive exactly what they need for optimum health and productivity. Historically, disease diagnosis relied on visual inspection by expert agronomists. Recently, Convolutional Neural Networks (CNNs) have automated this process. However, the agricultural domain is biologically complex; symptoms of abiotic stress (like extreme heat) often physically mimic biotic stress (like bacterial infections).

**Research Direction:** Moving beyond image-only models toward multimodal systems that integrate biological and environmental context.

## 2. Multimodal Late Fusion in Deep Learning

In machine learning, "Multimodality" refers to an AI's ability to process and relate information from multiple different data types.

**Early Fusion:** Combining raw data before processing (highly inefficient for disparate data types like images and tabular weather).

**Late Fusion (Our Approach):** We process images through a vision backbone (EfficientNet-B0) to generate a visual feature vector and independently process meteorological tabular data through an MLP to generate an environmental feature vector. These vectors are mathematically concatenated just before the final classification head. This allows the network to learn synergistic correlations between climate history and disease presentation.

## 3. Edge Computing & Quantization

Edge computing refers to processing data near the source of generation (the smartphone) rather than relying on a centralized cloud server.

**The Constraint:** Rural farms often suffer from "data deserts" where high-speed internet is unavailable. Cloud-dependent AI fails in these operational environments.

**The Domain Solution:** We utilize TensorFlow Lite (TFLite) to apply Post-Training Quantization (converting 32-bit floats to 8-bit integers). This compresses large neural networks (like YOLOv8 and EfficientNet) by up to 4x, allowing them to run inference on budget-friendly Android devices (sub-$50) in milliseconds, requiring zero internet bandwidth.

## 4. Cache-First Asynchronous Telemetry

A major bottleneck in offline agricultural AI is obtaining dynamic weather data without constant API calls. Our project solves this through a "Cache-First" hybrid architecture:

**Opportunistic Sync:** The Progressive Web App (PWA) uses service workers to silently perform background fetches of regional weather data (via Open-Meteo) whenever the device detects an intermittent internet connection.

**Local Caching:** This data is serialized and stored locally in IndexedDB.

**Offline Context:** When the farmer is offline in the field, the app queries this local cache rather than making network requests, ensuring the Contradiction Score Network (CSN) always has access to the most recent climate history to validate diagnoses.

## 5. Human-Computer Interaction (HCI) in Rural Tech

AI is only as effective as its user interface. This project integrates research in rural HCI:

**Voice UI:** Circumventing literacy barriers through native browser Text-to-Speech (TTS) APIs to provide actionable treatment prescriptions.

**Active Shutter Locking:** Preventing user error through passive UI gating, utilizing MediaPipe to analyze real-time video frames for focus and lighting before allowing image capture.

**Explainable AI (XAI):** Generating natural language logs explaining why the AI reached a specific diagnosis, fostering transparency and trust between rural communities and advanced diagnostic technology.
