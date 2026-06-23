# 🌱 Multimodal Plant Disease Prediction at the Edge

Overcoming Context Blindness and Connectivity Constraints in Rural Agriculture

## 📌 Project Overview

This repository contains the architecture and implementation of Offline, Context-Aware Diagnostic Ecosystem for smallholder farmers.

Standard agricultural AI models fail in rural deployment due to the "Reality Gap": they require high-speed internet, assume perfect photography, and ignore local weather context. This project solves these constraints by shifting inference entirely to the Edge and utilizing asynchronous SMS telemetry for environmental awareness.

## 🚀 Key Innovations

- **Active UX Gating (MediaPipe):** An active shutter lock prevents "Garbage-In, Garbage-Out" (GIGO) by evaluating image focus and lighting in real-time before the capture.
- **Mixed-Symptom Segmentation (YOLOv8-seg):** Isolates multiple co-occurring stress factors on a single leaf to prevent background bias.
- **SMS Micro-Climate Matrix:** A zero-internet telemetry payload that delivers 7-day hexadecimal-encoded weather matrices over standard 2G/GSM networks to local device storage.
- **Contradiction Score Network (CSN) & HITL:** An Explainable AI (XAI) circuit breaker that cross-references visual symptoms with offline weather data. If a paradox is detected (e.g., fungal prediction during a drought), it triggers a Human-in-the-Loop (HITL) prompt for validation.
- **Accessible PWA:** Voice UI and icon-driven output deployed via a lightweight Progressive Web App for rural users with low literacy.

## 🧠 System Architecture

The pipeline consists of four localized stages:

1. **Quality Gate:** MediaPipe (Focus/Lighting validation).
2. **Vision Branch:** YOLOv8-seg (Isolation) -> EfficientNet-B0 (Visual Feature Vector).
3. **Context Branch:** Intercepted SMS Matrix -> Local SQLite/IndexedDB -> MLP (Environmental Feature Vector).
4. **Late Fusion & CSN:** Concatenated Multi-modal Tensor -> Paradox Detection -> Voice UI Output.

## 🛠️ Tech Stack

- **Frontend / Edge:** Progressive Web App (PWA), React.js, TensorFlow.js / TFLite
- **Computer Vision:** Ultralytics YOLOv8-seg, MediaPipe (Face/Objectron logic adapted for leaves), EfficientNet-B0
- **Backend (Telemetry Sender):** Node.js / Python Flask, Twilio/GSM Gateway (for SMS dispatch), Open-Meteo API
- **Data Storage:** IndexedDB (Local PWA storage)

## 📖 Documentation

Please refer to the following documents for deep technical context:

- `PROBLEM_STATEMENT.md` - The "Reality Gap" and architectural solutions.
- `DOMAIN_DOCUMENT.md` - Background research on Multimodal Fusion and Edge AI in agriculture.
