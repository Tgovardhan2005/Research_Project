# 📚 Literature Review: Evolution of Agricultural Diagnostic AI

This document summarizes the current state of the field, the limitations of recent research, and how this project bridges the "Reality Gap" in rural agricultural deployment.

## 1. The State of the Field (2026)

Recent research has shifted from unimodal (image-only) CNNs toward more complex systems that attempt to integrate environmental variables. However, these systems often fail to address the "disconnected field" constraint, choosing instead to rely on high-speed internet to bridge the gap.

## 2. Validating Core Architectures (2024-2026)

Recent studies explicitly validate our architectural choices:
- **YOLOv8-Nano for Edge Deployment:** A 2024 study by Chin et al. validated optimized YOLOv8 architectures (incorporating GhostNet and attention mechanisms) to drastically cut parameters while achieving a 69.3% to 72.2% mAP on plant disease datasets. Another recent study integrated a Ghost model with YOLOv8, reducing the parameter count to 5.5 million while achieving a 95.4% mAP@50. This strongly validates our use of the aggressively quantized YOLOv8-Nano (1.54 MB) for offline Android environments.
- **Multimodal / Weather Integration:** A 2025 study from Midocean University criticized vision-only models, demonstrating that fusing MobileNetV2 with an MLP to process environmental and soil data "achieved nearly 100% precision, significantly outperforming unimodal models". A subsequent 2026 study introduced an ensemble framework combining CNNs and RNNs for time-series environmental data, confirming that integrating diverse data streams provides a "holistic view of plant health". These directly support our fusion of a vision extractor (YOLO) with an MLP-based Contradiction Score Network (CSN) handling cached weather and soil data.

## 3. Comparative Analysis of Current Research

| Paper | Contribution / Focus | Relevance to this Project | Key Limitation Exploited |
| :--- | :--- | :--- | :--- |
| **WeatherLeaf** (IJARSCT, 2026) | Fuses leaf images with real-time weather data using a CNN branch and a neural weather branch. | Validates the core multimodal (image + environment) fusion approach adopted in this project. | Relies on live internet/API-based weather data; unsuitable for offline rural environments. |
| **YOLOv8 + Swin Transformer** (IJRT, 2026) | Uses YOLOv8 as an end-to-end model for plant disease detection and localization. | Confirms YOLOv8 as an efficient and edge-capable backbone for disease detection. | Uses YOLOv8 only as a monolithic end-to-end detector; lacks pre-inference image quality assessment. |
| **LeafAI** (PMC/NIH, 2026) | Proposes an interpretable edge-computing framework for plant disease detection with emphasis on user trust. | Demonstrates that explainability and trust are as essential as raw prediction accuracy. | Provides basic explanations but lacks logic-based contradiction checking between image and context. |

## 4. Bridging the Reality Gap: Our Contribution

Unlike the research listed above, our project advances the field by evolving beyond passive interpretability into a robust, context-aware systems engineering pipeline:

**Active Quality-Gating:** Moving image validation to the UX layer via MediaPipe heuristics to stop the "Garbage-In, Garbage-Out" cycle before it reaches the model.

**The Contradiction Score Network (CSN):** Providing a logic-based circuit breaker that reconciles AI predictions with cached meteorological history, effectively resolving abiotic-biotic ambiguities.

**Cache-First Telemetry:** Solving the connectivity void without requiring real-time network reliance, making the system viable for the disconnected, rural user.
