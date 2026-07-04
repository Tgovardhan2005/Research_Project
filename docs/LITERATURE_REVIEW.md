# 📚 Literature Review

This document contains two versions of the literature review. **Version A (Thematic)** groups papers by technology and is best for a standard thesis. **Version B (Evolutionary)** tells the story of how AI has failed and evolved, which is excellent for a project pitch or introduction.

---

## Version A: Thematic Literature Review

### 1. Multimodal Fusion in Agriculture

Recent studies have unequivocally demonstrated that unimodal (image-only) AI is insufficient for accurate agronomic diagnostics. Research such as *WeatherLeaf: Multi-Modal Plant Disease Risk Prediction (2026)* confirms that fusing leaf images with meteorological data significantly improves prediction accuracy. They utilize a methodology similar to our baseline, employing a CNN for images and a Neural Network branch for weather data. 

However, the WeatherLeaf SOTA framework, like many others, relies on "real-time meteorological data," assuming constant high-speed internet. This critical assumption renders such systems impractical for rural deployment where connectivity is volatile or non-existent.

### 2. Edge Computing and Object Detection

To address the need for localized processing, the academic focus has shifted toward robust edge architectures. The IJRT 2026 study on *A Smart Agriculture Framework Using YOLOv8 and Swin Transformer Networks* validates the use of YOLOv8 as a powerhouse for plant disease localization and detection. 

While highly effective, current literature predominantly utilizes YOLO models in an "End-to-End" classification manner. This approach often falls victim to the "Garbage-In, Garbage-Out" (GIGO) problem, attempting to process poorly captured images. Our architecture advances this by utilizing YOLOv8 as an "Intelligent Pre-processor," isolating the region of interest before heavy feature extraction.

### 3. Human-in-the-Loop (HITL) and Explainable AI (XAI)

Model trust is increasingly recognized as being just as critical as accuracy. The study *LeafAI: Interpretable plant disease detection for edge computing (2026)* highlights the necessity of edge computing combined with interpretability to foster user trust. 

While LeafAI successfully demonstrates that offline, interpretable inference is possible, it remains a passive system. Our proposed architecture moves beyond basic interpretability by introducing the Contradiction Score Network (CSN) and a Human-in-the-Loop (HITL) circuit breaker, creating an active, collaborative diagnostic system.

---

## Version B: The Evolutionary Literature Review (Narrative Format)

The application of Deep Learning in agriculture has undergone three distinct evolutionary phases:

### Phase 1: The Unimodal Era (2018–2023)
Early research heavily utilized single-label CNN classifiers trained on isolated datasets. While achieving high accuracy in labs, these models suffered severe accuracy drops in the field due to **"Context Blindness"** (e.g., confusing abiotic heat stress with biotic fungal infections).

### Phase 2: The Multimodal & Edge Era (2024–2026)
To solve Context Blindness, researchers began fusing images with weather data, as seen in the 2026 *WeatherLeaf* framework. Concurrently, object detection models like YOLOv8 (validated by IJRT, 2026) were deployed to localize diseases, and frameworks like *LeafAI (2026)* pushed these models to edge devices. 

However, this created a paradox: the visual model was offline, but multimodal systems still required a 4G/5G connection to fetch real-time weather APIs, breaking the "offline" promise in rural areas. Furthermore, YOLO models were used end-to-end, passively accepting bad data (GIGO).

### Phase 3: Trust and Systems Engineering (2026-Present)
Current literature demands trustworthy, interpretable systems (*LeafAI*). Yet, a massive reality gap remains: no existing architecture actively prevents bad image capture at the UX level, nor do they possess a zero-internet method for fetching environmental context, nor do they actively prompt the user when uncertain. 

Our proposed architecture directly addresses these final, critical deployment hurdles through Active UX Gating, SMS Telemetry, and a CSN-driven HITL circuit breaker.
