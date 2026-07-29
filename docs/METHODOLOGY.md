# ⚙️ Methodology: System Pipeline & Edge Implementation

The methodology of this project adopts a Modular Systems Engineering Pipeline. By decoupling environmental data acquisition from real-time inference, the system maintains high performance in "data desert" environments. The pipeline processes data through five distinct, localized stages.

## 1. Phase 1: Pre-Inference Quality Gating (MediaPipe)

To solve the "Garbage-In, Garbage-Out" (GIGO) problem at the source:

**Real-time Heuristics:** The camera viewfinder initializes a lightweight MediaPipe landmark loop.

**Validation Criteria:** The algorithm evaluates frames for Laplacian variance (blur detection) and pixel luminance (lighting).

**Active Gating:** The capture button remains locked until thresholds are met. Real-time UI overlays guide the user (e.g., "Too dark", "Move closer"). The heavy diagnostic AI is only triggered once a scientifically viable image is captured.

## 2. Phase 2: Intelligent Pre-Processing (YOLOv8-seg)

To eliminate background noise and isolate the plant tissue:

**Instance Segmentation:** Validated images are passed to a quantized YOLOv8-seg model.

**Tissue Isolation:** The model segments the leaf from the background (soil, shadows, hands).

**Multi-Stress Bounding:** If multiple co-occurring stress factors exist (e.g., a necrotic spot and a chlorotic halo), YOLOv8 segments them as distinct Regions of Interest (ROIs).

## 3. Phase 3: Cache-First Environmental Context

To provide accurate climate data without requiring active internet in the field:

**Opportunistic Syncing:** The PWA uses Service Workers to silently fetch weather data from Open-Meteo whenever the device detects intermittent connectivity.

**Storage Layer:** Weather variables (Temperature, Humidity, VPD) are serialized into a 7-day matrix and stored in the device's local IndexedDB.

**Offline Retrieval:** When offline in the field, the app queries this local cache, ensuring the system has the most recent climate history without API calls.

## 4. Phase 4: Multimodal Late Fusion

Combining biological visuals with environmental reality:

**Vision Backbone:** Segmented leaf ROIs pass through a quantized EfficientNet-B0 to generate a 512-D visual feature vector.

**Environmental Branch:** The cached weather matrix passes through a Multi-Layer Perceptron (MLP) to generate a 64-D environmental feature vector.

**Synergistic Concatenation:** These vectors are mathematically concatenated into a unified diagnostic tensor before the final classification head.

## 5. Phase 5: Explainable Validation (CSN & HITL)

Ensuring diagnostic transparency and trust:

**Contradiction Score Network (CSN):** A logic-based layer that calculates the dissonance between visual predictions and climate reality (e.g., Visual: Fungal Pathogen; Environment: Extended Drought).

**HITL Circuit Breaker:** If the CSN detects a paradox, the system pauses classification and triggers an Human-in-the-Loop (HITL) prompt, asking the farmer for binary physical ground-truth (e.g., "Is the plant under a greenhouse?").

**Final Recalibration:** Farmer input overrides the visual bias, correcting the diagnosis from pathogen to abiotic stress.

## 6. Phase 6: Accessible Output

**Voice Synthesis:** Validated diagnoses and treatment steps are synthesized into local dialects using Web Speech APIs.

**Iconographic Treatment:** Actionable steps are presented through standardized iconography, completely bypassing literacy-based barriers.
