# ⚙️ Methodology & System Pipeline

The methodology of this project abandons standard image classification in favor of an Edge-Optimized, Attention-Driven Object Detection pipeline. To overcome the chaotic nature of in-the-wild agricultural imagery and hardware constraints, the system processes data through six distinct, localized stages.

---

## Phase 1: High-Resolution Input Processing

Standard mobile detection models typically downsample images to 640x640, which frequently destroys the pixel geometry of microscopic, early-stage fungal lesions.

- **Action:** The system captures the camera feed and processes the input at a scaled 800x800 spatial resolution.
- **Purpose:** This explicitly preserves high-frequency textural details of tiny disease spots, allowing them to survive the pooling layers of the deep neural network.

---

## Phase 2: Lightweight Feature Extraction (YOLOv8-Nano)

Instead of utilizing massive, battery-draining architectures, the image enters an ultra-lightweight feature extractor.

- The input is passed through the Cross Stage Partial (C2f) bottleneck layers of a YOLOv8-Nano backbone.
- The backbone aggressively extracts foundational geometric features (edges, color gradients, and structural shapes) while maintaining an exceptionally low parameter count to ensure future edge compatibility.

---

## Phase 3: Attentional Noise Filtering (CBAM Injection)

To solve the issue of "Context Blindness" (where models confuse background soil or weeds with plant diseases), the pipeline interrupts the standard YOLOv8 flow:

- The extracted feature maps are passed into a custom-injected Convolutional Block Attention Module (CBAM) positioned at Layer 10 (post-SPPF).
- **Channel Attention:** Computes "what" is important, locking onto the specific color and textural frequencies of biological decay.
- **Spatial Attention:** Computes "where" the target is, creating a mathematical mask that actively suppresses background dirt, shadows, and overlapping healthy leaves.

---

## Phase 4: Super-Class Bounding Box Regression

The attention-filtered data enters the detection head for localization and classification.

- To prevent the model from failing on minority classes due to long-tail data starvation, the detection head utilizes a mathematically grouped 18 Super-Class structure.
- By algorithmically mapping biologically similar strains into cohesive crop-level categories (e.g., aggregating 7 distinct tomato infections into a singular `Tomato_Diseased` class), the network guarantees mathematical convergence and localized bounding box prediction with extreme confidence.

---

## Phase 5: Edge Compression (INT8 Quantization)

Because the target environment (rural farms) lacks reliable internet infrastructure, the model must be completely severed from the cloud.

- The finalized PyTorch (`.pt`) weights are subjected to Post-Training Quantization via TensorFlow Lite.
- The network's 32-bit floating-point (FP32) weights are mathematically compressed into 8-bit integers (INT8).
- This shrinks the model's memory footprint by roughly 75%, allowing the sophisticated CBAM-augmented architecture to execute locally on a smartphone CPU/NPU with zero latency.

---

## Phase 6: Accessible Human-Computer Interaction (HCI)

The final mathematical outputs are translated into an actionable, user-friendly interface:

- **Visual Localization:** The UI renders precise bounding boxes directly over the live camera feed, physically proving to the user which specific leaf spots triggered the diagnosis.
- **Literacy Bypass:** The final diagnosis and treatment prescriptions are synthesized into the local regional dialect using native browser Text-to-Speech (TTS) APIs, ensuring the system is highly accessible to non-technical or illiterate farming communities.

