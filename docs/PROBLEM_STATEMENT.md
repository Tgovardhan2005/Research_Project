# 🎯 Problem Statement & Objectives

Despite significant advancements in computer vision, deploying highly accurate plant disease diagnostics to rural agricultural environments remains an unsolved challenge. Existing solutions face critical bottlenecks in processing unconstrained field data and executing on low-power hardware.

---

## 1. The Core Problems

- **Context Blindness in the Wild:** Standard deep learning models are primarily trained on lab-curated images (e.g., PlantVillage). When deployed "in-the-wild," these networks suffer severe accuracy drops because they cannot mathematically separate the biological leaf tissue from complex background clutter (soil, weeds, shadows, and overlapping healthy leaves).
- **Micro-Lesion Feature Degradation:** Standard mobile object detection architectures (like base YOLOv8) aggressively downsample input images to standard 640x640 resolutions. This process destroys the pixel geometry of microscopic, early-stage fungal lesions, rendering them invisible to the deep convolutional layers.
- **Long-Tail Data Starvation:** Real-world agricultural datasets (e.g., PlantDoc) are naturally imbalanced. While common diseases have thousands of images, rare but destructive strains suffer from "data starvation." Standard multi-class training algorithms fail to converge on these minority classes, leading to catastrophic misdiagnoses for severe infections.
- **The Cloud-Dependency Bottleneck:** State-of-the-art attention-based networks typically run in 32-bit floating-point (FP32) formats on high-end cloud GPUs. Because rural farms lack reliable, high-speed internet infrastructure, cloud-dependent diagnostic tools become entirely inaccessible to the farmers who need them most.

---

## 2. Project Objectives

To bridge the gap between academic AI and practical agricultural application, this project engineered a localized, edge-optimized object detection pipeline. The core objectives were:

- **Engineer a Hardware-Bound Architecture:** Build a high-precision object detection model that strictly adheres to the computational limits of low-end mobile CPUs, eliminating all reliance on cloud-based API calls.
- **Implement Dual-Domain Attention:** Solve context blindness by injecting a Convolutional Block Attention Module (CBAM) into an ultra-lightweight backbone, forcing the network to dynamically filter out background soil and lock onto lesion boundaries.
- **Preserve High-Frequency Geometry:** Configure the input pipeline to ingest 800x800 spatial resolutions, ensuring microscopic disease patterns survive the pooling layers of the network.
- **Resolve Minority Class Failure:** Engineer a mathematical Super-Class grouping strategy that aggregates visually and biologically similar strains, ensuring deep feature convergence and eliminating data starvation without artificial image generation.
- **Optimize for Edge Deployment:** Compress the finalized architecture using Post-Training Quantization (INT8), dramatically shrinking the memory footprint and enabling real-time, zero-latency execution directly on a smartphone.