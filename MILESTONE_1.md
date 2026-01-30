# Course Project — Milestone I

**Team:** AI Pinnacle

## Team Composition

| Name                                          | Year     | Major            |
| --------------------------------------------- | -------- | ---------------- |
| Dmytro Avdieienko (Авдєєнко Дмитро Максимович) | 3rd year | Computer Science |
| Andrii Shalaiev (Шалаєв Андрій Юрійович)      | 3rd year | Computer Science |
| Andrii Valenia (Валеня Андрій Іванович)       | 3rd year | Computer Science |

## Topic

**Raspberry Pi Modular Face Recognition System with Face Memory**

A fully functional face recognition system running on an edge device (Raspberry Pi 5) with the ability to memorize faces and execute predefined actions depending on the recognition result. The system is built as a modular pipeline, allowing individual components to be swapped, extended, or improved independently.

### Core Pipeline

| Stage | Component | Options Under Consideration |
|-------|-----------|----------------------------|
| 1. Frame Capture | Picamera2 | RGB888 at 640x480, downscaled to 320x240 for detection |
| 2. Face Detection | **SCRFD-500M** / Ultra-Light-Fast | SCRFD-500M: 90.57% WIDER Easy, ~20 FPS on RPi 5 / UltraFace-slim: ~65% Hard AP, 65 FPS — fallback for max speed |
| 3. Inference Framework | **MNN** / ncnn | MNN: 2.4x faster on ARM for same model (65 vs 26 FPS) / ncnn: well-documented, INT8 support — both viable |
| 4. Face Alignment | 5-point landmarks | ArcFace template → affine transform → 112x112 normalized crop |
| 5. Embedding Extraction | **MobileFaceNet** / EdgeFace-XS | MobileFaceNet: 99.55% LFW, 0.99M params / EdgeFace-XS: 99.73% LFW, 1.77M params |
| 6. Multi-Face Tracking | **Norfair** / DeepSORT / centroid-IoU | Norfair: BSD 3-Clause, flexible / DeepSORT: GPL, fewer ID switches / simple IoU: fastest |
| 7. Matching & Storage | FAISS + SQLite | Cosine similarity search + metadata storage |
| 8. Action & Response | TTS-based greetings | pyttsx3 or espeak, context-aware logic |
| 9. Anti-Spoofing | MiniFASNet (optional) | 97.8% TPR; add for access control scenarios |

### Target Goals

- Single and multi-face tracking and recognition
- Face enrollment ("Remembering Mode") with quality control
- All inference local on-device (no cloud APIs)
- Real-time performance: 10-15+ FPS target
- Person greeting with TTS and context-aware responses
- Anti-spoofing and potential continuous learning modules

## Supplementary Materials

### Models & Architectures

- [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) — edge face detection (ICLR 2022)
- [Ultra-Light-Fast-Generic-Face-Detector](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) — alternative face detection for max FPS
- [MobileFaceNet](https://arxiv.org/abs/1804.07573) — lightweight face embedding extraction
- [EdgeFace](https://github.com/otroshi/edgeface) — SOTA lightweight recognition (IJCB 2023 winner)
- [ArcFace / InsightFace](https://github.com/deepinsight/insightface) — face analysis toolbox; buffalo_sc/buffalo_s model packs
- [MiniFASNet (Silent-Face-Anti-Spoofing)](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) — passive anti-spoofing
- [Norfair](https://github.com/tryolabs/norfair) — multi-object tracking (BSD 3-Clause)
- [DeepSORT](https://github.com/nwojke/deep_sort) — multi-object tracking with appearance features (GPL)

### Frameworks & Libraries

- [MNN](https://github.com/alibaba/MNN) — Alibaba's inference engine with ARM NEON optimizations
- [ncnn](https://github.com/Tencent/ncnn) — Tencent's inference engine for ARM
- [FAISS](https://github.com/facebookresearch/faiss) — similarity search for embeddings
- [Picamera2](https://github.com/raspberrypi/picamera2) — camera interface with libcamera backend
- [OpenCV](https://opencv.org/) — image preprocessing and visualization
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) — offline text-to-speech

### Datasets & Benchmarks

- [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) — face verification benchmark
- [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) — face detection benchmark
- [CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) — anti-spoofing dataset
- [REPLAY-ATTACK](https://www.idiap.ch/en/scientific-research/data/replayattack) — anti-spoofing benchmark

### Key References

1. Guo et al., "Sample and Computation Redistribution for Efficient Face Detection" (SCRFD), arXiv:2105.04714, ICLR 2022.
2. Chen et al., "MobileFaceNets: Efficient CNNs for Accurate Real-time Face Verification on Mobile Devices", arXiv:1804.07573, 2018.
3. George et al., "EdgeFace: Efficient Face Recognition Model for Edge Devices", IEEE TBIOM, 2024.
4. Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR, 2019.
5. Wojke et al., "Simple Online and Realtime Tracking with a Deep Association Metric" (DeepSORT), ICIP, 2017.
6. Wang & Deng, "Deep Face Recognition: A Survey", Neurocomputing, 2021.

### Reference Implementations

- [LiveFaceReco_RaspberryPi](https://github.com/XinghaoChen9/LiveFaceReco_RaspberryPi) — ~20 FPS with 2800+ faces using ncnn
- [Face-Recognition-Raspberry-Pi-64-bits](https://github.com/Qengineering/Face-Recognition-Raspberry-Pi-64-bits) — 2000+ face database with anti-spoofing
- [face_recognition](https://github.com/ageitgey/face_recognition) — simple Python API, good for prototyping
