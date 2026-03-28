# Milestone II - Intermediary report on the progress


Team: **AI Pinnacle**:

1. Dmytro Avdieienko

1. Andrii Shalaiev

1. Andrii Valenia


This report covers the intermediary progrss on the Raspberry Pi face recognition with identity memory and additional modules.

After the first milestone on the project and discussing with the team, **we have decided to split our efforts** by each implementing an end-to-end pipeline all by ourselves. This would motivate us to research the findings more about each of the stages, and combine the findings and merge the best performance stages after this milestone.

Since the efforts were split and no group meeting to review the whole pipeline was done, we are currently unaware of the progresses and contributions to the final score, but we can say that the basic end-to-end pipeline was implemented by each of the members.

Personally, I have spent around 10-12 hours over the span of the last week (unfortunately, couldn't dedicate more time).

**Link to repository (my branch)**: https://github.com/dl4cv-ai-pinnacle/rpi-face-recognition/tree/feat/v1-shalaiev

More on the planned actions before the end of the project are described in the end of the document.


## Pipeline Overview


| Stage                   | Component                                        | Implementation                                                                    |
| ----------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------- |
| 1. Frame Capture        | Picamera2                                        | RGB888 at 640x480, downscaled to 320x240 for detection                            |
| 2. Face Detection       | SCRFD-500M (primary) / UltraFace-slim (fallback) | Swappable via `FaceDetector` protocol; coordinates mapped to original frame space |
| 3. Face Alignment       | ArcFace 5-point transform                        | Similarity transform from detected landmarks → 112x112 normalized crop            |
| 4. Embedding Extraction | MobileFaceNet (`w600k_mbf.onnx`)                 | 512-dim L2-normalized vectors via ONNX Runtime                                    |
| 5. Matching & Storage   | FAISS `IndexFlatIP` + SQLite                     | Cosine similarity search + identity metadata                                      |
| 6. Display & HUD        | OpenCV window                                    | Bounding boxes, landmarks, identity labels, confidence, FPS overlay               |


## Stage 1 — Frame Capture

The camera captures at **640x480** (configurable) in continuous video mode using Picamera2 with the libcamera backend. Detection runs at a reduced **320x240** resolution for performance. Each detector handles coordinate remapping internally — SCRFD via inverse scale factors, UltraFace via normalized `[0,1]` coordinates — so all downstream stages operate exclusively in original-frame pixel space.

## Stage 2 — Face Detection (SCRFD-500M)

**Why SCRFD-500M**: At 500 MFLOPs (comparable budget to UltraFace), SCRFD achieves significantly higher accuracy through computation redistribution — reallocating FLOPs via neural architecture search to the feature pyramid levels where faces are most commonly found. Critically, it provides 5-point facial landmarks required for alignment, which UltraFace lacks.


| Detector       | WIDER Easy | WIDER Medium | WIDER Hard | FPS (RPi 5) | Landmarks |
| -------------- | ---------- | ------------ | ---------- | ----------- | --------- |
| SCRFD-500M     | 90.57%     | 88.12%       | 68.51%     | ~20         | Yes       |
| UltraFace-slim | ~77%       | ~67%         | ~32%       | ~65         | No        |
| MTCNN          | ~85%       | ~82%         | ~60%       | 1-5         | Yes       |


UltraFace-slim remains as a high-speed fallback for detection-only use cases. Both detectors implement a shared `FaceDetector` protocol, allowing runtime switching via environment configuration (`FACE_DETECTOR=scrfd|ultraface`).

## Stage 3 — Face Alignment (ArcFace 5-Point Transform)

Raw bounding-box crops contain arbitrary rotation, scale, and translation — variation that either wastes embedding model capacity or degrades matching accuracy. The pipeline applies **standard ArcFace 5-point alignment**: a similarity transform (rotation + uniform scale + translation) computed via `cv2.estimateAffinePartial2D` from detected landmarks (eyes, nose, mouth corners) to a canonical 112x112 reference template.

This normalization reduces intra-identity embedding variation by approximately 40-60%, directly improving recognition accuracy at any threshold. When landmarks are unavailable (UltraFace detections), recognition is skipped — unaligned embeddings would produce unreliable matches.

## Stage 4 — Embedding Extraction (MobileFaceNet)

**Why MobileFaceNet**: Achieves 99.55% LFW accuracy with only 0.99M parameters, making it well-suited for edge inference. Three architectural features enable this efficiency: depthwise separable convolutions (~8-9x compute reduction over standard convolutions), inverted residual bottleneck blocks (MobileNetV2-style expand-process-compress with low-bandwidth residual connections), and Global Depthwise Convolution replacing average pooling with learned spatial attention that emphasizes identity-rich regions (eyes, nose bridge).


| Property      | Value                                            |
| ------------- | ------------------------------------------------ |
| Model         | `w600k_mbf.onnx` (InsightFace buffalo_sc pack)   |
| Input         | 112x112 aligned crop, BGR, normalized to [-1, 1] |
| Output        | 512-dim L2-normalized vector                     |
| Training data | WebFace600K (600K identities, ~10M images)       |
| Loss          | ArcFace                                          |
| LFW accuracy  | 99.55%                                           |
| Parameters    | 0.99M                                            |
| Inference     | ONNX Runtime (CPU)                               |


L2 normalization projects embeddings onto a unit hypersphere where cosine similarity equals the dot product, enabling efficient similarity computation downstream.

## Stage 5 — Matching & Storage (FAISS + SQLite)

The face database uses **FAISS** for vector similarity search and **SQLite** for identity metadata (names, enrollment timestamps), cleanly separating numerical and relational concerns.

FAISS uses `IndexFlatIP` (exact inner product), which equals cosine similarity for L2-normalized vectors. At the expected scale (tens of identities, hundreds of embeddings), exact search provides sub-millisecond latency — approximate indices like `IndexIVFFlat` would add complexity without measurable benefit below ~10K vectors.

**Threshold-based matching** converts similarity scores into identity decisions (default threshold: **0.4**, calibrated for household conditions with variable lighting):


| Similarity | Interpretation                                            |
| ---------- | --------------------------------------------------------- |
| > 0.6      | High-confidence match — same person, similar conditions   |
| 0.4 – 0.6  | Moderate match — same person, different angle or lighting |
| 0.3 – 0.4  | Ambiguous — high false-accept risk                        |
| < 0.3      | Different identity                                        |


When multiple embeddings exist per identity, FAISS returns the single nearest neighbor across all enrollments, naturally handling intra-identity variation from different poses and lighting. Faces below threshold are labeled **"Unknown"**. A numpy-based fallback is available when `faiss-cpu` cannot be installed.

## Face Enrollment

Enrollment builds a representative embedding gallery for each identity through **time-gated temporal sampling at 1-second intervals** (monotonic clock, not frame counting). This avoids storing redundant near-duplicate embeddings at 20+ FPS while naturally capturing variation in micro-expressions, head pose, and lighting over seconds of natural behavior.

**Per-sample pipeline:**

1. Select the detection with the **largest bounding box area** (proxy for closest-to-camera / highest quality)
2. Run ArcFace alignment → 112x112 crop
3. Extract 512-dim embedding via MobileFaceNet
4. Store embedding in FAISS index + identity metadata in SQLite

**Quality control:** Frames without detected faces or without landmarks are silently skipped without advancing the sampling timer, ensuring every stored embedding represents a complete detection → alignment → extraction chain. A typical enrollment session produces **5-15 diverse embeddings** per identity, covering the intra-identity variation space needed for robust recognition across conditions.

**Invocation:** `uv run main.py enroll --name "Name"` — the user faces the camera for 10-20 seconds while the system accumulates samples with live visual feedback (bounding box, landmark overlay, sample counter).

## Models & Datasets Summary

| Role | Model / Dataset | Source | Key Metric |
| ---- | --------------- | ------ | ---------- |
| Face detection | SCRFD-500M | InsightFace | 90.57% WIDER Easy, ~20 FPS RPi 5 |
| Face detection (fallback) | UltraFace-slim | ONNX Zoo | ~65 FPS RPi 5, no landmarks |
| Face embedding | MobileFaceNet `w600k_mbf` | InsightFace buffalo_sc | 99.55% LFW, 0.99M params |
| Embedding search | FAISS `IndexFlatIP` | Meta FAISS | Exact cosine, sub-ms at <10K vectors |
| Training set (embedding model) | WebFace600K | InsightFace | 600K identities, ~10M images |
| Detection benchmark | WIDER FACE | WIDER FACE project | Easy / Medium / Hard splits |
| Verification benchmark | LFW | UMass | 6,000 face pairs, standard protocol |

## What Worked and What Didn't

**What worked well:**
- The whole pipeline of facial detection and recognition worked with good performance at ~20 FPS on RPi 5 with 4 GB of RAM (as opposed to 8 GB in my fellow team members), even without the identity continuous tracking and simply run the entire recognition on each frame.
- While we haven't implemented identity tracking, multi-face problems were quite easy 
- The libraries selected mostly worked without major problems in terms of dependencies, the installation is pretty straightforward.
- The radiators along with the fans do a good job of keeping the entire Raspberry Pi cold, even under high load (40 degrees with no load and no fans, ~56-60 degrees during full load with fans working).
- Technical stuff (quite reliable quality in well-lit rooms, easy identity enrollment via the algorithm described above, etc.)

**What didn't work or posed challenges:**
- Actually, getting hands on a Raspberry Pi with all necessary components and assembling everything together took a long time, since it is the first time working with it.

Otherwise, everything worked smoothly with no major issues.

## Remaining Work

### Core pipeline completion (target requirements)

- **Multi-face tracking** — Integrate Norfair (BSD) or DeepSORT to maintain identity across frames, reducing per-frame recognition cost and eliminating ID flicker
- **TTS greeting system** — Context-aware spoken greetings via pyttsx3/espeak (e.g., "Good morning, Sarah — you're up early today!"), with last-seen tracking for differentiated responses
- **Performance optimization** — Migrate inference from ONNX Runtime to MNN for ~2-3x ARM speedup; explore INT8 quantization for both detection and embedding models
- **Robustness testing** — Evaluate recognition accuracy across lighting conditions, distances, and angles; tune thresholds per environment

### Additional modules (ideal-tier goals)

- **Anti-spoofing** — Passive liveness detection using MiniFASNet (cascade: LBP quick check → CNN deep check) to reject printed photos and screen replay attacks
- **Sign/gesture recognition** — Detect and distinguish hand signs per person, enabling gesture-based commands (e.g., wave to trigger a specific action)
- **Personal profiles with action bindings** — Per-identity profiles mapping recognized signs/gestures to custom actions (play music, toggle lights, etc.)
- **Continuous learning** — Online adaptation of the embedding gallery to better fit each person over time and adjust to deployment-specific lighting/background conditions
- **Multiprocessing pipeline** — Split capture, detection, and recognition across CPU cores using `multiprocessing.SharedMemory` for frame passing, bypassing the GIL bottleneck


In addition, we will also need presentation for the project (record or show live) demo, as well as running different benchmarks and trying and deploying it in a live settings (temporarily, but as a way to test real performance).

