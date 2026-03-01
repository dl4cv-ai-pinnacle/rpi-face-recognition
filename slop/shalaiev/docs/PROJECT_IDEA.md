# Raspberry Pi modular face recognition system with face memory

Idea: a fully functional face recognition system running on an edge device (like Raspberry Pi 4/5) with the ability to memorize faces and some predefined actions depending on the result of the detection \+ modular components that would allow to extend the system with additional behaviour/improved quality/QoL features.

## Requirements

1. **Minimal**:  
   1. Single face tracking and detection  
   2. Face recognition  
   3. Face remembering (recorded using “Remembering Mode”)  
   4. Local inference (no API requests to cloud)  
   5. Good performance (5+ FPS)  
   6. Predefined action on successful recognition  
2. **Target** (same as minimal, but also add):
   1. Simultaneous Multi-face tracking  
   2. Large memory base with faces and successful person distinction with high quality  
   3. High performance (10-15+ FPS)  
   4. Person greeting with TTS and some “smart assistant logic” (remembering when did the system last seen you, some if-else statements for differentiated responses)  
3. **Ideal \- implementing additional modules** (same as target, but also add):  
   1. Sign recognition with distinction between different persons  
   2. Personal profile with the ability to add actions on specific signs  
   3. Anti-spoofing (fraud detection) algorithms  
   4. Continuous learning (either to better fit the concrete person, or better fit the environment the system is deployed in)

## Hardware requirements

- **Edge device**: ideally Raspberry Pi (multiple, ideally one for each team member), testing can be started using old Android phones (simulate low performance)  
- **Edge device system parts** (do not confuse with pipeline modular features): cameras, speakers, wiring, casing, etc.

## Designing the core system pipeline

**Core Pipeline stages**:

1. **Picamera2** captures frames in RGB888 format at 640×480 resolution  
2. **Ultra-Light-Fast detector** (MNN) identifies face bounding boxes at 40-65 FPS  
3. **Face alignment** using 5-point landmarks normalizes faces to 112×112  
4. **EdgeFace-XS or MobileFaceNet** extracts 128/512-dimensional embeddings via ncnn  
5. **DeepSORT tracker** maintains identity consistency across frames  
6. **FAISS** searches enrolled embeddings with cosine similarity (0.42-0.5 threshold); new faces trigger anti-spoofing (**MiniFASNet**)  
7. **Resulting action** based on whether the face was successfully detected as a known person or a stranger (e.g. "Good morning, Sarah—you're up early today\!" or"Welcome back, haven't seen you since last Tuesday\!")

**Enrollment workflow** requires quality control to build a robust face database:

- Capture 5-10 images per person across different angles and lighting  
- Run anti-spoofing on enrollment images to prevent database poisoning  
- Compute average embedding from high-quality samples  
- Store in FAISS \+ SQLite with metadata (name, relationship, preferences)  
- Enable easy deletion for privacy compliance

The **software stack** recommendation for production:

- **ncnn**: Best inference performance on ARM, supports MTCNN, RetinaFace, MobileFaceNet, EdgeFace  
- **Picamera2**: Modern Python interface with libcamera backend and direct OpenCV integration  
- **OpenCV**: Image preprocessing, visualization, camera fallback  
- **SQLite \+ FAISS**: Hybrid storage for metadata and fast similarity search  
- **pyttsx3 or espeak**: Text-to-speech for greetings  
- **Google Coral USB**: possible hardware acceleration.

## Face detection architectures for embedded systems

Face detection forms the first critical stage of your pipeline, determining both the overall speed and accuracy ceiling of the system. The field has evolved dramatically from classical approaches to neural network-based methods optimized for edge devices.

**MTCNN (Multi-task Cascaded Convolutional Networks)** remains a foundational architecture worth understanding. Introduced by Zhang et al. in 2016, it uses a three-stage cascade that progressively refines face candidates. The **P-Net (Proposal Network)** rapidly scans an image pyramid using 12×12 sliding windows to generate candidate regions at \~30 MFLOPS. These candidates pass to **R-Net (Refine Network)**, which crops proposals to 24×24 pixels and applies a deeper CNN to reject false positives using Non-Maximum Suppression. Finally, **O-Net (Output Network)** processes 48×48 crops to produce final bounding boxes plus 5 facial landmarks (eyes, nose, mouth corners). This coarse-to-fine processing enables early rejection of obvious non-faces, but MTCNN achieves only **1-5 FPS on Raspberry Pi 4**—too slow for real-time butler operation.

For embedded deployment, **BlazeFace** from Google Research offers a compelling alternative. Using MobileNet-inspired depthwise separable convolutions and a GPU-friendly anchor scheme, it achieves **98.61% average precision** while running at sub-millisecond inference times on mobile GPUs. The model predicts 6 keypoints and weighs just 1-2 MB, though it's optimized for front-facing cameras and struggles with distant faces.

**RetinaFace** delivers the best accuracy at **91.4% AP on WIDER FACE hard test set**, using Feature Pyramid Networks for multi-scale detection. However, the full ResNet-50 backbone runs below 1 FPS on Raspberry Pi, making it impractical without hardware acceleration.

The recommended detector for Raspberry Pi butler applications is the **Ultra-Light-Fast-Generic Face Detector**. Specifically designed for embedded systems, it achieves remarkable performance using the MNN framework:

| Model Variant | Framework | Resolution | mAP | FPS on RPi 4 |
| :---- | :---- | :---- | :---- | :---- |
| Ultra-Light slim | MNN | 320×240 | 67.1% | **65 FPS** |
| Ultra-Light RFB | MNN | 320×240 | 69.8% | **56 FPS** |
| Ultra-Light slim | ncnn | 320×240 | 67.1% | 26 FPS |

Framework choice dramatically impacts performance—the same model runs **3-4× faster** on MNN compared to other frameworks due to ARM NEON optimizations and INT8 quantization support.

## How MobileFaceNet and EdgeFace extract discriminative embeddings

Once faces are detected, the recognition system must extract feature vectors (embeddings) that capture identity information. For someone familiar with CNNs, the key innovation in face recognition architectures lies in their efficiency-focused design choices and specialized loss functions.

**MobileFaceNet** (2018) achieves **99.55% accuracy on LFW with under 1 million parameters** by combining three architectural innovations. First, it uses **depthwise separable convolutions** that factorize standard convolutions into a depthwise operation (single filter per channel) followed by pointwise 1×1 convolutions—reducing computation by 8-9× while preserving feature quality. Second, it employs **inverted residual bottleneck** structures from MobileNetV2, expanding channels before depthwise convolutions for richer intermediate representations. Third, and most critically, it replaces global average pooling with **Global Depthwise Convolution (GDConv)**, which learns spatially-adaptive importance weights recognizing that facial center regions (eyes, nose) are more discriminative than corners.

**EdgeFace** (2024) represents the current state-of-the-art for lightweight face recognition, winning the IJCB 2023 Efficient Face Recognition Competition. Its innovation is a **hybrid CNN-Transformer architecture** using Split Depth-wise Transpose Attention (STDA) encoders. This design captures both local texture features (CNN strength) and global context relationships (Transformer strength) while maintaining linear computational complexity through transposed attention mechanisms. EdgeFace-XS achieves **99.73% LFW accuracy with only 1.77M parameters and 154 MFLOPs**—actually fewer floating-point operations than MobileFaceNet while delivering higher accuracy.

| Model | Parameters | MFLOPs | LFW Accuracy | Best For |
| :---- | :---- | :---- | :---- | :---- |
| MobileFaceNet | 0.99M | 439.8 | 99.55% | Proven, well-documented |
| EdgeFace-XS | 1.77M | 154 | 99.73% | SOTA edge deployment |
| EdgeFace-S | 3.65M | 306 | 99.78% | Higher accuracy needs |

The remarkable discriminative power of modern face embeddings comes from specialized **margin-based loss functions** rather than standard softmax. **ArcFace (Additive Angular Margin Loss)** normalizes both feature vectors and class weights to unit length, then adds a constant angular margin *m* (typically 0.5) directly to the angle between features and their target class weights. Mathematically: `L = -log(exp(s·cos(θ + m)) / [exp(s·cos(θ + m)) + Σ exp(s·cos(θⱼ))])` where *s* is a scale factor (\~64) and *θ* is the angle between embedding and class center. This forces embeddings of the same identity to cluster tightly on a hypersphere while maintaining geodesic distance gaps between different identities. **CosFace** achieves similar results by subtracting margin from the cosine value itself rather than adding to the angle, with slightly faster convergence but marginally lower performance at very low false acceptance rates.

For embedding dimensions, research surprisingly shows diminishing returns beyond **128 dimensions**. The original FaceNet paper found accuracy plateaued or even declined with 512 dimensions due to overfitting. For a household butler with 100 identities, 128-dimensional embeddings (512 bytes per face in float32) provide excellent accuracy while keeping storage and similarity computation efficient.

## Storing embeddings and searching for matches at scale

Your butler needs to store enrolled face embeddings and efficiently search for matches. The choice between storage backends depends on scale and query patterns.

**Pickle files** work for prototyping with under 100 identities—simply serialize a Python dictionary mapping names to embedding arrays. However, this approach requires loading everything into memory and performs O(n) linear search, making it unsuitable for larger databases.

**SQLite** provides structured storage with metadata (enrollment date, notes, last seen) and ACID compliance in a single-file database perfect for Raspberry Pi deployment. The newer **sqlite-vec extension** adds native vector operations with SIMD acceleration for ARM NEON, enabling KNN search directly in SQL queries. For 100-10,000 identities with complex metadata requirements, this offers the best balance.

**FAISS (Facebook AI Similarity Search)** excels when you need fast similarity search at scale. Even its simplest `IndexFlatL2` provides highly optimized exact search, while `IndexIVFFlat` uses Voronoi cell partitioning for approximate search on larger databases. For a butler application, FAISS handles the similarity search while a companion SQLite database stores identity metadata:

FAISS Index (embeddings \+ row IDs) ←→ SQLite (rowid → name, metadata)

For face matching, **cosine similarity is preferred** over Euclidean distance because most face embedding models produce L2-normalized vectors, where cosine similarity reduces to a simple dot product (computationally efficient). For normalized vectors, the metrics are mathematically equivalent: `Euclidean² = 2(1 - Cosine_Similarity)`, producing identical rankings.

**Threshold selection** critically impacts the false acceptance/rejection trade-off. Based on LFW benchmarks with modern embeddings, practical thresholds are:

| Use Case | Cosine Threshold | Behavior |
| :---- | :---- | :---- |
| High security (low FAR) | \> 0.6 | Few false accepts, more rejections |
| Balanced verification | \> 0.42 | Standard operating point |
| High recall (grouping) | \> 0.35 | Catches more matches, some false accepts |

For a butler greeting family members, **0.42-0.5** provides appropriate balance—confident enough to avoid embarrassing misidentifications while not requiring perfect lighting conditions.

## Maintaining identity across video frames with DeepSORT

Multi-face tracking prevents your butler from re-running expensive recognition on every frame while maintaining consistent identity assignments as people move through scenes.

**SORT (Simple Online and Realtime Tracking)** uses Kalman Filters to predict face positions and the Hungarian algorithm to match predictions with new detections based on Intersection-over-Union (IoU). Running at **260 Hz**, it's extremely fast but suffers high identity switches during occlusions because it relies purely on motion without appearance information.

**DeepSORT** extends SORT with **appearance feature matching**, reducing identity switches by 45%. Each track maintains a gallery of recent appearance embeddings (conveniently, you already have face embeddings from your recognition model). The association cost combines Mahalanobis distance for motion consistency with cosine distance for appearance similarity:

cost \= λ × d\_mahalanobis \+ (1-λ) × d\_appearance

Tracks progress through states: **tentative** (new detection, awaiting confirmation), **confirmed** (detected in consecutive frames), and **deleted** (not matched for max\_age frames). A matching cascade prioritizes recently-seen tracks, and unmatched detections after the cascade fall back to IoU-based matching.

For face tracking specifically, **FaceSORT** combines biometric face embeddings with generic appearance features, handling cases where faces are partially occluded or at extreme angles where recognition fails but tracking should continue.

Practical parameters for Raspberry Pi implementation:

| Parameter | Recommended Value | Purpose |
| :---- | :---- | :---- |
| Appearance threshold | 0.5 cosine | Similarity for appearance match |
| IoU threshold | 0.3 | Fallback geometric matching |
| Max age | 30-60 frames | Frames before track deletion |
| Min hits | 3 frames | Detections to confirm track |
| Gallery size | 10-50 embeddings | Per-track appearance history |

Re-identification after occlusion compares reappeared faces against stored gallery embeddings of "lost" tracks, with an exponential moving average update strategy: `new_emb = α × detection_emb + (1-α) × track_emb`.

## Achieving real-time performance through quantization and acceleration

Running neural networks in real-time on Raspberry Pi demands aggressive optimization. The good news: well-optimized systems achieve **10-25 FPS** with proper techniques.

**INT8 quantization** converts 32-bit floating-point weights and activations to 8-bit integers using learned scale factors. This yields **4× model size reduction** and **2-3× inference speedup** with typically less than 1-2% accuracy drop when using a representative calibration dataset. TensorFlow Lite's post-training quantization makes this straightforward:

converter.optimizations \= \[tf.lite.Optimize.DEFAULT\]

converter.representative\_dataset \= calibration\_data\_generator

converter.target\_spec.supported\_ops \= \[tf.lite.OpsSet.TFLITE\_BUILTINS\_INT8\]

**Neural network pruning** removes redundant weights or entire filters. Structured pruning (removing complete channels) is recommended for Raspberry Pi because it doesn't require sparse matrix operations. Research shows MobileNetV2 can be pruned **\~30%** with under 1% accuracy loss, while larger networks like ResNet50 can be pruned up to 99% to match lightweight alternatives.

**Hardware accelerators** provide the most dramatic speedups:

| Accelerator | Compute | Price | Face Detection FPS | Notes |
| :---- | :---- | :---- | :---- | :---- |
| Google Coral USB | 4 TOPS | \~$60 | 10-25 | INT8 only, easy setup |
| Intel NCS2 | \~1 TOPS | \~$70 | 11-24 | FP16 support, OpenVINO |
| Hailo-8L (RPi 5\) | 13 TOPS | \~$70 | 150+ | Newest option, best performance |

The **Google Coral USB Accelerator** is recommended for face recognition due to excellent TensorFlow Lite integration, power efficiency (2 TOPS/watt), and straightforward Raspberry Pi support. Without an accelerator, CPU-only achieves **0.5-1.5 FPS** for complete face detection; with Coral, this jumps to **10-25 FPS**.

**Threading and multiprocessing** patterns critically impact real-time performance. Python's GIL limits threading benefits for CPU-bound inference, but I/O-bound camera capture releases the GIL. A dedicated capture thread improved frame rates from **14.97 FPS to 51.83 FPS** in PyImageSearch benchmarks—a 246% improvement by eliminating I/O latency:

\# Producer-consumer pattern across processes

\# Core 0: Camera capture (I/O bound)

\# Core 1: Face detection \+ tracking (CPU/accelerator)

\# Core 2: Recognition inference (CPU/accelerator)

\# Core 3: Butler logic \+ TTS

Use multiprocessing for compute-intensive stages and shared memory (`multiprocessing.SharedMemory`) for efficient frame passing between processes.

## Defending against spoofing attacks with lightweight liveness detection

A production butler must distinguish real faces from photographs or screen replays. The most common attacks—**printed photos** and **phone replay attacks**—are fortunately detectable with lightweight methods.

**Texture-based detection** exploits the fact that recaptured images (photos of photos, screens displaying faces) contain subtle artifacts. **Local Binary Patterns (LBP)** analyze micro-texture by comparing pixel intensities with neighbors, achieving **92.4% HTER on REPLAY-ATTACK** with negligible computational cost. This serves as an excellent first-stage filter running in under 5ms per frame.

**CNN-based anti-spoofing models** dramatically improve accuracy. The **MiniFASNet** family from Minivision AI provides silent (passive) detection requiring no user interaction:

| Model | Size | Input | Best For |
| :---- | :---- | :---- | :---- |
| MiniFASNetV1 | \~600KB | 80×80 | Fastest, resource-constrained |
| MiniFASNetV2SE | \~1.2MB | 80×80 | Best accuracy, recommended |

**MobileNetV3-based anti-spoofing** achieves **99.1% AUC** on CelebA-Spoof with only 0.6M parameters and 0.03 GFLOPs—negligible overhead for Raspberry Pi 4\.

**Depth-based methods** using stereo cameras or depth sensors provide definitive 2D attack rejection. The **OAK-D Lite** (\~$150) integrates depth sensing with onboard neural inference, running anti-spoofing directly on its Intel Movidius VPU without burdening the Raspberry Pi CPU.

**Challenge-response methods** (blink detection, head movement) achieve near-100% accuracy against photos but add 2-5 seconds of latency and require user cooperation—inappropriate for seamless butler greetings but valuable for high-security functions like door unlocking.

The recommended integration strategy uses a **cascade architecture**:

Frame → Face Detection → LBP Quick Check (5ms)

                              ↓ \[uncertain\]

                         CNN Deep Check (30-50ms)

                              ↓ \[still uncertain\]  

                         Depth/Challenge Fallback

Apply anti-spoofing selectively: always check new faces, periodically verify continuous presence (every 30 seconds), and recheck when confidence drops. **Temporal voting** over 5 frames reduces false positives without adding significant latency.

## Essential academic papers and implementation references

The theoretical foundations for this system come from several key papers:

**Face detection**: Zhang et al., "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks" (2016) introduced MTCNN's cascade architecture. Bazarevsky et al., "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs" (2019) demonstrated edge-optimized detection.

**Face recognition**: Chen et al., "MobileFaceNets: Efficient CNNs for Accurate Real-time Face Verification on Mobile Devices" (arXiv:1804.07573, 2018\) established the lightweight recognition paradigm. Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019\) revolutionized embedding quality through angular margin losses. George et al., "EdgeFace: Efficient Face Recognition Model for Edge Devices" (IEEE TBIOM 2024\) represents current state-of-the-art, available at [https://github.com/otroshi/edgeface](https://github.com/otroshi/edgeface).

**Tracking**: Bewley et al., "Simple Online and Realtime Tracking" introduced SORT. Wojke et al., "Deep SORT: Deep Association Metric for Multi-Object Tracking" added appearance features for robust tracking.

**Survey papers**: Wang & Deng, "Deep Face Recognition: A Survey" (Neurocomputing 2021\) covers 330+ papers comprehensively. The FG2023 tutorial survey "A Survey of Face Recognition" (arXiv:2212.13038) provides practical implementation guidance.

**Open-source implementations** for Raspberry Pi:

- **LiveFaceReco\_RaspberryPi**: Achieves \~20 FPS with 2800+ faces using ncnn, RetinaFace detection, and ArcFace recognition  
- **Qengineering Face-Recognition-Raspberry-Pi-64-bits**: 2000+ face database with anti-spoofing and blur filtering  
- **face\_recognition library**: Simplest Python API (99.38% LFW), slower but excellent for prototyping  
- **InsightFace**: Complete 2D/3D face analysis toolbox with training support

The **ncnn framework** ([https://github.com/Tencent/ncnn](https://github.com/Tencent/ncnn)) from Tencent is essential for ARM deployment—it provides NEON assembly optimizations, zero third-party dependencies, and supports all major face models in quantized formats.

TF-JS

