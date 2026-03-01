# Technical Overview — Implemented Computer Vision Pipeline

This document describes the computer vision techniques and architectural decisions behind the implemented Butler face recognition pipeline. Where the initial project idea (PROJECT_IDEA.md) surveyed the landscape of possible approaches, this document focuses on what was actually built, why each technique was chosen, and how the components interact.

## Face detection with SCRFD-500M

The detection stage uses **SCRFD-500M (Sample and Computation Redistribution for Face Detection)**, an anchor-free detector from the InsightFace project that represents a significant architectural advance over the cascade-based MTCNN and SSD-based UltraFace alternatives surveyed in the initial design.

SCRFD's key innovation is its **computation redistribution strategy**. Rather than uniformly distributing computation across all feature pyramid levels (as RetinaFace does), SCRFD analyzes where compute is most valuable through neural architecture search and reallocates FLOPs toward the scales where faces are most commonly found. The "500M" variant operates at 500 MFLOPs — roughly equivalent to UltraFace in computational budget — but achieves dramatically higher accuracy by spending those FLOPs more efficiently.

The model operates on the **WIDER FACE benchmark** hierarchy, where detection difficulty increases from "Easy" (large, unoccluded, well-lit faces) through "Medium" to "Hard" (tiny, occluded, or extreme-pose faces):

| Detector | WIDER Easy | WIDER Medium | WIDER Hard | FPS (RPi 5) |
|----------|-----------|-------------|-----------|-------------|
| SCRFD-500M | 90.57% | 88.12% | 68.51% | ~20 |
| UltraFace-slim | ~77% | ~67% | ~32% | ~65 |
| MTCNN | ~85% | ~82% | ~60% | 1-5 |

SCRFD-500M was selected as the primary detector because it operates at the accuracy level needed for reliable landmark extraction — a prerequisite for downstream face alignment. UltraFace-slim remains available as a high-speed fallback, though it produces bounding boxes without landmarks, which prevents the recognition pipeline from running.

Both detectors implement a common `FaceDetector` protocol, accepting RGB frames and returning detections in the input frame's coordinate space. The protocol-based design allows runtime switching via environment configuration without code changes.

## Multi-resolution coordinate mapping

The camera captures at **640x480** but face detection runs at a reduced resolution (typically **320x240**) for performance. This creates a coordinate space mismatch that the pipeline must resolve transparently.

The mapping strategy differs by detector architecture:

**SCRFD (via insightface)** handles resolution internally. The `FaceAnalysis.get()` method receives the full-resolution frame, resizes it to the configured detection size, runs inference, and maps all output coordinates — bounding boxes and landmarks — back to the original frame's pixel space using the inverse scale factors. The caller receives detection results that can be drawn directly on the original frame without any manual transformation.

**UltraFace** outputs **normalized coordinates** in the `[0, 1]` range, where `(0, 0)` is the top-left corner and `(1, 1)` is the bottom-right of whatever image the model processed. This normalization is architecture-inherent: the SSD-style anchor boxes are defined in normalized space relative to the feature map. To obtain pixel coordinates in the original 640x480 frame, the pipeline multiplies by the original dimensions directly:

```
x_pixel = x_normalized × original_width
y_pixel = y_normalized × original_height
```

This bypasses the intermediate detection resolution entirely — the normalized coordinates describe proportional positions within the image content, not pixel offsets within the 320x240 tensor.

The result is that all downstream stages (alignment, display, enrollment) operate exclusively in original-frame pixel coordinates without awareness of the detection resolution.

## ArcFace alignment from detected landmarks

Face recognition accuracy depends critically on geometric normalization. Raw face crops from bounding boxes contain arbitrary rotation, scale, and translation — variations that the embedding model must either learn to ignore (wasting model capacity) or that must be removed through alignment.

The pipeline implements **standard ArcFace 5-point alignment**, which transforms detected landmark positions to a canonical reference template. The five landmarks are, in order: left eye center, right eye center, nose tip, left mouth corner, and right mouth corner. The reference template defines their expected positions in a 112x112 output image:

| Landmark | Template X | Template Y |
|----------|-----------|-----------|
| Left eye | 38.29 | 51.70 |
| Right eye | 73.53 | 51.50 |
| Nose tip | 56.03 | 71.74 |
| Left mouth | 41.55 | 92.37 |
| Right mouth | 70.73 | 92.20 |

The alignment computes a **similarity transform** (rotation + uniform scale + translation) from the detected landmarks to the template using `cv2.estimateAffinePartial2D`. This 4-degree-of-freedom transform preserves facial proportions while normalizing pose. The resulting 2x3 affine matrix is applied via `cv2.warpAffine` to produce a 112x112 aligned face crop.

This alignment is essential for the embedding model's discriminative power. Without it, the same person at different head angles produces embeddings that may differ more than two different people photographed at the same angle. With alignment, intra-identity variation drops by approximately 40-60%, directly improving recognition accuracy at any given threshold.

When landmarks are unavailable (as with UltraFace detections), alignment cannot be performed and the recognition pipeline is skipped for that detection. This is a deliberate design choice — attempting recognition on unaligned faces would produce unreliable embeddings that could cause false matches.

## Embedding extraction with MobileFaceNet

The aligned 112x112 face crops are processed by **MobileFaceNet** (`w600k_mbf.onnx` from the InsightFace buffalo_sc model pack) to produce **512-dimensional L2-normalized embedding vectors**. The model was trained on the **WebFace600K** dataset (600,000 identities, ~10M images) using ArcFace loss.

MobileFaceNet's architecture combines three efficiency innovations relevant to edge deployment:

**Depthwise separable convolutions** factor a standard K×K convolution with C_in input and C_out output channels into two steps: a K×K depthwise convolution (one filter per input channel, C_in operations) followed by a 1×1 pointwise convolution (mixing channels, C_in × C_out operations). For a 3×3 kernel, this reduces computation by approximately 8-9× compared to a standard convolution while preserving the representational capacity needed for facial feature extraction.

**Inverted residual bottleneck blocks** (from MobileNetV2) reverse the traditional bottleneck pattern. Instead of compressing → processing → expanding, they expand channels to a higher dimension, apply depthwise convolution in this expanded space for richer feature interaction, then project back to a compact representation. The residual connection bridges the compact representations, keeping memory bandwidth low.

**Global Depthwise Convolution (GDConv)** replaces the standard global average pooling at the network's output. Where average pooling treats all spatial positions equally — giving corner pixels the same weight as eye regions — GDConv applies a learned 7×7 depthwise kernel that discovers spatially-adaptive importance weights. In practice, these learned weights are highest around the eyes and nose bridge, encoding the insight that central facial features carry more identity information than peripheral regions.

The model outputs a 512-dimensional vector that is L2-normalized to unit length. This normalization is critical: it projects all embeddings onto a unit hypersphere where cosine similarity equals the dot product, enabling efficient similarity computation.

The preprocessing pipeline for inference is:
1. Receive 112x112 RGB aligned crop
2. Convert RGB → BGR (matching the model's training color space)
3. Normalize pixel values: `(pixel - 127.5) / 127.5` → range `[-1, 1]`
4. Transpose from HWC to CHW layout, add batch dimension → shape `(1, 3, 112, 112)`
5. Run ONNX Runtime inference on CPU
6. L2-normalize the output vector

## Similarity search with FAISS and threshold-based matching

The pipeline maintains a face database using **FAISS (Facebook AI Similarity Search)** for embedding retrieval and **SQLite** for identity metadata. This dual-backend design separates the numerical computation (vector similarity) from the relational data (names, enrollment timestamps).

FAISS uses an **`IndexFlatIP` (Flat Inner Product)** index, which computes exact dot products between a query vector and all stored vectors. For L2-normalized embeddings, the inner product equals the cosine similarity:

```
cos(a, b) = (a · b) / (‖a‖ · ‖b‖) = a · b    (when ‖a‖ = ‖b‖ = 1)
```

This equivalence also connects to Euclidean distance: `‖a - b‖² = 2(1 - cos(a, b))` for unit vectors, meaning cosine similarity and L2 distance produce identical rankings. The pipeline uses inner product directly because it avoids the subtraction and square root of L2 distance.

For the butler's expected scale (tens of identities, hundreds of enrolled embeddings), `IndexFlatIP` provides exact search with sub-millisecond latency. Approximate methods like `IndexIVFFlat` (Voronoi cell partitioning) would add complexity without measurable benefit below ~10,000 vectors.

**Threshold-based matching** converts raw similarity scores into identity decisions. The pipeline defaults to a cosine similarity threshold of **0.4**, calibrated for household recognition where lighting conditions vary but the identity set is small:

| Similarity | Interpretation |
|-----------|---------------|
| > 0.6 | High-confidence match — same person under similar conditions |
| 0.4 – 0.6 | Moderate match — same person, different angle or lighting |
| 0.3 – 0.4 | Ambiguous zone — possible match, high false-accept risk |
| < 0.3 | Different identity — reject match |

When the database contains multiple embeddings per identity (the expected case after enrollment), FAISS searches across all embeddings and returns the single nearest neighbor. The identity associated with that embedding determines the recognition result. This **maximum-similarity selection** naturally handles intra-identity variation: if a person was enrolled from multiple angles, the embedding closest to the current observation wins regardless of which angle it came from.

When no embeddings exist in the database (or when the best similarity falls below the threshold), the pipeline labels the face as **"Unknown"**. A numpy-based fallback search is available for environments where the `faiss-cpu` package cannot be installed, providing identical results through direct matrix multiplication.

## Face enrollment through temporal sampling

The enrollment mode captures face embeddings at a controlled rate to build a representative gallery for each identity. Rather than storing every frame (which would produce redundant near-duplicate embeddings at 20+ FPS), the pipeline uses **time-gated sampling at 1-second intervals**.

Each second, the pipeline:
1. Selects the detection with the **largest bounding box area** (proxy for closest-to-camera, highest quality)
2. Extracts the aligned 112x112 crop and computes the 512-dim embedding
3. Stores the embedding in the FAISS index and SQLite database under the specified identity name

This time-based approach (checked via monotonic clock comparison, not frame counting) produces samples that naturally span different micro-expressions, slight head movements, and lighting fluctuations that occur over seconds of natural behavior. The result is a gallery of 5-15 embeddings per person that covers the intra-identity variation space, improving recognition robustness across conditions.

Frames where no face is detected or landmarks are unavailable are silently skipped without advancing the sampling timer, ensuring every stored embedding represents a successful detection-alignment-extraction chain.
