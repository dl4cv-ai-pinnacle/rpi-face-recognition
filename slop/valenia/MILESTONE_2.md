# Milestone 2 -- General Overview

**Project:** Real-Time Face Recognition on Raspberry Pi 5
**Team member (this report):** Andrii (valenia setup)
**Date:** 2026-03-01
**Repository:** [github.com/dl4cv-ai-pinnacle/rpi-face-recognition](https://github.com/dl4cv-ai-pinnacle/rpi-face-recognition)
**Branch:** [feat/#1-valenia](https://github.com/dl4cv-ai-pinnacle/rpi-face-recognition/tree/feat/%231-valenia)

*This report was generated with the help of Claude Opus 4.6.*

---

## Team Strategy

For this milestone the team agreed that each member would independently bring
up the full pipeline, try it end-to-end, and explore their own ideas for what
the system should look like. After this deadline we will compare results and
merge the strongest ideas into one improved pipeline. Each of us therefore has
a separate report focused on personal contributions -- but my teammates worked
no less than me and have their own results; the split is about exploration
breadth, not effort difference.

## Development Approach

The codebase was heavily implemented using **Claude Opus 4.6** (via Claude
Code) and **GPT 5.3 Codex** (via OpenAI Codex), both running directly on the
Raspberry Pi. They operated autonomously most of the time -- implementing
features, writing tests, fixing lint/type errors -- while I steered the
architecture, evaluated behavior on the live camera feed, and decided what to
build next. This let one person cover a lot of ground quickly: the AI agents
handled the volume of code while I focused on the harder questions (what
actually works on real faces, what thresholds make sense, where the system
breaks).

## What Was Done

### End-to-End Pipeline

A complete face-recognition system running on a Raspberry Pi 5 (8 GB):
**detect -> align -> embed -> match**, using SCRFD-500M for detection and
ArcFace MobileFaceNet for 512-d embeddings, both via ONNX Runtime CPU. The
match threshold (0.228) was calibrated on LFW View2 cross-validation.

### Live Camera System

An HTTP server streams MJPEG from the Pi camera with real-time multi-face
tracking and recognition. The key optimizations that make it usable on the Pi:

- **Detect cadence** (`--det-every N`) -- run the detector only every Nth
  frame; between passes the tracker holds and smooths boxes. `det-every 2`
  halves detector cost with negligible visual impact
- **Selective embedding refresh** -- re-embed only new, stale, or moved
  tracks; stable known faces reuse cached results, saving ~29 ms per face
- **Identity grace period** -- holds a known label through brief occlusions
  (one-frame drop tolerance) to prevent label flickering
- **Track smoothing** -- exponential smoothing on boxes and landmarks reduces
  jitter and makes the refresh IoU check less noisy
- **RAM cap** -- configurable memory limit checked every frame, clean exit
  before OOM

The dashboard surfaces FPS, CPU, SoC temperature, memory, per-stage latencies,
and detector cadence in real-time. A `systemd` wrapper runs it unattended.

### Gallery & Identity Management

A browser-based gallery for enrollment, review, and curation:

- Auto-capture of unknown faces into a review inbox
- Promote unknowns to named identities; merge duplicate unknowns
- Per-identity detail pages with sample viewing, deletion, and photo upload
- **Auto-enrichment** from live frames -- gated by quality score (box size x
  confidence), diversity filter (cosine < 0.95 vs existing samples), 30s
  cooldown, and 48-sample cap with worst-quality eviction. Templates are
  quality-weighted averages, so better samples contribute more.

### Benchmarks & Validation

- LFW View2 10-fold: **94.75%** accuracy (FP32), unchanged after INT8
- INT8 quantization: 35% smaller model, ~0.6 ms/face faster, 6 MB less RAM,
  zero accuracy loss
- Image benchmark: ~14 FPS (FP32), ~18 FPS (INT8) on the Pi
- A/B variant comparison script for like-for-like latency tests

## What Worked

- The pipeline runs comfortably on the Pi in real-time
- The live system is stable for multi-hour sessions
- The full workflow (auto-capture -> review -> promote -> recognition) works
  end-to-end from a phone browser
- INT8 quantization was a free lunch -- smaller, faster, no quality loss

## What Didn't Work / Open Problems

### Identity Confusion

The biggest pain point. My mum was repeatedly labelled as me during initial
setups before she was enrolled -- the system was too confident matching her
face against my template. Our main fix was to collect more diverse embeddings
per person (auto-enrichment from live frames), which makes each identity's
template more robust and discriminative over time. We also added quality
gating, diversity filters, and merge-unknowns for the review inbox. These
changes helped noticeably, but the tuning was done quite in a rush -- the
thresholds, the enrichment heuristics, the quality cutoffs were all chosen
empirically without proper literature review. This needs to be thought out
more carefully, with actual reading of face recognition literature on
template adaptation and open-set identification.

### Unknown Fragmentation

Different sessions, angles, or lighting cause the system to create separate
unknown entries for the same person. Merge-unknowns helps clean up after the
fact, but the underlying unknown-matching threshold (0.36) needs more tuning.

### No Video Benchmark

We have solid image and LFW benchmarks, but no way to measure recognition
accuracy and stability on a realistic video clip. The image benchmark doesn't
capture the interplay of tracking, embedding refresh cadence, and identity
assignment across frames. This is a priority gap.

## Contribution

This report covers my work on the `valenia` setup. My teammates worked on
their own setups in parallel and will have their own reports.

## Time Allocated

Approximately **15-17 hours** total:

- ~10 hours prompting, steering, and validating AI agents + research carried
  over from Milestone 1
- ~5-7 hours hardware setup, ordering components, physical Pi configuration

## What's Next

- **Video benchmark** -- measure recognition accuracy on recorded clips, not
  just single images
- **Better auto-detection** -- per-identity adaptive thresholds, stronger
  face-quality filtering, possibly user-confirmed enrichment
- **Merge team ideas** -- combine the best approaches from each member's setup
- **Demo & final report** -- record a live demo, compile visuals and
  architecture diagrams
- **Stretch** -- anti-spoofing, alternate runtimes (MNN/ncnn), action triggers

## Screenshots

<!-- TODO: add screenshots from the live interface here -->
<!-- Suggested:
  - Live dashboard with metrics panel
  - Gallery page with confirmed identities and unknown review inbox
  - Identity detail page with per-sample grid
  - Enrollment flow
-->
