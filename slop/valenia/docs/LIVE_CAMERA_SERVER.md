# Live Camera MJPEG Server

Use `slop/valenia/scripts/live_camera_server.py` to expose the Pi camera over a simple stdlib HTTP server.

## Before You Start

1. Confirm the camera is detected:

```bash
rpicam-hello --list-cameras
```

2. Make sure the baseline models are present:

```bash
./slop/valenia/scripts/download_models.sh
```

## Start

```bash
python3 slop/valenia/scripts/live_camera_server.py \
  --track-max-missed 3 \
  --track-iou-thresh 0.3 \
  --track-smoothing 0.65 \
  --gallery-dir data/gallery \
  --match-threshold 0.228 \
  --embed-refresh-frames 5 \
  --embed-refresh-iou 0.85 \
  --metrics-json data/metrics/live_camera_metrics.json \
  --ram-cap-mb 4096
```

The server binds to `0.0.0.0:8000` by default.

## Endpoints

- `/` returns a tiny HTML page with an `<img>` tag pointed at the stream.
- `/` also includes a simple upload form for enrolling identities by name.
- `/stream.mjpg` returns an MJPEG stream (`multipart/x-mixed-replace`).
- `/metrics.json` returns the current rolling live metrics snapshot.

## Useful flags

```bash
python3 slop/valenia/scripts/live_camera_server.py \
  --port 8080 \
  --width 640 \
  --height 480 \
  --fps 8 \
  --jpeg-quality 80 \
  --ram-cap-mb 4096
```

Other model and detector flags match the existing benchmark script defaults.
Use `--disable-embed-refresh` to force the older behavior and recompute face
embeddings on every fresh tracked frame.
Use `--metrics-json ""` to disable writing the rolling metrics file.

## Notes

- The capture loop keeps only the latest encoded JPEG in memory, so clients do not build up frame queues.
- Frames are annotated with stable track IDs, smoothed boxes/landmarks, match labels, and per-frame timings.
- By default, identity embeddings are refreshed only for new, stale, or moved tracks.
- Uploaded photos are stored under `slop/valenia/data/gallery/<identity>/`.
- Rolling live metrics are written to `slop/valenia/data/metrics/live_camera_metrics.json` by default.
- The script exits early with a clear message if no camera is detected.
- The default RAM cap is `4096 MiB`.
