# Live Camera MJPEG Server

Use `slop/valenia/scripts/live_camera_server.py` to expose the Pi camera over a simple stdlib HTTP server.

## Start

```bash
python3 slop/valenia/scripts/live_camera_server.py
```

The server binds to `0.0.0.0:8000` by default.

## Endpoints

- `/` returns a tiny HTML page with an `<img>` tag pointed at the stream.
- `/stream.mjpg` returns an MJPEG stream (`multipart/x-mixed-replace`).

## Useful flags

```bash
python3 slop/valenia/scripts/live_camera_server.py \
  --port 8080 \
  --width 640 \
  --height 480 \
  --fps 8 \
  --jpeg-quality 80
```

Other model and detector flags match the existing benchmark script defaults.

## Notes

- The capture loop keeps only the latest encoded JPEG in memory, so clients do not build up frame queues.
- Frames are annotated with face boxes, landmarks, and per-frame pipeline timings from `FacePipeline`.
- The script expects the baseline models to exist under `slop/valenia/models`; if not, run `./slop/valenia/scripts/download_models.sh` first.
