"""Live camera server — main entry point.

Builds the pipeline from config, starts the camera streamer, and serves HTTP.
Origin: Valenia scripts/live_camera_server.py — main() + build helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import AppConfig, load_config
from src.gallery import GalleryStore
from src.live import LiveRuntime
from src.pipeline import build_pipeline

from server.handlers import LiveCameraHandler, LiveCameraHTTPServer
from server.streamer import CameraStreamer

logger = logging.getLogger(__name__)


def build_runtime(config: AppConfig) -> LiveRuntime:
    """Build a LiveRuntime from the unified AppConfig."""
    pipeline = build_pipeline(config)
    gallery = GalleryStore(
        Path(config.gallery.root_dir),
        embedding_dim=config.embedding.embedding_dim,
    )
    return LiveRuntime(pipeline=pipeline, gallery=gallery, config=config)


def main(config_path: str = "config.yaml") -> int:
    """Start the live camera server."""
    config = load_config(config_path)

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        runtime = build_runtime(config)
    except FileNotFoundError as exc:
        logger.error("Missing model files: %s", exc)
        return 2
    except MemoryError as exc:
        logger.error("Memory cap exceeded: %s", exc)
        return 4

    streamer = CameraStreamer(runtime, config)
    try:
        streamer.start()
    except RuntimeError as exc:
        logger.error("Camera error: %s", exc)
        return 3

    server: LiveCameraHTTPServer | None = None
    try:
        server = LiveCameraHTTPServer(
            (config.server.host, config.server.port), LiveCameraHandler
        )
        server.streamer = streamer
        logger.info(
            "Serving on http://%s:%d/", config.server.host, config.server.port
        )
        server.serve_forever(poll_interval=0.5)
    except OSError as exc:
        logger.error("Failed to bind %s:%d: %s", config.server.host, config.server.port, exc)
        return 4
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        streamer.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
