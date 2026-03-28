from __future__ import annotations

import argparse
import logging

import cv2

from src.capture.picamera2 import PiCamera2Capture
from src.config import load_config
from src.detection.scrfd import SCRFDDetector
from src.display.renderer import Renderer
from src.embedding.mobilefacenet import MobileFaceNetExtractor
from src.matching.database import FaceDatabase
from src.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="RPi Face Recognition")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("main")

    capture = PiCamera2Capture(config.capture)
    detector = SCRFDDetector(config.detection)
    embedder = MobileFaceNetExtractor(config.embedding)
    database = FaceDatabase(
        db_path=config.matching.db_path,
        index_path=config.matching.index_path,
        embedding_dim=config.embedding.embedding_dim,
        threshold=config.matching.threshold,
    )
    renderer = Renderer(config.display)

    database.load()

    pipeline = Pipeline(
        capture=capture,
        detector=detector,
        embedder=embedder,
        database=database,
        renderer=renderer,
        config=config,
    )

    logger.info("Starting face recognition pipeline")
    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        database.save()
        database.close()
        capture.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
