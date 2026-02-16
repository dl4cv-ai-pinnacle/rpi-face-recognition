from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from src.alignment.align import align_face
from src.capture.base import FrameCapture
from src.config import AppConfig
from src.detection.base import Detection, FaceDetector
from src.display.renderer import Renderer
from src.embedding.base import EmbeddingExtractor
from src.matching.database import FaceDatabase, MatchResult

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    detection: Detection
    aligned_face: np.ndarray | None
    embedding: np.ndarray | None
    match: MatchResult | None


class Pipeline:
    """Chains capture → detect → align → embed → match → display."""

    def __init__(
        self,
        capture: FrameCapture,
        detector: FaceDetector,
        embedder: EmbeddingExtractor,
        database: FaceDatabase,
        renderer: Renderer,
        config: AppConfig,
    ) -> None:
        self.capture = capture
        self.detector = detector
        self.embedder = embedder
        self.database = database
        self.renderer = renderer
        self.config = config

    def step(self) -> tuple[np.ndarray | None, list[RecognitionResult]]:
        """Run one pipeline iteration.

        Returns:
            (frame, results) — frame is None if capture failed.
        """
        frame = self.capture.read()
        if frame is None:
            return None, []

        det_w, det_h = self.config.capture.detection_resolution
        det_frame = cv2.resize(frame, (det_w, det_h))

        scale_x = frame.shape[1] / det_w
        scale_y = frame.shape[0] / det_h

        detections = self.detector.detect(det_frame)

        # Scale detection coords back to full resolution
        for det in detections:
            det.bbox[[0, 2]] *= scale_x
            det.bbox[[1, 3]] *= scale_y
            det.landmarks[:, 0] *= scale_x
            det.landmarks[:, 1] *= scale_y

        results: list[RecognitionResult] = []
        for det in detections:
            aligned = align_face(
                frame, det.landmarks, self.config.alignment.output_size
            )
            embedding = self.embedder.extract(aligned)
            match = self.database.search(embedding)

            results.append(
                RecognitionResult(
                    detection=det,
                    aligned_face=aligned,
                    embedding=embedding,
                    match=match,
                )
            )

        return frame, results

    def run(self) -> None:
        """Main loop. Blocks until 'q' is pressed or capture fails."""
        logger.info("Pipeline started")
        while True:
            frame, results = self.step()
            if frame is None:
                logger.warning("Capture returned None, stopping")
                break

            render_data = [(r.detection, r.match) for r in results]
            self.renderer.render(frame, render_data)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("'q' pressed, stopping")
                break
