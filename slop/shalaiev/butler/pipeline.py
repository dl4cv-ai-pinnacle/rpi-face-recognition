"""Pipeline orchestrator: capture → detect → recognize → display loop."""

from __future__ import annotations

import time

from .camera import Camera
from .detectors.base import Detection, FaceDetector
from .display import Display
from .recognition.base import FaceEmbedder, RecognizedFace
from .recognition.database import FaceDatabase

# Number of recent frame times used for rolling FPS average.
_FPS_WINDOW = 30


class Pipeline:
    """Wire camera capture to the display with an FPS-metered loop."""

    def __init__(
        self,
        resolution: tuple[int, int] = (640, 480),
        window_name: str = "Butler",
        detector: FaceDetector | None = None,
        detect: bool = True,
        embedder: FaceEmbedder | None = None,
        database: FaceDatabase | None = None,
        recognize: bool = True,
    ):
        self._camera = Camera(resolution=resolution)
        self._display = Display(window_name=window_name)
        self._detector = detector
        self._detect = detect
        self._embedder = embedder
        self._database = database
        self._recognize = recognize

    def _ensure_detector(self) -> FaceDetector | None:
        """Lazily create the detector on first call if none was injected."""
        if self._detector is not None or not self._detect:
            return self._detector
        try:
            from .detectors import create_detector

            self._detector = create_detector()
        except Exception as exc:
            print(f"[WARNING] Failed to load face detector: {exc}")
            self._detect = False
        return self._detector

    def _ensure_embedder(self) -> FaceEmbedder | None:
        """Lazily create the embedder on first call if none was injected."""
        if self._embedder is not None or not self._recognize:
            return self._embedder
        try:
            from .recognition import create_embedder

            self._embedder = create_embedder()
        except Exception as exc:
            print(f"[WARNING] Failed to load face embedder: {exc}")
            self._recognize = False
        return self._embedder

    def _ensure_database(self) -> FaceDatabase | None:
        """Lazily create the face database if none was injected."""
        if self._database is not None or not self._recognize:
            return self._database
        try:
            from butler import settings

            self._database = FaceDatabase(
                settings.DATABASE_DIR,
                embedding_dim=self._embedder.embedding_dim if self._embedder else 512,
            )
        except Exception as exc:
            print(f"[WARNING] Failed to open face database: {exc}")
            self._recognize = False
        return self._database

    def run(self) -> None:
        detector = self._ensure_detector()
        embedder = self._ensure_embedder()
        database = self._ensure_database()
        timestamps: list[float] = []
        with self._camera:
            try:
                while True:
                    frame = self._camera.capture()

                    # --- detection ---
                    detections: list[Detection] = []
                    detection_ms = 0.0
                    if detector is not None:
                        t0 = time.monotonic()
                        detections = detector.detect(frame)
                        detection_ms = (time.monotonic() - t0) * 1000.0

                    # --- recognition ---
                    recognized_faces: list[RecognizedFace] | None = None
                    recognition_ms = 0.0
                    if embedder is not None and database is not None and detections:
                        from butler import settings

                        t0 = time.monotonic()
                        recognized_faces = []
                        for det in detections:
                            embedding = embedder.extract(frame, det)
                            if embedding is not None:
                                match = database.search(
                                    embedding, settings.RECOGNITION_THRESHOLD,
                                )
                                recognized_faces.append(
                                    RecognizedFace(det, match.identity, match.similarity),
                                )
                            else:
                                recognized_faces.append(
                                    RecognizedFace(det, "Unknown", 0.0),
                                )
                        recognition_ms = (time.monotonic() - t0) * 1000.0

                    # --- FPS ---
                    now = time.monotonic()
                    timestamps.append(now)
                    if len(timestamps) > _FPS_WINDOW:
                        timestamps = timestamps[-_FPS_WINDOW:]
                    if len(timestamps) >= 2:
                        fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
                    else:
                        fps = 0.0

                    # --- display ---
                    self._display.show(
                        frame,
                        fps,
                        detections=detections or None,
                        recognized_faces=recognized_faces,
                        detector_name=detector.name if detector else None,
                        detection_ms=detection_ms,
                        embedder_name=embedder.name if embedder else None,
                        recognition_ms=recognition_ms,
                    )
                    if self._display.should_quit():
                        break
            except KeyboardInterrupt:
                pass
            finally:
                self._display.close()
                if self._database is not None:
                    self._database.close()

    def enroll(self, name: str) -> None:
        """Enrollment loop: detect → embed → time-gated store at ~1 Hz."""
        detector = self._ensure_detector()
        embedder = self._ensure_embedder()
        database = self._ensure_database()

        sample_count = 0
        last_store_time = 0.0
        timestamps: list[float] = []

        with self._camera:
            try:
                while True:
                    frame = self._camera.capture()

                    # --- detect ---
                    detections: list[Detection] = []
                    if detector is not None:
                        detections = detector.detect(frame)

                    # --- pick largest face (closest to camera) ---
                    best_det = max(
                        detections,
                        key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1),
                        default=None,
                    )

                    # --- extract embedding ---
                    embedding = None
                    if best_det is not None and embedder is not None:
                        embedding = embedder.extract(frame, best_det)

                    # --- time-gated store (1 sample per second) ---
                    captured_now = False
                    now = time.monotonic()
                    if (
                        embedding is not None
                        and database is not None
                        and (now - last_store_time >= 1.0)
                    ):
                        database.enroll(name, embedding)
                        sample_count += 1
                        last_store_time = now
                        captured_now = True

                    # --- FPS ---
                    timestamps.append(now)
                    if len(timestamps) > _FPS_WINDOW:
                        timestamps = timestamps[-_FPS_WINDOW:]
                    if len(timestamps) >= 2:
                        fps = (len(timestamps) - 1) / (
                            timestamps[-1] - timestamps[0]
                        )
                    else:
                        fps = 0.0

                    # --- display ---
                    self._display.show_enrollment(
                        frame,
                        fps,
                        detection=best_det,
                        name=name,
                        sample_count=sample_count,
                        captured=captured_now,
                        detector_name=detector.name if detector else None,
                    )
                    if self._display.should_quit():
                        break
            except KeyboardInterrupt:
                pass
            finally:
                self._display.close()
                if self._database is not None:
                    self._database.close()
                print(f"Enrolled '{name}' with {sample_count} sample(s).")
