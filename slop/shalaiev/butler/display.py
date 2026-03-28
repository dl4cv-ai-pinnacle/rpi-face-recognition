"""OpenCV display window with HUD overlay."""

from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np

from butler.detectors.base import Detection
from butler.recognition.base import RecognizedFace


def draw_hud(frame: np.ndarray, fps: float) -> None:
    """Draw FPS, resolution, and timestamp onto the frame (in-place, BGR)."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (0, 255, 0)
    thickness = 1

    y = 20
    cv2.putText(frame, f"FPS: {fps:.1f}", (8, y), font, scale, color, thickness)
    y += 22
    cv2.putText(frame, f"{w}x{h}", (8, y), font, scale, color, thickness)
    y += 22
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (8, y), font, scale, color, thickness)


def draw_recognitions(frame_bgr: np.ndarray, faces: list[RecognizedFace]) -> None:
    """Draw bounding boxes, landmarks, and identity labels (in-place, BGR)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    for face in faces:
        det = face.detection
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

        if face.identity != "Unknown":
            # Known face: green.
            color = (0, 255, 0)
            label = f"{face.identity} ({face.similarity:.2f})"
        elif face.similarity > 0.0:
            # Unknown with embedding extracted: cyan.
            color = (255, 255, 0)
            label = f"Unknown ({face.similarity:.2f})"
        elif det.landmarks is not None:
            # Unknown, no match score (empty db): cyan.
            color = (255, 255, 0)
            label = "Unknown"
        else:
            # No landmarks (e.g. UltraFace): yellow confidence only.
            color = (0, 255, 255)
            label = f"{det.confidence:.2f}"

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_bgr, label, (x1, max(y1 - 6, 12)), font, 0.45, color, 1)

        # Red landmark dots.
        if det.landmarks:
            for lm in det.landmarks:
                cv2.circle(frame_bgr, (int(lm.x), int(lm.y)), 2, (0, 0, 255), -1)


def draw_detections(frame_bgr: np.ndarray, detections: list[Detection]) -> None:
    """Draw bounding boxes, landmarks, and confidence labels (in-place, BGR).

    Kept for backward compatibility when recognition is disabled.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for det in detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.confidence:.2f}"
        cv2.putText(frame_bgr, label, (x1, max(y1 - 6, 12)), font, 0.45, (0, 255, 255), 1)
        if det.landmarks:
            for lm in det.landmarks:
                cv2.circle(frame_bgr, (int(lm.x), int(lm.y)), 2, (0, 0, 255), -1)


def draw_detection_hud(
    frame_bgr: np.ndarray,
    detector_name: str,
    face_count: int,
    detection_ms: float,
    embedder_name: str | None = None,
    recognition_ms: float = 0.0,
) -> None:
    """Draw detector/embedder metadata in the bottom-left corner (in-place, BGR)."""
    h = frame_bgr.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    color = (255, 200, 0)  # cyan-ish
    thickness = 1

    lines: list[str] = []
    lines.append(f"Det: {detector_name}")
    det_info = f"Faces: {face_count}  ({detection_ms:.0f} ms)"
    if embedder_name:
        det_info += f"  |  Rec: {embedder_name} ({recognition_ms:.0f} ms)"
    lines.append(det_info)

    y = h - 18 * len(lines) + 4
    for line in lines:
        cv2.putText(frame_bgr, line, (8, y), font, scale, color, thickness)
        y += 18


def draw_enrollment_overlay(
    frame_bgr: np.ndarray,
    detection: Detection | None,
    name: str,
    sample_count: int,
    captured: bool,
    detector_name: str | None,
) -> None:
    """Draw enrollment-specific overlay elements (in-place, BGR)."""
    h, w = frame_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- top banner: semi-transparent dark bar ---
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, 56), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)

    cv2.putText(
        frame_bgr, f"Enrolling: {name}",
        (10, 22), font, 0.6, (255, 255, 255), 1,
    )
    cv2.putText(
        frame_bgr, f"Samples: {sample_count}",
        (10, 46), font, 0.5, (200, 200, 200), 1,
    )

    if detection is not None:
        x1, y1 = int(detection.x1), int(detection.y1)
        x2, y2 = int(detection.x2), int(detection.y2)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if detection.landmarks:
            for lm in detection.landmarks:
                cv2.circle(frame_bgr, (int(lm.x), int(lm.y)), 2, (0, 0, 255), -1)
    else:
        cv2.putText(
            frame_bgr, "No face detected",
            (10, h - 20), font, 0.55, (0, 255, 255), 1,
        )

    # --- capture flash: green border ---
    if captured:
        cv2.rectangle(frame_bgr, (0, 0), (w - 1, h - 1), (0, 255, 0), 4)

    # --- detector info ---
    if detector_name:
        cv2.putText(
            frame_bgr, f"Det: {detector_name}",
            (w - 160, h - 10), font, 0.4, (255, 200, 0), 1,
        )


class Display:
    """OpenCV-based display window."""

    def __init__(self, window_name: str = "Butler"):
        self._window_name = window_name
        self._opened = False

    def show(
        self,
        frame_rgb: np.ndarray,
        fps: float = 0.0,
        detections: list[Detection] | None = None,
        recognized_faces: list[RecognizedFace] | None = None,
        detector_name: str | None = None,
        detection_ms: float = 0.0,
        embedder_name: str | None = None,
        recognition_ms: float = 0.0,
    ) -> None:
        """Convert RGB frame to BGR, draw overlays, and display."""
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if recognized_faces:
            draw_recognitions(bgr, recognized_faces)
        elif detections:
            draw_detections(bgr, detections)
        draw_hud(bgr, fps)
        if detector_name is not None:
            face_count = len(recognized_faces or detections or [])
            draw_detection_hud(
                bgr, detector_name, face_count, detection_ms,
                embedder_name=embedder_name,
                recognition_ms=recognition_ms,
            )
        cv2.imshow(self._window_name, bgr)
        self._opened = True

    def show_enrollment(
        self,
        frame_rgb: np.ndarray,
        fps: float,
        detection: Detection | None,
        name: str,
        sample_count: int,
        captured: bool = False,
        detector_name: str | None = None,
    ) -> None:
        """Convert RGB frame to BGR, draw enrollment overlay, and display."""
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        draw_enrollment_overlay(
            bgr, detection, name, sample_count, captured, detector_name,
        )
        draw_hud(bgr, fps)
        cv2.imshow(self._window_name, bgr)
        self._opened = True

    def should_quit(self) -> bool:
        """Return True if user pressed ESC or 'q'."""
        key = cv2.waitKey(1) & 0xFF
        return key in (27, ord("q"))

    def close(self) -> None:
        if self._opened:
            cv2.destroyWindow(self._window_name)
            self._opened = False
