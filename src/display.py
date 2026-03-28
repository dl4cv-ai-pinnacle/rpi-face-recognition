"""Frame overlay rendering — HUD, bounding boxes, labels, landmarks.

Origins:
- draw_hud, draw_detection_hud, draw_enrollment_overlay, Display class: Shalaiev display.py
- annotate_in_place double-pass text rendering: Valenia live_runtime.py
- GUI-backend fallback detection: Avdieienko display/renderer.py
"""

from __future__ import annotations

from datetime import datetime

import cv2

from src.contracts import UInt8Array
from src.gallery import GalleryMatch
from src.tracking import Track


def draw_hud(frame: UInt8Array, fps: float) -> None:
    """Draw FPS, resolution, and timestamp onto the frame (in-place, BGR)."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 20
    cv2.putText(frame, f"FPS: {fps:.1f}", (8, y), font, 0.5, (0, 255, 0), 1)
    y += 22
    cv2.putText(frame, f"{w}x{h}", (8, y), font, 0.5, (0, 255, 0), 1)
    y += 22
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (8, y), font, 0.5, (0, 255, 0), 1)


def draw_detection_hud(
    frame_bgr: UInt8Array,
    detector_name: str,
    face_count: int,
    detection_ms: float,
    embedder_name: str | None = None,
    recognition_ms: float = 0.0,
) -> None:
    """Draw detector/embedder metadata in the bottom-left corner (in-place, BGR)."""
    h = frame_bgr.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX

    lines: list[str] = [f"Det: {detector_name}"]
    det_info = f"Faces: {face_count}  ({detection_ms:.0f} ms)"
    if embedder_name:
        det_info += f"  |  Rec: {embedder_name} ({recognition_ms:.0f} ms)"
    lines.append(det_info)

    y = h - 18 * len(lines) + 4
    for line in lines:
        cv2.putText(frame_bgr, line, (8, y), font, 0.45, (255, 200, 0), 1)
        y += 18


def annotate_faces(
    frame_bgr: UInt8Array,
    tracks: list[Track],
    matches: list[GalleryMatch | None],
) -> None:
    """Draw bounding boxes, labels (double-pass), and landmarks (in-place, BGR)."""
    for track, match in zip(tracks, matches, strict=True):
        box = track.box
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        color = (0, 220, 140)
        if match is not None and match.matched:
            color = (64, 224, 208)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color=color, thickness=2)

        label: str | None = None
        if match is not None and match.name is not None:
            label = match.name

        if label is not None:
            # Double-pass: dark outline then colored foreground (Valenia pattern).
            pos = (x1, max(16, y1 - 6))
            cv2.putText(frame_bgr, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(
                frame_bgr, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (12, 18, 20), 1
            )

        if track.kps is not None:
            for point in track.kps:
                cv2.circle(frame_bgr, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)


class Display:
    """OpenCV-based display window with GUI-backend fallback (Avdieienko pattern)."""

    def __init__(self, window_name: str = "Face Recognition") -> None:
        self._window_name = window_name
        self._available = True
        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        except cv2.error:
            self._available = False

    def show(self, frame_bgr: UInt8Array) -> None:
        if self._available:
            cv2.imshow(self._window_name, frame_bgr)

    def should_quit(self) -> bool:
        if not self._available:
            return False
        key = cv2.waitKey(1) & 0xFF
        return key in (27, ord("q"))

    def close(self) -> None:
        if self._available:
            cv2.destroyWindow(self._window_name)
