from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from src.contracts import Float32Array

from ._base import Track, box_iou

# ---------------------------------------------------------------------------
# Kalman filter internals
# ---------------------------------------------------------------------------


def _box_to_state(box: Float32Array) -> Float32Array:
    """Convert [x1, y1, x2, y2, conf] to [cx, cy, w, h, 0, 0, 0, 0]."""
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _state_to_box(state: Float32Array, confidence: float) -> Float32Array:
    """Convert [cx, cy, w, h, ...] back to [x1, y1, x2, y2, conf]."""
    cx, cy = float(state[0]), float(state[1])
    w = max(float(state[2]), 1.0)
    h = max(float(state[3]), 1.0)
    return np.array(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, confidence],
        dtype=np.float32,
    )


# Constant-velocity state transition (8x8): position += velocity per step.
_F = np.eye(8, dtype=np.float32)
_F[0, 4] = 1.0  # cx += vx
_F[1, 5] = 1.0  # cy += vy
_F[2, 6] = 1.0  # w  += vw
_F[3, 7] = 1.0  # h  += vh

# Measurement maps observed [cx, cy, w, h] from state.
_H = np.eye(4, 8, dtype=np.float32)

# Process noise — tuned for face tracking at surveillance distances.
_Q = np.diag(np.array([10.0, 10.0, 10.0, 10.0, 25.0, 25.0, 10.0, 10.0], dtype=np.float32) ** 2)

# Measurement noise.
_R = np.diag(np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32) ** 2)

# Initial covariance — high uncertainty on velocity, moderate on position.
_P0 = np.diag(np.array([10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float32) ** 2)


@dataclass
class _KalmanTrackState:
    """Per-track Kalman filter state."""

    track_id: int
    x: Float32Array  # (8,) state vector
    P: Float32Array  # (8, 8) covariance
    confidence: float
    kps: Float32Array | None  # latest landmarks
    age: int
    hits: int
    missed: int


class KalmanFaceTracker:
    """Face tracker with constant-velocity Kalman filter and Hungarian assignment."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_missed: int = 3,
        smoothing: float = 0.65,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missed = max(0, int(max_missed))
        # smoothing is accepted for API compatibility but not used (Kalman smooths natively).
        self._next_track_id = 1
        self._states: list[_KalmanTrackState] = []

    def reset(self) -> None:
        self._states.clear()
        self._next_track_id = 1

    def predict(self) -> list[Track]:
        """Propagate all Kalman states one step forward; return predicted tracks."""
        result: list[Track] = []
        for s in self._states:
            x_pred = np.asarray(_F @ s.x, dtype=np.float32)
            P_pred = np.asarray(_F @ s.P @ _F.T + _Q, dtype=np.float32)
            # Enforce symmetry for numerical stability.
            s.P = np.asarray((P_pred + P_pred.T) / 2.0, dtype=np.float32)
            s.x = x_pred
            result.append(
                Track(
                    track_id=s.track_id,
                    box=_state_to_box(s.x, s.confidence),
                    kps=None if s.kps is None else np.asarray(s.kps, dtype=np.float32),
                    age=s.age + 1,
                    hits=s.hits,
                    missed=s.missed,
                    matched=False,
                )
            )
            s.age += 1
        return result

    def update(
        self,
        boxes: Float32Array,
        kps: Float32Array | None,
        *,
        max_tracks: int | None = None,
    ) -> list[Track]:
        if max_tracks is not None and max_tracks >= 0:
            boxes = np.asarray(boxes[:max_tracks], dtype=np.float32)
            if kps is not None:
                kps = np.asarray(kps[:max_tracks], dtype=np.float32)

        # --- Predict all existing tracks forward ---
        for s in self._states:
            x_pred = np.asarray(_F @ s.x, dtype=np.float32)
            P_pred = np.asarray(_F @ s.P @ _F.T + _Q, dtype=np.float32)
            s.P = np.asarray((P_pred + P_pred.T) / 2.0, dtype=np.float32)
            s.x = x_pred

        # --- Build IoU cost matrix and run Hungarian assignment ---
        n_tracks = len(self._states)
        n_dets = len(boxes)
        matches, unmatched_tracks, unmatched_dets = self._hungarian_match(boxes, n_tracks, n_dets)

        # --- Update matched tracks with Kalman correction ---
        next_states: list[_KalmanTrackState] = []
        for track_idx, det_idx in matches:
            s = self._states[track_idx]
            det_box = np.asarray(boxes[det_idx], dtype=np.float32)
            z = np.array(
                [
                    (float(det_box[0]) + float(det_box[2])) / 2.0,
                    (float(det_box[1]) + float(det_box[3])) / 2.0,
                    float(det_box[2]) - float(det_box[0]),
                    float(det_box[3]) - float(det_box[1]),
                ],
                dtype=np.float32,
            )
            # Kalman correction.
            S = _H @ s.P @ _H.T + _R
            K = s.P @ _H.T @ np.linalg.inv(S)
            s.x = np.asarray(s.x + K @ (z - _H @ s.x), dtype=np.float32)
            P_upd = np.asarray((np.eye(8, dtype=np.float32) - K @ _H) @ s.P, dtype=np.float32)
            s.P = np.asarray((P_upd + P_upd.T) / 2.0, dtype=np.float32)

            s.confidence = float(det_box[4])
            s.kps = None if kps is None else np.asarray(kps[det_idx], dtype=np.float32)
            s.age += 1
            s.hits += 1
            s.missed = 0
            next_states.append(s)

        # --- Carry forward unmatched tracks ---
        for track_idx in unmatched_tracks:
            s = self._states[track_idx]
            s.missed += 1
            if s.missed <= self.max_missed:
                s.age += 1
                next_states.append(s)

        # --- Start new tracks for unmatched detections ---
        for det_idx in unmatched_dets:
            det_box = np.asarray(boxes[det_idx], dtype=np.float32)
            x0 = _box_to_state(det_box)
            new_state = _KalmanTrackState(
                track_id=self._next_track_id,
                x=x0,
                P=_P0.copy(),
                confidence=float(det_box[4]),
                kps=None if kps is None else np.asarray(kps[det_idx], dtype=np.float32),
                age=1,
                hits=1,
                missed=0,
            )
            self._next_track_id += 1
            next_states.append(new_state)

        # --- Rank and limit ---
        next_states.sort(
            key=lambda s: (-(s.missed == 0), s.missed, -s.confidence, -s.hits),
        )
        if max_tracks is not None and max_tracks >= 0:
            next_states = next_states[:max_tracks]
        next_states.sort(key=lambda s: s.track_id)
        self._states = next_states

        return [
            Track(
                track_id=s.track_id,
                box=_state_to_box(s.x, s.confidence),
                kps=None if s.kps is None else np.asarray(s.kps, dtype=np.float32),
                age=s.age,
                hits=s.hits,
                missed=s.missed,
                matched=s.missed == 0,
            )
            for s in self._states
        ]

    def _hungarian_match(
        self,
        boxes: Float32Array,
        n_tracks: int,
        n_dets: int,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Run Hungarian assignment on IoU cost matrix.

        Returns (matches, unmatched_track_indices, unmatched_det_indices).
        """
        if n_tracks == 0 or n_dets == 0:
            return [], list(range(n_tracks)), list(range(n_dets))

        # Build IoU matrix.
        iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for t_idx, s in enumerate(self._states):
            pred_box = _state_to_box(s.x, s.confidence)
            for d_idx in range(n_dets):
                det_box = np.asarray(boxes[d_idx], dtype=np.float32)
                iou_matrix[t_idx, d_idx] = box_iou(pred_box[:4], det_box[:4])

        # Cost = 1 - IoU; impossible assignments get a large sentinel.
        _SENTINEL = 1e5
        cost = np.where(iou_matrix >= self.iou_threshold, 1.0 - iou_matrix, _SENTINEL)

        row_ind, col_ind = linear_sum_assignment(cost)

        matches: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for r, c in zip(row_ind, col_ind, strict=True):
            if cost[r, c] < _SENTINEL:
                matches.append((int(r), int(c)))
                matched_tracks.add(int(r))
                matched_dets.add(int(c))

        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
        unmatched_dets = [i for i in range(n_dets) if i not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets
