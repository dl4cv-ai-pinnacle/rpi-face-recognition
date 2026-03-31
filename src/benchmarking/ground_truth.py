"""Ground-truth types, dataset loaders, and enrollment helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from src.contracts import PipelineLike
from src.gallery import GalleryStore

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GTFace:
    """Single annotated face in a frame."""

    subject_id: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass(frozen=True)
class GTFrame:
    """All annotated faces in one frame."""

    frame_index: int
    faces: list[GTFace]


@dataclass(frozen=True)
class VideoClip:
    """A single video clip with frame paths and per-frame ground truth."""

    clip_id: str
    identity: str
    split: str
    frame_paths: list[Path]
    gt_frames: list[GTFrame]


# ---------------------------------------------------------------------------
# Generic JSON ground truth
# ---------------------------------------------------------------------------


def load_gt_json(gt_path: Path) -> list[GTFrame]:
    """Load ground truth from simple JSON format.

    Expected:
    ``[{"frame": 0, "faces": [{"id": "alice", "bbox": [x1, y1, x2, y2]}, ...]}, ...]``
    """
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    frames: list[GTFrame] = []
    for entry in data:
        faces = [
            GTFace(subject_id=f["id"], bbox=tuple(f["bbox"]))  # type: ignore[arg-type]
            for f in entry["faces"]
        ]
        frames.append(GTFrame(frame_index=entry["frame"], faces=faces))
    return frames


# ---------------------------------------------------------------------------
# FANVID dataset loader
# ---------------------------------------------------------------------------


def load_fanvid_gt(
    fanvid_dir: Path,
    *,
    max_clips: int = 0,
    split: str = "both",
) -> list[VideoClip]:
    """Load FANVID celebrity dataset: clips with per-frame bbox + identity GT.

    Expected layout::

        fanvid_dir/
            annotations/
                dataset_celebs.csv
                celebrity_annotations_LR.csv
            frames/
                train/<identity>/<clip_id>/<frame>.png
                test/<identity>/<clip_id>/<frame>.png
    """
    ann_dir = fanvid_dir / "annotations"
    clips_csv = ann_dir / "dataset_celebs.csv"
    annotations_csv = ann_dir / "celebrity_annotations_LR.csv"

    if not clips_csv.exists():
        msg = f"FANVID clips CSV not found: {clips_csv}"
        raise FileNotFoundError(msg)
    if not annotations_csv.exists():
        msg = f"FANVID annotations CSV not found: {annotations_csv}"
        raise FileNotFoundError(msg)

    # Parse clip metadata.
    clip_rows = _read_csv(clips_csv)
    if split != "both":
        clip_rows = [r for r in clip_rows if r["Split"].strip().lower() == split]

    # Parse annotations: (video_id, frame_no) -> faces.
    ann_by_key: dict[tuple[str, int], list[GTFace]] = {}
    for row in _read_csv(annotations_csv):
        video_id = row["VideoID"].strip()
        frame_no = int(row["FrameNo"].strip())
        face = GTFace(
            subject_id=row["IdentityOrText"].strip(),
            bbox=(
                float(row["BoxLeft"].strip()),
                float(row["BoxTop"].strip()),
                float(row["BoxRight"].strip()),
                float(row["BoxBottom"].strip()),
            ),
        )
        ann_by_key.setdefault((video_id, frame_no), []).append(face)

    # Build clips from downloaded frames.
    clips: list[VideoClip] = []
    frames_dir = fanvid_dir / "frames"

    for row in clip_rows:
        name = row["Name"].strip()
        clip_id = row["Clip ID"].strip()
        video_id = row["Video ID"].strip()
        clip_split = row["Split"].strip().lower()

        clip_dir = frames_dir / clip_split / name / clip_id
        if not clip_dir.exists():
            continue  # Clip not downloaded yet.

        frame_paths = sorted(clip_dir.glob("*.jpg"))
        if not frame_paths:
            # Fall back to legacy PNG extraction.
            frame_paths = sorted(clip_dir.glob("*.png"))
        if not frame_paths:
            continue

        # Build GT frames aligned to frame_paths order.
        gt_frames: list[GTFrame] = []
        for seq_idx, fp in enumerate(frame_paths):
            frame_no = int(fp.stem)
            faces = ann_by_key.get((video_id, frame_no), [])
            gt_frames.append(GTFrame(frame_index=seq_idx, faces=faces))

        clips.append(
            VideoClip(
                clip_id=f"{name}/{clip_id}",
                identity=name,
                split=clip_split,
                frame_paths=frame_paths,
                gt_frames=gt_frames,
            )
        )

        if 0 < max_clips <= len(clips):
            break

    return clips


# ---------------------------------------------------------------------------
# Gallery enrollment
# ---------------------------------------------------------------------------


def enroll_gallery_from_dir(
    enroll_dir: Path,
    gallery: GalleryStore,
    pipeline: PipelineLike,
) -> list[str]:
    """Enroll identities from ``enroll_dir/<subject>/<image>.jpg``.

    Returns list of successfully enrolled subject names.
    """
    enrolled: list[str] = []
    for subject_dir in sorted(enroll_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        name = subject_dir.name
        uploads: list[tuple[str, bytes]] = []
        for img_path in sorted(subject_dir.glob("*")):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                uploads.append((img_path.name, img_path.read_bytes()))
        if uploads:
            try:
                gallery.enroll(name, uploads, pipeline)
                enrolled.append(name)
            except ValueError:
                pass  # Skip subjects where no face is detected.
    return enrolled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
