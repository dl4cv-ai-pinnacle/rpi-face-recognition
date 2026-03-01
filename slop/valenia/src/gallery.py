"""Filesystem-backed gallery enrollment and matching."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import cv2
import numpy as np
from contracts import PipelineLike
from runtime_utils import Float32Array, UInt8Array

_SLUG_CLEAN_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class IdentityRecord:
    name: str
    slug: str
    template: Float32Array
    sample_count: int


@dataclass(frozen=True)
class GalleryMatch:
    name: str | None
    slug: str | None
    score: float
    matched: bool


@dataclass(frozen=True)
class EnrollmentResult:
    name: str
    slug: str
    accepted_files: tuple[str, ...]
    rejected_files: tuple[str, ...]
    sample_count: int


class GalleryStore:
    """Manage simple on-disk identity templates."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._records: dict[str, IdentityRecord] = {}
        self.load()

    def load(self) -> None:
        records: dict[str, IdentityRecord] = {}
        for identity_dir in sorted(self.root_dir.iterdir()):
            if not identity_dir.is_dir():
                continue
            record = self._load_record(identity_dir)
            if record is not None:
                records[record.slug] = record
        with self._lock:
            self._records = records

    def enroll(
        self,
        name: str,
        uploads: list[tuple[str, bytes]],
        pipeline: PipelineLike,
    ) -> EnrollmentResult:
        clean_name = " ".join(name.strip().split())
        if not clean_name:
            raise ValueError("Name is required")
        if not uploads:
            raise ValueError("At least one photo is required")

        slug = _slugify(clean_name)
        accepted_files: list[str] = []
        rejected_files: list[str] = []
        embeddings: list[Float32Array] = []
        stored_uploads: list[tuple[str, bytes]] = []

        for filename, payload in uploads:
            label = filename or "upload"
            if not payload:
                rejected_files.append(f"{label}: empty file")
                continue
            frame_bgr = _decode_image(payload)
            if frame_bgr is None:
                rejected_files.append(f"{label}: unreadable image")
                continue

            det, _ = pipeline.detect(frame_bgr)
            if det.kps is None or len(det.boxes) != 1:
                rejected_files.append(f"{label}: expected exactly one face")
                continue

            emb, _ = pipeline.embed_from_kps(frame_bgr, det.kps[0])
            embeddings.append(np.asarray(emb, dtype=np.float32))
            stored_uploads.append((label, payload))
            accepted_files.append(label)

        if not embeddings:
            raise ValueError("No usable photos. Each upload must contain exactly one clear face.")

        samples = np.asarray(np.stack(embeddings, axis=0), dtype=np.float32)
        template = np.asarray(np.mean(samples, axis=0), dtype=np.float32)
        norm = float(np.linalg.norm(template))
        if norm > 0.0:
            template = np.asarray(template / norm, dtype=np.float32)

        self._write_record(
            name=clean_name,
            slug=slug,
            template=template,
            samples=samples,
            uploads=stored_uploads,
        )
        return EnrollmentResult(
            name=clean_name,
            slug=slug,
            accepted_files=tuple(accepted_files),
            rejected_files=tuple(rejected_files),
            sample_count=len(accepted_files),
        )

    def match(self, embedding: Float32Array, threshold: float) -> GalleryMatch:
        with self._lock:
            records = list(self._records.values())

        if not records:
            return GalleryMatch(name=None, slug=None, score=0.0, matched=False)

        probe = np.asarray(embedding, dtype=np.float32)
        best_record: IdentityRecord | None = None
        best_score = -1.0
        for record in records:
            score = float(np.dot(probe, record.template))
            if score > best_score:
                best_score = score
                best_record = record

        if best_record is None or best_score < threshold:
            return GalleryMatch(name=None, slug=None, score=max(0.0, best_score), matched=False)

        return GalleryMatch(
            name=best_record.name,
            slug=best_record.slug,
            score=best_score,
            matched=True,
        )

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def _load_record(self, identity_dir: Path) -> IdentityRecord | None:
        meta_path = identity_dir / "meta.json"
        template_path = identity_dir / "template.npy"
        if not meta_path.exists() or not template_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            template = np.asarray(np.load(template_path, allow_pickle=False), dtype=np.float32)
        except (OSError, ValueError, TypeError):
            return None

        template = np.asarray(template.reshape(-1), dtype=np.float32)
        norm = float(np.linalg.norm(template))
        if norm > 0.0:
            template = np.asarray(template / norm, dtype=np.float32)

        name = str(meta.get("name", identity_dir.name))
        slug = str(meta.get("slug", identity_dir.name))
        sample_count = int(meta.get("sample_count", 0))
        return IdentityRecord(
            name=name,
            slug=slug,
            template=template,
            sample_count=sample_count,
        )

    def _write_record(
        self,
        *,
        name: str,
        slug: str,
        template: Float32Array,
        samples: Float32Array,
        uploads: list[tuple[str, bytes]],
    ) -> None:
        identity_dir = self.root_dir / slug
        identity_dir.mkdir(parents=True, exist_ok=True)
        for stale_file in identity_dir.glob("upload_*"):
            stale_file.unlink(missing_ok=True)

        np.save(identity_dir / "template.npy", template, allow_pickle=False)
        np.save(identity_dir / "samples.npy", samples, allow_pickle=False)
        meta = {
            "name": name,
            "slug": slug,
            "sample_count": int(samples.shape[0]),
        }
        (identity_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for idx, (filename, payload) in enumerate(uploads, start=1):
            suffix = _safe_suffix(filename)
            (identity_dir / f"upload_{idx:03d}{suffix}").write_bytes(payload)

        record = IdentityRecord(
            name=name,
            slug=slug,
            template=np.asarray(template, dtype=np.float32),
            sample_count=int(samples.shape[0]),
        )
        with self._lock:
            self._records[slug] = record


def _decode_image(payload: bytes) -> UInt8Array | None:
    array = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    return np.asarray(frame, dtype=np.uint8)


def _slugify(value: str) -> str:
    slug = _SLUG_CLEAN_RE.sub("-", value.strip().lower()).strip("-")
    if not slug:
        slug = "person"
    return slug


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if not suffix:
        return ".jpg"
    if 1 < len(suffix) <= 10 and suffix[1:].isalnum():
        return suffix
    return ".jpg"
