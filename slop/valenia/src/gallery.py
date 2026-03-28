"""Filesystem-backed gallery enrollment, review, and matching."""

from __future__ import annotations

import json
import mimetypes
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import cv2
import numpy as np
from contracts import PipelineLike
from runtime_utils import Float32Array, UInt8Array

_SLUG_CLEAN_RE = re.compile(r"[^a-z0-9]+")
_UNKNOWN_PREFIX = "unknown-"


@dataclass(frozen=True)
class IdentityRecord:
    name: str
    slug: str
    template: Float32Array
    sample_count: int
    preview_filename: str | None


@dataclass(frozen=True)
class UnknownRecord:
    slug: str
    display_name: str
    template: Float32Array
    sample_count: int
    first_seen_epoch: float
    last_seen_epoch: float
    preview_filename: str | None


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
    """Manage confirmed identities and an auto-captured unknown review inbox."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._unknown_root = self.root_dir / "_unknowns"
        self._unknown_root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._records: dict[str, IdentityRecord] = {}
        self._unknown_records: dict[str, UnknownRecord] = {}
        self.load()

    def load(self) -> None:
        records: dict[str, IdentityRecord] = {}
        unknown_records: dict[str, UnknownRecord] = {}
        for identity_dir in sorted(self.root_dir.iterdir()):
            if not identity_dir.is_dir() or identity_dir.name.startswith("_"):
                continue
            record = self._load_identity_record(identity_dir)
            if record is not None:
                records[record.slug] = record

        if self._unknown_root.exists():
            for unknown_dir in sorted(self._unknown_root.iterdir()):
                if not unknown_dir.is_dir():
                    continue
                record = self._load_unknown_record(unknown_dir)
                if record is not None:
                    unknown_records[record.slug] = record

        with self._lock:
            self._records = records
            self._unknown_records = unknown_records

    def enroll(
        self,
        name: str,
        uploads: list[tuple[str, bytes]],
        pipeline: PipelineLike,
    ) -> EnrollmentResult:
        clean_name = _normalize_name(name)
        if not clean_name:
            raise ValueError("Name is required")
        if not uploads:
            raise ValueError("At least one photo is required")

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

        result = self._upsert_identity(
            name=clean_name,
            samples=np.asarray(np.stack(embeddings, axis=0), dtype=np.float32),
            uploads=stored_uploads,
            replace_existing=False,
        )
        return EnrollmentResult(
            name=result.name,
            slug=result.slug,
            accepted_files=tuple(accepted_files),
            rejected_files=tuple(rejected_files),
            sample_count=len(accepted_files),
        )

    def match(self, embedding: Float32Array, threshold: float) -> GalleryMatch:
        with self._lock:
            records = list(self._records.values())

        if not records:
            return GalleryMatch(name=None, slug=None, score=0.0, matched=False)

        probe = _normalize_embedding(embedding)
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

    def capture_unknown(self, embedding: Float32Array, crop_bgr: UInt8Array) -> GalleryMatch:
        probe = _normalize_embedding(embedding)
        crop = np.asarray(crop_bgr, dtype=np.uint8)
        now = time.time()
        with self._lock:
            unknown_records = list(self._unknown_records.values())

        best_record: UnknownRecord | None = None
        best_score = -1.0
        for record in unknown_records:
            score = float(np.dot(probe, record.template))
            if score > best_score:
                best_score = score
                best_record = record

        if best_record is None or best_score < 0.36:
            slug = self._next_unknown_slug()
            samples = np.asarray(np.expand_dims(probe, axis=0), dtype=np.float32)
            record = self._write_unknown_record(
                slug=slug,
                samples=samples,
                crop_bgr=crop,
                first_seen_epoch=now,
                last_seen_epoch=now,
            )
            return GalleryMatch(
                name=record.display_name,
                slug=record.slug,
                score=0.0,
                matched=False,
            )

        sample_count = best_record.sample_count
        first_seen = best_record.first_seen_epoch
        last_seen = best_record.last_seen_epoch
        samples = self._load_samples(self._unknown_dir(best_record.slug) / "samples.npy")
        if samples.size == 0:
            samples = np.asarray(np.expand_dims(probe, axis=0), dtype=np.float32)
            sample_count = 0

        should_store_crop = (now - last_seen) >= 1.5 and sample_count < 24
        if should_store_crop:
            samples = np.asarray(np.vstack([samples, probe]), dtype=np.float32)
            sample_count = int(samples.shape[0])
        record = self._write_unknown_record(
            slug=best_record.slug,
            samples=samples,
            crop_bgr=crop if should_store_crop else None,
            first_seen_epoch=first_seen,
            last_seen_epoch=now,
        )
        return GalleryMatch(
            name=record.display_name,
            slug=record.slug,
            score=max(0.0, best_score),
            matched=False,
        )

    def identities(self) -> list[IdentityRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda record: record.name.lower())

    def unknowns(self) -> list[UnknownRecord]:
        with self._lock:
            records = list(self._unknown_records.values())
        return sorted(records, key=lambda record: record.last_seen_epoch, reverse=True)

    def upload_to_identity(
        self,
        slug: str,
        uploads: list[tuple[str, bytes]],
        pipeline: PipelineLike,
    ) -> EnrollmentResult:
        """Add uploaded photos to an existing identity."""
        with self._lock:
            record = self._records.get(slug)
        if record is None:
            raise ValueError(f"Unknown identity: {slug}")
        if not uploads:
            raise ValueError("At least one photo is required")

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

        result = self._upsert_identity(
            name=record.name,
            samples=np.asarray(np.stack(embeddings, axis=0), dtype=np.float32),
            uploads=stored_uploads,
            replace_existing=False,
            target_slug=slug,
        )
        return EnrollmentResult(
            name=result.name,
            slug=result.slug,
            accepted_files=tuple(accepted_files),
            rejected_files=tuple(rejected_files),
            sample_count=result.sample_count,
        )

    def rename_identity(self, slug: str, new_name: str) -> IdentityRecord:
        clean_name = _normalize_name(new_name)
        if not clean_name:
            raise ValueError("Name is required")

        with self._lock:
            record = self._records.get(slug)
        if record is None:
            raise ValueError(f"Unknown identity: {slug}")

        samples = self._load_samples(self._identity_dir(slug) / "samples.npy")
        uploads = self._load_uploads(self._identity_dir(slug), prefix="upload_")
        if samples.size == 0:
            samples = np.asarray(np.expand_dims(record.template, axis=0), dtype=np.float32)
        return self._write_identity_record(
            name=clean_name,
            slug=slug,
            samples=samples,
            uploads=uploads,
        )

    def promote_unknown(self, unknown_slug: str, name: str) -> EnrollmentResult:
        clean_name = _normalize_name(name)
        if not clean_name:
            raise ValueError("Name is required")

        with self._lock:
            unknown_record = self._unknown_records.get(unknown_slug)
        if unknown_record is None:
            raise ValueError(f"Unknown review item: {unknown_slug}")

        unknown_dir = self._unknown_dir(unknown_slug)
        unknown_samples = self._load_samples(unknown_dir / "samples.npy")
        if unknown_samples.size == 0:
            unknown_samples = np.asarray(
                np.expand_dims(unknown_record.template, axis=0),
                dtype=np.float32,
            )
        uploads = self._load_uploads(unknown_dir, prefix="capture_")
        target_slug = _slugify(clean_name)

        result = self._upsert_identity(
            name=clean_name,
            samples=unknown_samples,
            uploads=uploads,
            replace_existing=False,
            target_slug=target_slug,
        )
        self.delete_unknown(unknown_slug)
        return EnrollmentResult(
            name=result.name,
            slug=result.slug,
            accepted_files=tuple(filename for filename, _ in uploads),
            rejected_files=(),
            sample_count=int(unknown_samples.shape[0]),
        )

    def merge_unknowns(self, target_slug: str, source_slug: str) -> UnknownRecord:
        """Merge source unknown into target unknown, combining samples and captures."""
        if target_slug == source_slug:
            raise ValueError("Cannot merge an unknown into itself")
        with self._lock:
            target_record = self._unknown_records.get(target_slug)
            source_record = self._unknown_records.get(source_slug)
        if target_record is None:
            raise ValueError(f"Unknown review item: {target_slug}")
        if source_record is None:
            raise ValueError(f"Unknown review item: {source_slug}")

        target_dir = self._unknown_dir(target_slug)
        source_dir = self._unknown_dir(source_slug)

        target_samples = self._load_samples(target_dir / "samples.npy")
        source_samples = self._load_samples(source_dir / "samples.npy")
        if target_samples.size == 0:
            target_samples = np.asarray(
                np.expand_dims(target_record.template, axis=0), dtype=np.float32
            )
        if source_samples.size == 0:
            source_samples = np.asarray(
                np.expand_dims(source_record.template, axis=0), dtype=np.float32
            )
        combined_samples = np.asarray(np.vstack([target_samples, source_samples]), dtype=np.float32)

        first_seen = min(target_record.first_seen_epoch, source_record.first_seen_epoch)
        last_seen = max(target_record.last_seen_epoch, source_record.last_seen_epoch)

        source_captures = self._load_uploads(source_dir, prefix="capture_")

        record = self._write_unknown_record(
            slug=target_slug,
            samples=combined_samples,
            crop_bgr=None,
            first_seen_epoch=first_seen,
            last_seen_epoch=last_seen,
        )

        existing_count = len(list(target_dir.glob("capture_*")))
        for idx, (filename, payload) in enumerate(source_captures, start=existing_count + 1):
            suffix = _safe_suffix(filename)
            (target_dir / f"capture_{idx:03d}{suffix}").write_bytes(payload)

        record = self._load_unknown_record(target_dir)
        if record is None:
            raise RuntimeError(f"Failed to reload merged unknown: {target_slug}")
        with self._lock:
            self._unknown_records[target_slug] = record

        self.delete_unknown(source_slug)
        return record

    def delete_unknown(self, unknown_slug: str) -> None:
        unknown_dir = self._unknown_dir(unknown_slug)
        with self._lock:
            self._unknown_records.pop(unknown_slug, None)
        _remove_dir_tree(unknown_dir)

    def list_identity_images(self, slug: str) -> list[str]:
        identity_dir = self._identity_dir(slug)
        if not identity_dir.exists():
            return []
        return sorted(path.name for path in identity_dir.glob("upload_*") if path.is_file())

    def delete_identity(self, slug: str) -> None:
        with self._lock:
            if slug not in self._records:
                raise ValueError(f"Unknown identity: {slug}")
            self._records.pop(slug)
        _remove_dir_tree(self._identity_dir(slug))

    def delete_identity_sample(self, slug: str, filename: str) -> IdentityRecord:
        with self._lock:
            record = self._records.get(slug)
        if record is None:
            raise ValueError(f"Unknown identity: {slug}")

        identity_dir = self._identity_dir(slug)
        upload_files = sorted(path.name for path in identity_dir.glob("upload_*") if path.is_file())
        if filename not in upload_files:
            raise ValueError(f"Sample not found: {filename}")
        if len(upload_files) <= 1:
            raise ValueError("Cannot delete last sample — delete the identity instead")

        sample_idx = upload_files.index(filename)
        samples = self._load_samples(identity_dir / "samples.npy")
        qualities = self._load_sample_qualities(identity_dir, sample_count=int(samples.shape[0]))

        (identity_dir / filename).unlink()

        if samples.size > 0 and sample_idx < samples.shape[0]:
            samples = np.asarray(np.delete(samples, sample_idx, axis=0), dtype=np.float32)
            del qualities[sample_idx]

        remaining_uploads = self._load_uploads(identity_dir, prefix="upload_")
        return self._write_identity_record(
            name=record.name,
            slug=slug,
            samples=samples,
            uploads=remaining_uploads,
            qualities=qualities if any(q != 1.0 for q in qualities) else None,
        )

    def read_image(self, kind: str, slug: str, filename: str) -> tuple[bytes, str]:
        if not filename or Path(filename).name != filename:
            raise ValueError("Invalid filename")
        if kind == "identity":
            base_dir = self._identity_dir(slug)
        elif kind == "unknown":
            base_dir = self._unknown_dir(slug)
        else:
            raise ValueError("Unknown image kind")

        path = base_dir / filename
        if not path.exists() or not path.is_file():
            raise ValueError("Image not found")

        content_type, _ = mimetypes.guess_type(path.name)
        return path.read_bytes(), content_type or "application/octet-stream"

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def _upsert_identity(
        self,
        *,
        name: str,
        samples: Float32Array,
        uploads: list[tuple[str, bytes]],
        replace_existing: bool,
        target_slug: str | None = None,
    ) -> IdentityRecord:
        slug = target_slug or _slugify(name)
        existing_samples = self._load_samples(self._identity_dir(slug) / "samples.npy")
        existing_uploads = (
            []
            if replace_existing
            else self._load_uploads(self._identity_dir(slug), prefix="upload_")
        )
        combined_samples = np.asarray(samples, dtype=np.float32)
        if not replace_existing and existing_samples.size > 0:
            combined_samples = np.asarray(
                np.vstack([existing_samples, combined_samples]),
                dtype=np.float32,
            )
        combined_uploads = list(existing_uploads)
        combined_uploads.extend(uploads)
        return self._write_identity_record(
            name=name,
            slug=slug,
            samples=combined_samples,
            uploads=combined_uploads,
        )

    def _load_identity_record(self, identity_dir: Path) -> IdentityRecord | None:
        meta_path = identity_dir / "meta.json"
        template_path = identity_dir / "template.npy"
        if not meta_path.exists() or not template_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            template = np.asarray(np.load(template_path, allow_pickle=False), dtype=np.float32)
        except (OSError, ValueError, TypeError):
            return None

        preview_filename = self._pick_preview_filename(identity_dir, prefix="upload_")
        return IdentityRecord(
            name=str(meta.get("name", identity_dir.name)),
            slug=str(meta.get("slug", identity_dir.name)),
            template=_normalize_embedding(template.reshape(-1)),
            sample_count=int(meta.get("sample_count", 0)),
            preview_filename=preview_filename,
        )

    def _load_unknown_record(self, unknown_dir: Path) -> UnknownRecord | None:
        meta_path = unknown_dir / "meta.json"
        template_path = unknown_dir / "template.npy"
        if not meta_path.exists() or not template_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            template = np.asarray(np.load(template_path, allow_pickle=False), dtype=np.float32)
        except (OSError, ValueError, TypeError):
            return None

        preview_filename = self._pick_preview_filename(unknown_dir, prefix="capture_")
        slug = str(meta.get("slug", unknown_dir.name))
        return UnknownRecord(
            slug=slug,
            display_name=str(meta.get("display_name", slug)),
            template=_normalize_embedding(template.reshape(-1)),
            sample_count=int(meta.get("sample_count", 0)),
            first_seen_epoch=float(meta.get("first_seen_epoch", 0.0)),
            last_seen_epoch=float(meta.get("last_seen_epoch", 0.0)),
            preview_filename=preview_filename,
        )

    def enrich_identity(
        self,
        slug: str,
        embedding: Float32Array,
        quality: float,
        /,
        *,
        max_samples: int = 48,
        diversity_threshold: float = 0.95,
        crop_bgr: UInt8Array | None = None,
    ) -> bool:
        """Add a new embedding sample to an existing identity if diverse enough."""
        probe = _normalize_embedding(embedding)
        with self._lock:
            record = self._records.get(slug)
        if record is None:
            return False

        identity_dir = self._identity_dir(slug)
        samples = self._load_samples(identity_dir / "samples.npy")
        if samples.size == 0:
            samples = np.asarray(np.expand_dims(record.template, axis=0), dtype=np.float32)

        if not _check_diversity(probe, samples, max_similarity=diversity_threshold):
            return False

        qualities = self._load_sample_qualities(identity_dir, sample_count=int(samples.shape[0]))
        uploads = self._load_uploads(identity_dir, prefix="upload_")

        if samples.shape[0] >= max_samples:
            worst_idx = int(np.argmin(qualities))
            samples = np.asarray(np.delete(samples, worst_idx, axis=0), dtype=np.float32)
            del qualities[worst_idx]
            if worst_idx < len(uploads):
                del uploads[worst_idx]

        samples = np.asarray(np.vstack([samples, probe[np.newaxis, :]]), dtype=np.float32)
        qualities.append(quality)

        if crop_bgr is not None:
            ok, encoded = cv2.imencode(".jpg", crop_bgr)
            if ok:
                uploads.append(("enrichment.jpg", encoded.tobytes()))

        self._write_identity_record(
            name=record.name,
            slug=slug,
            samples=samples,
            uploads=uploads,
            qualities=qualities,
        )
        return True

    def _load_sample_qualities(self, identity_dir: Path, *, sample_count: int) -> list[float]:
        meta_path = identity_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                raw = meta.get("sample_qualities")
                if isinstance(raw, list) and len(raw) == sample_count:
                    return [float(v) for v in raw]
            except (OSError, ValueError, TypeError):
                pass
        return [1.0] * sample_count

    def _write_identity_record(
        self,
        *,
        name: str,
        slug: str,
        samples: Float32Array,
        uploads: list[tuple[str, bytes]],
        qualities: list[float] | None = None,
    ) -> IdentityRecord:
        identity_dir = self._identity_dir(slug)
        identity_dir.mkdir(parents=True, exist_ok=True)
        for stale_file in identity_dir.glob("upload_*"):
            stale_file.unlink(missing_ok=True)

        normalized_samples = np.asarray(samples, dtype=np.float32)
        template = _template_from_samples(normalized_samples, qualities)
        np.save(identity_dir / "template.npy", template, allow_pickle=False)
        np.save(identity_dir / "samples.npy", normalized_samples, allow_pickle=False)
        meta: dict[str, object] = {
            "name": name,
            "slug": slug,
            "sample_count": int(normalized_samples.shape[0]),
        }
        if qualities is not None:
            meta["sample_qualities"] = qualities
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
            template=template,
            sample_count=int(normalized_samples.shape[0]),
            preview_filename=self._pick_preview_filename(identity_dir, prefix="upload_"),
        )
        with self._lock:
            self._records[slug] = record
        return record

    def _write_unknown_record(
        self,
        *,
        slug: str,
        samples: Float32Array,
        crop_bgr: UInt8Array | None,
        first_seen_epoch: float,
        last_seen_epoch: float,
    ) -> UnknownRecord:
        unknown_dir = self._unknown_dir(slug)
        unknown_dir.mkdir(parents=True, exist_ok=True)

        normalized_samples = np.asarray(samples, dtype=np.float32)
        template = _template_from_samples(normalized_samples)
        np.save(unknown_dir / "template.npy", template, allow_pickle=False)
        np.save(unknown_dir / "samples.npy", normalized_samples, allow_pickle=False)
        if crop_bgr is not None:
            existing_count = len(list(unknown_dir.glob("capture_*")))
            suffix = ".jpg"
            ok, encoded = cv2.imencode(suffix, crop_bgr)
            if ok:
                output_path = unknown_dir / f"capture_{existing_count + 1:03d}{suffix}"
                output_path.write_bytes(encoded.tobytes())

        meta = {
            "slug": slug,
            "display_name": slug,
            "sample_count": int(normalized_samples.shape[0]),
            "first_seen_epoch": float(first_seen_epoch),
            "last_seen_epoch": float(last_seen_epoch),
        }
        (unknown_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        record = UnknownRecord(
            slug=slug,
            display_name=slug,
            template=template,
            sample_count=int(normalized_samples.shape[0]),
            first_seen_epoch=float(first_seen_epoch),
            last_seen_epoch=float(last_seen_epoch),
            preview_filename=self._pick_preview_filename(unknown_dir, prefix="capture_"),
        )
        with self._lock:
            self._unknown_records[slug] = record
        return record

    def _identity_dir(self, slug: str) -> Path:
        return self.root_dir / slug

    def _unknown_dir(self, slug: str) -> Path:
        return self._unknown_root / slug

    def _next_unknown_slug(self) -> str:
        with self._lock:
            existing = set(self._unknown_records)
        next_index = 1
        while True:
            slug = f"{_UNKNOWN_PREFIX}{next_index:04d}"
            if slug not in existing and not self._unknown_dir(slug).exists():
                return slug
            next_index += 1

    def _pick_preview_filename(self, base_dir: Path, *, prefix: str) -> str | None:
        files = sorted(path.name for path in base_dir.glob(f"{prefix}*") if path.is_file())
        if not files:
            return None
        return files[0]

    def _load_uploads(self, base_dir: Path, *, prefix: str) -> list[tuple[str, bytes]]:
        if not base_dir.exists():
            return []
        uploads: list[tuple[str, bytes]] = []
        for path in sorted(base_dir.glob(f"{prefix}*")):
            if not path.is_file():
                continue
            uploads.append((path.name, path.read_bytes()))
        return uploads

    def _load_samples(self, path: Path) -> Float32Array:
        if not path.exists():
            return np.asarray([], dtype=np.float32)
        try:
            samples = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        except (OSError, ValueError):
            return np.asarray([], dtype=np.float32)
        if samples.size == 0:
            return np.asarray([], dtype=np.float32)
        if samples.ndim == 1:
            samples = np.asarray(np.expand_dims(samples, axis=0), dtype=np.float32)
        return np.asarray(samples, dtype=np.float32)


def _decode_image(payload: bytes) -> UInt8Array | None:
    array = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    return np.asarray(frame, dtype=np.uint8)


def _normalize_name(value: str) -> str:
    return " ".join(value.strip().split())


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


def _normalize_embedding(embedding: Float32Array) -> Float32Array:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector
    return np.asarray(vector / norm, dtype=np.float32)


def _template_from_samples(
    samples: Float32Array,
    qualities: list[float] | None = None,
) -> Float32Array:
    matrix = np.asarray(samples, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = np.asarray(np.expand_dims(matrix, axis=0), dtype=np.float32)
    if qualities is not None and len(qualities) == matrix.shape[0]:
        weights = np.asarray(qualities, dtype=np.float32)
        weight_sum = float(np.sum(weights))
        if weight_sum > 0.0:
            template = np.asarray(
                np.sum(matrix * weights[:, np.newaxis] / weight_sum, axis=0),
                dtype=np.float32,
            )
            return _normalize_embedding(template)
    template = np.asarray(np.mean(matrix, axis=0), dtype=np.float32)
    return _normalize_embedding(template)


def _check_diversity(
    probe: Float32Array,
    existing_samples: Float32Array,
    *,
    max_similarity: float = 0.95,
) -> bool:
    """Return True if the probe is diverse enough (max cosine sim < threshold)."""
    matrix = np.asarray(existing_samples, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = np.asarray(np.expand_dims(matrix, axis=0), dtype=np.float32)
    similarities = matrix @ np.asarray(probe, dtype=np.float32)
    return bool(float(np.max(similarities)) < max_similarity)


def _remove_dir_tree(path: Path) -> None:
    if not path.exists():
        return
    shutil.rmtree(path, ignore_errors=True)
