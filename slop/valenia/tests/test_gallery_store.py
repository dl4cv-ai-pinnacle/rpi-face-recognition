from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from contracts import FrameResultLike
from gallery import GalleryStore
from runtime_utils import Float32Array, UInt8Array


def _embedding(values: list[float]) -> Float32Array:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector = np.asarray(vector / norm, dtype=np.float32)
    return vector


def _crop() -> UInt8Array:
    return np.ones((8, 8, 3), dtype=np.uint8)


@dataclass(frozen=True)
class _FakeDetection:
    boxes: Float32Array
    kps: Float32Array | None


class _StubPipeline:
    """Minimal pipeline that always returns one face with a deterministic embedding."""

    def __init__(self, emb: Float32Array) -> None:
        self._emb = emb

    def detect(self, frame_bgr: UInt8Array) -> tuple[_FakeDetection, float]:
        del frame_bgr
        boxes = np.asarray([[10, 10, 50, 50, 0.99]], dtype=np.float32)
        kps = np.asarray(
            [[[20, 20], [30, 20], [25, 30], [22, 40], [28, 40]]],
            dtype=np.float32,
        )
        return _FakeDetection(boxes=boxes, kps=kps), 1.0

    def embed_from_kps(
        self, frame_bgr: UInt8Array, landmarks: Float32Array
    ) -> tuple[Float32Array, float]:
        del frame_bgr, landmarks
        return self._emb, 1.0

    def process_frame(
        self, frame_bgr: UInt8Array, /, max_faces: int | None = None
    ) -> FrameResultLike:
        del frame_bgr, max_faces
        raise NotImplementedError


def test_gallery_store_can_capture_promote_and_rename_unknowns(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "gallery")

    unknown_match = store.capture_unknown(_embedding([1.0, 0.0, 0.0]), _crop())

    assert unknown_match.matched is False
    assert unknown_match.slug == "unknown-0001"
    assert unknown_match.name == "unknown-0001"
    assert len(store.unknowns()) == 1

    promote_result = store.promote_unknown("unknown-0001", "Alice")

    assert promote_result.slug == "alice"
    assert store.count() == 1
    assert len(store.unknowns()) == 0

    renamed = store.rename_identity("alice", "Alice Cooper")

    assert renamed.name == "Alice Cooper"
    identities = store.identities()
    assert len(identities) == 1
    assert identities[0].name == "Alice Cooper"


def test_gallery_store_enrich_identity_adds_diverse_sample(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "gallery")

    store.capture_unknown(_embedding([1.0, 0.0, 0.0]), _crop())
    store.promote_unknown("unknown-0001", "Alice")

    different_emb = _embedding([0.6, 0.8, 0.0])
    added = store.enrich_identity("alice", different_emb, 0.7)

    assert added is True
    identities = store.identities()
    assert identities[0].sample_count == 2


def test_gallery_store_enrich_rejects_near_duplicate(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "gallery")

    emb = _embedding([1.0, 0.0, 0.0])
    store.capture_unknown(emb, _crop())
    store.promote_unknown("unknown-0001", "Alice")

    near_dup = _embedding([1.0, 0.001, 0.0])
    added = store.enrich_identity("alice", near_dup, 0.7)

    assert added is False
    identities = store.identities()
    assert identities[0].sample_count == 1


def test_gallery_store_enrich_drops_lowest_quality_at_cap(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "gallery")

    store.capture_unknown(_embedding([1.0, 0.0, 0.0]), _crop())
    store.promote_unknown("unknown-0001", "Alice")

    store.enrich_identity("alice", _embedding([0.6, 0.8, 0.0]), 0.3, max_samples=2)
    store.enrich_identity("alice", _embedding([0.0, 0.6, 0.8]), 0.9, max_samples=2)

    identities = store.identities()
    assert identities[0].sample_count == 2


def test_gallery_store_enrich_returns_false_for_unknown_slug(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "gallery")
    added = store.enrich_identity("nonexistent", _embedding([1.0, 0.0, 0.0]), 0.5)
    assert added is False


def test_gallery_store_delete_identity(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "gallery")

    store.capture_unknown(_embedding([1.0, 0.0, 0.0]), _crop())
    store.promote_unknown("unknown-0001", "Alice")
    assert store.count() == 1

    store.delete_identity("alice")

    assert store.count() == 0
    assert len(store.identities()) == 0


def test_gallery_store_delete_identity_unknown_slug_raises(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "gallery")
    import pytest

    with pytest.raises(ValueError, match="Unknown identity"):
        store.delete_identity("nonexistent")


def test_gallery_store_list_identity_images(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "gallery")

    store.capture_unknown(_embedding([1.0, 0.0, 0.0]), _crop())
    store.promote_unknown("unknown-0001", "Alice")

    images = store.list_identity_images("alice")
    assert len(images) >= 1
    assert all(name.startswith("upload_") for name in images)
    assert store.list_identity_images("nonexistent") == []


def test_gallery_store_delete_identity_sample(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "gallery")

    store.capture_unknown(_embedding([1.0, 0.0, 0.0]), _crop())
    store.promote_unknown("unknown-0001", "Alice")

    store.capture_unknown(_embedding([0.0, 1.0, 0.0]), _crop())
    store.promote_unknown("unknown-0001", "Alice")

    images = store.list_identity_images("alice")
    assert len(images) == 2

    updated = store.delete_identity_sample("alice", images[0])
    assert updated.sample_count == 1

    remaining = store.list_identity_images("alice")
    assert len(remaining) == 1


def test_gallery_store_enrich_stores_crop_image(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "gallery")

    store.capture_unknown(_embedding([1.0, 0.0, 0.0]), _crop())
    store.promote_unknown("unknown-0001", "Alice")

    images_before = store.list_identity_images("alice")
    assert len(images_before) == 1

    different_emb = _embedding([0.6, 0.8, 0.0])
    crop = np.ones((16, 16, 3), dtype=np.uint8) * 128
    added = store.enrich_identity("alice", different_emb, 0.7, crop_bgr=crop)

    assert added is True
    images_after = store.list_identity_images("alice")
    assert len(images_after) == 2


def test_gallery_store_enrich_evicts_upload_at_cap(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "gallery")

    store.capture_unknown(_embedding([1.0, 0.0, 0.0]), _crop())
    store.promote_unknown("unknown-0001", "Alice")

    crop = np.ones((16, 16, 3), dtype=np.uint8) * 128
    store.enrich_identity("alice", _embedding([0.6, 0.8, 0.0]), 0.3, max_samples=2, crop_bgr=crop)
    store.enrich_identity("alice", _embedding([0.0, 0.6, 0.8]), 0.9, max_samples=2, crop_bgr=crop)

    identities = store.identities()
    assert identities[0].sample_count == 2
    images = store.list_identity_images("alice")
    assert len(images) == 2


def test_gallery_store_promoting_to_existing_name_merges_samples(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "gallery")

    first_unknown = store.capture_unknown(_embedding([1.0, 0.0, 0.0]), _crop())
    store.promote_unknown(str(first_unknown.slug), "Alice")

    second_unknown = store.capture_unknown(_embedding([1.0, 0.1, 0.0]), _crop())
    store.promote_unknown(str(second_unknown.slug), "Alice")

    identities = store.identities()
    assert len(identities) == 1
    assert identities[0].slug == "alice"
    assert identities[0].sample_count >= 2


def test_gallery_store_upload_to_identity(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "gallery")

    store.capture_unknown(_embedding([1.0, 0.0, 0.0]), _crop())
    store.promote_unknown("unknown-0001", "Alice")

    images_before = store.list_identity_images("alice")
    assert len(images_before) == 1

    pipeline = _StubPipeline(_embedding([0.6, 0.8, 0.0]))
    face_jpeg = b"fake-image-data"
    result = store.upload_to_identity("alice", [("extra.jpg", face_jpeg)], pipeline)

    assert result.slug == "alice"
    assert len(result.accepted_files) == 1
    assert result.sample_count == 2

    images_after = store.list_identity_images("alice")
    assert len(images_after) == 2

    identities = store.identities()
    assert identities[0].sample_count == 2


def test_gallery_store_merge_unknowns_combines_samples(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "gallery")

    store.capture_unknown(_embedding([1.0, 0.0, 0.0]), _crop())
    store.capture_unknown(_embedding([0.0, 1.0, 0.0]), _crop())

    assert len(store.unknowns()) == 2
    slugs = [u.slug for u in store.unknowns()]
    target, source = sorted(slugs)

    result = store.merge_unknowns(target, source)

    assert result.slug == target
    assert result.sample_count == 2
    assert len(store.unknowns()) == 1
    assert store.unknowns()[0].slug == target


def test_gallery_store_merge_unknowns_into_self_raises(tmp_path: Path) -> None:
    import pytest

    store = GalleryStore(tmp_path / "gallery")
    store.capture_unknown(_embedding([1.0, 0.0, 0.0]), _crop())

    with pytest.raises(ValueError, match="Cannot merge an unknown into itself"):
        store.merge_unknowns("unknown-0001", "unknown-0001")


def test_gallery_store_merge_unknowns_nonexistent_raises(tmp_path: Path) -> None:
    import pytest

    store = GalleryStore(tmp_path / "gallery")
    store.capture_unknown(_embedding([1.0, 0.0, 0.0]), _crop())

    with pytest.raises(ValueError, match="Unknown review item"):
        store.merge_unknowns("unknown-0001", "nonexistent")

    with pytest.raises(ValueError, match="Unknown review item"):
        store.merge_unknowns("nonexistent", "unknown-0001")


def test_gallery_store_upload_to_identity_unknown_slug_raises(tmp_path: Path) -> None:
    import pytest

    store = GalleryStore(tmp_path / "gallery")
    pipeline = _StubPipeline(_embedding([1.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="Unknown identity"):
        store.upload_to_identity("nonexistent", [("face.jpg", b"fake-image-data")], pipeline)
