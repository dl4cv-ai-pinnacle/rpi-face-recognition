from __future__ import annotations

from pathlib import Path

import numpy as np
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
