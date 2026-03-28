"""Tests for GalleryStore — FAISS-backed matching, enrollment, unknowns lifecycle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from src.gallery import GalleryStore, normalize_embedding


def _random_embedding(dim: int = 512) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return normalize_embedding(v)


def _similar_embedding(base: np.ndarray, noise: float = 0.05) -> np.ndarray:
    v = base + np.random.randn(*base.shape).astype(np.float32) * noise
    return normalize_embedding(v)


class TestGalleryMatch:
    def test_empty_gallery_returns_no_match(self, tmp_path: Path) -> None:
        gallery = GalleryStore(tmp_path / "gallery")
        result = gallery.match(_random_embedding(), threshold=0.4)
        assert not result.matched
        assert result.name is None

    def test_match_finds_enrolled_identity(self, tmp_path: Path) -> None:
        gallery = GalleryStore(tmp_path / "gallery", embedding_dim=8)
        emb = normalize_embedding(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))

        # Manually enroll by writing an identity record.
        gallery._write_identity_record(
            name="Alice",
            slug="alice",
            samples=emb.reshape(1, -1),
            uploads=[],
        )

        # Query with the same embedding should match.
        result = gallery.match(emb, threshold=0.4)
        assert result.matched
        assert result.name == "Alice"
        assert result.score > 0.9

    def test_match_rejects_below_threshold(self, tmp_path: Path) -> None:
        gallery = GalleryStore(tmp_path / "gallery", embedding_dim=8)
        emb_a = normalize_embedding(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        emb_b = normalize_embedding(np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32))

        gallery._write_identity_record(
            name="Alice", slug="alice", samples=emb_a.reshape(1, -1), uploads=[]
        )

        result = gallery.match(emb_b, threshold=0.4)
        assert not result.matched

    def test_match_distinguishes_two_identities(self, tmp_path: Path) -> None:
        gallery = GalleryStore(tmp_path / "gallery", embedding_dim=8)
        emb_a = normalize_embedding(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        emb_b = normalize_embedding(np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32))

        gallery._write_identity_record(
            name="Alice", slug="alice", samples=emb_a.reshape(1, -1), uploads=[]
        )
        gallery._write_identity_record(
            name="Bob", slug="bob", samples=emb_b.reshape(1, -1), uploads=[]
        )

        result_a = gallery.match(emb_a, threshold=0.4)
        result_b = gallery.match(emb_b, threshold=0.4)

        assert result_a.name == "Alice"
        assert result_b.name == "Bob"


class TestGalleryCount:
    def test_count_increases_on_enroll(self, tmp_path: Path) -> None:
        gallery = GalleryStore(tmp_path / "gallery", embedding_dim=8)
        assert gallery.count() == 0

        emb = _random_embedding(8)
        gallery._write_identity_record(
            name="Alice", slug="alice", samples=emb.reshape(1, -1), uploads=[]
        )
        assert gallery.count() == 1


class TestGalleryUnknowns:
    def test_capture_unknown_creates_new_entry(self, tmp_path: Path) -> None:
        gallery = GalleryStore(tmp_path / "gallery", embedding_dim=8)
        emb = _random_embedding(8)
        crop = np.zeros((64, 64, 3), dtype=np.uint8)

        result = gallery.capture_unknown(emb, crop)

        assert not result.matched
        assert result.slug is not None
        assert result.slug.startswith("unknown-")
        assert len(gallery.unknowns()) == 1

    def test_capture_unknown_merges_similar(self, tmp_path: Path) -> None:
        gallery = GalleryStore(tmp_path / "gallery", embedding_dim=8)
        emb = _random_embedding(8)
        crop = np.zeros((64, 64, 3), dtype=np.uint8)

        result1 = gallery.capture_unknown(emb, crop)
        # Same embedding should merge into the same unknown.
        result2 = gallery.capture_unknown(emb, crop)

        assert result1.slug == result2.slug
        assert len(gallery.unknowns()) == 1

    def test_capture_unknown_separates_different(self, tmp_path: Path) -> None:
        gallery = GalleryStore(tmp_path / "gallery", embedding_dim=8)
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        emb_a = normalize_embedding(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        emb_b = normalize_embedding(np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32))

        gallery.capture_unknown(emb_a, crop)
        gallery.capture_unknown(emb_b, crop)

        assert len(gallery.unknowns()) == 2


class TestGalleryDelete:
    def test_delete_identity_removes_from_gallery(self, tmp_path: Path) -> None:
        gallery = GalleryStore(tmp_path / "gallery", embedding_dim=8)
        emb = _random_embedding(8)
        gallery._write_identity_record(
            name="Alice", slug="alice", samples=emb.reshape(1, -1), uploads=[]
        )
        assert gallery.count() == 1

        gallery.delete_identity("alice")

        assert gallery.count() == 0
        assert not gallery.match(emb, threshold=0.4).matched

    def test_delete_unknown_removes_from_inbox(self, tmp_path: Path) -> None:
        gallery = GalleryStore(tmp_path / "gallery", embedding_dim=8)
        emb = _random_embedding(8)
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        result = gallery.capture_unknown(emb, crop)

        gallery.delete_unknown(result.slug)

        assert len(gallery.unknowns()) == 0


class TestGalleryPersistence:
    def test_gallery_survives_reload(self, tmp_path: Path) -> None:
        root = tmp_path / "gallery"
        gallery = GalleryStore(root, embedding_dim=8)
        emb = normalize_embedding(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        gallery._write_identity_record(
            name="Alice", slug="alice", samples=emb.reshape(1, -1), uploads=[]
        )

        # Create a new GalleryStore pointing at the same directory.
        gallery2 = GalleryStore(root, embedding_dim=8)

        assert gallery2.count() == 1
        result = gallery2.match(emb, threshold=0.4)
        assert result.matched
        assert result.name == "Alice"
