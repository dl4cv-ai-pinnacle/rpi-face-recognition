from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    face_id: int
    name: str
    score: float  # cosine similarity


class FaceDatabase:
    """Face embedding store using FAISS for similarity search and SQLite for metadata."""

    def __init__(
        self,
        db_path: str,
        index_path: str,
        embedding_dim: int,
        threshold: float,
    ) -> None:
        self.db_path = db_path
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.threshold = threshold

        # FAISS inner-product index (cosine similarity with L2-normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)

        # SQLite connection
        self._conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                face_id   INTEGER PRIMARY KEY,
                name      TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        self._conn.commit()

    def load(self) -> None:
        """Load FAISS index from disk if it exists."""
        if Path(self.index_path).exists():
            self.index = faiss.read_index(self.index_path)
            logger.info("Loaded FAISS index with %d vectors", self.index.ntotal)
        else:
            logger.info("No existing FAISS index found, starting empty")

    def save(self) -> None:
        """Persist FAISS index and commit SQLite."""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        self._conn.commit()
        logger.info("Saved FAISS index with %d vectors", self.index.ntotal)

    def add(self, name: str, embedding: np.ndarray) -> int:
        """Add a face to the database.

        Args:
            name: Person's name.
            embedding: L2-normalized embedding, shape (embedding_dim,).

        Returns:
            Assigned face_id.
        """
        face_id = self.index.ntotal
        vec = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        self.index.add(vec)

        self._conn.execute(
            "INSERT INTO faces (face_id, name) VALUES (?, ?)",
            (face_id, name),
        )
        self._conn.commit()
        logger.info("Added face_id=%d name='%s'", face_id, name)
        return face_id

    def search(
        self, embedding: np.ndarray, top_k: int = 1
    ) -> MatchResult | None:
        """Search for the closest face.

        Args:
            embedding: L2-normalized query embedding, shape (embedding_dim,).
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            Best MatchResult if score >= threshold, else None.
        """
        if self.index.ntotal == 0:
            return None

        vec = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        scores, indices = self.index.search(vec, top_k)

        best_score = float(scores[0, 0])
        best_idx = int(indices[0, 0])

        if best_score < self.threshold:
            return None

        row = self._conn.execute(
            "SELECT name FROM faces WHERE face_id = ?", (best_idx,)
        ).fetchone()
        if row is None:
            return None

        return MatchResult(face_id=best_idx, name=row[0], score=best_score)

    def close(self) -> None:
        """Close SQLite connection."""
        self._conn.close()
