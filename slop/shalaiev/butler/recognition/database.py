"""FAISS + SQLite face database for identity matching and storage."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from .base import MatchResult

try:
    import faiss
except ImportError:
    faiss = None  # fall back to numpy dot-product search


class FaceDatabase:
    """Manages face embeddings in a FAISS index backed by SQLite metadata."""

    def __init__(self, db_path: Path, embedding_dim: int = 512) -> None:
        self._db_path = db_path
        self._embedding_dim = embedding_dim
        self._db_path.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path / "faces.db"))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

        # FAISS index (or None for numpy fallback).
        self._index: object | None = None
        # Mapping: FAISS row index → (identity_name, embedding_id).
        self._id_map: list[tuple[str, int]] = []

        self._rebuild_index()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS identities (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                identity_id INTEGER NOT NULL REFERENCES identities(id),
                vector BLOB NOT NULL
            );
        """)

    def _rebuild_index(self) -> None:
        """Reload all embeddings from SQLite into the FAISS index."""
        rows = self._conn.execute("""
            SELECT e.id, i.name, e.vector
            FROM embeddings e JOIN identities i ON e.identity_id = i.id
        """).fetchall()

        self._id_map = []
        vectors = []

        for emb_id, name, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32).copy()
            vectors.append(vec)
            self._id_map.append((name, emb_id))

        if faiss is not None:
            self._index = faiss.IndexFlatIP(self._embedding_dim)
            if vectors:
                matrix = np.stack(vectors).astype(np.float32)
                self._index.add(matrix)
        else:
            # Numpy fallback: store as (N, dim) matrix.
            if vectors:
                self._index = np.stack(vectors).astype(np.float32)
            else:
                self._index = np.empty((0, self._embedding_dim), dtype=np.float32)

    def search(self, embedding: np.ndarray, threshold: float) -> MatchResult:
        """Find the best matching identity above threshold.

        Returns MatchResult("Unknown", 0.0) for empty database or no match.
        """
        if len(self._id_map) == 0:
            return MatchResult("Unknown", 0.0)

        query = embedding.reshape(1, -1).astype(np.float32)

        if faiss is not None and not isinstance(self._index, np.ndarray):
            scores, indices = self._index.search(query, 1)
            best_score = float(scores[0][0])
            best_idx = int(indices[0][0])
        else:
            # Numpy dot-product fallback.
            similarities = self._index @ query.T  # (N, 1)
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx, 0])

        if best_score >= threshold:
            name, _ = self._id_map[best_idx]
            return MatchResult(name, best_score)
        return MatchResult("Unknown", best_score)

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        """Add an embedding for the given identity (creates identity if needed)."""
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO identities (name) VALUES (?)", (name,),
        )
        if cursor.lastrowid:
            identity_id = cursor.lastrowid
        else:
            row = self._conn.execute(
                "SELECT id FROM identities WHERE name = ?", (name,),
            ).fetchone()
            identity_id = row[0]

        blob = embedding.astype(np.float32).tobytes()
        self._conn.execute(
            "INSERT INTO embeddings (identity_id, vector) VALUES (?, ?)",
            (identity_id, blob),
        )
        self._conn.commit()
        self._rebuild_index()

    def list_identities(self) -> list[str]:
        rows = self._conn.execute("SELECT name FROM identities ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def remove_identity(self, name: str) -> None:
        """Remove an identity and all its embeddings."""
        row = self._conn.execute(
            "SELECT id FROM identities WHERE name = ?", (name,),
        ).fetchone()
        if row is None:
            return
        identity_id = row[0]
        self._conn.execute("DELETE FROM embeddings WHERE identity_id = ?", (identity_id,))
        self._conn.execute("DELETE FROM identities WHERE id = ?", (identity_id,))
        self._conn.commit()
        self._rebuild_index()

    def close(self) -> None:
        self._conn.close()
