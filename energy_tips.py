"""Persist and search energy guide embeddings in SQLite."""

import json
import sqlite3
from contextlib import contextmanager, closing
from hashlib import sha256
from pathlib import Path

import sqlite_vec
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class EnergyTipsIndex:
    def __init__(self, path: Path, embeddings: Embeddings):
        self.path = path
        self.embeddings = embeddings

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS energy_tip_chunks (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            """)
            yield connection

    def index(self, chunks: list[Document]) -> int:
        """Keep cached embeddings for unchanged chunks and replace edited ones."""
        if not chunks:
            raise ValueError("No energy guide text to index.")

        chunks_by_id = {}
        for chunk in chunks:
            key = json.dumps([chunk.metadata["source"], chunk.page_content])
            chunks_by_id[sha256(key.encode("utf-8")).hexdigest()] = chunk

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = {row[0] for row in connection.execute("SELECT id FROM energy_tip_chunks")}
            missing = {key: chunk for key, chunk in chunks_by_id.items() if key not in existing}
            if missing:
                vectors = self.embeddings.embed_documents([chunk.page_content for chunk in missing.values()])
                connection.executemany(
                    "INSERT INTO energy_tip_chunks VALUES (?, ?, ?, ?)",
                    (
                        (key, chunk.metadata["source"], chunk.page_content, sqlite_vec.serialize_float32(vector))
                        for (key, chunk), vector in zip(missing.items(), vectors, strict=True)
                    ),
                )
            connection.executemany(
                "DELETE FROM energy_tip_chunks WHERE id = ?",
                ((key,) for key in existing.difference(chunks_by_id)),
            )

        return len(chunks_by_id)

    def sources(self) -> set[str]:
        with self._connect() as connection:
            return {row[0] for row in connection.execute("SELECT DISTINCT source FROM energy_tip_chunks")}

    def search(self, query: str, limit: int) -> list[Document]:
        if limit < 1:
            return []

        vector = sqlite_vec.serialize_float32(self.embeddings.embed_query(query))
        with self._connect() as connection:
            connection.enable_load_extension(True)
            try:
                sqlite_vec.load(connection)
            finally:
                connection.enable_load_extension(False)
            rows = connection.execute(
                """
                SELECT content, source
                FROM energy_tip_chunks
                ORDER BY vec_distance_L2(embedding, ?), id
                LIMIT ?
                """,
                (vector, limit),
            ).fetchall()
        return [Document(page_content=content, metadata={"source": source}) for content, source in rows]
