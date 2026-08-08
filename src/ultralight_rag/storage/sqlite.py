"""SQLite and sqlite-vec storage implementation."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from functools import wraps
from numbers import Real
from types import TracebackType
from typing import Any, TypeVar

import sqlite_vec

T = TypeVar("T")


def _synchronized(method: Callable[..., T]) -> Callable[..., T]:
    @wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class DocumentNotFoundError(KeyError):
    """Raised when a document id is not present in the index."""


class SQLiteStore:
    def __init__(self, database_path: str = ":memory:") -> None:
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(database_path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.row_factory = sqlite3.Row
        self.connection.enable_load_extension(True)
        sqlite_vec.load(self.connection)
        self.connection.enable_load_extension(False)
        self._vector_dimension: int | None = None
        self._create_schema()
        self._load_vector_dimension()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                UNIQUE(document_id, ordinal)
            );
            CREATE TABLE IF NOT EXISTS index_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                embedding_provider TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_fingerprint TEXT
            );
            """
        )
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(index_metadata)")}
        if "embedding_fingerprint" not in columns:
            try:
                self.connection.execute(
                    "ALTER TABLE index_metadata ADD COLUMN embedding_fingerprint TEXT"
                )
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.commit()

    def _load_vector_dimension(self) -> None:
        self._vector_dimension = None
        row = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'vec_chunks'"
        ).fetchone()
        if row:
            match = re.search(r"FLOAT\[(\d+)\]", row[0])
            if match:
                self._vector_dimension = int(match.group(1))

    def _ensure_vector_table(self, dimension: int) -> None:
        self._load_vector_dimension()
        if self._vector_dimension == dimension:
            return
        if self._vector_dimension is not None and self._vector_dimension != dimension:
            raise ValueError("All embeddings in one index must have the same dimension")
        self.connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks "
            f"USING vec0(chunk_id INTEGER PRIMARY KEY, "
            f"embedding FLOAT[{dimension}] distance_metric=cosine)"
        )
        self._vector_dimension = dimension

    def _drop_empty_vector_table(self) -> None:
        self._load_vector_dimension()
        if self._vector_dimension is None:
            return
        if self.connection.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]:
            raise ValueError("Cannot replace embedding identity while vector data remains")
        self.connection.execute("DROP TABLE vec_chunks")
        self._vector_dimension = None

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _metadata(value: dict[str, Any] | None) -> str:
        return json.dumps(value or {}, separators=(",", ":"))

    @staticmethod
    def _decode_metadata(value: str) -> dict[str, Any]:
        return json.loads(value) if value else {}

    @staticmethod
    def _normalize_vector(vector: list[float]) -> list[float]:
        if any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in vector
        ):
            raise ValueError("Embeddings must contain only finite numbers")
        return [float(value) for value in vector]

    @classmethod
    def _validate_embeddings(
        cls, chunks: list[str], embeddings: list[list[float]]
    ) -> list[list[float]]:
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk must have one embedding")
        if embeddings:
            dimension = len(embeddings[0])
            if dimension == 0 or any(len(vector) != dimension for vector in embeddings):
                raise ValueError("All embeddings must have the same non-zero dimension")
        return [cls._normalize_vector(vector) for vector in embeddings]

    @_synchronized
    def create_document(
        self,
        title: str,
        content: str,
        metadata: dict[str, Any],
        chunks: Iterable[str],
        embeddings: Iterable[list[float]],
        document_id: str | None = None,
        expected_embedding_identity: tuple[str, str, str] | None = None,
    ) -> dict[str, Any]:
        document_id = document_id or str(uuid.uuid4())
        chunks = list(chunks)
        embeddings = list(embeddings)
        embeddings = self._validate_embeddings(chunks, embeddings)
        now = self._now()
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            self._require_embedding_configuration(expected_embedding_identity)
            if embeddings:
                self._ensure_vector_table(len(embeddings[0]))
            self.connection.execute(
                "INSERT INTO documents "
                "(id,title,content,metadata,created_at,updated_at) VALUES (?,?,?,?,?,?)",
                (document_id, title, content, self._metadata(metadata), now, now),
            )
            if chunks:
                chunk_data = [
                    (document_id, ordinal, text, "{}") for ordinal, text in enumerate(chunks)
                ]
                self.connection.executemany(
                    "INSERT INTO chunks (document_id,ordinal,text,metadata) VALUES (?,?,?,?)",
                    chunk_data,
                )

                chunk_ids = [
                    row[0]
                    for row in self.connection.execute(
                        "SELECT id FROM chunks WHERE document_id = ? ORDER BY ordinal",
                        (document_id,),
                    )
                ]

                # strict=True documents (and enforces) an invariant already guaranteed by
                # _validate_embeddings above, which raises ValueError on a length mismatch
                # before this block ever runs.
                vec_data = [
                    (chunk_id, sqlite_vec.serialize_float32(vector))
                    for chunk_id, vector in zip(chunk_ids, embeddings, strict=True)
                ]
                self.connection.executemany(
                    "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?,?)",
                    vec_data,
                )
        return self.get_document(document_id)

    @_synchronized
    def list_documents(self) -> list[dict[str, Any]]:
        # ⚡ Bolt: Replaced a `LEFT JOIN ... GROUP BY d.id` with a correlated scalar subquery.
        # This optimizes performance significantly (removes grouping overhead, uses index lookups)
        # by leveraging the existing UNIQUE(document_id, ordinal) index on the chunks table.
        rows = self.connection.execute(
            """SELECT d.*, (SELECT COUNT(id) FROM chunks WHERE document_id = d.id) AS chunk_count
               FROM documents d ORDER BY d.created_at"""
        ).fetchall()
        return [self._document_summary(row) for row in rows]

    @_synchronized
    def get_document(self, document_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        if row is None:
            raise DocumentNotFoundError(document_id)
        result = self._document_summary(row)
        result["chunks"] = [
            {
                "id": item["id"],
                "ordinal": item["ordinal"],
                "text": item["text"],
                "metadata": self._decode_metadata(item["metadata"]),
            }
            for item in self.connection.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY ordinal", (document_id,)
            )
        ]
        result["content"] = row["content"]
        return result

    def _document_summary(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "metadata": self._decode_metadata(row["metadata"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "chunk_count": row["chunk_count"]
            if "chunk_count" in row.keys()
            else self._chunk_count(row["id"]),
        }

    def _chunk_count(self, document_id: str) -> int:
        return self.connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (document_id,)
        ).fetchone()[0]

    def _chunk_count_all(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    @_synchronized
    def is_persistent(self) -> bool:
        return bool(
            next(
                (
                    row[2]
                    for row in self.connection.execute("PRAGMA database_list")
                    if row[1] == "main"
                ),
                "",
            )
        )

    @_synchronized
    def ensure_embedding_configuration(self, provider: str, model: str, fingerprint: str) -> None:
        """Bind an index to its complete embedding identity."""
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            existing = self.connection.execute(
                "SELECT embedding_provider, embedding_model, embedding_fingerprint "
                "FROM index_metadata WHERE id = 1"
            ).fetchone()
            if existing is not None and tuple(existing) == (provider, model, fingerprint):
                return
            if self._chunk_count_all():
                if existing is None:
                    raise ValueError(
                        "Index has no embedding configuration metadata; reindex the database "
                        "before using it with the current embedding model."
                    )
                self._raise_embedding_configuration_error()
            self._drop_empty_vector_table()
            if existing is None:
                self.connection.execute(
                    "INSERT INTO index_metadata "
                    "(id, embedding_provider, embedding_model, embedding_fingerprint) "
                    "VALUES (1, ?, ?, ?)",
                    (provider, model, fingerprint),
                )
            else:
                self.connection.execute(
                    "UPDATE index_metadata SET embedding_provider=?, embedding_model=?, "
                    "embedding_fingerprint=? WHERE id=1",
                    (provider, model, fingerprint),
                )

    @staticmethod
    def _raise_embedding_configuration_error() -> None:
        raise ValueError(
            "Index embedding configuration does not match this service; "
            "reindex the database with the configured embedding model."
        )

    def _require_embedding_configuration(self, expected: tuple[str, str, str] | None) -> None:
        if expected is None:
            return
        existing = self.connection.execute(
            "SELECT embedding_provider, embedding_model, embedding_fingerprint "
            "FROM index_metadata WHERE id = 1"
        ).fetchone()
        if existing is None or tuple(existing) != expected:
            self._raise_embedding_configuration_error()

    @_synchronized
    def preflight_embedding_configuration(self, expected: tuple[str, str, str] | None) -> None:
        """Reject stale embedding identities before provider work begins."""
        self._require_embedding_configuration(expected)

    @_synchronized
    def replace_document(
        self,
        document_id: str,
        title: str,
        content: str,
        metadata: dict[str, Any],
        chunks: Iterable[str],
        embeddings: Iterable[list[float]],
        expected_embedding_identity: tuple[str, str, str] | None = None,
    ) -> dict[str, Any]:
        chunks, embeddings = list(chunks), list(embeddings)
        embeddings = self._validate_embeddings(chunks, embeddings)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            if (
                self.connection.execute(
                    "SELECT 1 FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
                is None
            ):
                raise DocumentNotFoundError(document_id)
            self._require_embedding_configuration(expected_embedding_identity)
            if embeddings:
                self._ensure_vector_table(len(embeddings[0]))
            old_ids = [
                row[0]
                for row in self.connection.execute(
                    "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
                )
            ]
            if old_ids:
                self.connection.executemany(
                    "DELETE FROM vec_chunks WHERE chunk_id = ?", [(item,) for item in old_ids]
                )
            self.connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self.connection.execute(
                "UPDATE documents SET title=?, content=?, metadata=?, updated_at=? WHERE id=?",
                (title, content, self._metadata(metadata), self._now(), document_id),
            )
            if chunks:
                chunk_data = [
                    (document_id, ordinal, text, "{}") for ordinal, text in enumerate(chunks)
                ]
                self.connection.executemany(
                    "INSERT INTO chunks (document_id,ordinal,text,metadata) VALUES (?,?,?,?)",
                    chunk_data,
                )

                chunk_ids = [
                    row[0]
                    for row in self.connection.execute(
                        "SELECT id FROM chunks WHERE document_id = ? ORDER BY ordinal",
                        (document_id,),
                    )
                ]

                # strict=True documents (and enforces) an invariant already guaranteed by
                # _validate_embeddings above, which raises ValueError on a length mismatch
                # before this block ever runs.
                vec_data = [
                    (chunk_id, sqlite_vec.serialize_float32(vector))
                    for chunk_id, vector in zip(chunk_ids, embeddings, strict=True)
                ]
                self.connection.executemany(
                    "INSERT INTO vec_chunks (chunk_id,embedding) VALUES (?,?)",
                    vec_data,
                )
        return self.get_document(document_id)

    @_synchronized
    def delete_document(
        self,
        document_id: str,
        expected_embedding_identity: tuple[str, str, str] | None = None,
    ) -> None:
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            if (
                self.connection.execute(
                    "SELECT 1 FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
                is None
            ):
                raise DocumentNotFoundError(document_id)
            self._require_embedding_configuration(expected_embedding_identity)
            ids = [
                row[0]
                for row in self.connection.execute(
                    "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
                )
            ]
            if ids:
                self.connection.executemany(
                    "DELETE FROM vec_chunks WHERE chunk_id = ?", [(item,) for item in ids]
                )
            self.connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    @_synchronized
    def search(
        self,
        vector: list[float],
        top_k: int,
        filter_metadata: dict[str, Any] | None = None,
        expected_embedding_identity: tuple[str, str, str] | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        vector = self._normalize_vector(vector)
        self._require_embedding_configuration(expected_embedding_identity)
        if self._vector_dimension is None:
            self._load_vector_dimension()
        if self._vector_dimension is None:
            return []
        if len(vector) != self._vector_dimension:
            raise ValueError("Search embedding dimension does not match indexed embeddings")
        if filter_metadata:
            # Metadata filtering happens after KNN, adequate for small local indexes.
            candidate_count = max(1, self._chunk_count_all())
        else:
            candidate_count = max(top_k * 10, 50)
        rows = self.connection.execute(
            """SELECT c.id, c.document_id, c.ordinal, c.text, c.metadata,
                      d.title, d.metadata AS document_metadata, v.distance
               FROM vec_chunks v JOIN chunks c ON c.id = v.chunk_id
               JOIN documents d ON d.id = c.document_id
               WHERE v.embedding MATCH ? AND v.k = ? ORDER BY v.distance""",
            (sqlite_vec.serialize_float32(vector), candidate_count),
        ).fetchall()
        result = []
        for row in rows:
            metadata = self._decode_metadata(row["document_metadata"])
            if filter_metadata and any(
                key not in metadata or metadata[key] != value
                for key, value in filter_metadata.items()
            ):
                continue
            result.append(
                {
                    "chunk_id": row["id"],
                    "document_id": row["document_id"],
                    "title": row["title"],
                    "text": row["text"],
                    "metadata": metadata,
                    "score": 0.0 if row["distance"] is None else 1.0 - row["distance"],
                }
            )
            if len(result) >= top_k:
                break
        return result

    @_synchronized
    def close(self) -> None:
        """Release the underlying sqlite3 connection.

        Safe to call more than once: `sqlite3.Connection.close()` is itself
        idempotent, so a second call here is a no-op rather than an error.
        """
        self.connection.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
