"""PostgreSQL storage for immutable document versions and their passages."""

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path

import numpy as np
from transformers.utils.import_utils import requires

DEFAULT_DATABASE_URL = "postgresql://postgres@localhost:5432/AI-Service-vectordb"


@dataclass(frozen=True)
class IndexedDocument:
    version_id: str
    source_path: str
    source_type: str
    title: str
    source_url: str | None
    published_at: datetime | None
    content_hash: str
    pipeline_signature: str
    embedding_model: str
    embedding_revision: str
    page_count: int
    pipeline_config: dict


class PostgresStore:
    def __init__(self, database_url=None):
        self.database_url = (
            os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
            if database_url is None else database_url
        )
        if not self.database_url:
            raise ValueError("A PostgreSQL connection URL is required")

    def _connect(self):
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(
            self.database_url, autocommit=True, row_factory=dict_row, connect_timeout=5
        )

    def initialize(self):
        migration = Path(__file__).resolve().parents[2] / "migrations/001_rag_index.sql"
        with self._connect() as connection, connection.transaction():
            connection.execute(migration.read_text(encoding="utf-8"), prepare=False)

    def get_current(self, source_path):
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM rag_documents WHERE source_path = %s AND is_current",
                (source_path,),
            ).fetchone()

    def save_document(self, document, chunks, vectors):
        chunks = list(chunks)
        embeddings = np.asarray(vectors, dtype=np.float32)
        if not chunks or embeddings.shape != (len(chunks), 1024):
            raise ValueError("Expected one 1024-dimensional vector per nonempty chunk")
        if not np.isfinite(embeddings).all() or (np.linalg.norm(embeddings, axis=1) == 0).any():
            raise ValueError("Embeddings must contain finite, nonzero vectors")
        if any(chunk.source_path != document.source_path for chunk in chunks):
            raise ValueError("All chunks must belong to the indexed document")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Chunk IDs must be unique within a document version")
        if any(
            not chunk.text or chunk.start_char < 0 or chunk.end_char <= chunk.start_char
            or chunk.token_count <= 0 or chunk.extraction_method not in {"text", "ocr"}
            or (chunk.page_number is not None and chunk.page_number <= 0)
            for chunk in chunks
        ):
            raise ValueError("Chunks must have nonempty text and valid citation metadata")

        from pgvector.psycopg import register_vector
        from psycopg.types.json import Jsonb

        with self._connect() as connection:
            register_vector(connection)
            with connection.transaction():
                connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (document.source_path,),
                )
                connection.execute(
                    "UPDATE rag_documents SET is_current = false "
                    "WHERE source_path = %s AND is_current", (document.source_path,),
                )
                existing = connection.execute(
                    "SELECT chunk_count FROM rag_documents WHERE version_id = %s",
                    (document.version_id,),
                ).fetchone()
                if existing is not None:
                    connection.execute(
                        "UPDATE rag_documents SET is_current = true WHERE version_id = %s",
                        (document.version_id,),
                    )
                    return existing["chunk_count"]
                connection.execute(
                    """INSERT INTO rag_documents (
                        version_id, source_path, source_type, title, source_url, published_at,
                        content_hash, pipeline_signature, embedding_model, embedding_revision,
                        page_count, chunk_count, pipeline_config
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        document.version_id, document.source_path, document.source_type,
                        document.title, document.source_url, document.published_at,
                        document.content_hash, document.pipeline_signature,
                        document.embedding_model, document.embedding_revision,
                        document.page_count, len(chunks), Jsonb(document.pipeline_config),
                    ),
                )
                with connection.cursor() as cursor:
                    cursor.executemany(
                        """INSERT INTO document_chunks (
                            source_id, chunk_index, content, metadata, embedding
                        ) VALUES (%s, %s, %s, %s, %s)""",
                        [
                            (
                                document.version_id, position, chunk.text,
                                Jsonb({
                                    "chunk_id": chunk.chunk_id,
                                    "page_number": chunk.page_number,
                                    "page_chunk_index": chunk.chunk_index,
                                    "start_char": chunk.start_char,
                                    "end_char": chunk.end_char,
                                    "token_count": chunk.token_count,
                                    "extraction_method": chunk.extraction_method,
                                }),
                                vector,
                            )
                            for position, (chunk, vector) in enumerate(zip(chunks, embeddings))
                        ],
                    )
                return len(chunks)
