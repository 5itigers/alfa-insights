"""PostgreSQL storage for immutable document versions and their passages."""

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path

import numpy as np

from .embeddings import EMBEDDING_DIMENSION

DEFAULT_DATABASE_URL = "postgresql://postgres@localhost:5432/AI-Service-vectordb"


def validate_search_options(
    query, *, top_k, candidate_limit, rrf_k, source_type, published_from, published_before,
):
    """Validate retrieval inputs before opening a connection or loading a model."""
    if not isinstance(query, str) or not query.strip() or "\x00" in query:
        raise ValueError("query must be a nonempty string without null characters")
    for name, value in (("top_k", top_k), ("candidate_limit", candidate_limit), ("rrf_k", rrf_k)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if candidate_limit < top_k:
        raise ValueError("candidate_limit must be at least top_k")
    if source_type is not None and source_type not in ("news", "bank_report"):
        raise ValueError("source_type must be 'news', 'bank_report', or None")
    for name, value in (("published_from", published_from), ("published_before", published_before)):
        if value is not None and (
            not isinstance(value, datetime) or value.utcoffset() is None
        ):
            raise ValueError(f"{name} must be a timezone-aware datetime")
    if published_from is not None and published_before is not None:
        if published_from >= published_before:
            raise ValueError("published_from must be earlier than published_before")


_HYBRID_SEARCH_SQL = """
WITH eligible_documents AS MATERIALIZED (
    SELECT d.* FROM rag_documents d
    WHERE d.is_current
      AND d.embedding_model = %(embedding_model)s
      AND d.embedding_revision = %(embedding_revision)s
      AND (%(source_type)s::text IS NULL OR d.source_type = %(source_type)s)
      AND (%(published_from)s::timestamptz IS NULL OR d.published_at >= %(published_from)s)
      AND (%(published_before)s::timestamptz IS NULL OR d.published_at < %(published_before)s)
), search_query AS (
    SELECT websearch_to_tsquery('russian', %(query)s) AS terms
), vector_candidates AS (
    SELECT c.source_id, c.chunk_index,
           c.embedding <=> %(query_vector)s::vector AS distance
    FROM document_chunks c JOIN eligible_documents d ON d.version_id = c.source_id
    ORDER BY distance, c.source_id, c.chunk_index
    LIMIT %(candidate_limit)s
), keyword_candidates AS (
    SELECT c.source_id, c.chunk_index,
           ts_rank_cd(to_tsvector('russian', c.content), q.terms) AS keyword_score
    FROM document_chunks c JOIN eligible_documents d ON d.version_id = c.source_id
    CROSS JOIN search_query q
    WHERE to_tsvector('russian', c.content) @@ q.terms
    ORDER BY keyword_score DESC, c.source_id, c.chunk_index
    LIMIT %(candidate_limit)s
), vector_ranked AS (
    SELECT *, row_number() OVER (ORDER BY distance, source_id, chunk_index) AS vector_rank
    FROM vector_candidates
), keyword_ranked AS (
    SELECT *, row_number() OVER (
        ORDER BY keyword_score DESC, source_id, chunk_index
    ) AS keyword_rank FROM keyword_candidates
), fused AS (
    SELECT coalesce(v.source_id, k.source_id) AS source_id,
           coalesce(v.chunk_index, k.chunk_index) AS chunk_index,
           1.0 - v.distance AS vector_score, k.keyword_score,
           v.vector_rank, k.keyword_rank,
           coalesce(1.0::double precision / (%(rrf_k)s + v.vector_rank), 0.0)
             + coalesce(1.0::double precision / (%(rrf_k)s + k.keyword_rank), 0.0) AS score
    FROM vector_ranked v FULL JOIN keyword_ranked k
      ON v.source_id = k.source_id AND v.chunk_index = k.chunk_index
)
SELECT d.version_id, d.source_path, d.source_type, d.title, d.source_url, d.published_at,
       c.chunk_index, c.content AS text, c.metadata,
       f.score, f.vector_score, f.keyword_score, f.vector_rank, f.keyword_rank
FROM fused f JOIN eligible_documents d ON d.version_id = f.source_id
JOIN document_chunks c ON c.source_id = f.source_id AND c.chunk_index = f.chunk_index
ORDER BY f.score DESC, f.source_id, f.chunk_index
LIMIT %(top_k)s
"""


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

    def search_chunks(
        self, query, query_vector, *, embedding_model, embedding_revision,
        top_k=5, candidate_limit=30, rrf_k=60, source_type=None,
        published_from=None, published_before=None,
    ):
        """Fuse exact cosine and Russian keyword rankings in one database snapshot.

        Publication bounds form [published_from, published_before). Sources with
        unknown dates are excluded whenever a publication bound is provided.
        Each search contributes at most candidate_limit candidates to RRF.
        """
        validate_search_options(
            query, top_k=top_k, candidate_limit=candidate_limit, rrf_k=rrf_k,
            source_type=source_type, published_from=published_from,
            published_before=published_before,
        )
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.shape != (EMBEDDING_DIMENSION,):
            raise ValueError("Expected a single 1024-dimensional query vector")
        if not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError("The query vector must contain finite, nonzero values")
        if any(not isinstance(value, str) or not value.strip()
               for value in (embedding_model, embedding_revision)):
            raise ValueError("The embedding model and revision must be nonempty strings")

        from pgvector.psycopg import register_vector

        with self._connect() as connection:
            register_vector(connection)
            return connection.execute(_HYBRID_SEARCH_SQL, {
                "query": query.strip(), "query_vector": vector,
                "embedding_model": embedding_model, "embedding_revision": embedding_revision,
                "top_k": top_k, "candidate_limit": candidate_limit, "rrf_k": rrf_k,
                "source_type": source_type, "published_from": published_from,
                "published_before": published_before,
            }).fetchall()

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
