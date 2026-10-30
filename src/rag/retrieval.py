"""Retrieve citation-ready passages with BGE-M3 and PostgreSQL hybrid search."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json

from .embeddings import BGEEmbedder, EMBEDDING_DIMENSION
from .storage import PostgresStore, validate_search_options


@dataclass(frozen=True)
class RetrievedChunk:
    version_id: str
    chunk_id: str
    chunk_index: int
    text: str
    source_path: str
    source_type: str
    title: str
    source_url: str | None
    published_at: datetime | None
    page_number: int | None
    start_char: int
    end_char: int
    token_count: int
    extraction_method: str
    score: float
    vector_score: float | None
    keyword_score: float | None
    vector_rank: int | None
    keyword_rank: int | None

    @property
    def citation_id(self):
        """Identify the exact chunk within its immutable document version."""
        return f"{self.version_id}:{self.chunk_id}"


class Retriever:
    """Reuse one lazy embedding model across queries against the same store."""

    def __init__(self, store=None, *, embedder=None):
        self.store = store if store is not None else PostgresStore()
        self.embedder = embedder if embedder is not None else BGEEmbedder()
        if self.embedder.dimension != EMBEDDING_DIMENSION:
            raise ValueError("This index requires 1024-dimensional embeddings")

    def retrieve(
        self, query, *, top_k=5, candidate_limit=30, rrf_k=60,
        source_type=None, published_from=None, published_before=None,
    ) -> list[RetrievedChunk]:
        """Return current passages ranked by reciprocal rank fusion (RRF).

        Both search branches use the query encoder's model name and revision.
        Dates are publication dates, with an inclusive lower and exclusive upper
        bound. Missing dates do not match date filters. An empty eligible corpus
        returns []; a nonempty corpus can return weak vector matches, so the
        scores must not be treated as confidence or proof that an answer exists.
        """
        options = dict(
            top_k=top_k, candidate_limit=candidate_limit, rrf_k=rrf_k,
            source_type=source_type, published_from=published_from,
            published_before=published_before,
        )
        validate_search_options(query, **options)
        query = query.strip()
        vector = self.embedder.encode_query(query)
        rows = self.store.search_chunks(
            query, vector, embedding_model=self.embedder.model_name,
            embedding_revision=self.embedder.revision, **options,
        )
        results = []
        for row in rows:
            values = dict(row)
            metadata = values.pop("metadata")
            results.append(RetrievedChunk(
                **values,
                **{key: metadata[key] for key in (
                    "chunk_id", "page_number", "start_char", "end_char",
                    "token_count", "extraction_method",
                )},
            ))
        return results


def _publication_bound(value):
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use an ISO date or timestamp") from error
    return parsed if parsed.utcoffset() is not None else parsed.replace(tzinfo=timezone.utc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--source-type", choices=["news", "bank_report"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--published-from", type=_publication_bound)
    parser.add_argument("--published-before", type=_publication_bound)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    retriever = Retriever(embedder=BGEEmbedder(device=args.device))
    try:
        results = retriever.retrieve(
            args.query, top_k=args.top_k, candidate_limit=args.candidate_limit,
            rrf_k=args.rrf_k, source_type=args.source_type,
            published_from=args.published_from, published_before=args.published_before,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps([
        dict(asdict(result), citation_id=result.citation_id) for result in results
    ], ensure_ascii=False, indent=2, default=lambda value: value.isoformat()))


if __name__ == "__main__":
    main()
