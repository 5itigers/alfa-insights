"""Load, chunk, embed, and persist documents in PostgreSQL."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from ..embeddings import BGEEmbedder
from ..storage import IndexedDocument, PostgresStore
from .chunking import chunk_pages
from .loaders import SUPPORTED_EXTENSIONS, load_document


PIPELINE_VERSION = "1"


@dataclass(frozen=True)
class IndexResult:
    source_path: str
    version_id: str
    chunk_count: int
    skipped: bool


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _file_hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_document(
    path, store, *, source_type, embedder=None, title=None, source_url=None,
    published_at=None, chunk_size=400, chunk_overlap=40, batch_size=4,
    ocr=True, ocr_language="rus+eng", tessdata_dir=None,
):
    """Index one local source, skipping unchanged inputs and retaining old versions.

    Source type and publication date are caller-supplied metadata. The pipeline
    version must change when extraction or chunking behavior changes. Changes
    to OCR models/settings beyond the language also require a version bump.
    """
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise IsADirectoryError(f"Expected a document file: {source}")
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document format: {source.suffix}")
    if source_type not in {"news", "bank_report"}:
        raise ValueError("source_type must be 'news' or 'bank_report'")
    if published_at is not None and (
        not isinstance(published_at, datetime) or published_at.utcoffset() is None
    ):
        raise ValueError("published_at must be a timezone-aware datetime")

    encoder = embedder if embedder is not None else BGEEmbedder()
    if encoder.dimension != 1024:
        raise ValueError("This index requires 1024-dimensional embeddings")
    pipeline_config = {
        "pipeline_version": PIPELINE_VERSION,
        "model": encoder.model_name, "revision": encoder.revision,
        "dimension": encoder.dimension, "max_length": encoder.max_length,
        "normalized": True, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
        "ocr": ocr, "ocr_language": ocr_language,
    }
    signature = _digest(pipeline_config)
    content_hash = _file_hash(source)
    document_title = title if title is not None else source.stem
    version_id = _digest({
        "source": str(source), "content_hash": content_hash, "pipeline": signature,
        "source_type": source_type, "title": document_title, "source_url": source_url,
        "published_at": published_at.astimezone(timezone.utc).isoformat() if published_at else None,
    })
    current = store.get_current(str(source))
    if current is not None and current["version_id"] == version_id:
        return IndexResult(str(source), version_id, current["chunk_count"], True)

    pages = load_document(source, ocr=ocr, ocr_language=ocr_language, tessdata_dir=tessdata_dir)
    chunks = chunk_pages(
        pages, encoder.tokenizer, chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, max_length=encoder.max_length,
    )
    if not chunks:
        raise ValueError(f"No searchable chunks found in {source}")
    vectors = encoder.encode_passages([chunk.text for chunk in chunks], batch_size=batch_size)
    if _file_hash(source) != content_hash:
        raise RuntimeError(f"Document changed during indexing; retry: {source}")

    document = IndexedDocument(
        version_id, str(source), source_type, document_title, source_url, published_at,
        content_hash, signature, encoder.model_name, encoder.revision, len(pages), pipeline_config,
    )
    saved_count = store.save_document(document, chunks, vectors)
    return IndexResult(str(source), version_id, saved_count, False)


def index_directory(path, store, *, source_type, recursive=True, embedder=None, **options):
    """Index a folder containing a single source type; unreadable files stop the run."""
    directory = Path(path).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a document directory: {directory}")
    encoder = embedder if embedder is not None else BGEEmbedder()
    paths = directory.rglob("*") if recursive else directory.glob("*")
    return [
        index_document(source, store, source_type=source_type, embedder=encoder, **options)
        for source in sorted(paths)
        if source.is_file() and source.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def _publication_date(value):
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use an ISO date or timestamp") from error
    return parsed if parsed.utcoffset() is not None else parsed.replace(tzinfo=timezone.utc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--source-type", choices=["news", "bank_report"], required=True)
    parser.add_argument("--init-db", action="store_true", help="Create the index schema if needed")
    parser.add_argument("--published-at", type=_publication_date)
    parser.add_argument("--title")
    parser.add_argument("--source-url")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--chunk-overlap", type=int, default=40)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--tessdata-dir")
    args = parser.parse_args()
    store = PostgresStore()
    if args.init_db:
        store.initialize()
    operation = index_directory if args.path.is_dir() else index_document
    result = operation(
        args.path, store, source_type=args.source_type,
        embedder=BGEEmbedder(device=args.device), title=args.title,
        source_url=args.source_url, published_at=args.published_at,
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size, ocr=not args.no_ocr, tessdata_dir=args.tessdata_dir,
    )
    for item in result if isinstance(result, list) else [result]:
        print(json.dumps(asdict(item), ensure_ascii=False))


if __name__ == "__main__":
    main()
