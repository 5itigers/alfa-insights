import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.embeddings import BGEEmbedder
from rag.ingestion.indexing import index_directory, index_document
from rag.storage import PostgresStore
if __package__:
    from .test_chunking import make_tokenizer
else:
    from test_chunking import make_tokenizer


class FixtureEncoder:
    model_name = "fixture-encoder"
    revision = "fixture-v1"
    dimension = 1024
    max_length = 512

    def __init__(self):
        self.tokenizer = make_tokenizer()
        self.calls = 0

    def encode_passages(self, passages, *, batch_size=4):
        self.calls += 1
        vectors = np.zeros((len(passages), 1024), dtype=np.float32)
        vectors[:, 0] = 1
        return vectors


class MemoryStore:
    def __init__(self):
        self.current = {}
        self.saved = []

    def get_current(self, source_path):
        return self.current.get(source_path)

    def save_document(self, document, chunks, vectors):
        self.saved.append((document, chunks, vectors))
        self.current[document.source_path] = dict(asdict(document), chunk_count=len(chunks))
        return len(chunks)


class IndexingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "report.txt"
        self.path.write_text("Ключевая ставка влияет на стоимость кредитования.", encoding="utf-8")
        self.encoder = FixtureEncoder()
        self.store = MemoryStore()

    def index(self, **options):
        return index_document(
            self.path, self.store, source_type="bank_report", embedder=self.encoder, **options
        )

    def test_indexes_loaded_chunks_with_metadata_and_vectors(self):
        published = datetime(2026, 9, 14, tzinfo=timezone.utc)
        result = self.index(published_at=published, source_url="https://example.org/report")
        document, chunks, vectors = self.store.saved[0]
        self.assertFalse(result.skipped)
        self.assertEqual(document.source_type, "bank_report")
        self.assertEqual(document.published_at, published)
        self.assertEqual(document.source_url, "https://example.org/report")
        self.assertEqual(vectors.shape, (len(chunks), 1024))
        self.assertTrue(all(chunk.source_path == str(self.path.resolve()) for chunk in chunks))
        self.assertEqual(document.pipeline_config["chunk_size"], 400)

    def test_unchanged_file_skips_loading_and_embedding(self):
        first = self.index()
        with patch("rag.ingestion.indexing.load_document", side_effect=AssertionError("Must skip OCR")):
            second = self.index()
        self.assertTrue(second.skipped)
        self.assertEqual(first.version_id, second.version_id)
        self.assertEqual(self.encoder.calls, 1)

    def test_changed_contents_produce_a_new_version(self):
        first = self.index()
        self.path.write_text("Новое событие и новая аналитика.", encoding="utf-8")
        second = self.index()
        self.assertNotEqual(first.version_id, second.version_id)
        self.assertEqual(len(self.store.saved), 2)

    def test_chunk_settings_model_revision_and_metadata_invalidate_cache(self):
        first = self.index()
        changed_settings = self.index(chunk_size=128)
        self.encoder.revision = "fixture-v2"
        changed_model = self.index(chunk_size=128)
        changed_metadata = self.index(chunk_size=128, title="Updated title")
        self.assertEqual(len({
            first.version_id, changed_settings.version_id,
            changed_model.version_id, changed_metadata.version_id,
        }), 4)

    def test_encoding_failure_does_not_replace_current_version(self):
        first = self.index()
        self.path.write_text("Changed text", encoding="utf-8")
        with patch.object(self.encoder, "encode_passages", side_effect=RuntimeError("Model failed")):
            with self.assertRaisesRegex(RuntimeError, "Model failed"):
                self.index()
        self.assertEqual(self.store.get_current(str(self.path.resolve()))["version_id"], first.version_id)

    def test_file_changes_during_encoding_are_rejected(self):
        first = self.index()
        self.path.write_text("Version two", encoding="utf-8")
        original_encode = self.encoder.encode_passages

        def changing_encode(passages, **options):
            self.path.write_text("Version three", encoding="utf-8")
            return original_encode(passages, **options)

        with patch.object(self.encoder, "encode_passages", side_effect=changing_encode):
            with self.assertRaisesRegex(RuntimeError, "changed during indexing"):
                self.index()
        self.assertEqual(self.store.get_current(str(self.path.resolve()))["version_id"], first.version_id)

    def test_missing_publication_date_remains_unknown(self):
        self.index()
        self.assertIsNone(self.store.saved[0][0].published_at)

    def test_rejects_naive_publication_date(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            self.index(published_at=datetime(2026, 9, 14))

    def test_rejects_unknown_source_type(self):
        with self.assertRaisesRegex(ValueError, "source_type"):
            index_document(self.path, self.store, source_type="brief", embedder=self.encoder)

    def test_rejects_incompatible_embedding_dimension(self):
        self.encoder.dimension = 384
        with self.assertRaisesRegex(ValueError, "1024"):
            self.index()

    def test_directory_ignores_other_formats_and_reuses_encoder(self):
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "article.md").write_text("one two three", encoding="utf-8")
        (self.root / "company.csv").write_text("id,revenue", encoding="utf-8")
        results = index_directory(
            self.root, self.store, source_type="news", embedder=self.encoder
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(self.encoder.calls, 2)

    def test_storage_rejects_invalid_vectors_before_connecting(self):
        self.index()
        document, chunks, _ = self.store.saved[0]
        store = PostgresStore("postgresql://unused")
        invalid = [
            np.ones((len(chunks), 384)),
            np.full((len(chunks), 1024), np.nan),
            np.zeros((len(chunks), 1024)),
        ]
        with patch.object(store, "_connect", side_effect=AssertionError("Must validate first")):
            for vectors in invalid:
                with self.subTest(shape=vectors.shape):
                    with self.assertRaises(ValueError):
                        store.save_document(document, chunks, vectors)

    def test_empty_embeddings_do_not_load_model(self):
        encoder = BGEEmbedder()
        self.assertEqual(encoder.encode_passages([]).shape, (0, 1024))
        self.assertNotIn("model", encoder.__dict__)

    def test_storage_rejects_invalid_page_metadata_before_connecting(self):
        self.index()
        document, chunks, vectors = self.store.saved[0]
        invalid_chunks = [replace(chunk, page_number=0) for chunk in chunks]
        store = PostgresStore("postgresql://unused")
        with patch.object(store, "_connect", side_effect=AssertionError("Must validate first")):
            with self.assertRaisesRegex(ValueError, "citation metadata"):
                store.save_document(document, invalid_chunks, vectors)

    def test_embedding_adapter_rejects_overlong_inputs_before_encoding(self):
        encoder = BGEEmbedder(max_length=4)

        class Backend:
            tokenizer = self.encoder.tokenizer

            def encode(self, *args, **kwargs):
                raise AssertionError("Must reject truncation")

        encoder.__dict__["model"] = Backend()
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            encoder.encode_passages(["one two three four five"])

    def test_embedding_adapter_encodes_query_as_one_vector(self):
        encoder = BGEEmbedder()

        class Backend:
            tokenizer = self.encoder.tokenizer

            def encode(self, texts, **kwargs):
                return np.ones((len(texts), 1024), dtype=np.float32) / 32

        encoder.__dict__["model"] = Backend()
        self.assertEqual(encoder.encode_query("one two").shape, (1024,))


if __name__ == "__main__":
    unittest.main()
