import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.ingestion.indexing import index_document
from rag.storage import PostgresStore
from test_indexing import FixtureEncoder


@unittest.skipUnless(os.environ.get("RAG_TEST_DATABASE_URL"), "No disposable PostgreSQL test database")
class StorageIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = PostgresStore(os.environ["RAG_TEST_DATABASE_URL"])
        cls.store.initialize()
        cls.store.initialize()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "bank's report.txt"
        self.path.write_text("Ключевая ставка влияет на стоимость кредитования.", encoding="utf-8")
        self.encoder = FixtureEncoder()

    def index(self):
        return index_document(self.path, self.store, source_type="bank_report", embedder=self.encoder)

    def query(self, sql, parameters):
        with self.store._connect() as connection:
            return connection.execute(sql, parameters).fetchall()

    def test_roundtrip_vectors_keywords_and_metadata(self):
        result = self.index()
        rows = self.query(
            """SELECT c.text, vector_dims(c.embedding) AS dimensions, c.start_char,
                c.end_char, c.page_number, d.source_type, d.pipeline_config,
                c.search_vector @@ plainto_tsquery('russian', 'кредитование') AS keyword_match,
                c.embedding <=> c.embedding AS cosine_distance
                FROM rag_chunks c JOIN rag_documents d
                ON d.version_id = c.document_version_id WHERE d.version_id = %s""",
            (result.version_id,),
        )
        self.assertEqual(len(rows), result.chunk_count)
        self.assertTrue(rows[0]["keyword_match"])
        self.assertEqual(rows[0]["dimensions"], 1024)
        self.assertEqual(rows[0]["cosine_distance"], 0)
        self.assertEqual(rows[0]["source_type"], "bank_report")
        self.assertEqual(rows[0]["pipeline_config"]["revision"], "fixture-v1")
        self.assertEqual(rows[0]["text"], self.path.read_text()[rows[0]["start_char"]:rows[0]["end_char"]])
        self.assertIsNone(rows[0]["page_number"])

    def test_unchanged_file_does_not_duplicate_rows(self):
        first = self.index()
        second = self.index()
        self.assertTrue(second.skipped)
        self.assertEqual(self.encoder.calls, 1)
        rows = self.query(
            "SELECT count(*) AS count FROM rag_chunks WHERE document_version_id = %s",
            (first.version_id,),
        )
        self.assertEqual(rows[0]["count"], first.chunk_count)

    def test_updates_preserve_old_chunks_and_only_one_current_version(self):
        first = self.index()
        self.path.write_text("Обновлённая аналитика и новое событие.", encoding="utf-8")
        second = self.index()
        rows = self.query(
            "SELECT version_id, is_current FROM rag_documents WHERE source_path = %s",
            (str(self.path.resolve()),),
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["version_id"]: row["is_current"] for row in rows}, {
            first.version_id: False, second.version_id: True,
        })
        old_chunks = self.query(
            "SELECT text FROM rag_chunks WHERE document_version_id = %s", (first.version_id,)
        )
        self.assertIn("Ключевая ставка", old_chunks[0]["text"])

    def test_restored_old_version_reuses_immutable_chunks(self):
        original = self.path.read_text()
        first = self.index()
        self.path.write_text("one two three", encoding="utf-8")
        self.index()
        self.path.write_text(original, encoding="utf-8")
        restored = self.index()
        self.assertEqual(restored.version_id, first.version_id)
        versions = self.query(
            "SELECT count(*) AS count FROM rag_documents WHERE source_path = %s",
            (str(self.path.resolve()),),
        )
        self.assertEqual(versions[0]["count"], 2)
        self.assertEqual(self.store.get_current(str(self.path.resolve()))["version_id"], first.version_id)

    def test_database_failure_rolls_back_new_version_and_current_marker(self):
        from psycopg.errors import CheckViolation
        from rag.ingestion.chunking import chunk_pages
        from rag.ingestion.loaders import load_text
        from rag.storage import IndexedDocument

        first = self.index()
        current = self.store.get_current(str(self.path.resolve()))
        document = IndexedDocument(
            "failed-" + first.version_id, str(self.path.resolve()), "bank_report",
            "Failure fixture", None, None, "changed", current["pipeline_signature"],
            self.encoder.model_name, self.encoder.revision, 1, current["pipeline_config"],
        )
        chunk = chunk_pages(load_text(self.path), self.encoder.tokenizer)[0]
        invalid_chunk = replace(chunk, chunk_id="invalid-page", page_number=0)
        vectors = self.encoder.encode_passages([chunk.text, invalid_chunk.text])
        with self.assertRaises(CheckViolation):
            self.store.save_document(document, [chunk, invalid_chunk], vectors)
        self.assertEqual(self.store.get_current(str(self.path.resolve()))["version_id"], first.version_id)
        self.assertEqual(self.query(
            "SELECT count(*) AS count FROM rag_documents WHERE version_id = %s",
            (document.version_id,),
        )[0]["count"], 0)

    def test_model_revision_change_creates_new_version(self):
        first = self.index()
        self.encoder.revision = "fixture-v2"
        second = self.index()
        self.assertNotEqual(first.version_id, second.version_id)
        self.assertEqual(self.store.get_current(str(self.path.resolve()))["embedding_revision"], "fixture-v2")


if __name__ == "__main__":
    unittest.main()
