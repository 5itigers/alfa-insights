import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.ingestion.indexing import index_document
from rag.storage import DEFAULT_DATABASE_URL, IndexedDocument, PostgresStore
if __package__:
    from .test_indexing import FixtureEncoder
else:
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
            """SELECT c.content, c.metadata, vector_dims(c.embedding) AS dimensions,
                d.source_type, d.pipeline_config,
                to_tsvector('russian', c.content)
                    @@ plainto_tsquery('russian', 'кредитование') AS keyword_match,
                c.embedding <=> c.embedding AS cosine_distance
                FROM document_chunks c JOIN rag_documents d
                ON d.version_id = c.source_id WHERE d.version_id = %s""",
            (result.version_id,),
        )
        self.assertEqual(len(rows), result.chunk_count)
        self.assertTrue(rows[0]["keyword_match"])
        self.assertEqual(rows[0]["dimensions"], 1024)
        self.assertEqual(rows[0]["cosine_distance"], 0)
        self.assertEqual(rows[0]["source_type"], "bank_report")
        self.assertEqual(rows[0]["pipeline_config"]["revision"], "fixture-v1")
        metadata = rows[0]["metadata"]
        self.assertEqual(rows[0]["content"],
                         self.path.read_text()[metadata["start_char"]:metadata["end_char"]])
        self.assertIsNone(metadata["page_number"])
        self.assertEqual(metadata["extraction_method"], "text")
        self.assertGreater(metadata["token_count"], 0)
        self.assertTrue(metadata["chunk_id"])

    def test_unchanged_file_does_not_duplicate_rows(self):
        first = self.index()
        second = self.index()
        self.assertTrue(second.skipped)
        self.assertEqual(self.encoder.calls, 1)
        rows = self.query(
            "SELECT count(*) AS count FROM document_chunks WHERE source_id = %s",
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
            "SELECT content FROM document_chunks WHERE source_id = %s", (first.version_id,)
        )
        self.assertIn("Ключевая ставка", old_chunks[0]["content"])

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

        first = self.index()
        current = self.store.get_current(str(self.path.resolve()))
        document = IndexedDocument(
            "failed-" + first.version_id, str(self.path.resolve()), "invalid-source-type",
            "Failure fixture", None, None, "changed", current["pipeline_signature"],
            self.encoder.model_name, self.encoder.revision, 1, current["pipeline_config"],
        )
        chunk = chunk_pages(load_text(self.path), self.encoder.tokenizer)[0]
        vectors = self.encoder.encode_passages([chunk.text])
        with self.assertRaises(CheckViolation):
            self.store.save_document(document, [chunk], vectors)
        self.assertEqual(self.store.get_current(str(self.path.resolve()))["version_id"], first.version_id)
        self.assertEqual(self.query(
            "SELECT count(*) AS count FROM rag_documents WHERE version_id = %s",
            (document.version_id,),
        )[0]["count"], 0)

    def test_all_pages_use_unique_document_wide_positions(self):
        from rag.ingestion.chunking import chunk_pages
        from rag.ingestion.loaders import DocumentPage

        first = self.index()
        current = self.store.get_current(str(self.path.resolve()))
        document = IndexedDocument(
            "pages-" + first.version_id, str(self.path.resolve()), "bank_report",
            "Two pages", None, None, "pages", current["pipeline_signature"],
            self.encoder.model_name, self.encoder.revision, 2, current["pipeline_config"],
        )
        chunks = chunk_pages([
            DocumentPage("one two", document.source_path, page_number=1),
            DocumentPage("three four", document.source_path, page_number=2),
        ], self.encoder.tokenizer)
        self.assertEqual([chunk.chunk_index for chunk in chunks], [0, 0])
        vectors = self.encoder.encode_passages([chunk.text for chunk in chunks])
        self.assertEqual(self.store.save_document(document, chunks, vectors), 2)
        rows = self.query(
            "SELECT chunk_index, content, metadata FROM document_chunks "
            "WHERE source_id = %s ORDER BY chunk_index", (document.version_id,),
        )
        self.assertEqual([row["chunk_index"] for row in rows], [0, 1])
        self.assertEqual([row["content"] for row in rows], ["one two", "three four"])
        self.assertEqual([row["metadata"]["page_number"] for row in rows], [1, 2])
        self.assertEqual([row["metadata"]["page_chunk_index"] for row in rows], [0, 0])

    def test_initialization_preserves_existing_table_rows(self):
        from pgvector.psycopg import register_vector
        from psycopg.types.json import Jsonb

        source_id = "external-" + str(self.path)
        with self.store._connect() as connection:
            register_vector(connection)
            connection.execute(
                "INSERT INTO document_chunks (source_id, chunk_index, content, metadata, embedding) "
                "VALUES (%s, 0, %s, %s, %s)",
                (source_id, "Existing row", Jsonb({"original_field": "kept"}),
                 self.encoder.encode_passages(["Existing row"])[0]),
            )
        self.store.initialize()
        rows = self.query(
            "SELECT content, metadata FROM document_chunks WHERE source_id = %s", (source_id,),
        )
        self.assertEqual(rows, [{"content": "Existing row", "metadata": {"original_field": "kept"}}])

    def test_model_revision_change_creates_new_version(self):
        first = self.index()
        self.encoder.revision = "fixture-v2"
        second = self.index()
        self.assertNotEqual(first.version_id, second.version_id)
        self.assertEqual(self.store.get_current(str(self.path.resolve()))["embedding_revision"], "fixture-v2")




class StorageConfigurationTests(unittest.TestCase):
    def test_defaults_to_requested_local_database(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(PostgresStore().database_url, DEFAULT_DATABASE_URL)
            self.assertEqual(DEFAULT_DATABASE_URL,
                             "postgresql://postgres@localhost:5432/AI-Service-vectordb")

    def test_environment_and_explicit_connection_overrides(self):
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://environment"}):
            self.assertEqual(PostgresStore().database_url, "postgresql://environment")
            self.assertEqual(PostgresStore("postgresql://explicit").database_url,
                             "postgresql://explicit")


if __name__ == "__main__":
    unittest.main()
