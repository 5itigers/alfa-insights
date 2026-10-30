import math
import os
from pathlib import Path
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.ingestion.chunking import Chunk
from rag.retrieval import Retriever
from rag.storage import IndexedDocument, PostgresStore


class QueryEncoder:
    model_name = "retrieval-fixture"
    revision = "fixture-v1"
    dimension = 1024

    def __init__(self):
        self.queries = []

    def encode_query(self, query):
        self.queries.append(query)
        return vector(1)


def vector(similarity):
    result = np.zeros(1024, dtype=np.float32)
    result[:2] = similarity, math.sqrt(1 - similarity ** 2)
    return result


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.encoder = QueryEncoder()
        self.store = Mock()
        self.store.search_chunks.return_value = []
        self.retriever = Retriever(self.store, embedder=self.encoder)

    def test_encodes_query_and_forwards_model_revision_and_filters(self):
        start = datetime(2026, 9, 1, tzinfo=timezone.utc)
        end = datetime(2026, 10, 1, tzinfo=timezone.utc)
        self.assertEqual(self.retriever.retrieve(
            "  Ключевая ставка  ", top_k=3, candidate_limit=12, rrf_k=40,
            source_type="bank_report", published_from=start, published_before=end,
        ), [])
        self.assertEqual(self.encoder.queries, ["Ключевая ставка"])
        args, options = self.store.search_chunks.call_args
        self.assertEqual(args[0], "Ключевая ставка")
        np.testing.assert_array_equal(args[1], vector(1))
        self.assertEqual(options, dict(
            embedding_model=self.encoder.model_name, embedding_revision="fixture-v1",
            top_k=3, candidate_limit=12, rrf_k=40, source_type="bank_report",
            published_from=start, published_before=end,
        ))

    def test_result_preserves_text_citation_metadata_and_separate_scores(self):
        row = dict(
            version_id="version-1", chunk_index=7, text="Ставка снижена.",
            source_path="/reports/bank.pdf", source_type="bank_report", title="Bond brief",
            source_url="https://example.org/brief", published_at=None,
            metadata=dict(chunk_id="chunk-1", page_number=3, start_char=8, end_char=22,
                          token_count=6, extraction_method="ocr", page_chunk_index=0),
            score=1 / 61, vector_score=0.8, keyword_score=None,
            vector_rank=1, keyword_rank=None,
        )
        self.store.search_chunks.return_value = [row]
        result = self.retriever.retrieve("ставка")[0]
        self.assertEqual(result.citation_id, "version-1:chunk-1")
        self.assertEqual(result.page_number, 3)
        self.assertEqual(result.extraction_method, "ocr")
        self.assertEqual(result.chunk_index, 7)
        self.assertEqual(result.text, "Ставка снижена.")
        self.assertEqual(result.vector_score, 0.8)
        self.assertIsNone(result.keyword_score)
        self.assertIn("metadata", row)

    def test_invalid_inputs_fail_before_embedding_or_database_access(self):
        cases = [
            (None, {}), (5, {}), (" \n\t", {}), ("null\x00byte", {}),
            ("query", {"top_k": 0}), ("query", {"top_k": True}),
            ("query", {"candidate_limit": 4}), ("query", {"candidate_limit": 2.5}),
            ("query", {"rrf_k": -1}), ("query", {"source_type": "brief"}),
            ("query", {"published_from": datetime(2026, 9, 1)}),
            ("query", {"published_before": "2026-10-01"}),
            ("query", dict(published_from=datetime(2026, 10, 1, tzinfo=timezone.utc),
                           published_before=datetime(2026, 9, 1, tzinfo=timezone.utc))),
        ]
        for query, options in cases:
            with self.subTest(query=query, options=options):
                with self.assertRaises(ValueError):
                    self.retriever.retrieve(query, **options)
        self.assertEqual(self.encoder.queries, [])
        self.store.search_chunks.assert_not_called()

    def test_incompatible_encoder_dimension_is_rejected(self):
        self.encoder.dimension = 384
        with self.assertRaisesRegex(ValueError, "1024"):
            Retriever(self.store, embedder=self.encoder)

    def test_multiple_queries_reuse_the_encoder(self):
        self.retriever.retrieve("first")
        self.retriever.retrieve("second")
        self.assertEqual(self.encoder.queries, ["first", "second"])

    def test_store_rejects_invalid_vectors_and_model_metadata_before_connecting(self):
        store = PostgresStore("postgresql://unused")
        options = dict(embedding_model="fixture", embedding_revision="v1")
        with patch.object(store, "_connect", side_effect=AssertionError("Must validate first")):
            for invalid in (np.zeros(1024), np.ones(384), np.ones((1, 1024)),
                            np.full(1024, np.nan), np.full(1024, np.inf)):
                with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                    store.search_chunks("query", invalid, **options)
            with self.assertRaises(ValueError):
                store.search_chunks("query", vector(1), embedding_model="fixture",
                                    embedding_revision="")
            with self.assertRaises(ValueError):
                store.search_chunks("query", vector(1), top_k=0, **options)

    def test_import_and_construction_do_not_load_inference_or_database_libraries(self):
        script = (
            "import sys; from src.rag.retrieval import Retriever; r = Retriever(); "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules; "
            "assert 'sentence_transformers' not in sys.modules; "
            "assert 'psycopg' not in sys.modules; "
            "assert 'model' not in r.embedder.__dict__"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_help_does_not_need_a_database_or_model_download(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.rag.retrieval", "--help"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--source-type", result.stdout)
        self.assertIn("--published-before", result.stdout)


@unittest.skipUnless(os.environ.get("RAG_TEST_DATABASE_URL"), "No disposable PostgreSQL test database")
class RetrievalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = PostgresStore(os.environ["RAG_TEST_DATABASE_URL"])
        cls.store.initialize()

    def setUp(self):
        self.encoder = QueryEncoder()
        self.encoder.model_name = "retrieval-fixture-" + uuid4().hex
        self.retriever = Retriever(self.store, embedder=self.encoder)
        self.published = datetime(2026, 9, 14, tzinfo=timezone.utc)

    def add(
        self, name, text, similarity=1, *, source_type="bank_report", published_at=None,
        model=None, revision="fixture-v1", version=None,
    ):
        source_path = f"/fixtures/{self.encoder.model_name}/{name}.pdf"
        version_id = f"{self.encoder.model_name}-{version or name}"
        chunk = Chunk("chunk-" + name, text, source_path, 3, 0, 0, len(text), 20, "ocr")
        document = IndexedDocument(
            version_id=version_id, source_path=source_path, source_type=source_type,
            title=name, source_url=f"https://example.org/{name}", published_at=published_at,
            content_hash=version_id, pipeline_signature="fixture", embedding_model=model or self.encoder.model_name,
            embedding_revision=revision, page_count=1,
            pipeline_config=dict(model=self.encoder.model_name, revision=revision),
        )
        self.store.save_document(document, [chunk], [vector(similarity)])
        return version_id

    def test_rrf_promotes_agreement_and_keeps_each_chunk_once(self):
        self.add("a-semantic", "Рост стоимости заимствований.", 1)
        agreement = self.add("b-agreement", "Ключевая ставка снижена.", 0.8)
        self.add("c-keyword", "Ключевая ставка повышена.", 0.2)
        results = self.retriever.retrieve("ключевая ставка", top_k=3, candidate_limit=3)
        self.assertEqual(results[0].version_id, agreement)
        self.assertEqual(len({result.citation_id for result in results}), 3)
        for result in results:
            expected = 1 / (60 + result.vector_rank)
            if result.keyword_rank is not None:
                expected += 1 / (60 + result.keyword_rank)
            self.assertAlmostEqual(result.score, expected)

    def test_keyword_candidate_can_survive_outside_vector_candidate_limit(self):
        self.add("semantic", "Стоимость заимствований растёт.", 1)
        keyword = self.add("keyword", "Ключевая ставка снижена.", 0)
        results = self.retriever.retrieve("ключевая ставка", top_k=1, candidate_limit=1)
        self.assertEqual(results[0].version_id, keyword)
        self.assertIsNone(results[0].vector_rank)
        self.assertIsNone(results[0].vector_score)
        self.assertEqual(results[0].keyword_rank, 1)

    def test_no_keywords_falls_back_to_cosine_order(self):
        self.add("distant", "Промышленное производство.", 0.1)
        nearest = self.add("nearest", "Рынок облигаций.", 1)
        results = self.retriever.retrieve("несуществующийтермин", top_k=2)
        self.assertEqual(results[0].version_id, nearest)
        self.assertAlmostEqual(results[0].vector_score, 1)
        self.assertTrue(all(result.keyword_rank is None for result in results))

    def test_russian_inflection_matches_and_stopword_queries_still_work(self):
        self.add("credit", "Стоимость кредитования увеличилась.")
        result = self.retriever.retrieve("кредитование")[0]
        self.assertEqual(result.keyword_rank, 1)
        self.assertGreater(result.keyword_score, 0)
        self.assertIsNone(self.retriever.retrieve("и в на")[0].keyword_rank)

    def test_source_filter_applies_before_both_candidate_limits(self):
        self.add("news", "Ключевая ставка.", 1, source_type="news")
        bank = self.add("bank", "Стоимость заимствований.", 0)
        results = self.retriever.retrieve(
            "ключевая ставка", source_type="bank_report", top_k=1, candidate_limit=1,
        )
        self.assertEqual([result.version_id for result in results], [bank])
        self.assertIsNone(results[0].keyword_rank)
        self.assertEqual(self.retriever.retrieve("ставка", source_type="news")[0].source_type, "news")

    def test_old_versions_and_incompatible_embeddings_are_excluded_from_both_searches(self):
        self.add("changed", "Ключевая ставка.", 1, version="old")
        current = self.add("changed", "Стоимость заимствований.", 0.1, version="current")
        self.add("other-revision", "Ключевая ставка.", 1, revision="v2")
        self.add("other-model", "Ключевая ставка.", 1, model="other-encoder")
        results = self.retriever.retrieve("ключевая ставка")
        self.assertEqual([result.version_id for result in results], [current])
        self.assertIsNone(results[0].keyword_rank)

    def test_publication_window_is_inclusive_exclusive_and_excludes_unknown_dates(self):
        self.add("unknown", "Ставка.")
        self.add("early", "Ставка.", published_at=self.published - timedelta(seconds=1))
        start = self.add("start", "Ставка.", published_at=self.published)
        inside = self.add("inside", "Ставка.", published_at=self.published + timedelta(hours=1))
        self.add("end", "Ставка.", published_at=self.published + timedelta(days=1))
        results = self.retriever.retrieve(
            "ставка", published_from=self.published.astimezone(timezone(timedelta(hours=3))),
            published_before=self.published + timedelta(days=1),
        )
        self.assertEqual({result.version_id for result in results}, {start, inside})
        lower_only = self.retriever.retrieve("ставка", published_from=self.published)
        self.assertEqual({result.title for result in lower_only}, {"start", "inside", "end"})
        upper_only = self.retriever.retrieve("ставка", published_before=self.published)
        self.assertEqual({result.title for result in upper_only}, {"early"})

    def test_database_result_preserves_immutable_citation_and_source_details(self):
        version_id = self.add("report", "Ключевая ставка.", published_at=self.published)
        result = self.retriever.retrieve("ставка")[0]
        self.assertEqual(result.citation_id, version_id + ":chunk-report")
        self.assertEqual(result.page_number, 3)
        self.assertEqual(result.published_at, self.published)
        self.assertEqual(result.source_url, "https://example.org/report")
        self.assertEqual(result.text, "Ключевая ставка.")
        self.assertEqual(result.end_char, len(result.text))
        self.assertEqual(result.extraction_method, "ocr")
        self.add("report", "Новая ставка.", version="updated")
        self.assertNotEqual(self.retriever.retrieve("ставка")[0].citation_id, result.citation_id)

    def test_empty_corpus_and_filters_with_no_matches_return_empty_list(self):
        self.assertEqual(self.retriever.retrieve("ставка"), [])
        self.add("bank", "Ставка.")
        self.assertEqual(self.retriever.retrieve("ставка", source_type="news"), [])

    def test_query_quotes_and_sql_characters_are_bound_as_data(self):
        self.add("report", "Ставка.")
        self.assertEqual(len(self.retriever.retrieve("'\" ); DROP TABLE rag_documents; --")), 1)
        self.assertEqual(len(self.retriever.retrieve("ставка")), 1)

    def test_equal_scores_have_repeatable_order_and_top_k_is_respected(self):
        for name in ("c", "b", "a"):
            self.add(name, "Ключевая ставка.")
        first = self.retriever.retrieve("ставка", top_k=2)
        second = self.retriever.retrieve("ставка", top_k=2)
        self.assertEqual([result.title for result in first], ["a", "b"])
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
