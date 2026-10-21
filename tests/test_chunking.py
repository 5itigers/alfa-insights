import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.ingestion.chunking import chunk_page, chunk_pages
from rag.ingestion.loaders import DocumentPage, load_text


def make_tokenizer(byte_level=False):
    if byte_level:
        backend = Tokenizer(models.BPE(unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=260,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"],
        )
    else:
        backend = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"]
        )
    backend.train_from_iterator([
        " ".join(f"w{index}" for index in range(100)),
        "one two three four five six seven eight nine ten eleven twelve.",
        "Ключевая ставка влияет на стоимость кредитования.\n\nПример документа.",
    ], trainer=trainer)
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", backend.token_to_id("[CLS]")),
            ("[SEP]", backend.token_to_id("[SEP]")),
        ],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]",
        model_max_length=8192,
    )


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = make_tokenizer()
        cls.byte_tokenizer = make_tokenizer(byte_level=True)

    def assert_valid_chunks(self, page, chunks, tokenizer, limit):
        covered = set()
        for index, chunk in enumerate(chunks):
            self.assertTrue(chunk.text)
            self.assertEqual(chunk.text, page.text[chunk.start_char:chunk.end_char])
            self.assertEqual(chunk.source_path, page.source_path)
            self.assertEqual(chunk.page_number, page.page_number)
            self.assertEqual(chunk.extraction_method, page.extraction_method)
            self.assertEqual(chunk.chunk_index, index)
            self.assertEqual(chunk.token_count, len(tokenizer.encode(chunk.text)))
            self.assertLessEqual(chunk.token_count, limit)
            covered.update(range(chunk.start_char, chunk.end_char))
        for position, character in enumerate(page.text):
            if not character.isspace():
                self.assertIn(position, covered, f"Lost source character at {position}")

    def test_short_page_preserves_text_and_metadata(self):
        page = DocumentPage("  Ключевая ставка.\n", "report.pdf", 7, "ocr")
        chunks = chunk_page(page, self.tokenizer)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Ключевая ставка.")
        self.assert_valid_chunks(page, chunks, self.tokenizer, 400)

    def test_long_page_stays_within_token_budget(self):
        page = DocumentPage(" ".join(f"w{index}" for index in range(100)), "report.pdf", 1)
        chunks = chunk_page(page, self.tokenizer, chunk_size=18, chunk_overlap=4)
        self.assertGreater(len(chunks), 1)
        self.assert_valid_chunks(page, chunks, self.tokenizer, 18)

    def test_overlap_repeats_source_tokens(self):
        page = DocumentPage(" ".join(f"w{index}" for index in range(20)), "report.pdf", 1)
        chunks = chunk_page(page, self.tokenizer, chunk_size=8, chunk_overlap=2)
        self.assertEqual(chunks[0].text.split(), [f"w{index}" for index in range(6)])
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous.text.split()[-2:], current.text.split()[:2])
        self.assert_valid_chunks(page, chunks, self.tokenizer, 8)

    def test_zero_overlap_does_not_lose_text(self):
        page = DocumentPage(" ".join(f"w{index}" for index in range(25)), "report.pdf", 1)
        chunks = chunk_page(page, self.tokenizer, chunk_size=8, chunk_overlap=0)
        self.assert_valid_chunks(page, chunks, self.tokenizer, 8)
        self.assertEqual(" ".join(chunk.text for chunk in chunks), page.text)

    def test_prefers_paragraph_boundary(self):
        first = "one two three four five six"
        second = "seven eight nine ten eleven twelve"
        page = DocumentPage(first + "\n\n" + second, "report.pdf", 1)
        chunks = chunk_page(page, self.tokenizer, chunk_size=10, chunk_overlap=0)
        self.assertEqual([chunk.text for chunk in chunks], [first, second])
        self.assert_valid_chunks(page, chunks, self.tokenizer, 10)

    def test_prefers_sentence_boundary_without_paragraphs(self):
        page = DocumentPage("one two three four. five six seven eight.", "report.pdf", 1)
        chunks = chunk_page(page, self.tokenizer, chunk_size=9, chunk_overlap=0)
        self.assertEqual(chunks[0].text, "one two three four.")
        self.assert_valid_chunks(page, chunks, self.tokenizer, 9)

    def test_multilingual_unicode_offsets_do_not_corrupt_characters(self):
        text = "Ставка выросла. 中文分析 🙂 café e\u0301 🇷🇺\n\n" * 8
        page = DocumentPage(text, "report.pdf", 3, "ocr")
        chunks = chunk_page(page, self.byte_tokenizer, chunk_size=48, chunk_overlap=8)
        self.assert_valid_chunks(page, chunks, self.byte_tokenizer, 48)
        self.assertTrue(all("\ufffd" not in chunk.text for chunk in chunks))

    def test_long_unbroken_text_is_split_without_loss(self):
        page = DocumentPage("абвгд" * 80, "report.pdf", 1)
        chunks = chunk_page(page, self.byte_tokenizer, chunk_size=32, chunk_overlap=4)
        self.assertGreater(len(chunks), 1)
        self.assert_valid_chunks(page, chunks, self.byte_tokenizer, 32)

    def test_chunk_ids_are_stable_and_source_specific(self):
        page = DocumentPage("one two three", "first.pdf", 1)
        first = chunk_page(page, self.tokenizer)[0]
        self.assertEqual(first, chunk_page(page, self.tokenizer)[0])
        variants = [
            replace(page, source_path="second.pdf"),
            replace(page, page_number=2),
            replace(page, text="one two four"),
        ]
        for variant in variants:
            self.assertNotEqual(first.chunk_id, chunk_page(variant, self.tokenizer)[0].chunk_id)

    def test_does_not_merge_pages_or_renumber_them(self):
        pages = [
            DocumentPage("one two", "report.pdf", 1),
            DocumentPage("three four", "report.pdf", 3, "ocr"),
        ]
        chunks = chunk_pages(iter(pages), self.tokenizer)
        self.assertEqual([chunk.page_number for chunk in chunks], [1, 3])
        self.assertEqual([chunk.chunk_index for chunk in chunks], [0, 0])
        self.assertEqual([chunk.extraction_method for chunk in chunks], ["text", "ocr"])

    def test_plain_text_loader_integrates_without_page_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "article.txt"
            path.write_text("Ключевая ставка влияет на стоимость кредитования.", encoding="utf-8")
            pages = load_text(path)
            chunks = chunk_pages(pages, self.byte_tokenizer, chunk_size=48, chunk_overlap=8)
            self.assertTrue(all(chunk.page_number is None for chunk in chunks))
            self.assert_valid_chunks(pages[0], chunks, self.byte_tokenizer, 48)

    def test_empty_pages_and_empty_input(self):
        self.assertEqual(chunk_page(DocumentPage(" \n\t", "report.pdf", 1), self.tokenizer), [])
        self.assertEqual(chunk_pages([], self.tokenizer), [])

    def test_rejects_invalid_budgets(self):
        page = DocumentPage("one two", "report.pdf", 1)
        for options in (
            {"chunk_size": 0},
            {"chunk_size": 2},
            {"chunk_overlap": -1},
            {"chunk_size": 10, "chunk_overlap": 8},
            {"chunk_size": 513},
            {"chunk_size": 100, "max_length": 64},
            {"chunk_size": 4.5},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    chunk_page(page, self.tokenizer, **options)

    def test_respects_tokenizers_own_model_limit(self):
        tokenizer = make_tokenizer()
        tokenizer.model_max_length = 16
        with self.assertRaisesRegex(ValueError, "encoder limit of 16"):
            chunk_page(DocumentPage("one two", "report.pdf", 1), tokenizer)

    def test_rejects_slow_tokenizer(self):
        class SlowTokenizer:
            is_fast = False

        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            chunk_page(DocumentPage("one two", "report.pdf", 1), SlowTokenizer())

    def test_tiny_budget_for_multibyte_character_fails_without_looping(self):
        with self.assertRaisesRegex(ValueError, "too small"):
            chunk_page(
                DocumentPage("🙂", "report.pdf", 1), self.byte_tokenizer,
                chunk_size=3, chunk_overlap=0,
            )


if __name__ == "__main__":
    unittest.main()
