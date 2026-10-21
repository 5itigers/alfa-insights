import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.ingestion.loaders import load_directory, load_document, load_pdf, load_text


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def make_pdf(self, name, texts, password=None):
        writer = PdfWriter()
        for text in texts:
            page = writer.add_blank_page(width=600, height=800)
            if text is not None:
                font = DictionaryObject({
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                })
                page[NameObject("/Resources")] = DictionaryObject({
                    NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
                })
                stream = DecodedStreamObject()
                stream.set_data(
                    f"BT /F1 18 Tf 50 700 Td ({text}) Tj ET".encode("ascii")
                )
                page[NameObject("/Contents")] = writer._add_object(stream)
        if password:
            writer.encrypt(password)
        path = self.root / name
        writer.write(path)
        writer.close()
        return path

    def make_scanned_pdf(self):
        path = self.root / "scanned.pdf"
        with Image.new("RGB", (1400, 250), "white") as image:
            draw = ImageDraw.Draw(image)
            draw.text(
                (50, 80), "LOCAL OCR TEST 123", fill="black",
                font=ImageFont.load_default(size=64),
            )
            image.save(path, "PDF", resolution=150)
        return path

    def test_pdf_preserves_original_page_numbers_and_source(self):
        path = self.make_pdf("report.pdf", ["First page", None, "Third page"])
        with self.assertLogs("rag.ingestion.loaders", level="WARNING"):
            pages = load_pdf(path, ocr=False)
        self.assertEqual([page.page_number for page in pages], [1, 3])
        self.assertEqual([page.text for page in pages], ["First page", "Third page"])
        self.assertTrue(all(page.source_path == str(path.resolve()) for page in pages))
        self.assertTrue(all(page.extraction_method == "text" for page in pages))

    def test_text_pdf_does_not_require_tesseract(self):
        path = self.make_pdf("report.pdf", ["Embedded text"])
        with patch("rag.ingestion.loaders.shutil.which", return_value=None):
            self.assertEqual(load_pdf(path)[0].text, "Embedded text")

    def test_loads_russian_utf8_with_bom_and_preserves_paragraphs(self):
        path = self.root / "article.TXT"
        path.write_text("\ufeffСтавка и кредитование.\n\nВторой абзац.\n", encoding="utf-8")
        page, = load_document(path)
        self.assertEqual(page.text, "Ставка и кредитование.\n\nВторой абзац.")
        self.assertIsNone(page.page_number)

    def test_markdown_is_supported(self):
        path = self.root / "article.md"
        path.write_text("# Заголовок\n\nНовость", encoding="utf-8")
        self.assertEqual(load_document(path)[0].text, "# Заголовок\n\nНовость")

    def test_rejects_empty_text(self):
        path = self.root / "empty.txt"
        path.write_text(" \n\t", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "No readable text"):
            load_text(path)

    def test_rejects_unsupported_format(self):
        path = self.root / "companies.csv"
        path.write_text("id,revenue\n1,100", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unsupported document format"):
            load_document(path)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_document(self.root / "missing.pdf")

    def test_file_loader_rejects_directory(self):
        with self.assertRaises(IsADirectoryError):
            load_text(self.root)

    def test_directory_loader_rejects_file(self):
        path = self.root / "article.txt"
        path.write_text("Text", encoding="utf-8")
        with self.assertRaises(NotADirectoryError):
            load_directory(path)

    def test_rejects_empty_pdf_without_ocr(self):
        path = self.make_pdf("blank.pdf", [None])
        with self.assertLogs("rag.ingestion.loaders", level="WARNING"):
            with self.assertRaisesRegex(ValueError, "No readable text"):
                load_pdf(path, ocr=False)

    def test_rejects_encrypted_pdf(self):
        path = self.make_pdf("encrypted.pdf", ["Secret"], password="password")
        with self.assertRaisesRegex(ValueError, "Encrypted PDFs"):
            load_pdf(path)

    def test_rejects_corrupt_pdf_with_source_in_error(self):
        path = self.root / "corrupt.pdf"
        path.write_bytes(b"not a PDF")
        with self.assertLogs("pypdf", level="WARNING"):
            with self.assertRaisesRegex(ValueError, "Could not read PDF.*corrupt.pdf"):
                load_pdf(path)

    def test_directory_order_and_recursion(self):
        (self.root / "a.txt").write_text("First", encoding="utf-8")
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "b.md").write_text("Second", encoding="utf-8")
        (self.root / "ignored.csv").write_text("Ignored", encoding="utf-8")
        self.assertEqual([page.text for page in load_directory(self.root)], ["First", "Second"])
        self.assertEqual(
            [page.text for page in load_directory(self.root, recursive=False)], ["First"]
        )

    def test_missing_ocr_executable_is_actionable(self):
        path = self.make_scanned_pdf()
        with patch("rag.ingestion.loaders.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "needs OCR.*Install Tesseract"):
                load_pdf(path)

    def test_ocr_failure_is_reported(self):
        path = self.make_scanned_pdf()
        result = subprocess.CompletedProcess([], 1, b"", b"Failed loading language 'rus'")
        with patch("rag.ingestion.loaders.shutil.which", return_value="tesseract"):
            with patch("rag.ingestion.loaders.subprocess.run", return_value=result):
                with self.assertRaisesRegex(RuntimeError, "OCR failed on page 1.*rus"):
                    load_pdf(path)

    def test_ocr_timeout_is_reported(self):
        path = self.make_scanned_pdf()
        with patch("rag.ingestion.loaders.shutil.which", return_value="tesseract"):
            with patch(
                "rag.ingestion.loaders.subprocess.run",
                side_effect=subprocess.TimeoutExpired("tesseract", 120),
            ):
                with self.assertRaisesRegex(RuntimeError, "OCR timed out on page 1"):
                    load_pdf(path)

    @unittest.skipUnless(shutil.which("tesseract"), "Tesseract is not installed")
    def test_scanned_pdf_uses_real_ocr(self):
        path = self.make_scanned_pdf()
        page, = load_pdf(path, ocr_language="eng")
        self.assertIn("LOCAL OCR TEST 123", page.text)
        self.assertEqual(page.page_number, 1)
        self.assertEqual(page.extraction_method, "ocr")
        self.assertEqual(page.source_path, str(path.resolve()))


if __name__ == "__main__":
    unittest.main()
