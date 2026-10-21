"""Load local documents while preserving their source and PDF page numbers."""

import logging
import shutil
import subprocess
from contextlib import closing
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Literal

from pypdf import PdfReader
from pypdf.errors import PyPdfError


logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
_PDF_RENDER_LOCK = Lock()


@dataclass(frozen=True)
class DocumentPage:
    text: str
    source_path: str
    page_number: int | None = None
    extraction_method: Literal["text", "ocr"] = "text"


def _file_path(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise IsADirectoryError(f"Expected a document file: {source}")
    return source


def _ocr_page(
    source: Path,
    page_index: int,
    language: str,
    tessdata_dir: str | Path | None,
) -> str:
    tesseract = shutil.which("tesseract")
    if tesseract is None:
        raise RuntimeError(
            f"Page {page_index + 1} of {source} needs OCR. "
            "Install Tesseract and the requested language data, "
            "or pass ocr=False to load only embedded text."
        )

    import pypdfium2 as pdfium

    with _PDF_RENDER_LOCK, pdfium.PdfDocument(str(source)) as document:
        with closing(document[page_index]) as page:
            with closing(page.render(scale=300 / 72)) as bitmap:
                with bitmap.to_pil() as image, BytesIO() as buffer:
                    image.save(buffer, format="PNG")
                    image_bytes = buffer.getvalue()

    command = [tesseract, "stdin", "stdout", "-l", language, "--dpi", "300"]
    if tessdata_dir is not None:
        command.extend(["--tessdata-dir", str(Path(tessdata_dir).expanduser())])

    try:
        result = subprocess.run(
            command, input=image_bytes, capture_output=True, timeout=120
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"OCR timed out on page {page_index + 1} of {source}"
        ) from error

    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"OCR failed on page {page_index + 1} of {source}. "
            f"Check that Tesseract language data for {language!r} is installed. {detail}"
        )
    return result.stdout.decode("utf-8").strip()


def load_pdf(
    path: str | Path,
    *,
    ocr: bool = True,
    ocr_language: str = "rus+eng",
    tessdata_dir: str | Path | None = None,
) -> list[DocumentPage]:
    """Return nonempty pages, using OCR when a page has no embedded text.

    Page numbers are one-based and still match the original PDF when blank
    pages are skipped. OCR requires Tesseract and its language data; recognition
    errors and complex table/chart layouts still need review.
    """
    source = _file_path(path)
    pages = []

    try:
        with source.open("rb") as stream:
            reader = PdfReader(stream)
            if reader.is_encrypted:
                raise ValueError(f"Encrypted PDFs are not supported: {source}")

            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                extraction_method: Literal["text", "ocr"] = "text"
                if not text and ocr:
                    text = _ocr_page(
                        source, page_number - 1, ocr_language, tessdata_dir
                    )
                    extraction_method = "ocr"

                if not text:
                    logger.warning(
                        "Skipping page %s of %s: no readable text.",
                        page_number,
                        source,
                    )
                    continue

                pages.append(
                    DocumentPage(text, str(source), page_number, extraction_method)
                )
    except PyPdfError as error:
        raise ValueError(f"Could not read PDF {source}: {error}") from error

    if not pages:
        raise ValueError(
            f"No readable text found in {source}. "
            "Image-only PDFs require OCR; enable it with ocr=True."
        )
    return pages


def load_text(path: str | Path) -> list[DocumentPage]:
    """Load one UTF-8 text or Markdown document without inventing page numbers."""
    source = _file_path(path)
    text = source.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"No readable text found in {source}")
    return [DocumentPage(text=text, source_path=str(source))]


def load_document(
    path: str | Path,
    *,
    ocr: bool = True,
    ocr_language: str = "rus+eng",
    tessdata_dir: str | Path | None = None,
) -> list[DocumentPage]:
    """Load a PDF, plain-text article, or Markdown file."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(
            path, ocr=ocr, ocr_language=ocr_language, tessdata_dir=tessdata_dir
        )
    if suffix in {".txt", ".md"}:
        return load_text(path)
    raise ValueError(f"Unsupported document format {suffix!r}: {path}")


def load_directory(
    path: str | Path,
    *,
    recursive: bool = True,
    ocr: bool = True,
    ocr_language: str = "rus+eng",
    tessdata_dir: str | Path | None = None,
) -> list[DocumentPage]:
    """Load supported files in path order, stopping if a document cannot be read.

    Point this at a news or bank-research folder rather than a directory that
    also contains the hackathon brief or other reference material.
    """
    directory = Path(path).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a document directory: {directory}")

    paths = directory.rglob("*") if recursive else directory.glob("*")
    documents = []
    for source in sorted(paths):
        if source.is_file() and source.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.extend(
                load_document(
                    source,
                    ocr=ocr,
                    ocr_language=ocr_language,
                    tessdata_dir=tessdata_dir,
                )
            )
    return documents
