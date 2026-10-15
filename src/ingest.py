"""PDF/image extraction. Uploaded bytes are held in memory only."""
from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

from src.chunking import PageText

LOGGER = logging.getLogger(__name__)
SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


class IngestError(ValueError):
    """A document cannot be read safely."""


def _ocr_image(image: Image.Image, claude: object | None) -> str:
    if claude is not None and hasattr(claude, "transcribe_image"):
        try:
            return claude.transcribe_image(image)  # type: ignore[attr-defined]
        except Exception:
            LOGGER.warning("Vision transcription failed; trying local OCR")
    try:
        import pytesseract
        return pytesseract.image_to_string(image, lang="eng")
    except Exception as exc:
        raise IngestError("Image transcription failed. Install Tesseract OCR for scanned PDFs and photos.") from exc


def extract_document(data: bytes, filename: str, claude: object | None = None) -> list[PageText]:
    """Extract document lines, preserving page numbers and rejecting empty input."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise IngestError("Unsupported file type. Upload PDF, PNG, JPG or WEBP.")
    if not data:
        raise IngestError("The file is empty.")
    if suffix != ".pdf":
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
            text = _ocr_image(image, claude)
        except (OSError, ValueError) as exc:
            raise IngestError("The image could not be read.") from exc
        pages = [PageText(1, text)]
    else:
        pages = []
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                texts = [page.extract_text(layout=False) or "" for page in pdf.pages]
        except Exception:
            LOGGER.warning("pdfplumber failed; trying pypdf")
            try:
                reader = PdfReader(io.BytesIO(data))
                texts = [page.extract_text() or "" for page in reader.pages]
            except Exception as exc:
                raise IngestError("The PDF could not be read.") from exc
        for index, text in enumerate(texts, 1):
            if not text.strip():
                try:
                    reader = PdfReader(io.BytesIO(data))
                    text = reader.pages[index - 1].extract_text() or ""
                except Exception:
                    pass
            if not text.strip():
                try:
                    import pypdfium2 as pdfium
                    pdf = pdfium.PdfDocument(data)
                    image = pdf[index - 1].render(scale=2).to_pil().convert("RGB")
                    text = _ocr_image(image, claude)
                except Exception as exc:
                    LOGGER.warning("OCR failed for PDF page %d: %s", index, type(exc).__name__)
            pages.append(PageText(index, text))
    if not any(page.text.strip() for page in pages):
        raise IngestError("No readable text was found. Try a clearer scan or photo.")
    LOGGER.info("Document extracted: %d pages", len(pages))
    return pages
