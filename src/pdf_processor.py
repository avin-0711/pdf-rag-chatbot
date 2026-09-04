from io import BytesIO
from pathlib import Path

import fitz
from pypdf import PdfReader


def extract_pages(pdf_source: bytes | str | Path) -> list[dict[str, str | int]]:
    """Extract non-empty PDF pages while preserving page numbers."""
    if isinstance(pdf_source, bytes):
        document = fitz.open(stream=pdf_source, filetype="pdf")
    else:
        document = fitz.open(str(pdf_source))

    try:
        return [
            {"page": page_number, "text": text}
            for page_number, page in enumerate(document, start=1)
            if (text := page.get_text("text").strip())
        ]
    finally:
        document.close()


def extract_text_from_pdf(pdf_path: str | Path) -> list[dict[str, str | int]]:
    """Compatibility wrapper returning filename and page-number metadata."""
    path = Path(pdf_path)
    return [
        {"filename": path.name, "page_number": page["page"], "text": page["text"]}
        for page in extract_pages(path)
    ]