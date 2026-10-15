"""OCR for PDF pages that have no usable text layer.

Pages are rasterised at 300 DPI (pdf2image / poppler) and read with
tesseract (pytesseract). The decision to use OCR is made per page by the PDF
parser from the text layer alone: there are no file-name or doc_id special
cases anywhere. ``pdf2image`` and ``pytesseract`` are imported lazily, so a
missing poppler/tesseract only matters when a page actually needs OCR.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median

OCR_DPI = 300
OCR_LANG = "eng"


@dataclass(frozen=True)
class OcrLine:
    text: str
    height: float  # median word height in pixels: a proxy for font size
    paragraph: tuple[int, int]  # (block, paragraph) ids from the OCR engine


def ocr_pdf_page(path: str | Path, page_number: int) -> list[OcrLine]:
    """OCR one 1-based page and return its lines in reading order.

    Raises if poppler or tesseract is unavailable or the page cannot be
    rendered; the caller decides what that means for the document.
    """
    import pytesseract
    from pdf2image import convert_from_path

    images = convert_from_path(
        str(path), dpi=OCR_DPI, first_page=page_number, last_page=page_number, grayscale=True
    )
    if not images:
        return []
    data = pytesseract.image_to_data(
        images[0], lang=OCR_LANG, output_type=pytesseract.Output.DICT
    )

    # Group recognised words into lines using the engine's own layout ids.
    grouped: dict[tuple[int, int, int], list[tuple[int, str, int]]] = {}
    for i, word in enumerate(data["text"]):
        word = (word or "").strip()
        if not word or float(data["conf"][i]) < 0:  # conf -1 = layout row, not a word
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        grouped.setdefault(key, []).append((data["left"][i], word, data["height"][i]))

    lines = []
    for (block, par, _line), words in grouped.items():  # dict keeps reading order
        words.sort(key=lambda w: w[0])
        lines.append(
            OcrLine(
                text=" ".join(w[1] for w in words),
                height=float(median(w[2] for w in words)),
                paragraph=(block, par),
            )
        )
    return lines
