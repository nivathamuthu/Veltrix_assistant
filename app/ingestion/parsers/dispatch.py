"""Choose a parser by file extension. Never raises."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.ingestion.parsers.docx_parser import parse_docx
from app.ingestion.parsers.html_parser import parse_html
from app.ingestion.parsers.markdown_parser import parse_markdown
from app.ingestion.parsers.models import ParsedDocument
from app.ingestion.parsers.pdf_parser import parse_pdf
from app.logging_setup import get_logger

log = get_logger(__name__)

PARSERS: dict[str, Callable[[Path], ParsedDocument]] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".html": parse_html,
    ".htm": parse_html,
}

UNSUPPORTED = "UnsupportedFormat"


def parse_document(path: str | Path) -> ParsedDocument:
    """Parse one file. A file that cannot be parsed yields a document with
    ``error`` set and no sections; it never raises, so one bad file cannot stop
    startup.

    Only the file name and the exception *type* are logged: exception messages
    and tracebacks can quote the document's own content.
    """
    path = Path(path)
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        log.error("skipped doc=%s error=%s", path.name, UNSUPPORTED)
        return ParsedDocument(path=path, parser="none", error=UNSUPPORTED)
    try:
        return parser(path)
    except Exception as exc:  # noqa: BLE001 - corrupt, encrypted, missing, ...
        error = type(exc).__name__
        log.error("skipped doc=%s error=%s", path.name, error)
        return ParsedDocument(path=path, parser=parser.__name__.removeprefix("parse_"), error=error)
