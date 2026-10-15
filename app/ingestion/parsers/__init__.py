"""Document parsers: file path in, ``ParsedDocument`` (sections) out."""
from app.ingestion.parsers.dispatch import PARSERS, parse_document
from app.ingestion.parsers.models import Kind, ParsedDocument, Section
from app.ingestion.parsers.tables import rows_to_text

__all__ = ["Kind", "PARSERS", "ParsedDocument", "Section", "parse_document", "rows_to_text"]
