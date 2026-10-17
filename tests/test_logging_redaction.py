import io
import logging

import pytest

from app.logging_setup import (
    REDACTED_EMAIL,
    REDACTED_NAME,
    REDACTED_PHONE,
    RedactingFormatter,
    LOG_FORMAT,
    mask_email,
    redact,
    register_pii,
    scrub_visible_text,
)


def test_any_email_is_redacted_even_if_unregistered():
    out = redact("contact someone.else@veltrix.example now")
    assert "someone.else" not in out and REDACTED_EMAIL in out


@pytest.mark.parametrize(
    "phone",
    ["+91 98221 51875", "+1 512-555-0182", "+49 30 5550 2681", "(512) 555-0182", "9822151875"],
)
def test_phone_shapes_are_redacted(phone):
    out = redact(f"call {phone} please")
    assert REDACTED_PHONE in out
    assert "555" not in out and "98221" not in out and "5550" not in out


def test_registered_name_is_redacted_case_insensitively():
    register_pii(names=["Zed Testperson"])
    out = redact("user zed TESTPERSON asked about leave")
    assert "Testperson".lower() not in out.lower()
    assert REDACTED_NAME in out


def test_registered_values_are_labelled_by_kind():
    register_pii(emails=["zed.testperson@veltrix.example"], phones=["0123 456"])
    out = redact("mail zed.testperson@veltrix.example phone 0123 456")
    assert out == f"mail {REDACTED_EMAIL} phone {REDACTED_PHONE}"


def test_longest_name_wins():
    register_pii(names=["Anna Lee", "Anna Lee Smith"])
    assert redact("Anna Lee Smith joined") == f"{REDACTED_NAME} joined"


@pytest.mark.parametrize(
    "safe",
    [
        "employee_id=E1001 region=IN",
        "date 2026-10-15 to 2026-10-19",
        "request 3f2b8c1e-9d4a-4f7e-8a21-5c6d7e8f9a0b created",
        "corpus_tokens=2874 working_days=3",
    ],
)
def test_ids_dates_uuids_and_counts_are_left_alone(safe):
    assert redact(safe) == safe


def _capture(logger_name: str):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter(LOG_FORMAT))
    logger = logging.getLogger(logger_name)
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger, stream


def test_formatter_redacts_message_args_and_tracebacks():
    register_pii(names=["Zed Testperson"])
    logger, stream = _capture("test.redaction")
    try:
        raise ValueError("bad record for Zed Testperson <zed@veltrix.example>")
    except ValueError:
        logger.error("failed for %s", "Zed Testperson", exc_info=True)
    text = stream.getvalue()
    assert "Testperson" not in text
    assert "zed@veltrix.example" not in text
    assert REDACTED_NAME in text and REDACTED_EMAIL in text


def test_mask_email_is_first_character_stars_and_domain():
    assert mask_email("ravi.kumar@veltrix.example") == "r***@veltrix.example"
    assert mask_email("Eli.Fenn@Veltrix.example") == "E***@Veltrix.example"


def test_visible_text_masks_emails_and_drops_phones():
    text = scrub_visible_text("write to eli.fenn@veltrix.example or call +91 90000 00004")
    assert "eli.fenn@veltrix.example" not in text
    assert "e***@veltrix.example" in text
    assert "90000" not in text and "+91" not in text
