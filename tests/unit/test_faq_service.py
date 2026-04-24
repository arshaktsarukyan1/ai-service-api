from datetime import date

import pytest

from app.application.faq_service import (
    build_faq_user_prompt,
    parse_faq_model_content,
)
from app.domain.exceptions import FaqResponseParseError
from app.domain.location import ConstructionSiteLocation

_SITE = ConstructionSiteLocation(
    id="test-1",
    name="Test Bridge",
    start_date=date(2025, 1, 1),
    expected_end_date=date(2026, 1, 1),
    description="A short test project for unit tests.",
    costs="EUR 1",
    initiator="Test Org",
    address="1 Test St",
    area="Test area",
    latitude=48.0,
    longitude=16.0,
)


def test_build_faq_user_prompt_includes_id_and_name() -> None:
    p = build_faq_user_prompt(_SITE)
    assert "test-1" in p
    assert "Test Bridge" in p
    assert '"costs": "EUR 1"' in p or "EUR 1" in p


def test_parse_raw_json_faqs() -> None:
    raw = (
        '{"faqs": ['
        '{"question": "Q1?", "answer": "A1."},'
        '{"question": "Q2?", "answer": "A2."}'
        "]}"
    )
    out = parse_faq_model_content(raw)
    assert len(out) == 2
    assert out[0].question == "Q1?"


def test_parse_top_level_array() -> None:
    raw = '[{"question": "One?", "answer": "Yes."}]'
    out = parse_faq_model_content(raw)
    assert len(out) == 1
    assert out[0].question == "One?"


def test_parse_json_fenced_in_markdown() -> None:
    raw = """
Here you go:
```json
{"faqs": [{"question": "Hi?", "answer": "Bye."}]}
```
"""
    out = parse_faq_model_content(raw)
    assert len(out) == 1
    assert out[0].answer == "Bye."


def test_parse_invalid_json_raises() -> None:
    with pytest.raises(FaqResponseParseError, match="Invalid JSON"):
        parse_faq_model_content("not json")

    with pytest.raises(FaqResponseParseError, match="Invalid JSON"):
        parse_faq_model_content('{"faqs": tr}')


def test_parse_empty_faqs_raises() -> None:
    with pytest.raises(FaqResponseParseError, match="empty"):
        parse_faq_model_content('{"faqs": []}')
