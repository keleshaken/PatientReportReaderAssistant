import pytest

from src.chunking import SourceLine
from src.safety import FALLBACK, is_emergency, validate_payload


LINES = {"p1-l2": SourceLine("p1-l2", 1, "Medications", "Paracetamol 500 mg twice daily for 3 days.")}


def test_emergency_detection_in_both_languages():
    assert is_emergency("I have chest pain now")
    assert is_emergency("എനിക്ക് നെഞ്ചുവേദന ഉണ്ട്")
    assert not is_emergency("What is my follow-up date?")


def test_fallback_text_is_exact():
    assert FALLBACK["en"] == "This isn't in your report. Please ask your doctor."
    assert FALLBACK["ml"] == "ഈ വിവരം നിങ്ങളുടെ റിപ്പോർട്ടിൽ ഇല്ല. ദയവായി ഡോക്ടറോട് ചോദിക്കുക."


def test_valid_qa_json_and_exact_citation():
    payload = {"answer": "Paracetamol 500 mg for 3 days.", "language": "en",
               "source_chunk_ids": ["p1-c1"], "source_line_ids": ["p1-l2"], "grounded": True}
    assert validate_payload(payload, "qa", {"p1-c1"}, LINES) == payload


def test_rejects_uncited_number_and_missing_line():
    payload = {"answer": "Paracetamol 650 mg for 3 days.", "language": "en",
               "source_chunk_ids": ["p1-c1"], "source_line_ids": ["p1-l2"], "grounded": True}
    with pytest.raises(ValueError, match="Uncited number"):
        validate_payload(payload, "qa", {"p1-c1"}, LINES)
    payload["answer"] = "Paracetamol 500 mg"
    payload["source_line_ids"] = []
    with pytest.raises(ValueError, match="exact citations"):
        validate_payload(payload, "qa", {"p1-c1"}, LINES)


def test_valid_explanation_json_schema():
    payload = {"section": "medicines", "explanation_ml": "Paracetamol 500 mg കഴിക്കുക.",
               "source_chunk_ids": ["p1-c1"], "source_line_ids": ["p1-l2"], "found": True,
               "medicine_rows": [{"drug": "Paracetamol", "dose": "500 mg", "timing": "ദിവസം 2 നേരം", "duration": "3 ദിവസം"}]}
    with pytest.raises(ValueError, match="Uncited medicine number"):
        validate_payload(payload, "explain", {"p1-c1"}, LINES)
    payload["medicine_rows"][0]["timing"] = "ദിവസം രണ്ട് നേരം"
    assert validate_payload(payload, "explain", {"p1-c1"}, LINES) == payload
