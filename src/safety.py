"""Deterministic grounding checks, emergency routing and user messages."""
from __future__ import annotations

import re
from typing import Any

from jsonschema import ValidationError, validate

from src.chunking import SourceLine

FALLBACK = {
    "ml": "ഈ വിവരം നിങ്ങളുടെ റിപ്പോർട്ടിൽ ഇല്ല. ദയവായി ഡോക്ടറോട് ചോദിക്കുക.",
    "en": "This isn't in your report. Please ask your doctor.",
}
EMERGENCY = {
    "ml": "ഇത് അടിയന്തര പ്രശ്നമായിരിക്കാം. ഉടൻ ഡോക്ടറെയോ അടിയന്തര സേവനത്തെയോ ബന്ധപ്പെടുക.",
    "en": "This may be an emergency. Contact your doctor or emergency services immediately.",
}
DISCLAIMER = (
    "ഇത് റിപ്പോർട്ട് മനസ്സിലാക്കാനുള്ള സഹായം മാത്രമാണ്; ചികിത്സാ ഉപദേശം അല്ല. നിങ്ങളുടെ ഡോക്ടറോട് ചോദിക്കുക.\n"
    "This helps explain your report; it is not medical advice. Consult your doctor."
)
EMERGENCY_TERMS = re.compile(
    r"\b(chest pain|breathless(?:ness)?|trouble breathing|difficulty breathing|"
    r"heavy bleeding|severe bleeding|unconscious|seizure|stroke symptoms|"
    r"suicid(?:e|al))\b|നെഞ്ചുവേദന|ശ്വാസം\s*മുട്ട|ശ്വാസതടസ്സം|കടുത്ത\s*രക്തസ്രാവം|"
    r"ബോധം\s*പോയ|അപസ്മാരം", re.IGNORECASE)
NUMERIC = re.compile(r"\d+(?:[.,:/-]\d+)*")
ARRAY_OF_STRINGS = {"type": "array", "items": {"type": "string"}}
QA_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["answer", "language", "source_chunk_ids", "source_line_ids", "grounded"],
    "properties": {
        "answer": {"type": "string"}, "language": {"enum": ["ml", "en"]},
        "source_chunk_ids": ARRAY_OF_STRINGS, "source_line_ids": ARRAY_OF_STRINGS,
        "grounded": {"type": "boolean"},
    },
}
EXPLAIN_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["section", "explanation_ml", "source_chunk_ids", "source_line_ids", "found", "medicine_rows"],
    "properties": {
        "section": {"type": "string"}, "explanation_ml": {"type": "string"},
        "source_chunk_ids": ARRAY_OF_STRINGS, "source_line_ids": ARRAY_OF_STRINGS,
        "found": {"type": "boolean"},
        "medicine_rows": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["drug", "dose", "timing", "duration"],
            "properties": {key: {"type": "string"} for key in ("drug", "dose", "timing", "duration")},
        }},
    },
}


def is_emergency(question: str) -> bool:
    return bool(EMERGENCY_TERMS.search(question))


def validate_payload(payload: Any, kind: str, allowed_chunks: set[str],
                     lines: dict[str, SourceLine],
                     chunk_lines: dict[str, set[str]] | None = None) -> dict[str, Any]:
    """Validate model JSON, exact citation membership, and numerical evidence."""
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    if kind == "qa":
        schema = QA_SCHEMA
        required = {"answer", "language", "source_chunk_ids", "source_line_ids", "grounded"}
        text = payload.get("answer")
        supported = payload.get("grounded")
        if payload.get("language") not in {"ml", "en"}:
            raise ValueError("Invalid response language")
    elif kind == "explain":
        schema = EXPLAIN_SCHEMA
        required = {"section", "explanation_ml", "source_chunk_ids", "source_line_ids", "found"}
        text = payload.get("explanation_ml")
        supported = payload.get("found")
        if not isinstance(payload.get("section"), str):
            raise ValueError("Missing section")
    else:
        raise ValueError("Unknown response kind")
    try:
        validate(payload, schema)
    except ValidationError as exc:
        raise ValueError("Response does not match JSON schema") from exc
    if not required.issubset(payload) or not isinstance(text, str) or not isinstance(supported, bool):
        raise ValueError("Missing or invalid fields")
    chunk_ids = payload["source_chunk_ids"]
    line_ids = payload["source_line_ids"]
    if not isinstance(chunk_ids, list) or not isinstance(line_ids, list):
        raise ValueError("Citations must be lists")
    if any(not isinstance(x, str) or x not in allowed_chunks for x in chunk_ids):
        raise ValueError("Unknown chunk citation")
    if any(not isinstance(x, str) or x not in lines for x in line_ids):
        raise ValueError("Unknown line citation")
    if chunk_lines is not None and any(
        line_id not in set().union(*(chunk_lines.get(chunk_id, set()) for chunk_id in chunk_ids))
        for line_id in line_ids
    ):
        raise ValueError("Line citation is not in a cited chunk")
    if supported and (not text.strip() or not chunk_ids or not line_ids):
        raise ValueError("Supported text requires exact citations")
    if supported:
        evidence = " ".join(lines[x].text for x in line_ids)
        numeric_in_answer = set(NUMERIC.findall(text))
        # All numbers in a patient-facing answer must occur in the cited English lines.
        if numeric_in_answer - set(NUMERIC.findall(evidence)):
            raise ValueError("Uncited number or dose")
        if kind == "explain" and "medicine_rows" in payload:
            rows = payload["medicine_rows"]
            if not isinstance(rows, list) or any(
                not isinstance(row, dict) or not {"drug", "dose", "timing", "duration"}.issubset(row)
                for row in rows
            ):
                raise ValueError("Invalid medicine table")
            for row in rows:
                for value in row.values():
                    if not isinstance(value, str):
                        raise ValueError("Invalid medicine cell")
                    if set(NUMERIC.findall(value)) - set(NUMERIC.findall(evidence)):
                        raise ValueError("Uncited medicine number")
    return payload


def cited_lines(ids: list[str], lines: dict[str, SourceLine]) -> list[SourceLine]:
    return [lines[x] for x in dict.fromkeys(ids) if x in lines]
