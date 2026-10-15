"""Partial summaries must remain checked, and failed later sections must propagate."""
import json

import pytest

from src.chunking import SourceLine
from src.explain import UNVERIFIED, iter_explanation
from src.llm import LocalModelError


class ReportRetriever:
    def search_many(self, questions, k):
        return [[{"id": "p1-c1", "page": 1, "section": "Diagnosis",
                  "text": "[p1-l1] Diagnosis: Dengue fever.", "line_ids": ["p1-l1"]}]
                for _ in questions]


def configure_sections(tmp_path, monkeypatch):
    definitions = tmp_path / "sections.json"
    definitions.write_text(json.dumps([
        {"key": "diagnosis", "title": "Diagnosis", "query": "diagnosis"},
        {"key": "medicines", "title": "Medicines", "query": "medicines"},
    ]))
    monkeypatch.setattr("src.explain.QUERIES", definitions)
    return {"p1-l1": SourceLine("p1-l1", 1, "Diagnosis", "Diagnosis: Dengue fever.")}


def test_checked_first_section_is_available_before_later_model_failure(tmp_path, monkeypatch):
    lines = configure_sections(tmp_path, monkeypatch)

    class Client:
        calls = 0

        def json(self, system, user, max_tokens=1200):
            self.calls += 1
            if self.calls == 1:
                return {"section": "diagnosis", "explanation_ml": "Dengue fever",
                        "source_chunk_ids": ["p1-c1"], "source_line_ids": ["p1-l1"],
                        "found": True, "medicine_rows": []}
            if self.calls == 2:
                return {"supported": True}
            raise LocalModelError("disconnected")

    client = Client()
    results = iter_explanation(ReportRetriever(), client, lines)
    first = next(results)
    assert first.found and first.sources == [lines["p1-l1"]]
    assert client.calls == 2  # The first result includes the grounding audit.
    with pytest.raises(LocalModelError, match="disconnected"):
        next(results)


def test_bad_citations_never_appear_in_incremental_results(tmp_path, monkeypatch):
    lines = configure_sections(tmp_path, monkeypatch)

    class Client:
        def json(self, system, user, max_tokens=1200):
            return {"section": "diagnosis", "explanation_ml": "Dengue fever",
                    "source_chunk_ids": ["p1-c1"], "source_line_ids": ["made-up-line"],
                    "found": True, "medicine_rows": []}

    results = list(iter_explanation(ReportRetriever(), Client(), lines))
    assert len(results) == 2
    assert all(not section.found and section.explanation == UNVERIFIED and section.sources
               and section.status == "unverified"
               for section in results)
