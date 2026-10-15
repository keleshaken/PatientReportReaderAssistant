from types import SimpleNamespace
import json
import httpx
from src.llm import OllamaClient

from src.hybrid_qa import answer_from_report
from src.online_translation import TranslatedPage


PAGES = [SimpleNamespace(page=1, original="Medications\nParacetamol 500 mg for 3 days.")]


class Translator:
    def translate_page(self, page, text, source="en", target="ml"):
        if source == "ml":
            return TranslatedPage(page, text, "What medicines?", True)
        return TranslatedPage(page, text, "മരുന്ന്: " + text, True)


class Client:
    def __init__(self, answer="Paracetamol 500 mg for 3 days.", grounded=True):
        self.answer = answer
        self.grounded = grounded

    def json(self, system, prompt, max_tokens=400):
        if "auditor" in system:
            return {"supported": False}
        return {"answer": self.answer if self.grounded else "", "language": "en",
                "source_chunk_ids": ["p1-c1"] if self.grounded else [],
                "source_line_ids": ["p1-l2"] if self.grounded else [],
                "grounded": self.grounded}


def test_generated_english_answer_is_translated_and_cited():
    answer = answer_from_report("What medicines?", PAGES, Client(), Translator())
    assert answer.status == "answered" and "മരുന്ന്" in answer.malayalam
    assert answer.sources[0].text == "Paracetamol 500 mg for 3 days."


def test_malayalam_question_uses_english_generation():
    assert answer_from_report("മരുന്ന്?", PAGES, Client(), Translator()).status == "answered"


def test_invented_dose_is_rejected():
    result = answer_from_report("Dose?", PAGES, Client("Paracetamol 650 mg for 3 days."), Translator())
    assert result.status == "unverified" and not result.sources


def test_unsupported_paraphrase_is_rejected_by_audit():
    result = answer_from_report("Is it safe?", PAGES, Client("Paracetamol 500 mg is safe."), Translator())
    assert result.status == "source_excerpt"
    assert result.english == "Paracetamol 500 mg for 3 days."
    assert "safe" not in result.malayalam


def test_partial_quote_does_not_drop_negation():
    pages = [SimpleNamespace(page=1, original="Diagnosis\nNo evidence of cancer.")]
    result = answer_from_report("Diagnosis?", pages, Client("cancer."), Translator())
    assert result.status == "source_excerpt"
    assert result.english == "No evidence of cancer."


def test_missing_answer_and_emergency_do_not_become_generated_advice():
    assert answer_from_report("Surgery?", PAGES, Client(grounded=False), Translator()).status == "not_found"
    assert answer_from_report("I have chest pain", PAGES, None, None).status == "emergency"


def test_changed_number_in_translation_is_flagged():
    class ChangedTranslator:
        def translate_page(self, page, text):
            return TranslatedPage(page, text, "650 mg", False)
    assert answer_from_report("Dose?", PAGES, Client(), ChangedTranslator()).status == "translation_warning"


def test_ollama_chunk_and_line_citation_enums_are_separate():
    def respond(request):
        schema = json.loads(request.content)["format"]["properties"]
        assert schema["source_chunk_ids"]["items"]["enum"] == ["p1-c1"]
        assert schema["source_line_ids"]["items"]["enum"] == ["p1-l2"]
        payload = Client().json("", "")
        return httpx.Response(200, json={"message": {"content": json.dumps(payload)}})
    client = OllamaClient("qwen3:1.7b", transport=httpx.MockTransport(respond))
    assert answer_from_report("Dose?", PAGES, client, Translator()).status == "answered"
