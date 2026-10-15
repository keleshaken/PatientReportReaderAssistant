"""Qwen generates English document answers; Google translates the final answer."""
from copy import deepcopy
from dataclasses import dataclass
import re

from src.chunking import PageText, split_pages
from src.llm import OllamaClient, render_context, verify_claim
from src.safety import QA_SCHEMA, EMERGENCY, is_emergency, validate_payload, cited_lines


@dataclass
class HybridAnswer:
    english: str
    malayalam: str
    sources: list
    status: str


SYSTEM = """Answer a question using ONLY the supplied English report. Select the relevant source sentence(s) and copy them VERBATIM into answer. Do not paraphrase or add introductory words. For multiple sentences, preserve each source sentence unchanged and join with a newline. Copy medicine names, doses, frequency, dates and numbers exactly. Do not add medical advice, interpret tests, or infer missing facts. Use recent conversation only to resolve references, never as factual evidence. If the report does not answer the question, set grounded=false and answer="". Return JSON with answer, language="en", source_chunk_ids, source_line_ids, grounded. Cite only the exact source sentences you copied. When grounded=false, both citation arrays must be empty."""


def answer_from_report(question, pages, client, translator, history=None):
    if not question.strip():
        raise ValueError("Enter a question first.")
    if is_emergency(question):
        return HybridAnswer(EMERGENCY["en"], EMERGENCY["ml"], [], "emergency")
    english_question = question
    if re.search(r"[\u0d00-\u0d7f]", question):
        translated = translator.translate_page(0, question, source="ml", target="en")
        if not translated.numbers_match:
            raise ValueError("Question translation changed numbers. Please ask in English.")
        english_question = translated.translated
    if is_emergency(english_question):
        return HybridAnswer(EMERGENCY["en"], EMERGENCY["ml"], [], "emergency")
    chunks, lines = split_pages([PageText(page.page, page.original) for page in pages])
    # Small reports fit in full; longer reports use English lexical retrieval.
    if sum(len(chunk.text) for chunk in chunks) > 6000:
        words = set(re.findall(r"[a-z]+", english_question.lower())) - {"what", "which", "the", "is", "my", "a", "in", "report", "when", "how"}
        if words & {"medicine", "medicines", "drug", "dose"}:
            words.update({"medications", "prescriptions"})
        ranked = sorted(chunks, key=lambda chunk: sum(word in (chunk.section + " " + chunk.text).lower() for word in words), reverse=True)
        chunks = []
        for chunk in ranked:
            if sum(len(item.text) for item in chunks) + len(chunk.text) <= 6000:
                chunks.append(chunk)
            if len(chunks) == 6:
                break
    context = [{"id": chunk.id, "page": chunk.page, "section": chunk.section,
                "text": chunk.text, "line_ids": list(chunk.line_ids)} for chunk in chunks]
    ids = {chunk.id for chunk in chunks}
    chunk_lines = {chunk.id: set(chunk.line_ids) for chunk in chunks}
    allowed_lines = {key: line for key, line in lines.items() if any(key in values for values in chunk_lines.values())}
    schema = deepcopy(QA_SCHEMA)
    schema["properties"]["language"] = {"type": "string", "enum": ["en"]}
    for field, values in (("source_chunk_ids", ids), ("source_line_ids", set(allowed_lines))):
        if values:
            schema["properties"][field] = {"type": "array", "items": {"type": "string", "enum": sorted(values)}}
    prompt = f"Question: {english_question}\nRecent conversation: {(history or [])[-4:]}\nReport:\n{render_context(context)}"
    for attempt in range(2):
        candidate = client.json(SYSTEM, prompt + ("\nFix the exact citation IDs and include only directly supported facts." if attempt else ""),
                                max_tokens=400, **({"schema": schema} if isinstance(client, OllamaClient) else {}))
        try:
            validate_payload(candidate, "qa", ids, allowed_lines, chunk_lines)
            if candidate["language"] != "en":
                raise ValueError("Expected English generation")
            if not candidate["grounded"]:
                return HybridAnswer("I could not find a supported answer in this report.",
                                    "ഈ ചോദ്യത്തിന് ഉറപ്പുള്ള ഉത്തരം റിപ്പോർട്ടിൽ കണ്ടെത്താനായില്ല.", [], "not_found")
            sources = cited_lines(candidate["source_line_ids"], allowed_lines)
            evidence = "\n".join(line.text for line in sources)
            normalized = lambda value: " ".join(value.split()).casefold()
            exact_quote = normalized(candidate["answer"]) in {
                normalized(evidence), *(normalized(line.text) for line in sources)}
            if not exact_quote and not verify_claim(client, candidate["answer"], evidence):
                # Discard the unsupported generated wording. Display only the
                # actual cited report text, explicitly labeled as an excerpt.
                translated = translator.translate_page(0, evidence)
                if not translated.numbers_match:
                    return HybridAnswer(evidence, "മലയാളം വിവർത്തനത്തിലെ സംഖ്യകൾ പരിശോധിക്കാനായില്ല. ഇംഗ്ലീഷ് ഉറവിടം പരിശോധിക്കുക.", sources, "translation_warning")
                return HybridAnswer(evidence, translated.translated, sources, "source_excerpt")
            translated = translator.translate_page(0, candidate["answer"])
            if not translated.numbers_match:
                return HybridAnswer(candidate["answer"], "മലയാളം വിവർത്തനത്തിലെ സംഖ്യകൾ പരിശോധിക്കാനായില്ല. താഴെയുള്ള ഇംഗ്ലീഷ് ഉത്തരം പരിശോധിക്കുക.", sources, "translation_warning")
            return HybridAnswer(candidate["answer"], translated.translated, sources, "answered")
        except ValueError:
            continue
    return HybridAnswer("I could not verify an answer. Please check the source report.",
                        "ഉത്തരം പരിശോധിച്ച് ഉറപ്പാക്കാനായില്ല. ഇംഗ്ലീഷ് റിപ്പോർട്ട് പരിശോധിക്കുക.", [], "unverified")
