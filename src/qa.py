"""Question answering with fresh retrieval and strict source checks."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)

import anthropic

from src.chunking import SourceLine
from src.llm import ClaudeBillingError, LocalModelError, TextClient, load_prompt, render_context, verify_claim
from src.retriever import Retriever
from src.safety import EMERGENCY, FALLBACK, cited_lines, is_emergency, validate_payload


@dataclass(frozen=True)
class Answer:
    text: str
    language: str
    sources: list[SourceLine]
    confidence: str
    fallback: bool


def answer_question(question: str, language: str, retriever: Retriever, client: TextClient,
                    lines: dict[str, SourceLine], history: list[dict[str, str]] | None = None,
                    k: int = 4) -> Answer:
    """Retrieve anew even for follow-ups; use history only to resolve references."""
    language = language if language in {"ml", "en"} else "ml"
    if is_emergency(question):
        return Answer(EMERGENCY[language], language, [], "Emergency guidance", True)
    context = retriever.search(question, k)
    if not context:
        return Answer(FALLBACK[language], language, [], "Not found in report", True)
    allowed_chunks = {item["id"] for item in context}
    chunk_lines = {item["id"]: set(item["line_ids"]) for item in context}
    allowed_line_ids = {line_id for item in context for line_id in item["line_ids"]}
    allowed_lines = {key: lines[key] for key in allowed_line_ids if key in lines}
    short_history = (history or [])[-4:]
    api_failures = 0
    prompt = (f"Language: {language}\nQuestion: {question}\nRecent conversation for reference resolution only: "
              f"{short_history}\n\nRetrieved document context:\n{render_context(context)}")
    for attempt in range(2):
        try:
            result = client.json(load_prompt("qa_system.txt"), prompt +
                                 ("\nPrevious output was invalid or unsupported. Correct it or return grounded=false." if attempt else ""))
            validate_payload(result, "qa", allowed_chunks, allowed_lines, chunk_lines)
            if result["language"] != language:
                raise ValueError("Wrong language")
            if not result["grounded"]:
                break
            sources = cited_lines(result["source_line_ids"], allowed_lines)
            evidence = "\n".join(x.text for x in sources)
            if not verify_claim(client, result["answer"], evidence):
                raise ValueError("Claim not supported by cited lines")
            return Answer(result["answer"], language, sources, "Grounded in cited report lines", False)
        except (ClaudeBillingError, LocalModelError):
            raise
        except anthropic.APIError:
            api_failures += 1
            continue
        except Exception as exc:
            LOGGER.warning("QA attempt %d failed: %s — %s", attempt, type(exc).__name__, exc)
            continue
    if api_failures == 2:
        raise RuntimeError("Claude API unavailable")
    return Answer(FALLBACK[language], language, [], "Not enough support in report", True)
