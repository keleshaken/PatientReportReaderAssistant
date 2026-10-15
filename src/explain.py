"""Automatic six-part Malayalam summary grounded in retrieved lines."""
from __future__ import annotations

import json
import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator

LOGGER = logging.getLogger(__name__)

import anthropic

from src.chunking import SourceLine
from src.llm import ClaudeBillingError, LocalModelError, OllamaClient, TextClient, load_prompt, render_context, verify_claim
from src.retriever import Retriever
from src.safety import EXPLAIN_SCHEMA, cited_lines, validate_payload


QUERIES = Path(__file__).resolve().parent.parent / "prompts" / "extraction_queries.json"
ABSENT = "ഈ വിവരം റിപ്പോർട്ടിൽ ഇല്ല"
UNVERIFIED = "ഈ ഭാഗം വിശ്വസനീയമായി വിശദീകരിക്കാനായില്ല. താഴെയുള്ള ഇംഗ്ലീഷ് ഉറവിടം പരിശോധിക്കുക. / Could not verify this explanation. Please check the English source below."
SECTION_HEADINGS = {
    "diagnosis": ("diagnosis", "diagnoses", "clinical summary"),
    "medicines": ("medication", "medicine", "prescription"),
    "tests": ("investigation", "test", "lab"),
    "follow_up": ("follow",),
    "warnings": ("warning", "when to return"),
    "diet_rest": ("diet", "activity", "rest"),
}


@dataclass(frozen=True)
class ExplanationSection:
    key: str
    title: str
    explanation: str
    medicine_rows: list[dict[str, str]]
    sources: list[SourceLine]
    found: bool
    status: str = "ready"


def explain_document(retriever: Retriever, client: TextClient,
                     lines: dict[str, SourceLine], k: int = 4) -> list[ExplanationSection]:
    return list(iter_explanation(retriever, client, lines, k))


def iter_explanation(retriever: Retriever, client: TextClient,
                     lines: dict[str, SourceLine], k: int = 4) -> Iterator[ExplanationSection]:
    """Yield each section immediately after its citation and grounding checks."""
    definitions = json.loads(QUERIES.read_text(encoding="utf-8"))
    contexts = retriever.search_many([item["query"] for item in definitions], k)
    for item, context in zip(definitions, contexts):
        allowed_chunks = {x["id"] for x in context}
        chunk_lines = {x["id"]: set(x["line_ids"]) for x in context}
        allowed_line_ids = {line_id for x in context for line_id in x["line_ids"]}
        allowed_lines = {key: line for key, line in lines.items() if key in allowed_line_ids}
        section_sources = [line for line in allowed_lines.values()
                           if any(word in line.section.lower() for word in SECTION_HEADINGS[item["key"]])]
        result = None
        api_failures = 0
        for attempt in range(2):
            try:
                prompt = (f"Requested section: {item['key']} ({item['title']})\n"
                          f"Only include facts relevant to this section.\n\n{render_context(context)}")
                if attempt:
                    prompt += "\nPrevious output failed validation. Fix the JSON fields and exact citation IDs. Set found=false only if the requested facts are genuinely absent."
                if isinstance(client, OllamaClient):
                    schema = copy.deepcopy(EXPLAIN_SCHEMA)
                    schema["properties"]["section"] = {"type": "string", "enum": [item["key"]]}
                    for field, ids in (("source_chunk_ids", allowed_chunks), ("source_line_ids", allowed_line_ids)):
                        if ids:
                            schema["properties"][field] = {"type": "array", "items": {"type": "string", "enum": sorted(ids)}}
                    candidate = client.json(load_prompt("explain_system.txt"), prompt, schema=schema)
                else:
                    candidate = client.json(load_prompt("explain_system.txt"), prompt)
                validate_payload(candidate, "explain", allowed_chunks, allowed_lines, chunk_lines)
                if candidate["section"] != item["key"]:
                    raise ValueError("Wrong section")
                if candidate["found"]:
                    cited = cited_lines(candidate["source_line_ids"], allowed_lines)
                    claim = candidate["explanation_ml"] + " " + str(candidate.get("medicine_rows", []))
                    if not verify_claim(client, claim, "\n".join(x.text for x in cited)):
                        raise ValueError("Unsupported section — verify_claim returned false")
                result = candidate
                break
            except (ClaudeBillingError, LocalModelError):
                raise
            except anthropic.APIError:
                api_failures += 1
                continue
            except Exception as exc:
                LOGGER.warning("Section %s attempt %d failed validation: %s",
                               item["key"], attempt, type(exc).__name__)
                continue
        if api_failures == 2:
            raise RuntimeError("Claude API unavailable")
        if result and result["found"]:
            yield ExplanationSection(item["key"], item["title"], result["explanation_ml"],
                                               result.get("medicine_rows", []),
                                               cited_lines(result["source_line_ids"], allowed_lines), True)
        elif result is not None and not section_sources:
            yield ExplanationSection(item["key"], item["title"], ABSENT, [], [], False)
        else:
            yield ExplanationSection(item["key"], item["title"], UNVERIFIED, [],
                                     section_sources or list(allowed_lines.values()), False, "unverified")
