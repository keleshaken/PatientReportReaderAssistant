"""Evaluate fallback, citations, numerical matches, and LLM-judged faithfulness."""
from __future__ import annotations

import json
import re
from collections import defaultdict

from common import E5Embedder, make_client, prepare_document, questions, settings, update_results
from src.qa import answer_question
from src.safety import FALLBACK


JUDGE_SYSTEM = """You judge whether a patient-facing answer is faithful to cited English source lines.
Return only JSON: {"faithful": boolean, "reason": string}.
Rubric: faithful=true only when EVERY factual claim is explicitly supported by the cited lines; all medicine names,
numbers, units, doses, frequencies, durations and dates match exactly; there is no outside medical advice or
interpretation. A natural Malayalam translation of the same claim is acceptable. If citations are empty or any
part is uncertain, faithful=false. Do not use outside medical knowledge."""


def main() -> None:
    embedder = E5Embedder(settings.embedding_model)
    client = make_client(settings)
    grouped = defaultdict(list)
    for row in questions():
        grouped[row["document"]].append(row)
    n = fallback_ok = grounded_ok = numeric_ok = faithful_ok = 0
    for filename, rows in grouped.items():
        temp, retriever, lines = prepare_document(filename, embedder, client)
        try:
            for row in rows:
                language = "ml" if re.search(r"[\u0d00-\u0d7f]", row["question"]) else "en"
                answer = answer_question(row["question"], language, retriever, client, lines)
                n += 1
                if not row["answerable"]:
                    fallback_ok += answer.text == FALLBACK[language]
                    continue
                evidence = "\n".join(source.text for source in answer.sources)
                grounded_ok += bool(answer.sources) and not answer.fallback
                expected_numbers = set(re.findall(r"\d+(?:[.,:/-]\d+)*", row["expected_answer"]))
                numeric_ok += bool(answer.sources) and expected_numbers.issubset(
                    set(re.findall(r"\d+(?:[.,:/-]\d+)*", answer.text)))
                verdict = client.json(JUDGE_SYSTEM,
                                      f"Question: {row['question']}\nAnswer: {answer.text}\nCited evidence: {evidence}",
                                      max_tokens=160)
                faithful_ok += verdict.get("faithful") is True
        finally:
            temp.cleanup()
    answerable = sum(row["answerable"] for row in questions())
    unanswerable = n - answerable
    values = {
        "Answer citations and groundedness": f"{grounded_ok}/{answerable}",
        "Exact numeric match": f"{numeric_ok}/{answerable}",
        "Unanswerable fallback": f"{fallback_ok}/{unanswerable}",
        "LLM judge faithfulness": f"{faithful_ok}/{answerable}",
    }
    for key, value in values.items():
        print(f"{key}: {value}")
        update_results(key, value)


if __name__ == "__main__":
    main()
