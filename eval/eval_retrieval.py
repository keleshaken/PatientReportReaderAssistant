"""Hit@k and reciprocal rank of the annotated source section."""
from __future__ import annotations

from collections import defaultdict

from common import E5Embedder, ROOT, prepare_document, questions, settings, update_results


def main() -> None:
    embedder = E5Embedder(settings.embedding_model)
    grouped = defaultdict(list)
    for row in questions():
        if row["answerable"]:
            grouped[row["document"]].append(row)
    ranks = []
    skipped = []
    for filename, rows in grouped.items():
        try:
            temp, retriever, _ = prepare_document(filename, embedder)
        except Exception as exc:
            skipped.append(f"{filename}: {type(exc).__name__}")
            continue
        try:
            for row in rows:
                hits = retriever.search(row["question"], k=4)
                expected = row["expected_source_section"].casefold()
                rank = next((n for n, hit in enumerate(hits, 1)
                             if hit["section"].casefold() == expected), None)
                ranks.append(rank)
        finally:
            temp.cleanup()
    if not ranks:
        raise RuntimeError("No documents could be evaluated")
    metrics = {k: sum(rank is not None and rank <= k for rank in ranks) / len(ranks) for k in (1, 2, 4)}
    mrr = sum(1 / rank for rank in ranks if rank is not None) / len(ranks)
    print(f"Evaluated {len(ranks)} questions; skipped {len(skipped)} documents")
    for k, value in metrics.items():
        print(f"Hit@{k}: {value:.3f}")
        update_results(f"Retrieval hit@{k}", f"{value:.3f} (n={len(ranks)})")
    print(f"MRR@4: {mrr:.3f}")
    update_results("Retrieval MRR@4", f"{mrr:.3f} (n={len(ranks)})")
    if skipped:
        update_results("Retrieval skipped documents", ", ".join(skipped))


if __name__ == "__main__":
    main()
