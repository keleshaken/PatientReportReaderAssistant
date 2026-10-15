"""Stage 1 command-line smoke path: English PDF/image text RAG and sourced answer."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from src.chunking import split_pages  # noqa: E402
from src.embeddings import E5Embedder  # noqa: E402
from src.ingest import extract_document  # noqa: E402
from src.llm import make_client  # noqa: E402
from src.qa import answer_question  # noqa: E402
from src.retriever import Retriever  # noqa: E402
from src.vectorstore import VectorStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask one English question about a hospital document")
    parser.add_argument("document", type=Path)
    parser.add_argument("question")
    args = parser.parse_args()
    client = make_client(settings)
    pages = extract_document(args.document.read_bytes(), args.document.name, client)
    chunks, lines = split_pages(pages)
    embedder = E5Embedder(settings.embedding_model)
    with tempfile.TemporaryDirectory(prefix="patient-stage1-") as folder:
        store = VectorStore(Path(folder))
        store.add(chunks, embedder.passages([chunk.text for chunk in chunks]))
        answer = answer_question(args.question, "en", Retriever(embedder, store), client, lines)
        print(answer.text)
        for source in answer.sources:
            print(f"Page {source.page} | {source.section} | {source.text}")


if __name__ == "__main__":
    main()
