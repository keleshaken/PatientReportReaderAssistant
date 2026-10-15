"""Document retrieval independent of Streamlit."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.embeddings import E5Embedder
    from src.vectorstore import VectorStore


class Retriever:
    def __init__(self, embedder: E5Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def search(self, question: str, k: int = 4) -> list[dict[str, Any]]:
        if not question.strip():
            return []
        return self.store.query(self.embedder.queries([question.strip()])[0], k=k)

    def search_many(self, questions: list[str], k: int = 4) -> list[list[dict[str, Any]]]:
        """Encode the summary queries together instead of six separate model runs."""
        if not questions:
            return []
        embeddings = self.embedder.queries([question.strip() for question in questions])
        return [self.store.query(embedding, k=k) for embedding in embeddings]
