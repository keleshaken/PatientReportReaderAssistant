"""E5 embeddings with the required asymmetric prefixes."""
from __future__ import annotations

class E5Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        values = self.model.encode(["passage: " + x for x in texts], normalize_embeddings=True)
        return values.tolist()

    def queries(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        values = self.model.encode(["query: " + x for x in texts], normalize_embeddings=True)
        return values.tolist()
