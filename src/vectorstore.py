"""One local persistent Chroma collection per ephemeral user session."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.chunking import Chunk


class VectorStore:
    def __init__(self, path: Path) -> None:
        import chromadb
        from chromadb.config import Settings

        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            "patient_document", metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            raise ValueError("There is no readable text to index")
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk/embedding count mismatch")
        self.collection.add(
            ids=[x.id for x in chunks],
            documents=[x.text for x in chunks],
            embeddings=embeddings,
            metadatas=[{"page": x.page, "section": x.section,
                        "line_ids": ",".join(x.line_ids)} for x in chunks],
        )

    def query(self, embedding: list[float], k: int = 4) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(query_embeddings=[embedding], n_results=min(k, self.collection.count()),
                                       include=["documents", "metadatas", "distances"])
        return [{"id": cid, "text": doc, "page": meta["page"],
                 "section": meta["section"], "line_ids": meta["line_ids"].split(","),
                 "distance": distance}
                for cid, doc, meta, distance in zip(result["ids"][0], result["documents"][0],
                                                      result["metadatas"][0], result["distances"][0])]

    def reset(self) -> None:
        self.client.delete_collection("patient_document")
        self.collection = self.client.get_or_create_collection(
            "patient_document", metadata={"hnsw:space": "cosine"})
