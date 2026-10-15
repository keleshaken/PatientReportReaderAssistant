"""Use a bounded short report as complete evidence, without an embedding download."""
from src.chunking import Chunk

MAX_DIRECT_CHARACTERS = 6000


class DirectReportRetriever:
    def __init__(self, chunks: list[Chunk]):
        if sum(len(chunk.text) for chunk in chunks) > MAX_DIRECT_CHARACTERS:
            raise ValueError("Report is too long for direct context")
        self.context = [{"id": chunk.id, "text": chunk.text, "page": chunk.page,
                         "section": chunk.section, "line_ids": list(chunk.line_ids)}
                        for chunk in chunks]

    def search(self, question: str, k: int = 4) -> list[dict]:
        # All the evidence fits; retaining it also supports Malayalam questions.
        return list(self.context) if question.strip() else []

    def search_many(self, questions: list[str], k: int = 4) -> list[list[dict]]:
        return [self.search(question, k) for question in questions]
