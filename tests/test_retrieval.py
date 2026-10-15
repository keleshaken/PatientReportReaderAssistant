from src.retriever import Retriever


class FakeEmbedder:
    def queries(self, texts):
        assert texts == ["What medicine?"]
        return [[0.2, 0.4]]


class FakeStore:
    def query(self, embedding, k):
        assert embedding == [0.2, 0.4]
        assert k == 4
        return [{"id": "p1-c1", "section": "Medications", "page": 1}]


def test_retriever_passes_query_embedding_and_returns_metadata():
    result = Retriever(FakeEmbedder(), FakeStore()).search("  What medicine?  ")
    assert result[0]["section"] == "Medications"


def test_blank_question_does_not_search():
    assert Retriever(FakeEmbedder(), FakeStore()).search(" ") == []


def test_summary_queries_are_encoded_as_one_batch():
    class BatchEmbedder:
        def queries(self, texts):
            assert texts == ["diagnosis", "medicines"]
            return [[0.2, 0.4], [0.2, 0.4]]

    results = Retriever(BatchEmbedder(), FakeStore()).search_many(["diagnosis", "medicines"])
    assert len(results) == 2
    assert all(result[0]["section"] == "Medications" for result in results)
