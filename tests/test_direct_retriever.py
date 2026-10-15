import pytest

from src.chunking import PageText, split_pages
from src.direct_retriever import DirectReportRetriever, MAX_DIRECT_CHARACTERS


def test_short_report_retains_all_sources_for_malayalam_questions():
    chunks, lines = split_pages([PageText(1, "Diagnosis\nViral fever.\nMedications\nParacetamol 500 mg.")])
    retriever = DirectReportRetriever(chunks)
    result = retriever.search("എന്താണ് മരുന്ന്?", k=1)
    assert len(result) == len(chunks)
    assert {line for item in result for line in item["line_ids"]} == set(lines)
    assert retriever.search(" ") == []
    assert retriever.search_many(["diagnosis", "medicines"]) == [result, result]


def test_large_report_cannot_bypass_the_context_limit():
    chunks, _ = split_pages([PageText(1, "word " * MAX_DIRECT_CHARACTERS)])
    with pytest.raises(ValueError, match="too long"):
        DirectReportRetriever(chunks)
