from src.chunking import PageText, split_pages


def test_section_page_and_line_metadata():
    chunks, lines = split_pages([PageText(2, "Diagnosis\nDengue fever.\nMedications\nParacetamol 500 mg daily.")])
    assert [chunk.section for chunk in chunks] == ["Diagnosis", "Medications"]
    assert chunks[0].page == 2
    assert lines["p2-l2"].text == "Dengue fever."
    assert chunks[1].line_ids == ("p2-l4",)


def test_long_section_splits_with_overlap():
    text = "Advice\n" + "\n".join(f"Instruction {i} with several words." for i in range(20))
    chunks, _ = split_pages([PageText(1, text)], max_tokens=25, overlap=8)
    assert len(chunks) > 1
    assert set(chunks[0].line_ids) & set(chunks[1].line_ids)
