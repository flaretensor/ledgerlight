from ledgerlight.data import build_corpus


def test_corpus_keeps_first_occurrence_order_exact_text_and_source_order():
    first = {"text": " Revenue  2024. ", "page_number": 1}
    second = {"text": "Other passage", "page_number": 2}
    rows = [
        {
            "id": "q0",
            "question": "First?",
            "ticker": "GM",
            "year": "2024",
            "doc_type": "10-k",
            "passages": [first],
            "hard_negatives": [second],
        },
        {
            "id": "q1",
            "question": "Second?",
            "ticker": "F",
            "year": "2025",
            "doc_type": "10-q",
            "passages": [first],
            "hard_negatives": [],
        },
    ]
    corpus, queries, labels, mismatches, positives, negatives = build_corpus(rows)
    assert [(p["id"], p["text"]) for p in corpus] == [
        ("p00000", " Revenue  2024. "),
        ("p00001", "Other passage"),
    ]
    assert [source["ticker"] for source in corpus[0]["sources"]] == ["GM", "F"]
    assert queries == [{"id": "q0", "question": "First?"}, {"id": "q1", "question": "Second?"}]
    assert labels["q0"] == {"positives": ["p00000"], "hard_negatives": ["p00001"]}
    assert labels["q1"] == {"positives": ["p00000"], "hard_negatives": []}
    assert mismatches == 0 and positives == {"Revenue 2024."} and negatives == ["Other passage"]
