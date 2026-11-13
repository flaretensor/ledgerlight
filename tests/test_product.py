def test_citations_require_request_issued_markers():
    from ledgerlight.citations import ABSTENTION, issue, render

    item = {
        "passage_id": "p0",
        "windows": ["Revenue was ten."],
        "sources": [],
        "text": "Revenue was ten.",
    }
    issued = issue([item])
    marker = next(iter(issued))
    answer, sources = render("Revenue was ten. " + marker, issued)
    assert answer == "Revenue was ten. [1]"
    assert sources[0]["passage_id"] == "p0"
    for invalid in [
        "Revenue was ten. [1]",
        "Revenue [[SRC_unknown]]",
        "http://example.com " + marker,
        "ftp://example.com " + marker,
        "[invented](source) " + marker,
        "<b>Revenue</b> " + marker,
        "Revenue without citation",
    ]:
        assert render(invalid, issued) == (ABSTENTION, [])
    assert render("Revenue " + marker, issue([item])) == (ABSTENTION, [])


def test_filters_use_one_provenance_record():
    from ledgerlight.engine import source_matches

    sources = [{"ticker": "AAA", "year": "2024"}, {"ticker": "BBB", "year": "2025"}]
    assert not any(source_matches(s, {"ticker": "AAA", "year": "2025"}) for s in sources)
    assert any(source_matches(s, {"ticker": "AAA", "year": "2024"}) for s in sources)


def test_query_contract_rejects_oracle_fields():
    import pytest
    from pydantic import ValidationError

    from ledgerlight.api_models import SearchRequest

    for payload in [
        {"question": "q", "answer": "oracle"},
        {"question": "q", "filters": {"positive_passage_id": "p0"}},
    ]:
        with pytest.raises(ValidationError):
            SearchRequest.model_validate(payload)


def test_original_window_text_preserves_source_and_rejects_mismatch(tmp_path):
    import pytest
    from transformers import BertTokenizerFast

    from ledgerlight.excerpts import original_window_text

    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nintro\n.\nmed\n##tech\nsales\nwere\n"
        "$\n31\n9\nbillion\n;\nu\ns\nrevenue\nwas\n88\n,\n821\n"
    )
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
    text = "Intro. MedTech sales were $31.9 billion; U.S. revenue was 88,821.  \n"
    ids = tokenizer.encode(text, add_special_tokens=False)
    window = {"start": 2, "end": 11, "input_ids": ids[2:11]}
    assert original_window_text(text, window, tokenizer) == "MedTech sales were $31.9 billion"
    tail = {"start": 12, "end": len(ids), "input_ids": ids[12:]}
    assert original_window_text(text, tail, tokenizer) == "U.S. revenue was 88,821."
    piece = {"start": 3, "end": 4, "input_ids": ids[3:4]}
    assert original_window_text(text, piece, tokenizer) == "Tech"
    with pytest.raises(ValueError, match="token IDs do not match"):
        original_window_text(text.replace("31.9", "88.9"), window, tokenizer)
    with pytest.raises(ValueError, match="Invalid original passage token span"):
        original_window_text(text, {**window, "end": len(ids) + 1}, tokenizer)
