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
