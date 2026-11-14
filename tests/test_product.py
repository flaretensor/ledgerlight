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
        "Revenue " + marker + "\n[[SRC_unknown]]\nNot found in sources",
        "Revenue " + marker + "\n[[SRC_broken\nNot found in sources",
        "Revenue " + marker + "\nhttps://example.com\nNot found in sources",
        "Revenue " + marker + "\n<b>Not found in sources</b>",
        "Revenue " + marker + "\n[1]\nNot found in sources",
        "Revenue " + marker + "\n<think>unfinished",
    ]:
        assert render(invalid, issued) == (ABSTENTION, [])
    assert render("Revenue " + marker, issue([item])) == (ABSTENTION, [])


def test_cited_body_keeps_inline_abstention_and_drops_only_standalone_tail():
    from ledgerlight.citations import ABSTENTION, issue, render

    item = {"passage_id": "p0", "windows": ["Evidence"], "sources": [], "text": "Evidence"}
    other = {**item, "passage_id": "p1"}
    issued = issue([item, other])
    marker, orphan = issued
    for ending in ["Not found in sources", "  NOT FOUND IN SOURCES.!?  ", "not found in sources…"]:
        assert render(ending, issued) == (ABSTENTION, [])
        answer, citations = render("Revenue rose. " + marker + "\n\n" + ending, issued)
        assert answer == "Revenue rose. [1]"
        assert [c["passage_id"] for c in citations] == ["p0"]
    answer, citations = render(
        orphan
        + "\nRevenue rose. "
        + marker
        + "\n"
        + marker
        + " "
        + orphan
        + "\nNot found in sources.",
        issued,
    )
    assert answer == "Revenue rose. [2]"
    assert [c["passage_id"] for c in citations] == ["p1", "p0"]
    answer, citations = render("Revenue rose.\n" + marker, issued)
    assert answer == "Revenue rose."
    assert [c["passage_id"] for c in citations] == ["p0"]
    for text in [marker, marker + "\nNot found in sources"]:
        assert render(text, issued) == (ABSTENTION, [])
    for body in [
        "The phrase not found in sources is quoted here. ",
        "Revenue rose, not found in sources. ",
    ]:
        assert render(body + marker, issued)[0] == body + "[1]"
    assert render("<think>private</think>Revenue rose. " + marker, issued)[0] == "Revenue rose. [1]"


def test_gateway_length_retry_is_unchanged(monkeypatch):
    import httpx
    from fastapi.testclient import TestClient

    from ledgerlight import gateway

    item = {"passage_id": "p0", "windows": ["Evidence"], "sources": [], "text": "Evidence"}

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            return httpx.Response(
                200,
                json={"results": [item], "trace": [], "latencies": {}, "versions": {}},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(gateway.httpx, "Client", Client)
    monkeypatch.setattr(gateway, "record", lambda *args: None)
    for retry_finish in ["stop", "length"]:
        calls = []

        def complete(client, question, issued, thinking=True):
            calls.append(thinking)
            return "Revenue rose. " + next(iter(issued)) + "\nNot found in sources", {
                "finish_reason": "length" if thinking else retry_finish
            }

        monkeypatch.setattr(gateway, "complete", complete)
        response = TestClient(gateway.app).post("/ask", json={"question": "Synthetic retry check"})
        assert response.status_code == 200
        assert calls == [True, False]
        assert response.json()["abstained"] == (retry_finish == "length")


def test_abbvie_payload_filter_preserves_joint_provenance():
    from ledgerlight.engine import payload_filter, source_matches

    for ticker in ["ABBV", "ABbV"]:
        assert source_matches(
            {"ticker": ticker, "year": "2025"}, {"ticker": "ABBV", "year": "2025"}
        )
        assert not source_matches(
            {"ticker": ticker, "year": "2024"}, {"ticker": "ABBV", "year": "2025"}
        )
        nested = payload_filter({"ticker": ticker, "year": "2025"}).must[0].nested
        assert nested.key == "sources"
        assert nested.filter.must[0].match.any == ["ABBV", "ABbV"]
        assert nested.filter.must[1].match.value == "2025"
    assert payload_filter({"ticker": "GM"}).must[0].nested.filter.must[0].match.value == "GM"


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
