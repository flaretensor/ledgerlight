import logging

import httpx
import pytest
from fastapi.testclient import TestClient


def test_retrieval_failure_does_not_count_as_abstention(monkeypatch):
    from ledgerlight import gateway
    from ledgerlight.admission import Admission

    class FailedClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            assert kwargs["json"]["limit"] == 3
            raise httpx.ConnectError("synthetic retrieval failure")

    monkeypatch.setattr(gateway, "ADMISSION", Admission())
    monkeypatch.setattr(gateway.httpx, "Client", FailedClient)
    monkeypatch.setattr(gateway, "params", lambda: {"serving": {"top_k": 3}})
    before = gateway.ABSTENTIONS._value.get()
    response = TestClient(gateway.app).post("/ask", json={"question": "Synthetic failure"})
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "error"}}
    assert gateway.ABSTENTIONS._value.get() == before
    assert not gateway.ADMISSION.snapshot()["active"]


@pytest.mark.parametrize("error", [RuntimeError, httpx.ConnectError])
def test_failure_diagnostics_omit_exception_and_request_content(caplog, error):
    from ledgerlight.gateway import log_failure

    private = "Private question 198.18.99.2 https://secret.invalid"
    with caplog.at_level(logging.ERROR):
        try:
            raise error(private)
        except error as exc:
            log_failure(exc)
    record = caplog.records[-1]
    assert record.getMessage().startswith(error.__name__ + " test_gateway.py:")
    assert record.getMessage().endswith(
        "test_failure_diagnostics_omit_exception_and_request_content"
    )
    assert private not in caplog.text
    assert "198.18.99.2" not in caplog.text and "secret.invalid" not in caplog.text
    assert record.exc_info is None


@pytest.mark.parametrize("upstream", ["retrieval", "generation"])
def test_stream_upstream_failure_emits_error_and_releases_slot(monkeypatch, upstream):
    import asyncio

    from ledgerlight import gateway
    from ledgerlight.admission import Admission
    from ledgerlight.api_models import AskRequest

    real_client = httpx.AsyncClient
    item = {
        "passage_id": "p0",
        "windows": ["Synthetic evidence"],
        "sources": [],
        "text": "Synthetic evidence",
    }

    def response(request):
        if request.url.path == "/search":
            if upstream == "retrieval":
                raise httpx.ConnectError("private request 198.18.99.2", request=request)
            return httpx.Response(200, json={"results": [item], "versions": {}, "latencies": {}})
        return httpx.Response(503)

    monkeypatch.setattr(gateway, "ADMISSION", Admission())
    monkeypatch.setattr(gateway, "client_address", lambda request: "synthetic")
    monkeypatch.setattr(gateway, "pack_stream_prompt", lambda *args: ([], 0, False))
    monkeypatch.setattr(
        gateway.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(response), **kwargs),
    )

    class Connected:
        async def is_disconnected(self):
            return False

    async def run():
        response = await gateway.ask_stream(Connected(), AskRequest(question="Synthetic failure"))
        assert gateway.ADMISSION.snapshot()["active"]
        return [part async for part in response.body_iterator]

    events = asyncio.run(run())
    assert any('event: error\ndata: {"code": "error"}' in part for part in events)
    assert not any("event: final" in part for part in events)
    assert gateway.ADMISSION.snapshot() == {"active": False, "waiting": 0}


def test_active_stream_disconnect_releases_slot_and_closes_upstream(monkeypatch):
    import asyncio

    from ledgerlight import gateway
    from ledgerlight.admission import Admission
    from ledgerlight.api_models import AskRequest

    real_client = httpx.AsyncClient
    item = {
        "passage_id": "p0",
        "windows": ["Synthetic evidence"],
        "sources": [],
        "text": "Synthetic evidence",
    }
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"results": [item], "versions": {}, "latencies": {}}
        )
    )
    monkeypatch.setattr(gateway, "ADMISSION", Admission())
    monkeypatch.setattr(gateway, "client_address", lambda request: "synthetic")
    monkeypatch.setattr(
        gateway.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs)
    )
    closed = []

    async def completion(*args, **kwargs):
        try:
            yield "content", "Synthetic"
            await asyncio.sleep(60)
        finally:
            closed.append(True)

    monkeypatch.setattr(gateway, "stream_completion", completion)

    class Disconnected:
        async def is_disconnected(self):
            return True

    async def run():
        response = await gateway.ask_stream(
            Disconnected(), AskRequest(question="Synthetic disconnect")
        )
        assert gateway.ADMISSION.snapshot()["active"]
        with pytest.raises(asyncio.CancelledError):
            async for _ in response.body_iterator:
                pass

    asyncio.run(run())
    assert closed == [True]
    assert gateway.ADMISSION.snapshot() == {"active": False, "waiting": 0}
