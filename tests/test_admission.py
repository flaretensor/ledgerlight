import asyncio

import pytest
from fastapi.testclient import TestClient

from ledgerlight.admission import Admission, Rejected


def test_single_slot_bounded_fifo_queue_cancel_and_recovery():
    gate = Admission()
    tickets = [gate.submit(str(i)) for i in range(4)]
    assert gate.snapshot() == {"active": True, "waiting": 3}
    with pytest.raises(Rejected, match="busy"):
        gate.submit("fifth")
    gate.finish(tickets[1])
    last = gate.submit("replacement")
    gate.finish(tickets[0])
    assert tickets[2].granted and not last.granted
    for ticket in [tickets[2], tickets[3], last]:
        gate.finish(ticket)
    assert gate.snapshot() == {"active": False, "waiting": 0}


def test_rolling_rate_limit_expires_without_persistent_identity():
    now = [0.0]
    gate = Admission(clock=lambda: now[0])
    for _ in range(6):
        gate.finish(gate.submit("same"))
    with pytest.raises(Rejected, match="rate"):
        gate.submit("same")
    now[0] = 60
    gate.expire()
    assert "same" not in gate.clients
    gate.finish(gate.submit("new"))
    assert "same" not in gate.clients


def test_queued_disconnect_releases_waiter():
    gate = Admission()
    active, waiting = gate.submit("one"), gate.submit("two")

    async def disconnected():
        return True

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await gate.wait_async(waiting, disconnected)

    asyncio.run(run())
    assert gate.snapshot()["waiting"] == 0
    gate.finish(active)


def test_queue_timeout_and_request_caps_do_not_reach_models(monkeypatch):
    from ledgerlight import gateway

    gate = Admission()
    active, waiting = gate.submit("one"), gate.submit("two")
    with pytest.raises(Rejected, match="busy"):
        gate.wait(waiting, 0)
    gate.finish(active)
    monkeypatch.setattr(gateway, "ADMISSION", Admission())
    client = TestClient(gateway.app)
    assert client.post("/ask", json={"question": " "}).status_code == 422
    response = client.post("/ask", json={"question": "x" * 501})
    assert response.status_code == 422 and response.json()["detail"]["code"] == "too_long"
    assert client.post("/ask/stream", content=b"x" * 8193).status_code == 413
    assert not gateway.ADMISSION.snapshot()["active"]
