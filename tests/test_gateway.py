import httpx
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
            raise httpx.ConnectError("synthetic retrieval failure")

    monkeypatch.setattr(gateway, "ADMISSION", Admission())
    monkeypatch.setattr(gateway.httpx, "Client", FailedClient)
    before = gateway.ABSTENTIONS._value.get()
    response = TestClient(gateway.app).post("/ask", json={"question": "Synthetic failure"})
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "error"}}
    assert gateway.ABSTENTIONS._value.get() == before
    assert not gateway.ADMISSION.snapshot()["active"]
