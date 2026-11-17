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
            raise httpx.ConnectError("synthetic retrieval failure")

    monkeypatch.setattr(gateway, "ADMISSION", Admission())
    monkeypatch.setattr(gateway.httpx, "Client", FailedClient)
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
