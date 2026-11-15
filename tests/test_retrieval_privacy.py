import logging

import pytest
from fastapi import HTTPException

from ledgerlight.api_models import SearchRequest


def test_retrieval_error_log_omits_request_and_exception_content(monkeypatch, caplog):
    from ledgerlight import retrieval_api

    question = "Private visitor question used only in this synthetic test"
    address = "198.18.99.1"

    class FailedEngine:
        def search(self, *args):
            raise RuntimeError(question + " " + address + " failed to allocate")

    monkeypatch.setattr(retrieval_api, "refresh", lambda: FailedEngine())
    with caplog.at_level(logging.ERROR), pytest.raises(HTTPException) as failure:
        retrieval_api.search(SearchRequest(question=question))
    assert failure.value.status_code == 503
    assert "allocation_failure" in caplog.text
    assert question not in caplog.text and address not in caplog.text
    assert question not in failure.value.detail and address not in failure.value.detail
