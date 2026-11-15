import asyncio
import threading

import httpx
import pytest

from ledgerlight import ui_transport


def test_streaming_http_error_body_is_available_to_ui(monkeypatch):
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"detail":{"code":"busy"}}'

    async def handler(request):
        return httpx.Response(429, stream=Body(), request=request)

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    with pytest.raises(httpx.HTTPStatusError) as error:
        list(ui_transport.events("http://gateway/ask/stream", {}, {}))
    assert error.value.response.json()["detail"]["code"] == "busy"


def test_queue_phase_is_visible_before_admission_headers(monkeypatch):
    release = threading.Event()

    async def handler(request):
        while not release.is_set():
            await asyncio.sleep(0.01)
        return httpx.Response(200, text="event: final\ndata: {}\n\n", request=request)

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    stream = ui_transport.events("http://gateway/ask/stream", {}, {})
    try:
        assert next(stream) == ("queued", {})
        release.set()
        assert list(stream) == [("final", {})]
    finally:
        stream.close()
