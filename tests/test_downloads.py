import hashlib
import io
import urllib.request

import pytest

from ledgerlight.downloads import download


def test_interrupted_download_is_retried_atomically(tmp_path, monkeypatch):
    target = tmp_path / "weights"
    part = target.with_name("weights.part")
    part.write_bytes(b"stale")
    calls = []

    class Interrupted(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError()
            return super().read(3)

    def open_url(url, timeout):
        calls.append(timeout)
        return Interrupted(b"partial") if len(calls) == 1 else io.BytesIO(b"complete")

    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    with pytest.raises(OSError):
        download("https://example.invalid/weights", target)
    assert not target.exists() and not part.exists()
    download("https://example.invalid/weights", target)
    assert target.read_bytes() == b"complete"
    assert calls == [60, 60]


def test_invalid_destination_is_replaced_only_after_checksum_verification(tmp_path, monkeypatch):
    target = tmp_path / "weights"
    target.write_bytes(b"invalid")
    expected = hashlib.sha256(b"complete").hexdigest()

    def verify(candidate):
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            raise ValueError()

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong"))
    with pytest.raises(ValueError):
        download("https://example.invalid/weights", target, verify)
    assert target.read_bytes() == b"invalid"
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"complete"))
    download("https://example.invalid/weights", target, verify)
    assert target.read_bytes() == b"complete"
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail())
    download("https://example.invalid/weights", target, verify)
