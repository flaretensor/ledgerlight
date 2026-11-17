from urllib.parse import unquote, urlsplit

import pytest

from ledgerlight.common import database_url


@pytest.mark.parametrize("password", ["x@y", "x/y", "x:y", "x#y", "x%y", "@/:#%"])
def test_database_credentials_are_encoded(monkeypatch, password):
    monkeypatch.setenv("POSTGRES_USER", "user@/:#%")
    monkeypatch.setenv("POSTGRES_PASSWORD", password)
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "ledgerlight")
    parsed = urlsplit(database_url())
    assert parsed.hostname == "127.0.0.1" and parsed.port == 5433
    assert parsed.path == "/ledgerlight" and not parsed.fragment
    assert unquote(parsed.username) == "user@/:#%"
    assert unquote(parsed.password) == password
