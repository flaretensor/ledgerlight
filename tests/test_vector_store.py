from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.mark.parametrize("count", [1, 2])
def test_upload_recreates_only_incomplete_collections(monkeypatch, count):
    from ledgerlight import vector_store

    client = Mock()
    client.collection_exists.return_value = True
    client.count.side_effect = (
        [SimpleNamespace(count=count), SimpleNamespace(count=2)]
        if count == 1
        else [SimpleNamespace(count=2), SimpleNamespace(count=2)]
    )
    monkeypatch.setattr(vector_store, "QdrantClient", lambda *args, **kwargs: client)
    corpus = [{"id": "p0", "sources": []}]
    windows = [{"id": i, "passage_index": 0} for i in range(2)]
    monkeypatch.setattr(
        vector_store, "read_jsonl", lambda name: corpus if "corpus" in name else windows
    )
    monkeypatch.setattr(vector_store, "read_json", lambda name: {"components": {"encoder": ["1"]}})
    monkeypatch.setattr(vector_store.np, "load", lambda name: np.ones((2, 384)))
    monkeypatch.setattr(vector_store, "digest", lambda name: "synthetic")
    monkeypatch.setattr(vector_store, "write_json", lambda *args: None)
    vector_store.upload()
    if count == 1:
        client.delete_collection.assert_called_once_with("ledgerlight-windows-encoder-1")
        client.create_collection.assert_called_once()
        points = client.upsert.call_args.kwargs["points"]
        assert [p.id for p in points] == [0, 1]
    else:
        client.delete_collection.assert_not_called()
        client.create_collection.assert_not_called()
        client.upsert.assert_not_called()
