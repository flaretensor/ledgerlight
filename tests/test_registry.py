from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("prefix,log_metrics", [("register", True), ("bootstrap", False)])
def test_registration_primitives_preserve_names_requirements_and_manifests(
    monkeypatch, tmp_path, prefix, log_metrics
):
    from ledgerlight import common, registry

    monkeypatch.setattr(common, "ROOT", tmp_path)
    client = Mock()
    monkeypatch.setattr(registry, "client", lambda: client)
    starts, logs, metrics, promoted = [], [], [], []

    def start_run(**kwargs):
        starts.append(kwargs["run_name"])
        return nullcontext()

    versions = {}

    def log_model(**kwargs):
        logs.append(kwargs)
        name = kwargs["registered_model_name"]
        versions[name] = versions.get(name, 0) + 1
        return SimpleNamespace(registered_model_version=versions[name])

    monkeypatch.setattr(registry.mlflow, "start_run", start_run)
    monkeypatch.setattr(registry.mlflow.pyfunc, "log_model", log_model)
    monkeypatch.setattr(registry.mlflow, "log_metric", lambda *args: metrics.append(args))
    monkeypatch.setattr(registry, "promote", promoted.append)
    components = registry.register_components(
        {
            "encoder": ["base-e", "trained-e"],
            "cross-encoder": ["base-c", "trained-c"],
            "ranker": "ranker",
        },
        prefix,
    )
    config = {"index": "base", "mode": "bm25", "cross_encoder_path": None}
    trained = {"index": "trained", "mode": "ltr", "cross_encoder_path": "trained-c"}
    freeze = {
        "configurations": {"BM25": config, "CatBoost": trained},
        "dev": {"BM25": {"ndcg@10": 0.2}, "CatBoost": {"ndcg@10": 0.4}},
        "selected": "CatBoost",
    }
    assert registry.register_pipelines(freeze, components, prefix, log_metrics) == {
        "BM25": "1",
        "CatBoost": "2",
    }
    assert promoted == ["2"]
    assert components == {"encoder": ["1", "2"], "cross-encoder": ["1", "2"], "ranker": "1"}
    assert registry.read_json("data/registry/CatBoost/manifest.json")["components"] == {
        "encoder": "2",
        "cross_encoder": "2",
        "ranker": "1",
    }
    assert registry.read_json("data/registry/BM25/manifest.json")["components"] == {
        "encoder": "1",
        "cross_encoder": None,
        "ranker": None,
    }
    assert [x["registered_model_name"] for x in logs] == ["ledgerlight-encoder"] * 2 + [
        "ledgerlight-cross-encoder"
    ] * 2 + ["ledgerlight-ranker"] + ["ledgerlight-pipeline"] * 2
    assert starts == [
        prefix + "-" + suffix
        for suffix in [
            "encoder-base",
            "encoder-trained",
            "cross-encoder-base",
            "cross-encoder-trained",
            "ranker",
            "pipeline-BM25",
            "pipeline-CatBoost",
        ]
    ]
    assert metrics == ([("dev_ndcg10", 0.2), ("dev_ndcg10", 0.4)] if log_metrics else [])
    assert logs[0]["pip_requirements"] == ["sentence-transformers==5.2.3", "torch==2.8.0"]
    assert logs[4]["pip_requirements"] == ["catboost>=1.2.8,<1.3"]
    client.set_registered_model_alias.assert_any_call("ledgerlight-pipeline", "champion", "1")


@pytest.mark.parametrize("explicit_version", [None, 7])
@pytest.mark.parametrize("mode", ["bm25", "ce", "ltr"])
def test_manifest_loads_version_or_alias_and_resolves_only_used_components(
    monkeypatch, tmp_path, explicit_version, mode
):
    import json

    from ledgerlight import common, registry

    monkeypatch.setattr(common, "ROOT", tmp_path)
    client = Mock()
    client.get_model_version.return_value = SimpleNamespace(version="7")
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="7")
    monkeypatch.setattr(registry, "client", lambda: client)
    configuration = {
        "mode": mode,
        "components": {
            "encoder": "2",
            "cross_encoder": "2",
            "ranker": "1" if mode == "ltr" else None,
        },
        "encoder_path": None,
        "cross_encoder_path": None,
        "ltr_path": None,
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(configuration))
    roots = {}
    for name, version in [("encoder", "2"), ("cross-encoder", "2"), ("ranker", "1")]:
        root = tmp_path / (name + "-" + version)
        root.mkdir()
        (root / "MLmodel").write_text(
            json.dumps(
                {
                    "flavors": {
                        "python_function": {"artifacts": {"weights": {"path": "artifacts/weights"}}}
                    }
                }
            )
        )
        (root / "artifacts").mkdir()
        (root / "artifacts/weights").write_bytes(name.encode())
        roots[f"models:/ledgerlight-{name}/{version}"] = root
    calls = []

    def download(artifact_uri, dst_path):
        calls.append(artifact_uri)
        assert Path(dst_path).is_relative_to(tmp_path)
        return str(
            manifest_file
            if artifact_uri.endswith("/artifacts/manifest.json")
            else roots[artifact_uri]
        )

    from pathlib import Path

    monkeypatch.setattr(registry.mlflow.artifacts, "download_artifacts", download)
    result = registry.manifest(explicit_version)
    assert result["registry_version"] == "7"
    assert result["encoder_path"] == str(
        roots["models:/ledgerlight-encoder/2"] / "artifacts/weights"
    )
    assert ("models:/ledgerlight-cross-encoder/2" in calls) == (mode in {"ce", "ltr"})
    assert ("models:/ledgerlight-ranker/1" in calls) == (mode == "ltr")
    assert calls[0] == "models:/ledgerlight-pipeline/7/artifacts/manifest.json"
    assert json.loads(manifest_file.read_text()) == configuration
    if explicit_version is None:
        client.get_model_version_by_alias.assert_called_once_with(
            "ledgerlight-pipeline", "champion"
        )
        client.get_model_version.assert_not_called()
    else:
        client.get_model_version.assert_called_once_with("ledgerlight-pipeline", "7")
        client.get_model_version_by_alias.assert_not_called()
    assert not any(call[0].startswith("set_") for call in client.mock_calls)
