import importlib.util
import json
from pathlib import Path

import pytest


def test_downloaded_snapshots_can_be_registered_as_local_artifacts(monkeypatch, tmp_path):
    import huggingface_hub
    from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository

    from ledgerlight import common

    source = Path(__file__).resolve().parents[1] / "scripts/release_models.py"
    specification = importlib.util.spec_from_file_location("release_models", source)
    release = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(release)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    configs = {
        role: {"name": "example/" + role, "revision": str(index) * 40}
        for index, role in enumerate(["encoder", "cross_encoder"], 1)
    }
    monkeypatch.setattr(release, "params", lambda: configs)
    snapshots = {}
    for role, config in configs.items():
        cache = tmp_path / "hub" / role
        blob = cache / "blobs/weights"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(role.encode())
        snapshot = cache / "snapshots" / config["revision"]
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors").symlink_to(blob)
        (snapshot / "config.json").write_text("{}")
        snapshots[config["name"]] = snapshot
    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs["revision"]))
        return str(snapshots[repo])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    reused = []
    materialized = release.base_snapshots(None, reused)
    for role, target in materialized.items():
        destination = tmp_path / "restored" / role
        destination.mkdir(parents=True)
        restored = Path(
            LocalArtifactRepository(str(target.parent)).download_artifacts(
                target.name, dst_path=str(destination)
            )
        )
        assert (restored / "model.safetensors").read_bytes() == role.encode()
        assert not (target / "model.safetensors").is_symlink()
    assert calls == [(config["name"], config["revision"]) for config in configs.values()]
    assert reused == []


@pytest.fixture
def release(monkeypatch, tmp_path):
    from ledgerlight import common

    source = Path(__file__).resolve().parents[1] / "scripts/release_models.py"
    specification = importlib.util.spec_from_file_location("release_models_test", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def test_bundle_refuses_nonempty_destination(release, tmp_path):
    destination = tmp_path / "bundle"
    destination.mkdir()
    (destination / "keep").write_bytes(b"existing")
    with pytest.raises(ValueError) as failure:
        release.build_bundle(destination)
    assert str(failure.value) == "Bundle destination is not empty: " + str(destination)
    assert (destination / "keep").read_bytes() == b"existing"


def test_bundle_installs_tracked_card_and_checksums(release, tmp_path, monkeypatch):
    source_root = Path(__file__).resolve().parents[1]
    card = tmp_path / "config/model_card.md"
    card.parent.mkdir()
    card.write_bytes((source_root / "config/model_card.md").read_bytes())
    selected = {
        "encoder_path": "models/encoder/trial-2",
        "cross_encoder_path": "models/cross_encoder/trial-3",
        "ltr_path": "models/ltr/trial-37.cbm",
        "corpus_sha256": "synthetic",
    }
    for role in ["encoder", "cross_encoder"]:
        target = tmp_path / selected[role + "_path"] / "model.safetensors"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"weights")
        for version in ["1", "2"]:
            target = tmp_path / f"data/runtime/onnx/{role}/{version}/model.onnx"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"onnx")
    ranker = tmp_path / selected["ltr_path"]
    ranker.parent.mkdir(parents=True)
    ranker.write_bytes(b"ranker")
    split = tmp_path / "data/raw/splits.json"
    split.parent.mkdir(parents=True)
    split.write_text("{}")
    frozen = dict(
        selected="CatBoost",
        selection_metric="ndcg@10",
        dev_queries=118,
        test_tuning=False,
        configurations={},
        dev={},
    )
    values = {
        "models/production/config.json": selected,
        "data/evaluation/freeze.json": frozen,
        "data/evaluation/metrics.json": {
            "dev": {"CatBoost": {}},
            "test": {"CatBoost": {}},
            "bootstrap": {},
        },
    }
    monkeypatch.setattr(release, "read_json", values.__getitem__)
    monkeypatch.setattr(
        release,
        "params",
        lambda: {"data": {"revision": "a" * 40}, "encoder": {}, "cross_encoder": {}},
    )
    destination = tmp_path / "bundle"
    release.build_bundle(destination)
    assert (destination / "README.md").read_bytes() == card.read_bytes()
    records = dict(
        line.split("  ", 1)[::-1] for line in (destination / "SHA256SUMS").read_text().splitlines()
    )
    assert records["README.md"] == release.digest(card)
    assert all(release.digest(destination / name) == expected for name, expected in records.items())


@pytest.fixture
def payload(release, tmp_path):
    directory = tmp_path / "payload"
    directory.mkdir()
    selected = {
        "encoder_path": "models/encoder/trial-2",
        "cross_encoder_path": "models/cross_encoder/trial-3",
        "ltr_path": "models/ltr/trial-37.cbm",
    }
    (directory / "provenance.json").write_text(json.dumps({"selected_paths": selected}))
    (directory / "pipelines.json").write_text(
        json.dumps({"configurations": {"CatBoost": selected}})
    )
    (directory / "README.md").write_text("synthetic")
    names = release.bundle_requirements(directory)
    for name in names - {"provenance.json", "pipelines.json", "README.md"}:
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"synthetic")

    def sums():
        (directory / "SHA256SUMS").write_text(
            "".join(
                release.digest(p) + "  " + str(p.relative_to(directory)) + "\n"
                for p in sorted(directory.rglob("*"))
                if p.is_file() and p.name != "SHA256SUMS"
            )
        )

    sums()
    return directory, sums


def test_bundle_verifies_complete_payload(release, payload):
    directory, _ = payload
    assert len(release.verify_bundle(directory)) == 24


@pytest.mark.parametrize("listed", [True, False])
def test_bundle_rejects_missing_payload_with_sorted_paths(release, payload, listed):
    directory, sums = payload
    missing = [
        "models/encoder/trial-2/tokenizer.json",
        "data/runtime/onnx/cross_encoder/2/model.onnx",
    ]
    for name in missing:
        (directory / name).unlink()
    if not listed:
        sums()
    with pytest.raises(ValueError) as failure:
        release.verify_bundle(directory)
    assert str(failure.value) == "Bundle is missing required files: " + ", ".join(sorted(missing))


def test_bundle_rejects_corrupt_file(release, payload):
    directory, _ = payload
    (directory / "models/ltr/trial-37.cbm").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Bundle checksum or path violation"):
        release.verify_bundle(directory)


def test_bootstrap_validates_payload_before_restore_and_rebuild(release, payload, monkeypatch):
    from argparse import Namespace

    directory, _ = payload
    (directory / "models/ltr/trial-37.cbm").unlink()
    monkeypatch.setattr(release, "restore_payload", lambda *args: pytest.fail())
    monkeypatch.setattr(release, "rebuild_indexes", lambda *args: pytest.fail())
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    monkeypatch.delenv("HF_HUB_ETAG_TIMEOUT", raising=False)
    with pytest.raises(ValueError, match="Bundle is missing required files"):
        release.bootstrap(Namespace(repo=None, bundle=directory))


def test_bootstrap_restore_rebuild_registration_order(release, payload, monkeypatch):
    from argparse import Namespace

    directory, sums = payload
    provenance = json.loads((directory / "provenance.json").read_text())
    provenance["dataset"] = {"revision": "revision", "sha256": "sha256"}
    (directory / "provenance.json").write_text(json.dumps(provenance))
    sums()
    calls = []
    monkeypatch.setattr(release, "params", lambda: {"data": provenance["dataset"]})
    monkeypatch.setattr(
        release, "restore_payload", lambda *args: calls.append("restore") or {"encoder": "snapshot"}
    )
    monkeypatch.setattr(release, "rebuild_indexes", lambda *args: calls.append("rebuild"))
    monkeypatch.setattr(release, "prepare_services", lambda *args: calls.append("services"))
    monkeypatch.setattr(release, "register_fresh", lambda *args: calls.append("register"))
    monkeypatch.setattr(release, "read_json", lambda name: {"selected": "7"})
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "90")
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "45")
    release.bootstrap(Namespace(repo=None, bundle=directory, cache_root=None, prepare_only=True))
    assert calls == ["restore", "rebuild", "services", "register"]
    assert release.os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "90"
    assert release.os.environ["HF_HUB_ETAG_TIMEOUT"] == "45"
