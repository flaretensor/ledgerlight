import importlib.util
from pathlib import Path


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
