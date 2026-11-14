import json
import shutil


def test_training_fingerprint_tracks_sources_and_survives_relocation(monkeypatch, tmp_path):
    from ledgerlight import common
    from ledgerlight import training_identity as identity

    root = tmp_path / "first"
    for directory in ["src/ledgerlight", "config", "data/raw", "data/processed", "data/training"]:
        (root / directory).mkdir(parents=True)
    (root / "src/ledgerlight/features.py").write_text("feature = 1\n")
    (root / "config/company_aliases.json").write_text("{}")
    (root / "data/raw/FinRank.jsonl").write_text("dataset bytes")
    (root / "data/processed/splits.json").write_text(
        json.dumps({"train": ["q"], "dev": [], "test": []})
    )
    for stage, name in [("encoder", "encoder"), ("cross_encoder", "ce"), ("ranker", "ltr")]:
        (root / f"models/{stage}").mkdir(parents=True)
        (root / f"models/{stage}/model.safetensors").write_bytes(stage.encode())
        (root / f"data/training/{name}_selected.json").write_text(
            json.dumps({"path": f"models/{stage}"})
        )
    monkeypatch.setattr(common, "ROOT", root)
    monkeypatch.setattr(identity, "ROOT", root)
    monkeypatch.setattr(identity, "resolved_parameters", lambda: {"lr": 0.01})
    original = identity.training_fingerprint()
    relocated = tmp_path / "second"
    shutil.copytree(root, relocated)
    monkeypatch.setattr(common, "ROOT", relocated)
    monkeypatch.setattr(identity, "ROOT", relocated)
    assert identity.training_fingerprint() == original
    (relocated / "src/ledgerlight/features.py").write_text("feature = 2\n")
    changed = identity.training_fingerprint()
    assert changed != original
    (relocated / "models/encoder/model.safetensors").write_bytes(b"new weights")
    assert identity.training_fingerprint() != changed
    encoder_identity = identity.training_fingerprint("encoder")
    cross_identity = identity.training_fingerprint("cross_encoder")
    (relocated / "models/encoder/model.safetensors").write_bytes(b"other weights")
    assert identity.training_fingerprint("encoder") == encoder_identity
    assert identity.training_fingerprint("cross_encoder") != cross_identity
