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


def test_stage_resume_reuses_only_matching_outputs(monkeypatch, tmp_path):
    from ledgerlight import common, run_training

    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(run_training, "training_fingerprint", lambda _: "identity")
    calls = []

    def fit():
        calls.append(True)
        (tmp_path / "weights").write_bytes(b"selected")

    run_training.run_stage("fit", "encoder", fit, ["weights"])
    run_training.run_stage("fit", "encoder", fit, ["weights"])
    assert len(calls) == 1
    (tmp_path / "weights").write_bytes(b"damaged")
    run_training.run_stage("fit", "encoder", fit, ["weights"])
    assert len(calls) == 2
    monkeypatch.setattr(run_training, "training_fingerprint", lambda _: "changed")
    import pytest

    with pytest.raises(ValueError, match="inputs changed"):
        run_training.run_stage("fit", "encoder", fit, ["weights"])


def test_query_reader_does_not_decode_held_out_question(monkeypatch, tmp_path):
    from ledgerlight import common

    monkeypatch.setattr(common, "ROOT", tmp_path)
    target = tmp_path / "data/processed"
    target.mkdir(parents=True)
    (target / "splits.json").write_text(json.dumps({"train": ["q"], "dev": [], "test": ["held"]}))
    (target / "queries.jsonl").write_bytes(
        b'{"id":"q","question":"Allowed"}\n{"id":"held",INVALID UNREAD QUESTION}\n'
    )
    assert common.read_jsonl("data/processed/queries.jsonl") == [
        {"id": "q", "question": "Allowed"},
        {"id": "held", "question": ""},
    ]


def test_explicit_test_requires_frozen_selection_and_refuses_repeat(monkeypatch, tmp_path):
    import pytest

    from ledgerlight import common
    from ledgerlight.final_evaluation import evaluate_final

    monkeypatch.setattr(common, "ROOT", tmp_path)
    with pytest.raises(PermissionError):
        evaluate_final()
    with pytest.raises(FileNotFoundError, match="Missing DEV selection"):
        evaluate_final(allow_test=True)
    target = tmp_path / "data/evaluation"
    target.mkdir(parents=True)
    (target / "metrics.json").write_text("{}")
    with pytest.raises(FileExistsError):
        evaluate_final(allow_test=True)
