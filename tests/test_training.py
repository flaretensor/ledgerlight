import json


def test_stage_resume_reuses_only_matching_outputs(monkeypatch, tmp_path):
    from ledgerlight import common, run_training

    monkeypatch.setattr(common, "ROOT", tmp_path)
    calls = []

    def fit():
        calls.append(True)
        (tmp_path / "weights").write_bytes(b"selected")

    run_training.run_stage("fit", fit, ["weights"])
    run_training.run_stage("fit", fit, ["weights"])
    assert len(calls) == 1
    (tmp_path / "weights").write_bytes(b"damaged")
    run_training.run_stage("fit", fit, ["weights"])
    assert len(calls) == 2


def test_query_reader_does_not_decode_held_out_question(monkeypatch, tmp_path):
    from ledgerlight import common

    monkeypatch.setattr(common, "ROOT", tmp_path)
    target = tmp_path / "data/processed"
    target.mkdir(parents=True)
    (target / "splits.json").write_text(json.dumps({"train": ["q"], "dev": [], "test": ["held"]}))
    (target / "queries.jsonl").write_bytes(
        b'{"id":"q","question":"Allowed"}\n{"id":"held",INVALID UNREAD QUESTION}\n'
    )
    from ledgerlight.split_guard import read_queries

    assert read_queries() == [
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
    with pytest.raises(FileNotFoundError, match="Missing DEV selection.*freeze.json"):
        evaluate_final(allow_test=True)
    target = tmp_path / "data/evaluation"
    target.mkdir(parents=True)
    (target / "metrics.json").write_text("{}")
    with pytest.raises(FileExistsError):
        evaluate_final(allow_test=True)
    (target / "metrics.json").unlink()
    reservation = target / "test.started.json"
    reservation.write_text("{}")
    with pytest.raises(FileExistsError):
        evaluate_final(allow_test=True)
    assert reservation.read_text() == "{}"
    assert not (target / "freeze.json").exists()
