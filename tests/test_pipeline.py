import json
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def tokenizer(tmp_path):
    from transformers import BertTokenizerFast

    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nrevenue\nwas\n$\n31\n.\n9\nbillion\nmed\n##tech\n,\n2024\n"
    )
    return BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)


def test_windows_preserve_original_text_and_all_boundaries(tokenizer):
    from ledgerlight.retrieval import make_windows

    text = "MedTech revenue was $31.9 billion, 2024. " * 6
    windows = make_windows(tokenizer, [{"id": "p0", "text": text}], 12, 7)
    for item in windows:
        assert item["text"] == text[item["char_start"] : item["char_end"]]
        assert tokenizer.encode(item["text"], add_special_tokens=False) == item["input_ids"]
        assert len(item["input_ids"]) <= 12
        assert not tokenizer.convert_ids_to_tokens(item["input_ids"])[0].startswith("##")
    assert not text[windows[-1]["char_end"] :].strip()
    assert any("MedTech" in item["text"] for item in windows)


def test_ce_training_torch_and_ort_build_identical_overflowing_pairs(tokenizer):
    import torch

    from ledgerlight.onnx_backend import OnnxCrossEncoder
    from ledgerlight.pairs import pair_inputs, torch_pair_scores

    pairs = [("revenue " * 80, "2024 revenue " * 300), ("revenue", "2024")]
    training = pair_inputs(tokenizer, pairs)
    captured = {}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, **batch):
            captured["torch"] = batch["input_ids"].cpu().numpy()
            return SimpleNamespace(logits=torch.zeros(len(batch["input_ids"]), 1))

    class Runtime:
        def run(self, _, batch):
            captured["ort"] = batch["input_ids"]
            return [np.zeros(len(batch["input_ids"]))]

    torch_pair_scores(SimpleNamespace(model=Model(), tokenizer=tokenizer), pairs)
    encoder = OnnxCrossEncoder.__new__(OnnxCrossEncoder)
    encoder.tokenizer, encoder.runtime = tokenizer, Runtime()
    encoder.predict(pairs)
    assert training["input_ids"].shape[1] == 512
    assert np.array_equal(training["input_ids"].numpy(), captured["torch"])
    assert np.array_equal(captured["torch"], captured["ort"])
    assert int((training["token_type_ids"][0] == 0).sum()) == 82


def test_query_matches_preserve_feature_order_and_metadata(tokenizer):
    from ledgerlight.features import FEATURE_NAMES, features, query_tickers, query_years

    assert query_years("FY2024 Q42023 FY24 FY25") == {"2024", "2023", "2025"}
    assert {"JNJ"} <= query_tickers("Johnson & Johnson FY24 revenue")
    assert {"ABBV", "ABbV"} <= query_tickers("AbbVie FY2024 revenue")
    corpus = [{"sources": [{"ticker": "JNJ", "year": "2024", "form": "10-K"}]}]
    data = features(
        "Johnson & Johnson FY24 10-K", [0], *[np.ones(1)] * 4, corpus, np.ones(1), np.ones(1)
    )
    assert len(FEATURE_NAMES) == 14 and data.shape == (1, 14)
    assert data[0, 11:].tolist() == [1, 1, 1]


def test_ltr_drops_only_whitespace_duplicates_during_training(monkeypatch):
    from ledgerlight import training

    corpus = [
        {"id": f"p{i}", "text": text, "sources": []}
        for i, text in enumerate(["Revenue 10", "Revenue   10", "Unjudged passage"])
    ]
    context = {
        "corpus": corpus,
        "windows": [],
        "lookup": {"q": 0},
        "queries": [{"question": "Revenue"}],
        "lexical": np.ones((1, 3)),
        "dense": np.ones((1, 3)),
        "k": 10,
    }
    scored = {"q": {"candidates": [0, 1, 2], "scores": [1, 2, 3]}}
    monkeypatch.setattr(training, "read_json", lambda _: {"q": {"positives": ["p0"]}})
    audit = {"dropped_candidates": 0, "affected_queries": 0}
    matrix, y, groups, ids = training.ltr_inputs(scored, context, training=True, audit=audit)
    assert matrix.shape == (2, 14) and y.tolist() == [1, 0]
    assert audit == {"dropped_candidates": 1, "affected_queries": 1}
    monkeypatch.setattr(training, "read_json", lambda _: pytest.fail("Inference read labels"))
    assert training.ltr_inputs(scored, context)[0].shape == (3, 14)


def test_chat_template_budget_and_token_trimming():
    import httpx

    from ledgerlight.prompt_budget import pack_prompt

    seen = []

    def handler(request):
        data = json.loads(request.content)
        if request.url.path == "/apply-template":
            seen.append(data)
            return httpx.Response(
                200,
                json={
                    "prompt": "<begin> "
                    + " ".join(item["role"] + " " + item["content"] for item in data["messages"])
                    + " assistant <think>"
                },
            )
        assert data["add_special"] is False and data["parse_special"] is True
        return httpx.Response(200, json={"tokens": data["content"].split()})

    issued = {"[[SRC_a]]": {"sources": [], "windows": ["MedTech $31.9 billion " * 60]}}
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        messages, count, clipped = pack_prompt(
            client,
            "http://local",
            "SYSTEM",
            "Revenue?",
            issued,
            {"max_tokens": 32, "context": 330},
            True,
        )
    assert clipped and count <= 42
    assert messages == seen[-1]["messages"]
    assert all(item["chat_template_kwargs"] == {"enable_thinking": True} for item in seen)
    assert messages[1]["content"].split("\n")[-1] in issued["[[SRC_a]]"]["windows"][0]


def test_test_permission_and_pilot_output_isolation(monkeypatch, tmp_path):
    from ledgerlight import common
    from ledgerlight.evaluation import evaluate
    from ledgerlight.final_evaluation import evaluate_final
    from ledgerlight.test_access import allow_test

    monkeypatch.setattr(common, "ROOT", tmp_path)
    target = tmp_path / "data/processed/splits.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"train": ["train"], "dev": ["dev"], "test": ["held-out"]}))
    with pytest.raises(PermissionError):
        evaluate({"held-out": ["p0"]}, {"held-out": {"positives": ["p0"]}})
    with pytest.raises(PermissionError):
        evaluate_final()
    monkeypatch.setenv("LEDGERLIGHT_RUN_MODE", "pilot")
    with pytest.raises(PermissionError), allow_test(True):
        pass
    assert common.path("models/encoder/trial-0") == tmp_path / "data/pilot/models/encoder/trial-0"
    assert common.path("data/optuna.db") == tmp_path / "data/pilot/data/optuna.db"
    assert common.path("data/processed/corpus.jsonl") == tmp_path / "data/processed/corpus.jsonl"
