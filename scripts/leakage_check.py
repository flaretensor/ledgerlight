import argparse
import ast
import inspect
import json

import numpy as np
from catboost import CatBoostRanker

import ledgerlight.features as feature_module
from ledgerlight.common import digest, path, read_json, write_json
from ledgerlight.data import normalize
from ledgerlight.retrieval import fuse, order
from ledgerlight.split_guard import read_queries

SOURCES = {
    "bm25": "Query text and indexed passage text; bm25s Lucene score",
    "dense": "Query embedding and passage window embeddings; maximum cosine",
    "rrf": "BM25 and dense top-100 ranks; frozen reciprocal rank fusion",
    "ce": "Query text and the best dense passage window; CE logit",
    "bm25_rank": "Descending BM25 score with stable corpus-order ties",
    "dense_rank": "Descending maximum window cosine with stable ties",
    "rrf_rank": "Descending RRF score with stable ties",
    "ce_rank": "Descending CE logits within the RRF top-100 candidate set",
    "passage_tokens": "Passage WordPiece length from window end offsets",
    "window_count": "Number of indexed windows for the passage",
    "query_tokens": "Regex word count from query text",
    "ticker_match": "Explicit ticker tokens or verified static company aliases compared with ticker metadata",
    "year_match": "Explicit and fiscal-year expressions compared with source year metadata",
    "form_match": "Compact query text compared with passage form metadata",
}


def static_audit():
    assert feature_module.FEATURE_NAMES == list(SOURCES), "Unapproved LTR feature"
    function = feature_module.features
    assert list(inspect.signature(function).parameters) == [
        "question",
        "candidates",
        "lexical",
        "dense",
        "fusion",
        "ce",
        "corpus",
        "lengths",
        "counts",
    ], "Unexpected feature input"
    tree = ast.parse(inspect.getsource(feature_module))
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert [
        (node.module, [item.name for item in node.names])
        if isinstance(node, ast.ImportFrom)
        else (None, [item.name for item in node.names])
        for node in imports
    ] == [
        (None, ["re"]),
        ("functools", ["lru_cache"]),
        (None, ["numpy"]),
        ("ledgerlight.retrieval", ["order"]),
    ]
    forbidden = {
        "qrels",
        "gold",
        "labels",
        "positives",
        "answer",
        "popularity",
        "train_positive",
        "was_positive_in_train",
        "positive_count",
    }
    assert not ({node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} & forbidden)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                assert node.slice.value in {"sources", "ticker", "year", "form"}, (
                    "Forbidden metadata read"
                )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"open", "read_jsonl", "eval", "exec"}
            if node.func.id == "read_json":
                assert len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
                assert node.args[0].value == "config/company_aliases.json"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr in {
                "empty",
                "arange",
                "asarray",
                "column_stack",
                "astype",
                "casefold",
                "upper",
                "sub",
                "findall",
                "stack",
                "full",
                "append",
                "update",
                "items",
                "search",
                "escape",
            }, "Unreviewed feature data access"
    corpus = [
        {"sources": [{"ticker": "JNJ", "year": "2024", "form": "10-K"}]},
        {"sources": [{"ticker": "BBB", "year": "2023", "form": "10-Q"}]},
    ]
    args = (
        "JNJ 2024 10-K revenue",
        [0, 1],
        np.array([2.0, 1.0]),
        np.array([0.9, 0.2]),
        np.array([0.1, 0.05]),
        np.array([3.0, -2.0]),
        corpus,
        np.array([700, 400]),
        np.array([2, 1]),
    )
    expected = function(*args)
    for passage in corpus:
        passage.update({name: "forbidden oracle" for name in forbidden})
    assert np.array_equal(expected, function(*args)), "Oracle fields changed LTR features"
    assert expected.shape == (2, 14)
    assert expected[:, 11:].tolist() == [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    engine = ast.parse(path("src/ledgerlight/engine.py").read_text())
    for node in ast.walk(engine):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "labels.json" not in node.value and "qrels" not in node.value
    return {"features": SOURCES, "forbidden_feature_gate": True, "oracle_poison_invariance": True}


def full_audit():
    result = static_audit()
    split = read_json("data/training/split.json")
    official = read_json("data/processed/splits.json")
    a, b = set(split["A"]), set(split["B"])
    assert not a & b and a | b == set(official["train"])
    assert not (a | b) & (set(official["dev"]) | set(official["test"]))
    queries = {q["id"]: q["question"] for q in read_queries()}
    assert not {normalize(queries[q]) for q in a} & {normalize(queries[q]) for q in b}
    examples = read_json("data/training/examples.json")
    assert {row["qid"] for row in examples} <= a
    source = path("src/ledgerlight/training.py").read_text()
    parsed = ast.parse(source)
    ranking = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "rank_ltr"
    )
    prediction_pools = [
        node
        for node in ast.walk(ranking)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Pool"
    ]
    assert len(prediction_pools) == 1 and len(prediction_pools[0].args) == 1
    assert {item.arg for item in prediction_pools[0].keywords} == {"group_id"}
    for name in ["fit_encoder", "fit_cross_encoder"]:
        function = next(
            node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == name
        )
        strings = {
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert (
            "data/training/examples.json"
            if name == "fit_encoder"
            else "data/training/ce_examples.json"
        ) in strings and "data/processed/labels.json" not in strings
    scored = read_json("data/training/ltr_scores.json")
    assert set(scored["train"]) == b and set(scored["dev"]) == set(official["dev"])
    selected = read_json("data/training/ltr_selected.json")
    lexical = np.load(path("indices/bm25/scores.npy"))
    dense = np.load(path(f"indices/{selected['index']}/scores.npy"))
    lookup = {qid: i for i, qid in enumerate(queries)}
    k = read_json("data/baselines/metrics.json")["rrf_k"]
    for qid, row in scored["train"].items():
        i = lookup[qid]
        assert row["candidates"] == order(fuse(lexical[i], dense[i], k))[:100].tolist()
    models = list(path("models/ltr").glob("*.cbm"))
    assert len(read_json("data/training/ltr_trials.json")) == 50
    for filename in models:
        model = CatBoostRanker()
        model.load_model(str(filename))
        assert model.feature_names_ == list(SOURCES), (
            f"Unapproved serialized feature: {filename.name}"
        )
        assert model.get_all_params()["loss_function"] == "YetiRank"
    result.update(
        {
            "neural_partition": "A",
            "neural_queries": len(a),
            "neural_examples": len(examples),
            "neural_example_queries": len({row["qid"] for row in examples}),
            "ltr_partition": "B",
            "ltr_queries": len(b),
            "candidates_per_query": 100,
            "serialized_models_checked": len(models),
            "train_only_fit": True,
            "labels_usage": "Supervised training targets and offline evaluation only; not feature inputs",
            "offline_prediction_note": "The offline row helper also constructs an unused target vector. CatBoost prediction receives feature data and group IDs only; the live SearchEngine never loads labels.",
            "frozen_test_metrics_sha256": digest(path("data/evaluation/metrics.json")),
            "leak_found": False,
        }
    )
    write_json("data/audits/leakage.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true")
    print(json.dumps(static_audit() if parser.parse_args().static else full_audit(), indent=2))
