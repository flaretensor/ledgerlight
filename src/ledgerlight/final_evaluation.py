import json
import time

import numpy as np
from catboost import CatBoostRanker

from ledgerlight.common import digest, path, read_json, read_jsonl, write_json
from ledgerlight.evaluation import evaluate
from ledgerlight.retrieval import fuse, load_model, order
from ledgerlight.training import candidate_context, ce_rankings, rank_ltr, score_ce


def configuration(method, ltr, encoder, cross, baselines):
    trained = (
        method in {"BGE-ft", "CE-ft", "BGE-ft-dense", "RRF-ft"}
        or method == "CatBoost"
        and ltr["index"] == "trained"
    )
    ce_path = cross["path"] if method == "CE-ft" else None
    if method == "CatBoost":
        ce_path = ltr["cross_encoder_path"]
    mode = {
        "BM25": "bm25",
        "BGE": "dense",
        "BGE-ft-dense": "dense",
        "RRF-ft": "rrf",
        "RRF": "rrf",
        "CatBoost": "ltr",
    }.get(method, "ce")
    return {
        "method": method,
        "mode": mode,
        "index": "trained" if trained else "base",
        "encoder_path": encoder["path"] if trained else None,
        "cross_encoder_path": ce_path,
        "ltr_path": ltr["path"] if method == "CatBoost" else None,
        "rrf_k": baselines["rrf_k"],
        "ce_depth": baselines["ce_depth"],
        "window_count": len(read_jsonl("indices/base/windows.jsonl")),
        "windows_per_passage": 1,
        "corpus_sha256": digest(path("data/processed/corpus.jsonl")),
    }


def paired_bootstrap(left, right):
    delta = np.asarray(left) - np.asarray(right)
    rng = np.random.default_rng(42)
    means = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
    return {
        "delta": float(delta.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "query_bootstraps": 10000,
        "seed": 42,
        "unit": "fraction",
    }


def evaluate_dev():
    baseline = read_json("data/baselines/metrics.json")
    encoder = read_json("data/training/encoder_selected.json")
    cross = read_json("data/training/ce_selected.json")
    ltr = read_json("data/training/ltr_selected.json")
    context = candidate_context("trained")
    rankings = read_json("data/baselines/rankings.json")
    rankings["BGE-ft"] = ce_rankings(
        read_json("data/training/ft_encoder_zero_ce.json")["scores"], context
    )
    rankings["CE-ft"] = ce_rankings(
        read_json(f"data/training/ce_trial_{cross['trial']}.json"), context
    )
    model = CatBoostRanker()
    model.load_model(str(path(ltr["path"])))
    rankings["CatBoost"] = rank_ltr(
        model, read_json("data/training/ltr_scores.json")["dev"], candidate_context(ltr["index"])
    )
    for method in ["BGE-ft-dense", "RRF-ft"]:
        rankings[method] = {}
        for qid in read_json("data/processed/splits.json")["dev"]:
            index = context["lookup"][qid]
            scores = (
                context["dense"][index]
                if method == "BGE-ft-dense"
                else fuse(context["lexical"][index], context["dense"][index], context["k"])
            )
            rankings[method][qid] = [
                context["corpus"][p]["id"]
                for p in order(scores)
                if method == "BGE-ft-dense" or scores[p] > 0
            ]
    labels = read_json("data/processed/labels.json")
    ladder, per_query = {}, {}
    for name, ranking in rankings.items():
        ladder[name], per_query[name] = evaluate(ranking, labels)
    comparisons = [
        ("CatBoost", "BM25"),
        ("BGE-ft-dense", "BGE"),
        ("RRF-ft", "RRF"),
        ("BGE-ft", "CE-zero"),
        ("CE-ft", "BGE-ft"),
        ("CatBoost", "CE-ft"),
    ]
    bootstrap = {
        f"{left}-minus-{right}": paired_bootstrap(
            per_query[left]["ndcg@10"], per_query[right]["ndcg@10"]
        )
        for left, right in comparisons
    }
    selected = max(ladder, key=lambda name: ladder[name]["ndcg@10"])
    configs = {name: configuration(name, ltr, encoder, cross, baseline) for name in ladder}
    freeze = {
        "selected": selected,
        "configurations": configs,
        "dev": ladder,
        "selection_metric": "dev_ndcg@10",
        "dev_queries": len(rankings[selected]),
        "test_tuning": False,
    }
    report = {
        "dev": ladder,
        "bootstrap": bootstrap,
        "selected": selected,
        "dev_queries": freeze["dev_queries"],
        "test_evaluated": False,
        "method_note": "BGE-ft is tuned encoder + RRF + pretrained CE",
        "caveat": "Known-document random split and incomplete qrels; no answer-accuracy claim",
    }
    write_json("data/evaluation/freeze.json", freeze)
    write_json("data/evaluation/dev_metrics.json", report)
    write_json("data/evaluation/dev_rankings.json", rankings)
    write_json("data/evaluation/dev_per_query.json", per_query)
    write_json("models/production/config.json", configs[selected])
    print(json.dumps(report, indent=2), flush=True)
    return report


def _evaluate_final():
    if (
        path("data/evaluation/metrics.json").exists()
        or path("data/evaluation/test.started.json").exists()
    ):
        raise FileExistsError(
            "Frozen TEST evaluation already exists or was started; refusing another run"
        )
    for name in [
        "data/evaluation/freeze.json",
        "data/evaluation/dev_metrics.json",
        "data/training/encoder_selected.json",
        "data/training/ce_selected.json",
        "data/training/ltr_selected.json",
    ]:
        if not path(name).is_file():
            raise FileNotFoundError("Missing DEV selection receipt: " + name)
    freeze = read_json("data/evaluation/freeze.json")
    reservation = path("data/evaluation/test.started.json")
    with reservation.open("x") as stream:
        json.dump({}, stream)
    started = time.monotonic()
    import re

    import bm25s
    import torch

    from ledgerlight.retrieval import encode_queries, parent_scores

    contexts = {name: candidate_context(name) for name in ["base", "trained"]}
    ids = read_json("data/processed/splits.json")["test"]
    lexical = bm25s.BM25.load(str(path("indices/bm25")), load_corpus=False)
    for name, context in contexts.items():
        encoder_path = (
            read_json("data/training/encoder_selected.json")["path"] if name == "trained" else None
        )
        encoder = load_model(model_path=path(encoder_path) if encoder_path else None)
        queries = [context["queries"][context["lookup"][qid]] for qid in ids]
        qvectors = encode_queries(encoder, queries)
        dense = parent_scores(
            qvectors,
            context["vectors"],
            np.array([w["passage_index"] for w in context["windows"]]),
            len(context["corpus"]),
        )
        for number, qid in enumerate(ids):
            index = context["lookup"][qid]
            context["qvectors"][index] = qvectors[number]
            context["dense"][index] = dense[number]
            context["lexical"][index] = lexical.get_scores(
                re.findall(r"\w+", queries[number]["question"].lower())
            )
        del encoder
        torch.cuda.empty_cache()
    rankings = {}
    base, tuned = contexts["base"], contexts["trained"]
    for method, context, field in [
        ("BM25", base, "lexical"),
        ("BGE", base, "dense"),
        ("BGE-ft-dense", tuned, "dense"),
        ("RRF", base, None),
        ("RRF-ft", tuned, None),
    ]:
        rankings[method] = {}
        for qid in ids:
            index = context["lookup"][qid]
            values = (
                context[field][index]
                if field
                else fuse(context["lexical"][index], context["dense"][index], context["k"])
            )
            rankings[method][qid] = [
                context["corpus"][p]["id"] for p in order(values) if field or values[p] > 0
            ]
    zero = load_model("cross_encoder")
    zero_base = score_ce(zero, ids, base)
    zero_tuned = score_ce(zero, ids, tuned)
    rankings["CE-zero"] = ce_rankings(zero_base, base)
    rankings["BGE-ft"] = ce_rankings(zero_tuned, tuned)
    del zero
    torch.cuda.empty_cache()
    selected_ce = read_json("data/training/ce_selected.json")
    model = load_model("cross_encoder", path(selected_ce["path"]))
    trained_scores = score_ce(model, ids, tuned)
    rankings["CE-ft"] = ce_rankings(trained_scores, tuned)
    del model
    torch.cuda.empty_cache()
    ltr = read_json("data/training/ltr_selected.json")
    ranker = CatBoostRanker()
    ranker.load_model(str(path(ltr["path"])))
    scores = {"CE-zero": zero_base, "BGE-ft": zero_tuned, "CE-ft": trained_scores}[ltr["upstream"]]
    rankings["CatBoost"] = rank_ltr(ranker, scores, contexts[ltr["index"]])
    metrics, per_query = {}, {}
    labels = read_json("data/processed/labels.json")
    for name, values in rankings.items():
        metrics[name], per_query[name] = evaluate(values, labels)
    result = {
        "dev": freeze["dev"],
        "test": metrics,
        "selected": freeze["selected"],
        "bootstrap": paired_bootstrap(
            per_query[freeze["selected"]]["ndcg@10"], per_query["BM25"]["ndcg@10"]
        ),
        "seconds": time.monotonic() - started,
    }
    write_json("data/evaluation/metrics.json", result)
    write_json("data/evaluation/rankings.json", rankings)
    write_json("data/evaluation/per_query.json", per_query)
    print(json.dumps(result, indent=2))
    return result


def evaluate_final(*, allow_test=False):
    from ledgerlight.split_guard import allow_test as test_permission

    with test_permission(allow_test):
        return _evaluate_final()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-test", action="store_true")
    evaluate_final(allow_test=parser.parse_args().allow_test)
