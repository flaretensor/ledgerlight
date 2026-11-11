import hashlib
import json
import time

import numpy as np
from catboost import CatBoostRanker

from ledgerlight.common import digest, path, read_json, write_json
from ledgerlight.evaluation import evaluate, tracking
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
        "window_count": 11067,
        "windows_per_passage": 1,
        "corpus_sha256": digest(path("data/processed/corpus.jsonl")),
    }


def training_fingerprint():
    values = [
        path("data/training/" + name).read_bytes()
        for name in ["encoder_selected.json", "ce_selected.json", "ltr_selected.json"]
    ]
    return hashlib.sha256(b"".join(values)).hexdigest()


def evaluate_final():
    fingerprint = training_fingerprint()
    if path("data/evaluation/metrics.json").exists():
        previous = read_json("data/evaluation/metrics.json")
        if previous.get("training_fingerprint") != fingerprint:
            raise ValueError(
                "Training changed after the held-out evaluation was frozen; preserve that evaluation and use DEV for new candidates"
            )
        print("Frozen evaluation already exists; reusing recorded test results")
        return
    started = time.monotonic()
    baseline = read_json("data/baselines/metrics.json")
    encoder = read_json("data/training/encoder_selected.json")
    cross = read_json("data/training/ce_selected.json")
    ltr = read_json("data/training/ltr_selected.json")
    labels = read_json("data/processed/labels.json")
    splits = read_json("data/processed/splits.json")
    contexts = {"base": candidate_context("base"), "trained": candidate_context("trained")}
    ce_ft_dev = read_json(f"data/training/ce_trial_{cross['trial']}.json")
    ltr_scored = read_json("data/training/ltr_scores.json")
    ranker = CatBoostRanker()
    ranker.load_model(str(path(ltr["path"])))
    dev = dict(baseline["dev"])
    dev["BGE-ft"] = read_json("data/training/ft_encoder_zero_ce.json")["metrics"]
    dev["CE-ft"], _ = evaluate(ce_rankings(ce_ft_dev, contexts["trained"]), labels)
    dev["CatBoost"], _ = evaluate(
        rank_ltr(ranker, ltr_scored["dev"], contexts[ltr["index"]]), labels
    )
    ft = contexts["trained"]
    extra_rankings = {"BGE-ft-dense": {}, "RRF-ft": {}}
    for qid in splits["dev"]:
        index = ft["lookup"][qid]
        extra_rankings["BGE-ft-dense"][qid] = [
            ft["corpus"][p]["id"] for p in order(ft["dense"][index])
        ]
        fusion = fuse(ft["lexical"][index], ft["dense"][index], ft["k"])
        extra_rankings["RRF-ft"][qid] = [
            ft["corpus"][p]["id"] for p in order(fusion) if fusion[p] > 0
        ]
    for name, ranks in extra_rankings.items():
        dev[name], _ = evaluate(ranks, labels)
    selected = max(dev, key=lambda name: dev[name]["ndcg@10"])
    configs = {name: configuration(name, ltr, encoder, cross, baseline) for name in dev}
    freeze = {
        "selected": selected,
        "configurations": configs,
        "dev": dev,
        "training_fingerprint": fingerprint,
        "selection_metric": "full_pipeline_dev_ndcg@10",
        "test_tuning": False,
        "test_queries": len(splits["test"]),
        "dev_queries": len(splits["dev"]),
        "selection_note": "Configuration selected among many candidates on 118 DEV queries",
        "seed_policy": "seed 42 retained for selected configurations; others quantify variance",
    }
    write_json("data/evaluation/freeze.json", freeze)
    test_rankings = {}
    base = contexts["base"]
    for name, matrix in (("BM25", base["lexical"]), ("BGE", base["dense"])):
        test_rankings[name] = {
            qid: [base["corpus"][p]["id"] for p in order(matrix[base["lookup"][qid]])]
            for qid in splits["test"]
        }
    test_rankings["RRF"] = {}
    for qid in splits["test"]:
        idx = base["lookup"][qid]
        values = fuse(base["lexical"][idx], base["dense"][idx], base["k"])
        test_rankings["RRF"][qid] = [
            base["corpus"][p]["id"] for p in order(values) if values[p] > 0
        ]
    test_rankings["BGE-ft-dense"], test_rankings["RRF-ft"] = {}, {}
    ft = contexts["trained"]
    for qid in splits["test"]:
        index = ft["lookup"][qid]
        test_rankings["BGE-ft-dense"][qid] = [
            ft["corpus"][p]["id"] for p in order(ft["dense"][index])
        ]
        fusion = fuse(ft["lexical"][index], ft["dense"][index], ft["k"])
        test_rankings["RRF-ft"][qid] = [
            ft["corpus"][p]["id"] for p in order(fusion) if fusion[p] > 0
        ]
    zero_model = load_model("cross_encoder")
    zero_base = score_ce(zero_model, splits["test"], base)
    zero_trained = score_ce(zero_model, splits["test"], contexts["trained"])
    test_rankings["CE-zero"] = ce_rankings(zero_base, base)
    test_rankings["BGE-ft"] = ce_rankings(zero_trained, contexts["trained"])
    del zero_model
    import torch

    torch.cuda.empty_cache()
    ft_model = load_model("cross_encoder", path(cross["path"]))
    ft_trained = score_ce(ft_model, splits["test"], contexts["trained"])
    test_rankings["CE-ft"] = ce_rankings(ft_trained, contexts["trained"])
    del ft_model
    torch.cuda.empty_cache()
    scores_for_ltr = {"CE-zero": zero_base, "BGE-ft": zero_trained, "CE-ft": ft_trained}[
        ltr["upstream"]
    ]
    test_rankings["CatBoost"] = rank_ltr(ranker, scores_for_ltr, contexts[ltr["index"]])
    test, per_query = {}, {}
    for name, rankings in test_rankings.items():
        test[name], per_query[name] = evaluate(rankings, labels)
        tracking(
            name + "-frozen-test",
            test[name],
            {"split": "test", "method": name, "frozen": True},
            test[name],
        )
    for name, metrics in dev.items():
        tracking(name + "-ladder-dev", metrics, {"split": "dev", "method": name}, metrics)
    deltas = np.asarray(per_query[selected]["ndcg@10"]) - np.asarray(per_query["BM25"]["ndcg@10"])
    rng = np.random.default_rng(42)
    means = deltas[rng.integers(0, len(deltas), (10000, len(deltas)))].mean(axis=1)
    bootstrap = {
        "delta": float(deltas.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "query_bootstraps": 10000,
        "seed": 42,
        "selected": selected,
        "unit": "fraction",
        "caveat": "Published qrels and known-document random split; no answer truth inference",
    }
    result = {
        "dev": dev,
        "test": test,
        "selected": selected,
        "training_fingerprint": fingerprint,
        "bootstrap": bootstrap,
        "seeds": {
            "encoder": encoder["seeds"],
            "cross_encoder": cross["seeds"],
            "ltr": ltr["seeds"],
        },
        "seconds": time.monotonic() - started,
    }
    write_json("data/evaluation/metrics.json", result)
    write_json("data/evaluation/rankings.json", test_rankings)
    write_json("data/evaluation/per_query.json", per_query)
    write_json("models/production/config.json", configs[selected])
    import mlflow
    import pandas as pd

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("ledgerlight")
    with mlflow.start_run(run_name="frozen-ladder"):
        mlflow.log_params(
            {
                "selected": selected,
                "selection_queries": len(splits["dev"]),
                "test_queries": len(splits["test"]),
                "training_fingerprint": fingerprint,
                "selection_split": "dev",
            }
        )
        mlflow.log_table(
            pd.DataFrame(
                [
                    {"split": split, "method": name, **metrics}
                    for split, table in (("dev", dev), ("test", test))
                    for name, metrics in table.items()
                ]
            ),
            artifact_file="ladder.json",
        )
        mlflow.log_table(
            pd.DataFrame(
                [
                    {
                        "query_id": qid,
                        "method": name,
                        **{metric: scores[index] for metric, scores in values.items()},
                    }
                    for name, values in per_query.items()
                    for index, qid in enumerate(sorted(test_rankings[name]))
                ]
            ),
            artifact_file="test-per-query.json",
        )
        mlflow.log_dict(result, "metrics.json")
        mlflow.log_dict(freeze, "freeze.json")
        mlflow.log_metrics(
            {
                "selected_test_ndcg10": test[selected]["ndcg@10"],
                "delta_vs_bm25": bootstrap["delta"],
                "delta_ci95_low": bootstrap["ci95"][0],
                "delta_ci95_high": bootstrap["ci95"][1],
            }
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    evaluate_final()
