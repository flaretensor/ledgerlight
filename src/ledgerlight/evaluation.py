import argparse
import json
import re
import time

import numpy as np

from ledgerlight.common import namespace, params, path, read_json, read_jsonl, write_json
from ledgerlight.retrieval import cross_scores, fuse, load_model, order
from ledgerlight.split_guard import read_queries

METRICS = ["recall@10", "recall@20", "recall@100", "ndcg@10", "mrr"]


def evaluate(rankings, labels):
    from ledgerlight.split_guard import guard_queries

    guard_queries(rankings)

    from ranx import Qrels, Run
    from ranx import evaluate as ranx_evaluate

    qrels = Qrels({qid: dict.fromkeys(set(labels[qid]["positives"]), 1) for qid in rankings})
    run = Run(
        {
            qid: {pid: float(len(ids) - i) for i, pid in enumerate(ids)}
            for qid, ids in rankings.items()
        }
    )
    per_query = ranx_evaluate(qrels, run, METRICS, return_mean=False)
    values = {name: np.asarray(per_query[name]).tolist() for name in METRICS}
    return {name: float(np.mean(v)) for name, v in values.items()}, values


def tracking(name, metrics, parameters, artifact=None):
    import mlflow

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment = namespace()
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics({k.replace("@", "_"): float(v) for k, v in metrics.items()})
        if artifact is not None:
            mlflow.log_dict(artifact, "metrics.json")
        return run.info.run_id


def baseline():
    started = time.monotonic()
    corpus = read_jsonl("data/processed/corpus.jsonl")
    queries = read_queries()
    query_index = {q["id"]: i for i, q in enumerate(queries)}
    labels = read_json("data/processed/labels.json")
    dev = read_json("data/processed/splits.json")["dev"]
    lexical = np.load(path("indices/bm25/scores.npy"))
    dense = np.load(path("indices/base/scores.npy"))
    docids = [p["id"] for p in corpus]
    outputs, ranks = {}, {}
    for name, matrix in (("BM25", lexical), ("BGE", dense)):
        rankings = {qid: [docids[i] for i in order(matrix[query_index[qid]])] for qid in dev}
        score, _ = evaluate(rankings, labels)
        outputs[name] = score
        ranks[name] = rankings
        tracking(name + "-dev", score, {"split": "dev", "stage": "baselines"}, score)
    selected_k, best = None, -1
    fusion_grid = []
    for k in params()["retrieval"]["rrf_grid"]:
        rankings = {}
        for qid in dev:
            index = query_index[qid]
            values = fuse(lexical[index], dense[index], k)
            rankings[qid] = [docids[i] for i in order(values) if values[i] > 0]
        score, _ = evaluate(rankings, labels)
        fusion_grid.append({"k": k, **score})
        tracking(f"RRF-dev-k{k}", score, {"rrf_k": k, "split": "dev"})
        if score["ndcg@10"] > best:
            selected_k, best = k, score["ndcg@10"]
            outputs["RRF"], ranks["RRF"] = score, rankings
    vectors = np.load(path("indices/base/vectors.npy"))
    qvectors = np.load(path("indices/base/queries.npy"))
    windows = read_jsonl("indices/base/windows.jsonl")
    model = load_model("cross_encoder")
    ce_values, candidates_by_query = {}, {}
    for number, qid in enumerate(dev):
        index = query_index[qid]
        values = fuse(lexical[index], dense[index], selected_k)
        candidates = order(values)[:100]
        ce_values[qid] = cross_scores(
            model,
            queries[index]["question"],
            candidates,
            qvectors[index],
            vectors,
            windows,
            params()["retrieval"]["windows_per_passage"],
        ).tolist()
        candidates_by_query[qid] = candidates.tolist()
        if number % 10 == 0:
            print(
                json.dumps(
                    {"ce_dev": number + 1, "total": len(dev), "elapsed": time.monotonic() - started}
                ),
                flush=True,
            )
    depth_grid, selected_depth, best = [], None, -1
    for depth in params()["retrieval"]["rerank_grid"]:
        rankings = {}
        for qid in dev:
            candidates = np.asarray(candidates_by_query[qid])
            head = candidates[:depth][order(np.asarray(ce_values[qid])[:depth])]
            rankings[qid] = [docids[i] for i in np.concatenate([head, candidates[depth:]])]
        score, _ = evaluate(rankings, labels)
        tracking(f"CE-zero-dev-n{depth}", score, {"depth": depth, "split": "dev"})
        depth_grid.append({"depth": depth, **score})
        if score["ndcg@10"] > best:
            selected_depth, best = depth, score["ndcg@10"]
            outputs["CE-zero"], ranks["CE-zero"] = score, rankings
    priors = {
        "BM25": [33.90, 42.58, 62.50, 24.49, 28.69],
        "BGE": [39.19, 51.20, 72.95, 25.39, 27.73],
    }
    comparison = {
        name: {m: outputs[name][m] * 100 - prior for m, prior in zip(METRICS, values)}
        for name, values in priors.items()
    }
    if max(abs(x) for x in comparison["BM25"].values()) > 1.5:
        from rank_bm25 import BM25Okapi

        model_bm25 = BM25Okapi([re.findall(r"\w+", p["text"].lower()) for p in corpus])
        rankings = {
            qid: [
                docids[i]
                for i in order(
                    model_bm25.get_scores(
                        re.findall(r"\w+", queries[query_index[qid]]["question"].lower())
                    )
                )
            ]
            for qid in dev
        }
        diagnostic, _ = evaluate(rankings, labels)
        comparison["rank_bm25_diagnostic"] = diagnostic
        tracking("rank-bm25-dev-diagnostic", diagnostic, {"split": "dev"})
    result = {
        "dev": outputs,
        "rrf_k": selected_k,
        "ce_depth": selected_depth,
        "rrf_grid": fusion_grid,
        "ce_grid": depth_grid,
        "prior_delta_pp": comparison,
        "windows_per_passage": 1,
        "seconds": time.monotonic() - started,
    }
    write_json("data/baselines/metrics.json", result)
    write_json("data/baselines/rankings.json", ranks)
    write_json(
        "data/baselines/ce_scores.json", {"scores": ce_values, "candidates": candidates_by_query}
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    baseline()
