import argparse
import json
import time

import numpy as np

from ledgerlight.common import digest, path, read_json, read_jsonl, write_json
from ledgerlight.evaluation import evaluate
from ledgerlight.final_evaluation import paired_bootstrap
from ledgerlight.test_access import allow_test

NOTE = "post-hoc descriptive, not used for selection"
EXTRA = ["recall@5", "ndcg@5", "hit_rate@5", "hit_rate@10"]
GROUPS = {
    "metadata": ["ticker_match", "year_match", "form_match"],
    "cross_encoder": ["ce", "ce_rank"],
    "first_stage": ["bm25", "dense", "rrf", "bm25_rank", "dense_rank", "rrf_rank"],
    "length": ["passage_tokens", "window_count", "query_tokens"],
}


def metrics(rankings, labels):
    from ranx import Qrels, Run
    from ranx import evaluate as ranx_evaluate

    rankings = dict(sorted(rankings.items()))
    with allow_test(True):
        means, per_query = evaluate(rankings, labels)
    qrels = Qrels({q: dict.fromkeys(set(labels[q]["positives"]), 1) for q in rankings})
    run = Run({q: {p: float(len(v) - i) for i, p in enumerate(v)} for q, v in rankings.items()})
    extra = ranx_evaluate(qrels, run, EXTRA, return_mean=False)
    per_query.update({k: np.asarray(v).tolist() for k, v in extra.items()})
    means.update({k: float(np.mean(v)) for k, v in extra.items()})
    return means, per_query


def overlap_groups(ids, labels, texts, train_ids, normalized=False):
    def key(text):
        return " ".join(text.split()) if normalized else text

    seen = {key(texts[p]) for q in train_ids for p in labels[q]["positives"]}
    groups = {name: [] for name in ["any_seen", "none_seen", "all_seen", "not_all_seen"]}
    for q in ids:
        flags = [key(texts[p]) in seen for p in set(labels[q]["positives"])]
        groups["any_seen" if any(flags) else "none_seen"].append(q)
        groups["all_seen" if all(flags) else "not_all_seen"].append(q)
    return groups


def decompose(rankings, labels):
    rows = []
    for q, ranking in rankings.items():
        positives = set(labels[q]["positives"])
        pool, top5, top10 = set(ranking), set(ranking[:5]), set(ranking[:10])
        bucket = (
            "no_positive_in_pool"
            if not positives & pool
            else "positive_in_top5"
            if positives & top5
            else "pool_positive_outside_top5"
        )
        rows.append(
            {
                "qid": q,
                "bucket": bucket,
                "positive_not_in_pool": len(positives - pool),
                "positive_in_pool_outside_top10": len((positives & pool) - top10),
                "positive_in_top10": len(positives & top10),
            }
        )
    return {
        "queries": {
            k: sum(r["bucket"] == k for r in rows)
            for k in ["no_positive_in_pool", "pool_positive_outside_top5", "positive_in_top5"]
        },
        "positives": {
            k: sum(r[k] for r in rows)
            for k in [
                "positive_not_in_pool",
                "positive_in_pool_outside_top10",
                "positive_in_top10",
            ]
        },
        "rows": rows,
    }


def strata_report(groups, per_query, query_ids, methods=None):
    lookup = {q: i for i, q in enumerate(query_ids)}
    methods = list(per_query) if methods is None else methods
    output = {}
    for name, ids in groups.items():
        indices = [lookup[q] for q in ids]
        if not indices:
            output[name] = {"count": 0, "metrics": {}, "bootstrap": None}
            continue
        output[name] = {
            "count": len(ids),
            "query_ids": ids,
            "metrics": {
                method: {
                    metric: float(np.mean(np.asarray(per_query[method][metric])[indices]))
                    for metric in ["ndcg@10", "recall@10"]
                }
                for method in methods
            },
            "bootstrap": paired_bootstrap(
                np.asarray(per_query["CatBoost"]["ndcg@10"])[indices],
                np.asarray(per_query["BM25"]["ndcg@10"])[indices],
            ),
        }
    return output


def latency_summary(values):
    if not values:
        return None
    return {
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def summarize_replay(rows, labels):
    from collections import Counter

    ok = [r for r in rows if r["http_status"] == 200]
    payloads = [r["response"] for r in ok]
    return {
        "offered": len(rows),
        "http_statuses": dict(Counter(str(r["http_status"]) for r in rows)),
        "cited_answers": sum(bool(p.get("citations")) for p in payloads),
        "abstentions": sum(p.get("abstained", False) for p in payloads),
        "citing_positive_parent_overlap_proxy": sum(
            bool(
                {c["passage_id"] for c in r["response"].get("citations", [])}
                & set(labels[r["qid"]]["positives"])
            )
            for r in ok
        ),
        "length_retries": sum(
            p.get("generation", {}).get("retried_without_thinking", False) for p in payloads
        ),
        "context_clipped_final": sum(
            p.get("generation", {}).get("context_clipped", False) for p in payloads
        ),
        "context_clipped_any_attempt": sum(
            bool(
                p.get("generation", {}).get("context_clipped")
                or p.get("generation", {}).get("first_attempt", {}).get("context_clipped")
            )
            for p in payloads
        ),
        "e2e_seconds_all_offered": latency_summary([r["elapsed"] for r in rows]),
        "search_seconds": latency_summary(
            [p["latencies"]["retrieval"] for p in payloads if "retrieval" in p.get("latencies", {})]
        ),
        "errors": [
            {k: r.get(k) for k in ["qid", "http_status", "error"]}
            for r in rows
            if r["http_status"] != 200
        ],
        "timeouts": sum("Timeout" in (r.get("error") or "") for r in rows),
        "caveat": "Citation overlap and answer/abstention counts are proxies, not factual correctness",
    }


def posthoc():
    labels = read_json("data/processed/labels.json")
    splits = read_json("data/processed/splits.json")
    corpus = read_jsonl("data/processed/corpus.jsonl")
    texts = {p["id"]: p["text"] for p in corpus}
    with allow_test(True):
        queries = {q["id"]: q["question"] for q in read_jsonl("data/processed/queries.jsonl")}
    ranks = {
        "dev": read_json("data/evaluation/dev_rankings.json"),
        "test": read_json("data/evaluation/rankings.json"),
    }
    result = {"note": NOTE, "metrics": {}, "failure": {}, "inputs": {}}
    per_query = {}
    for split, methods in ranks.items():
        result["metrics"][split], per_query[split] = {}, {}
        for name, rankings in methods.items():
            means, values = metrics(rankings, labels)
            result["metrics"][split][name] = means
            per_query[split][name] = values
    test = result["metrics"]["test"]
    gate = paired_bootstrap(
        per_query["test"]["CatBoost"]["ndcg@10"], per_query["test"]["BM25"]["ndcg@10"]
    )
    observed = [test["CatBoost"]["ndcg@10"], test["BM25"]["ndcg@10"], gate["delta"], *gate["ci95"]]
    expected = [
        0.38581570063671955,
        0.22790842926837723,
        0.15790727136834226,
        0.10265189289627742,
        0.21440251007380864,
    ]
    result["sanity"] = {
        "observed": observed,
        "difference": np.subtract(observed, expected).tolist(),
    }
    if not np.allclose(observed, expected, rtol=0, atol=1e-14):
        write_json("data/analysis/posthoc/result.json", result)
        raise RuntimeError("Frozen ranking sanity gate failed")
    result["bootstrap"] = {
        f"{left}-minus-{right}": paired_bootstrap(
            per_query["test"][left]["ndcg@10"], per_query["test"][right]["ndcg@10"]
        )
        for left, right in [
            ("BGE-ft-dense", "BGE"),
            ("RRF-ft", "RRF"),
            ("BGE-ft", "CE-zero"),
            ("CE-ft", "BGE-ft"),
            ("CatBoost", "CE-ft"),
        ]
    }
    result["candidate_pool"] = (
        "Stable top 100 of frozen RRF(k=10, BM25 depth=100, dense depth=100); "
        "CatBoost reranks these same 100 parents, so its saved ranking is the pool."
    )
    for split in ranks:
        assert all(len(v) == 100 for v in ranks[split]["CatBoost"].values())
        assert all(
            set(v) == set(ranks[split]["RRF-ft"][q][:100])
            for q, v in ranks[split]["CatBoost"].items()
        )
        result["failure"][split] = decompose(ranks[split]["CatBoost"], labels)
    ids = sorted(ranks["test"]["CatBoost"])
    result["overlap"] = {
        name: strata_report(
            overlap_groups(ids, labels, texts, splits["train"], normalized), per_query["test"], ids
        )
        for name, normalized in [("exact", False), ("whitespace_normalized", True)]
    }
    from ledgerlight.features import query_tickers

    lengths = dict.fromkeys(texts, 0)
    for w in read_jsonl("indices/base/windows.jsonl"):
        pid = corpus[w["passage_index"]]["id"]
        lengths[pid] = max(lengths[pid], w["end"])
    ticker = [q for q in ids if query_tickers(queries[q])]
    long = [q for q in ids if all(lengths[p] > 512 for p in labels[q]["positives"])]
    groups = {
        "ticker_alias": ticker,
        "no_ticker_alias": [q for q in ids if q not in ticker],
        "all_positive_parents_over512": long,
        "other_parent_lengths": [q for q in ids if q not in long],
    }
    result["slices"] = strata_report(groups, per_query["test"], ids, ["CatBoost", "BM25"])
    result["length_definition"] = "Original BGE WordPiece content tokens (without special tokens)"
    for name in [
        "data/processed/labels.json",
        "data/processed/splits.json",
        "data/processed/corpus.jsonl",
        "data/raw/FinRank.jsonl",
        "data/evaluation/dev_rankings.json",
        "data/evaluation/rankings.json",
    ]:
        result["inputs"][name] = digest(path(name))
    assert (
        result["inputs"]["data/raw/FinRank.jsonl"]
        == read_json("data/raw/provenance.json")["sha256"]
    )
    write_json("data/analysis/posthoc/result.json", result)
    write_json("data/analysis/posthoc/per_query.json", per_query)
    print(json.dumps({"stage": "A", "sanity": result["sanity"], "bootstrap": result["bootstrap"]}))


def ablation():
    from catboost import CatBoostRanker, Pool

    from ledgerlight.features import FEATURE_NAMES
    from ledgerlight.retrieval import order
    from ledgerlight.training import candidate_context, ltr_inputs

    started = time.monotonic()
    selection = read_json("data/training/ltr_selected.json")
    scored = read_json("data/training/ltr_scores.json")
    context = candidate_context(selection["index"])
    audit = {"dropped_candidates": 0, "affected_queries": 0}
    x, y, groups, train_ids = ltr_inputs(scored["train"], context, training=True, audit=audit)
    dev_x, _, dev_groups, dev_ids = ltr_inputs(scored["dev"], context)
    labels = read_json("data/processed/labels.json")
    assert not set(train_ids + dev_ids) & set(read_json("data/processed/splits.json")["test"])
    assert audit == {k: selection["label_audit"][k] for k in audit}
    out = path("data/analysis/ablation")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "matrices.npz", train=x, targets=y, groups=groups, dev=dev_x, dev_groups=dev_groups
    )
    result = {
        "note": NOTE,
        "split": "DEV only",
        "config": selection["config"],
        "seed": 42,
        "rows": len(x),
        "train_queries": len(train_ids),
        "label_audit": audit,
        "matrix_source": "Reconstructed deterministically from cached LTR CE scores and frozen indices",
        "models": {},
    }
    full_values = None
    for name, dropped in [("full", []), *GROUPS.items()]:
        columns = [i for i, f in enumerate(FEATURE_NAMES) if f not in dropped]
        names = [FEATURE_NAMES[i] for i in columns]
        model = CatBoostRanker(
            **selection["config"],
            loss_function="YetiRank",
            random_seed=42,
            thread_count=8,
            verbose=False,
            allow_writing_files=False,
        )
        fit_started = time.monotonic()
        model.fit(Pool(x[:, columns], y, group_id=groups, feature_names=names))
        predictions = model.predict(
            Pool(dev_x[:, columns], group_id=dev_groups, feature_names=names)
        )
        rankings = {
            q: [
                context["corpus"][p]["id"]
                for p in np.asarray(scored["dev"][q]["candidates"])[
                    order(predictions[dev_groups == i])
                ]
            ]
            for i, q in enumerate(dev_ids)
        }
        means, values = metrics(rankings, labels)
        model.save_model(str(out / (name + ".cbm")))
        row = {"dropped": dropped, "metrics": means, "seconds": time.monotonic() - fit_started}
        if name == "full":
            full_values = values["ndcg@10"]
            row["sanity_difference"] = means["ndcg@10"] - 0.4367276797012511
        row["bootstrap_vs_full"] = paired_bootstrap(values["ndcg@10"], full_values)
        result["models"][name] = row
        result["seconds"] = time.monotonic() - started
        write_json("data/analysis/ablation/result.json", result)
        write_json(f"data/analysis/ablation/{name}_rankings.json", rankings)
        if name == "full" and abs(row["sanity_difference"]) > 0.005:
            raise RuntimeError("Full-feature retrain sanity gate failed")
        print(json.dumps({"ablation": name, **row}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["posthoc", "ablation"])
    args = parser.parse_args()
    {"posthoc": posthoc, "ablation": ablation}[args.stage]()
