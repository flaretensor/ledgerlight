import argparse
import json

import httpx
import numpy as np
from catboost import CatBoostRanker

from ledgerlight.common import path, read_json, write_json
from ledgerlight.evaluation import tracking
from ledgerlight.retrieval import fuse, order
from ledgerlight.split_guard import read_queries
from ledgerlight.training import candidate_context, ce_rankings, rank_ltr


def main():
    config = read_json("models/production/config.json")
    method = config["method"]
    context = candidate_context(config["index"])
    if method in ["BM25", "BGE", "RRF", "CE-zero"]:
        expected = read_json("data/baselines/rankings.json")[method]
    elif method in {"BGE-ft-dense", "RRF-ft"}:
        expected = {}
        for qid in read_json("data/processed/splits.json")["dev"]:
            i = context["lookup"][qid]
            values = (
                context["dense"][i]
                if method == "BGE-ft-dense"
                else fuse(context["lexical"][i], context["dense"][i], context["k"])
            )
            expected[qid] = [
                context["corpus"][p]["id"]
                for p in order(values)
                if method == "BGE-ft-dense" or values[p] > 0
            ]
    elif method == "BGE-ft":
        expected = ce_rankings(
            read_json("data/training/ft_encoder_zero_ce.json")["scores"], context
        )
    elif method == "CE-ft":
        trial = read_json("data/training/ce_selected.json")["trial"]
        expected = ce_rankings(read_json(f"data/training/ce_trial_{trial}.json"), context)
    else:
        model = CatBoostRanker()
        model.load_model(str(path(config["ltr_path"])))
        expected = rank_ltr(model, read_json("data/training/ltr_scores.json")["dev"], context)
    queries = {q["id"]: q["question"] for q in read_queries()}
    rows = []
    backend = None
    with httpx.Client(timeout=180) as client:
        for qid in read_json("data/processed/splits.json")["dev"][:20]:
            response = client.post(
                "http://127.0.0.1:8001/search", json={"question": queries[qid], "limit": 20}
            )
            response.raise_for_status()
            payload = response.json()
            backend = payload["versions"]["backend"]
            ids = [r["passage_id"] for r in payload["results"]]
            target = expected[qid][:20]
            rows.append(
                {
                    "qid": qid,
                    "exact_order": ids == target,
                    "overlap20": len(set(ids) & set(target)) / len(set(target)),
                    "served": ids,
                    "offline": target,
                }
            )
    result = {
        "method": method,
        "backend": backend,
        "experiment_split": "dev_only_post_freeze_serving_validation",
        "queries": len(rows),
        "exact_order_queries": sum(r["exact_order"] for r in rows),
        "mean_overlap20": float(np.mean([r["overlap20"] for r in rows])),
        "rows": rows,
    }
    if path("data/serving/ranking_parity.json").exists():
        previous = read_json("data/serving/ranking_parity.json")
        write_json(
            "data/serving/ranking_parity-" + previous.get("backend", "torch") + ".json", previous
        )
    write_json("data/serving/ranking_parity.json", result)
    write_json("data/serving/ranking_parity-" + backend + ".json", result)
    print(json.dumps(result, indent=2))

    tracking(
        "serving-ranking-parity-" + backend,
        {
            "mean_overlap20": result["mean_overlap20"],
            "exact_order_queries": result["exact_order_queries"],
        },
        {"split": "dev", "queries": 20, "backend": backend},
        result,
    )


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
