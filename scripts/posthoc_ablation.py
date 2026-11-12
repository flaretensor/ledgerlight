import gc
import hashlib
import json
import time

import mlflow
import numpy as np
from catboost import CatBoostRanker, Pool

from ledgerlight.common import digest, path, read_json, write_json
from ledgerlight.evaluation import evaluate
from ledgerlight.features import FEATURE_NAMES
from ledgerlight.retrieval import order
from ledgerlight.training import candidate_context, ltr_inputs

selection = read_json("data/training/ltr_selected.json")
scored = read_json("data/training/ltr_scores.json")
split = read_json("data/training/split.json")
official = read_json("data/processed/splits.json")
assert set(scored["train"]) == set(split["B"])
assert set(scored["dev"]) == set(official["dev"])
assert not set(split["B"]) & set(split["A"])
assert not (set(scored["train"]) | set(scored["dev"])) & set(official["test"])
inputs = [
    "data/training/ltr_selected.json",
    "data/training/ltr_scores.json",
    "data/training/split.json",
    "indices/bm25/scores.npy",
    f"indices/{selection['index']}/scores.npy",
    "src/ledgerlight/features.py",
]
input_hashes = {name: digest(path(name)) for name in inputs}
fingerprint = hashlib.sha256(json.dumps(input_hashes, sort_keys=True).encode()).hexdigest()
context = candidate_context(selection["index"])
x, y, groups, train_ids = ltr_inputs(scored["train"], context)
dev_x, _, dev_groups, dev_ids = ltr_inputs(scored["dev"], context)
assert len(train_ids) == 236 and len(dev_ids) == 118
assert x.shape == (23600, 14) and dev_x.shape == (11800, 14)
path(".cache/posthoc").mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    path(".cache/posthoc/ltr_matrices.npz"),
    train_x=x,
    train_y=y,
    train_groups=groups,
    dev_x=dev_x,
    dev_groups=dev_groups,
)
all_labels = read_json("data/processed/labels.json")
dev_labels = {qid: all_labels[qid] for qid in dev_ids}
del all_labels
variants = {
    "full_14": FEATURE_NAMES,
    "without_metadata": [
        name for name in FEATURE_NAMES if name not in {"ticker_match", "year_match", "form_match"}
    ],
    "without_ce": [name for name in FEATURE_NAMES if name not in {"ce", "ce_rank"}],
    "without_ranks": [
        name
        for name in FEATURE_NAMES
        if name not in {"bm25_rank", "dense_rank", "rrf_rank", "ce_rank"}
    ],
    "without_length_count": [
        name
        for name in FEATURE_NAMES
        if name not in {"passage_tokens", "window_count", "query_tokens"}
    ],
    "retrieval_only": ["bm25", "dense", "rrf", "bm25_rank", "dense_rank", "rrf_rank"],
}
frozen_path = path(selection["path"])
frozen_digest = digest(frozen_path)
frozen = CatBoostRanker()
frozen.load_model(str(frozen_path))
importance = frozen.get_feature_importance(type="PredictionValuesChange", thread_count=8)
assert frozen.feature_names_ == FEATURE_NAMES
feature_importance = sorted(
    [
        {"feature": name, "importance": float(value)}
        for name, value in zip(FEATURE_NAMES, importance)
    ],
    key=lambda row: -row["importance"],
)
report = {
    "status": "RUNNING",
    "scope": "post-hoc exploratory fixed-hyperparameter DEV-only feature ablation",
    "selection_changed": False,
    "test_scored": False,
    "hyperparameters": selection["config"],
    "hyperparameters_origin": "Frozen trial 26, tuned for all 14 features; no retuning for ablations",
    "training_partition": "B",
    "train_queries": len(train_ids),
    "dev_queries": len(dev_ids),
    "candidates_per_query": 100,
    "input_hashes": input_hashes,
    "input_fingerprint": fingerprint,
    "cached_neural_scores": True,
    "variants": variants,
    "frozen_feature_importance": feature_importance,
    "feature_importance_type": "PredictionValuesChange; descriptive model importance, not causal contribution",
    "frozen_model_sha256": frozen_digest,
    "runs": [],
}
result_path = "reports/posthoc/dev_ablation.json"
if path(result_path).exists():
    old = read_json(result_path)
    assert old["input_fingerprint"] == fingerprint
    report["runs"] = old["runs"]
mlflow.set_experiment("posthoc-analysis")
path("models/posthoc").mkdir(parents=True, exist_ok=True)
try:
    for variant, names in variants.items():
        columns = [FEATURE_NAMES.index(name) for name in names]
        train_pool = Pool(x[:, columns], y, group_id=groups, feature_names=names)
        dev_pool = Pool(dev_x[:, columns], group_id=dev_groups, feature_names=names)
        for seed in [42, 123, 2026]:
            if any(row["variant"] == variant and row["seed"] == seed for row in report["runs"]):
                continue
            with mlflow.start_run(run_name=f"dev-ablation-{variant}-seed-{seed}") as run:
                mlflow.log_params(
                    {
                        **selection["config"],
                        "variant": variant,
                        "features": ",".join(names),
                        "seed": seed,
                        "train_partition": "B",
                        "evaluation_split": "dev",
                        "retuned": False,
                    }
                )
                model = CatBoostRanker(
                    **selection["config"],
                    loss_function="YetiRank",
                    random_seed=seed,
                    thread_count=8,
                    verbose=False,
                    allow_writing_files=False,
                )
                started = time.perf_counter()
                model.fit(train_pool)
                seconds = time.perf_counter() - started
                predictions = model.predict(dev_pool)
                rankings = {}
                for group, qid in enumerate(dev_ids):
                    candidates = np.asarray(scored["dev"][qid]["candidates"])
                    ranked = candidates[order(predictions[dev_groups == group])]
                    rankings[qid] = [context["corpus"][p]["id"] for p in ranked]
                metrics, _ = evaluate(rankings, dev_labels)
                if variant == "full_14":
                    expected = next(row for row in selection["seeds"] if row["seed"] == seed)
                    assert all(abs(metrics[key] - expected[key]) < 1e-12 for key in metrics), (
                        "Full-feature reproduction differs from frozen DEV seed result"
                    )
                target = f"models/posthoc/{variant}-{seed}.cbm"
                model.save_model(str(path(target)))
                row = {
                    "variant": variant,
                    "features": names,
                    "seed": seed,
                    "metrics": metrics,
                    "training_seconds": seconds,
                    "run_id": run.info.run_id,
                    "model_path": target,
                }
                mlflow.log_metrics({key.replace("@", "_"): value for key, value in metrics.items()})
                mlflow.log_metric("training_seconds", seconds)
                mlflow.log_dict(row, "dev_result.json")
                if variant == "full_14" and seed == 42:
                    mlflow.log_dict(
                        {"type": report["feature_importance_type"], "rows": feature_importance},
                        "frozen_feature_importance.json",
                    )
            report["runs"].append(row)
            write_json(result_path, report)
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "seed": seed,
                        "ndcg10": metrics["ndcg@10"],
                        "mrr": metrics["mrr"],
                        "seconds": seconds,
                    }
                ),
                flush=True,
            )
            del model
            gc.collect()
    report["summary"] = []
    for variant in variants:
        rows = [row for row in report["runs"] if row["variant"] == variant]
        assert len(rows) == 3
        summary = {"variant": variant, "features": len(variants[variant])}
        for key in ["ndcg@10", "mrr"]:
            values = [row["metrics"][key] for row in rows]
            summary[key] = {"mean": float(np.mean(values)), "range": [min(values), max(values)]}
        report["summary"].append(summary)
    report["status"] = "FINISHED"
except BaseException:
    report["status"] = "INTERRUPTED_OR_FAILED"
    write_json(result_path, report)
    raise
assert digest(frozen_path) == frozen_digest
assert (
    digest("data/evaluation/metrics.json")
    == read_json("reports/rename/baseline.json")["metrics_sha256"]
)
write_json(result_path, report)
lines = [
    "# Post-hoc DEV feature ablation",
    "",
    report["scope"],
    "",
    report["hyperparameters_origin"],
    "",
    "Three fixed seeds per variant, TRAIN-B only, DEV118 only. No TEST scoring, registry update, alias change, or serving change.",
    "",
    "| Variant | Features | DEV nDCG@10 mean [range] (%) | MRR mean [range] (%) |",
    "|---|---:|---:|---:|",
]
for row in report["summary"]:
    values = [
        f"{100 * row[key]['mean']:.3f} [{100 * row[key]['range'][0]:.3f}, {100 * row[key]['range'][1]:.3f}]"
        for key in ["ndcg@10", "mrr"]
    ]
    lines.append(f"| {row['variant']} | {row['features']} | " + " | ".join(values) + " |")
lines += [
    "",
    "## Frozen model importance",
    "",
    "PredictionValuesChange, descriptive model importance; not causal attribution.",
    "",
    "| Feature | Importance |",
    "|---|---:|",
]
lines += [f"| {row['feature']} | {row['importance']:.4f} |" for row in feature_importance]
path("reports/posthoc/dev_ablation.md").write_text("\n".join(lines) + "\n")
print(json.dumps(report["summary"], indent=2))
