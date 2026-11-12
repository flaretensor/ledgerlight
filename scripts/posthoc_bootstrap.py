import json

import mlflow
import numpy as np

from ledgerlight.common import digest, path, read_json, write_json

baseline = read_json("reports/rename/baseline.json")
assert digest("data/evaluation/per_query.json") == baseline["per_query_sha256"]
assert digest("data/evaluation/metrics.json") == baseline["metrics_sha256"]
saved = read_json("data/evaluation/per_query.json")
rows = []
for comparator in ["RRF-ft", "CE-ft", "BGE-ft-dense"]:
    for metric in ["ndcg@10", "mrr"]:
        differences = np.asarray(saved["CatBoost"][metric]) - np.asarray(saved[comparator][metric])
        assert len(differences) == 119 and np.isfinite(differences).all()
        rng = np.random.default_rng(42)
        means = differences[rng.integers(0, len(differences), (10000, len(differences)))].mean(
            axis=1
        )
        rows.append(
            {
                "comparator": comparator,
                "metric": metric,
                "delta": float(differences.mean()),
                "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
            }
        )
report = {
    "status": "FINISHED",
    "scope": "post-hoc exploratory paired bootstrap of saved TEST query scores only",
    "queries": 119,
    "seed": 42,
    "resamples": 10000,
    "comparisons": rows,
    "multiplicity": "Three unadjusted model comparisons for primary nDCG@10; three additional secondary MRR intervals. No family-wise or simultaneous coverage claim.",
    "implementation": "Same difference-vector, default_rng integer resampling, mean and percentile quantiles as the frozen bootstrap",
    "input_sha256": digest("data/evaluation/per_query.json"),
    "new_predictions": False,
    "selection_changed": False,
    "unit": "fraction; multiply by 100 for percentage points",
}
mlflow.set_experiment("posthoc-analysis")
with mlflow.start_run(run_name="saved-test-paired-bootstrap") as run:
    mlflow.log_params(
        {"scope": "posthoc_saved_scores_only", "seed": 42, "resamples": 10000, "queries": 119}
    )
    for row in rows:
        mlflow.log_metric(
            row["comparator"] + "_" + row["metric"].replace("@", "_") + "_delta", row["delta"]
        )
    mlflow.log_dict(report, "paired_bootstrap.json")
    report["run_id"] = run.info.run_id
write_json("reports/posthoc/paired_bootstrap.json", report)
lines = [
    "# Post-hoc paired bootstrap",
    "",
    report["scope"],
    "",
    "Three unadjusted primary comparisons; secondary MRR intervals are also exploratory. No selection change or new TEST inference.",
    "",
    "| CatBoost minus | Metric | Difference (pp) | 95% CI (pp) |",
    "|---|---|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['comparator']} | {row['metric']} | {100 * row['delta']:.3f} | [{100 * row['ci95'][0]:.3f}, {100 * row['ci95'][1]:.3f}] |"
    )
path("reports/posthoc/paired_bootstrap.md").write_text("\n".join(lines) + "\n")
print(json.dumps(report, indent=2))
