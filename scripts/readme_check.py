import json
from pathlib import Path

from ledgerlight.common import read_json

text = Path("README.md").read_text()
metrics = read_json("data/evaluation/metrics.json")
for split in ["dev", "test"]:
    for name, values in metrics[split].items():
        label = "BGE-ft+RRF+CE-zero" if name == "BGE-ft" else name
        expected = (
            "| "
            + label
            + " | "
            + " | ".join(
                f"{100 * values[key]:.2f}"
                for key in ["recall@10", "recall@20", "recall@100", "ndcg@10", "mrr"]
            )
            + " |"
        )
        assert expected in text, (split, name)
backends = read_json("data/serving/backends.json")
for model in ["encoder", "cross_encoder"]:
    for batch in [1, 8]:
        values = [
            next(
                row["median_ms"]
                for row in backends["benchmarks"]
                if row["model"] == model and row["batch"] == batch and row["backend"] == provider
            )
            for provider in ["torch-cpu", "ort-cpu", "torch-cuda", "ort-cuda"]
        ]
        assert " | ".join(f"{value:.2f}" for value in values) in text
facts = read_json("reports/docs_facts.json")
assert f"{100 * facts['positive_hit_among_cited']:.2f}%" in text
for row in read_json("reports/licenses.json")["python"]:
    assert f"| {row['component']} | {row['version']} | {row['declared_license']} |" in text
for key in ["citation_present_rate", "positive_citation_hit_rate", "abstention_rate", "retry_rate"]:
    assert f"{100 * read_json('data/generation/metrics.json')[key]:.2f}%" in text
assert text.count("```") % 2 == 0
assert Path("docs/ui-example.png").read_bytes() == Path("logs/ui-dev.png").read_bytes()
assert Path("docs/ui-example.png").stat().st_size < 500000
print(
    json.dumps(
        {
            "ladder_rows": 18,
            "backend_cells": 16,
            "license_rows": len(read_json("reports/licenses.json")["python"]),
            "proxies_and_screenshot": "verified",
        }
    )
)
