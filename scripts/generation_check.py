import json
import time
from collections import Counter

import httpx
import numpy as np

from ledgerlight.common import path, read_json, read_jsonl, write_json
from ledgerlight.evaluation import tracking

queries = {q["id"]: q["question"] for q in read_jsonl("data/processed/queries.jsonl")}
ids = read_json("data/processed/splits.json")["test"]
labels = read_json("data/processed/labels.json")
output = path("data/generation/responses.jsonl")
output.parent.mkdir(parents=True, exist_ok=True)
rows = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
completed = {r["qid"] for r in rows}
with httpx.Client(timeout=300) as client, output.open("a") as stream:
    for qid in ids:
        if qid in completed:
            continue
        started = time.perf_counter()
        try:
            response = client.post("http://127.0.0.1:8000/ask", json={"question": queries[qid]})
            data = response.json()
            status = response.status_code
        except httpx.HTTPError as exc:
            data, status = {"error_kind": type(exc).__name__}, 0
        seconds = time.perf_counter() - started
        cited = {c["passage_id"] for c in data.get("citations", [])}
        row = {
            "qid": qid,
            "status": status,
            "seconds": seconds,
            "positive_citation_hit": bool(cited & set(labels[qid]["positives"])),
            "response": data,
        }
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        rows.append(row)
        print(
            json.dumps(
                {
                    "completed": len(rows),
                    "total": len(ids),
                    "status": status,
                    "seconds": seconds,
                    "hit": row["positive_citation_hit"],
                }
            ),
            flush=True,
        )
valid = [r for r in rows if r["status"] == 200]
latencies = [r["seconds"] for r in rows]
result = {
    "queries": len(ids),
    "completed": len(rows),
    "failed": len(rows) - len(valid),
    "citation_present_rate": sum(bool(r["response"].get("citations")) for r in valid) / len(ids),
    "positive_citation_hit_rate": sum(r["positive_citation_hit"] for r in rows) / len(ids),
    "abstention_rate": sum(r["response"].get("abstained", False) for r in valid) / len(ids),
    "retry_rate": sum(
        r["response"].get("generation", {}).get("retried_without_thinking", False) for r in valid
    )
    / len(ids),
    "completion_limit_rate": sum(
        r["response"]
        .get("generation", {})
        .get("first_attempt", r["response"].get("generation", {}))
        .get("finish_reason")
        == "length"
        for r in valid
    )
    / len(ids),
    "pipeline_versions": dict(
        Counter(
            r["response"].get("ranking", {}).get("versions", {}).get("pipeline", "missing")
            for r in valid
        )
    ),
    "latency_seconds": {
        "mean": float(np.mean(latencies)),
        "p50": float(np.quantile(latencies, 0.5)),
        "p95": float(np.quantile(latencies, 0.95)),
    },
    "interpretation": "Citation-positive overlap only; no answer-correctness evaluation",
    "rate_denominator": len(ids),
}
write_json("data/generation/metrics.json", result)
print(json.dumps(result, indent=2))


tracking(
    "generation-citation-check",
    {
        "positive_citation_hit_rate": result["positive_citation_hit_rate"],
        "citation_present_rate": result["citation_present_rate"],
        "abstention_rate": result["abstention_rate"],
        "retry_rate": result["retry_rate"],
        "failed": result["failed"],
        "p95_seconds": result["latency_seconds"]["p95"],
    },
    {"split": "test", "interpretation": "citation overlap only"},
    result,
)
