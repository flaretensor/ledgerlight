import json
import math
from datetime import UTC, datetime, timedelta

import httpx
import psycopg

from ledgerlight.citations import ABSTENTION
from ledgerlight.common import database_url, read_json, read_jsonl, write_json

queries = {q["id"]: q["question"] for q in read_jsonl("data/processed/queries.jsonl")}
question = queries[read_json("data/processed/splits.json")["dev"][0]]
rows = read_jsonl("data/generation/responses.jsonl")
assert len(rows) == 119 and len({row["qid"] for row in rows}) == 119
expected = read_json("data/registry/versions.json")["selected"]
assert all(
    row["response"]["ranking"]["versions"]["pipeline"] == expected
    for row in rows
    if row["status"] == 200
)
with httpx.Client(timeout=180) as client:
    rejected = client.post(
        "http://127.0.0.1:8000/ask", json={"question": question, "answer": "not permitted"}
    )
    assert rejected.status_code == 422
    empty = client.post(
        "http://127.0.0.1:8000/ask",
        json={"question": question, "filters": {"ticker": "LEDGERLIGHT_NONE"}},
    )
    empty.raise_for_status()
    value = empty.json()
    assert (
        value["answer"] == ABSTENTION
        and value["citations"] == []
        and value["ranking"]["results"] == []
    )
    write_json(
        "data/serving/contracts.json",
        {
            "extra_oracle_field_status": rejected.status_code,
            "empty_filter_response": value,
            "dev_only": True,
        },
    )
    identifiers = [row["response"]["request_id"] for row in rows if row["status"] == 200]
    with psycopg.connect(database_url()) as connection:
        first, last, count = connection.execute(
            "SELECT min(created_at),max(created_at),count(*) FROM ask_requests WHERE request_id=ANY(%s::uuid[])",
            (identifiers,),
        ).fetchone()
    assert count == len(identifiers)
    first -= timedelta(seconds=max(row["seconds"] for row in rows))
    duration = math.ceil((datetime.now(UTC) - first).total_seconds()) + 5
    peak = client.get(
        "http://127.0.0.1:9090/api/v1/query",
        params={"query": f"max_over_time(ledgerlight_gpu_memory_used_bytes[{duration}s])"},
    )
    peak.raise_for_status()
    results = peak.json()["data"]["result"]
    assert results
    samples = client.get(
        "http://127.0.0.1:9090/api/v1/query",
        params={"query": f"count_over_time(ledgerlight_gpu_memory_used_bytes[{duration}s])"},
    )
    samples.raise_for_status()
    report = {
        "generation_requests_logged": count,
        "pipeline_version": expected,
        "generation_window_utc": [first.isoformat(), last.isoformat()],
        "vram_peak_mib_sampled": max(float(row["value"][1]) for row in results) / 1024**2,
        "gpu_metric_samples": sum(
            float(row["value"][1]) for row in samples.json()["data"]["result"]
        ),
        "scrape_interval_seconds": 5,
        "scope": "Prometheus observed global GPU memory over the generation window plus a conservative request-duration margin; desktop included. Not an instantaneous peak or GPU-only allocation claim.",
    }
    write_json("data/generation/resources.json", report)
    print(json.dumps(report, indent=2))
