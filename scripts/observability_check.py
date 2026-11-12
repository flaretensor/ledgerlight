import json
import time

import httpx

from ledgerlight.common import read_json, read_jsonl, write_json

qid = read_json("data/processed/splits.json")["dev"][0]
question = next(q["question"] for q in read_jsonl("data/processed/queries.jsonl") if q["id"] == qid)
with httpx.Client(timeout=180) as client:
    for _ in range(90):
        try:
            response = client.get("http://127.0.0.1:8000/health", timeout=5)
            response.raise_for_status()
            break
        except httpx.HTTPError:
            time.sleep(1)
    else:
        raise RuntimeError("Gateway not ready")
    assert response.json()["retrieval"]["versions"]["pipeline"] == "7"
    answer = client.post("http://127.0.0.1:8000/ask", json={"question": question})
    answer.raise_for_status()
    write_json(
        "data/serving/s6_dev_ask.json", {"split": "dev", "qid": qid, "response": answer.json()}
    )
    invalid = client.post(
        "http://127.0.0.1:8000/ask", json={"question": question, "answer": "forbidden field"}
    )
    assert invalid.status_code == 422
    time.sleep(6)
    expressions = {
        "gateway_requests": 'sum(ledgerlight_requests_total{service="gateway"})',
        "e2e_observations": 'sum(ledgerlight_stage_seconds_count{service="gateway",stage="e2e"})',
        "ce_observations": 'sum(ledgerlight_stage_seconds_count{service="retrieval",stage="ce"})',
        "errors": "sum(ledgerlight_errors_total)",
        "abstentions": "ledgerlight_abstentions_total",
        "resident_memory": 'sum(process_resident_memory_bytes{job=~"gateway|retrieval"})',
        "cpu": 'sum(rate(process_cpu_seconds_total{job=~"gateway|retrieval"}[1m]))',
        "gpu_memory": "ledgerlight_gpu_memory_used_bytes",
        "triton_observed_inferences": 'max_over_time(nv_inference_count{job="triton"}[1h])',
        "scrapes_seen": 'max_over_time(up{job=~"gateway|retrieval|qdrant|llama|triton"}[1h])',
    }
    queries = {}
    for name, expression in expressions.items():
        response = client.get("http://127.0.0.1:9090/api/v1/query", params={"query": expression})
        response.raise_for_status()
        data = response.json()
        assert data["status"] == "success" and data["data"]["result"], name
        queries[name] = {"expression": expression, "result": data["data"]["result"]}
    seen = {r["metric"]["job"]: float(r["value"][1]) for r in queries["scrapes_seen"]["result"]}
    assert all(
        seen.get(name) == 1 for name in ["gateway", "retrieval", "qdrant", "llama", "triton"]
    )
    response = client.get("http://127.0.0.1:3000/api/dashboards/uid/ledgerlight-runtime")
    response.raise_for_status()
    dashboard = response.json()["dashboard"]
    response = client.get("http://127.0.0.1:3000/d/ledgerlight-runtime/ledgerlight-runtime")
    response.raise_for_status()
    result = {
        "dashboard_http": response.status_code,
        "panel_titles": [p["title"] for p in dashboard["panels"]],
        "queries": queries,
        "dev_ask_http": answer.status_code,
        "dev_ask_versions": answer.json()["ranking"]["versions"],
        "controlled_validation_status": invalid.status_code,
        "gpu_services_sequenced": True,
        "scrape_note": "Each target verified while active; Triton is intentionally stopped during RAG serving to avoid GPU contention.",
    }
write_json("data/serving/observability.json", result)
print(
    json.dumps(
        {
            "dashboard_http": result["dashboard_http"],
            "panels": len(result["panel_titles"]),
            "scrapes_seen": seen,
            "dev_ask_http": result["dev_ask_http"],
            "backend": result["dev_ask_versions"]["backend"],
        },
        indent=2,
    )
)
