import json
import time

import gevent
import numpy as np
from locust import HttpUser
from locust.env import Environment
from locust.stats import RequestStats

from ledgerlight.common import read_json, read_jsonl, write_json
from ledgerlight.evaluation import tracking


class LocalUser(HttpUser):
    host = "http://127.0.0.1:8000"


def quantiles(values):
    return (
        {"p50": float(np.quantile(values, 0.5)), "p95": float(np.quantile(values, 0.95))}
        if values
        else None
    )


environment = Environment(user_classes=[LocalUser])
environment.stats = RequestStats()
environment.create_local_runner()
user = LocalUser(environment)
lookup = {q["id"]: q["question"] for q in read_jsonl("data/processed/queries.jsonl")}
questions = [lookup[qid] for qid in read_json("data/processed/splits.json")["dev"][:20]]
records = []


def request(number, phase):
    started = time.perf_counter()
    try:
        response = user.client.post(
            "/ask",
            json={"question": questions[number % len(questions)]},
            name="/ask-" + phase,
            timeout=180,
        )
        ok = response.status_code == 200
        payload = response.json() if ok else {}
    except Exception as exc:
        ok, payload = False, {"error": type(exc).__name__}
    records.append(
        {
            "number": number,
            "phase": phase,
            "completed": ok,
            "seconds": time.perf_counter() - started,
            "stages": payload.get("latencies", {}),
            "retried_without_thinking": payload.get("generation", {}).get(
                "retried_without_thinking", False
            ),
            "http_status": response.status_code if "response" in locals() else None,
        }
    )


request(0, "cold")
rate, offered = 0.1, 12
start = time.perf_counter()
jobs = []
for i in range(offered):
    delay = max(0, start + i / rate - time.perf_counter())
    gevent.sleep(delay)
    jobs.append(gevent.spawn(request, i + 1, "warm"))
gevent.joinall(jobs)
wall = time.perf_counter() - start
warm = [r for r in records if r["phase"] == "warm"]
keys = sorted({key for r in warm for key in r["stages"]})
result = {
    "cold": records[0],
    "warm": {
        "offered": offered,
        "completed": sum(r["completed"] for r in warm),
        "failed": sum(not r["completed"] for r in warm),
        "retry_rate": sum(r["retried_without_thinking"] for r in warm) / len(warm),
        "offered_rate_per_second": rate,
        "wall_seconds": wall,
        "completed_throughput": sum(r["completed"] for r in warm) / wall,
        "e2e_seconds": quantiles([r["seconds"] for r in warm]),
        "stage_seconds": {
            key: quantiles([r["stages"][key] for r in warm if key in r["stages"]]) for key in keys
        },
    },
    "locust_requests": environment.stats.total.num_requests,
    "locust_failures": environment.stats.total.num_failures,
    "protocol": "Locust HttpSession, independently scheduled arrivals; cold request after manual service restart",
    "question_split": "dev_only",
    "requests": records,
}
write_json("data/serving/load.json", result)
print(json.dumps(result, indent=2))


tracking(
    "locust-local-load",
    {
        "offered": result["warm"]["offered"],
        "completed": result["warm"]["completed"],
        "failed": result["warm"]["failed"],
        "p95_seconds": result["warm"]["e2e_seconds"]["p95"],
    },
    {"offered_rate": rate, "arrival": "independent"},
    result,
)
