import argparse
import json
import math
import subprocess
import time

import httpx
from bench_runtime import OUT, dev_queries, mark_step, save, summary


def rollback():
    from live_proof import registered_hashes

    from ledgerlight.registry import PIPELINE_NAME, client
    from ledgerlight.registry import rollback as move

    c = client()
    names = [
        "ledgerlight-encoder",
        "ledgerlight-cross-encoder",
        "ledgerlight-ranker",
        PIPELINE_NAME,
    ]
    aliases = {name: dict(c.get_registered_model(name).aliases) for name in names}
    counts = {name: len(list(c.search_model_versions("name='" + name + "'"))) for name in names}
    result = {"before": aliases, "hashes_before": registered_hashes(), "steps": []}
    question = dev_queries()[0]["question"]
    try:
        for _ in range(2):
            clock = time.monotonic()
            changed = move()
            response = httpx.post(
                "http://127.0.0.1:8001/search",
                json={"question": question, "limit": 20},
                timeout=300,
            )
            response.raise_for_status()
            payload = response.json()
            result["steps"].append(
                {
                    "action": changed,
                    "http_status": response.status_code,
                    "versions": payload["versions"],
                    "seconds": time.monotonic() - clock,
                }
            )
    finally:
        for name, expected in aliases.items():
            current = c.get_registered_model(name).aliases
            for alias in set(current) - set(expected):
                c.delete_registered_model_alias(name, alias)
            for alias, version in expected.items():
                c.set_registered_model_alias(name, alias, version)
        response = httpx.post(
            "http://127.0.0.1:8001/search", json={"question": question}, timeout=300
        )
        response.raise_for_status()
    result["after"] = {name: dict(c.get_registered_model(name).aliases) for name in names}
    result["hashes_after"] = registered_hashes()
    result["versions_unchanged"] = counts == {
        name: len(list(c.search_model_versions("name='" + name + "'"))) for name in names
    }
    result["passed"] = (
        result["before"] == result["after"]
        and result["hashes_before"] == result["hashes_after"]
        and result["versions_unchanged"]
        and len(result["steps"]) == 2
    )
    save(OUT / "rollback/result.json", result)
    summary("rollback", "drill", result)
    mark_step(
        8,
        "8.6",
        "DONE" if result["passed"] else "FAILED",
        artifacts=["data/analysis/stage2/rollback/result.json"],
    )
    print(json.dumps(result, indent=2))


def observe():
    with httpx.Client(timeout=30) as client:
        targets = client.get("http://127.0.0.1:9090/api/v1/targets").json()["data"]["activeTargets"]
        dashboard = client.get("http://127.0.0.1:3000/api/dashboards/uid/ledgerlight-runtime")
        dashboard.raise_for_status()
        panels = []
        for panel in dashboard.json()["dashboard"]["panels"]:
            values = []
            for target in panel.get("targets", []):
                payload = client.get(
                    "http://127.0.0.1:9090/api/v1/query", params={"query": target["expr"]}
                ).json()
                values.extend(
                    float(row["value"][1]) for row in payload.get("data", {}).get("result", [])
                )
            panels.append(
                {
                    "id": panel["id"],
                    "title": panel["title"],
                    "series": len(values),
                    "finite": bool(values) and all(math.isfinite(v) for v in values),
                }
            )
        result = {
            "targets": [
                {"job": t["labels"]["job"], "health": t["health"], "error": t.get("lastError")}
                for t in targets
            ],
            "panels": panels,
            "passed": all(t["health"] == "up" for t in targets)
            and all(p["finite"] for p in panels),
            "gpu_panel_scope": "GPU memory visible to retrieval; host-sampled matrix/replay captures the full stack",
        }
    save(OUT / "observability/result.json", result)
    summary("observability", "final", result)
    mark_step(
        8,
        "8.8.observability",
        "DONE" if result["passed"] else "FAILED",
        artifacts=["data/analysis/stage2/observability/result.json"],
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("Observability checks failed")


def privacy():
    questions = [q["question"] for q in dev_queries()]
    ids = subprocess.check_output(
        ["docker", "ps", "-q", "--filter", "label=com.docker.compose.project=ledgerlight"],
        text=True,
    ).split()
    rows = []
    for identifier in ids:
        result = subprocess.run(["docker", "logs", identifier], capture_output=True, text=True)
        logs = result.stdout + result.stderr
        name = subprocess.check_output(
            ["docker", "inspect", "--format", "{{.Name}}", identifier], text=True
        ).strip()
        rows.append(
            {
                "container": name,
                "native_dev_questions_found": sum(question in logs for question in questions),
                "benchmark_client_addresses_found": "198.18." in logs or "198.19." in logs,
            }
        )
    import psycopg

    from ledgerlight.common import database_url

    with psycopg.connect(database_url()) as connection:
        exists = connection.execute(
            "SELECT to_regclass('public.ask_requests') IS NOT NULL"
        ).fetchone()[0]
        count = (
            connection.execute("SELECT count(*) FROM ask_requests").fetchone()[0] if exists else 0
        )
    result = {
        "logs": rows,
        "legacy_request_rows": count,
        "passed": count == 0
        and all(
            r["native_dev_questions_found"] == 0 and not r["benchmark_client_addresses_found"]
            for r in rows
        ),
        "scope": "Exact native DEV input strings and synthetic client prefixes in project container logs; no filesystem-wide tracing",
    }
    save(OUT / "privacy/result.json", result)
    summary("privacy", "final", result)
    mark_step(
        5,
        "5.privacy",
        "DONE" if result["passed"] else "FAILED",
        artifacts=["data/analysis/stage2/privacy/result.json"],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=["rollback", "observe", "privacy"])
    args = parser.parse_args()
    {"rollback": rollback, "observe": observe, "privacy": privacy}[args.check]()
