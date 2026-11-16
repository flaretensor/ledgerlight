import argparse
import json
import os
import subprocess
import sys
import time
from argparse import Namespace

import httpx
from runtime_helpers import (
    OUT,
    Sampler,
    ask_stream,
    dev_queries,
    lifecycle,
    save,
    summarize,
    summary,
)

from ledgerlight.analysis import latency_summary
from ledgerlight.common import ROOT, read_json
from ledgerlight.evaluation import evaluate

VARIANTS = ["torch", "ort", "openvino", "triton-ort", "triton-openvino"]


def capture_services(directory):
    from dotenv import dotenv_values

    directory.mkdir(parents=True, exist_ok=True)
    secrets = [
        value
        for name, value in dotenv_values(ROOT / ".env").items()
        if value and name.endswith(("_PASSWORD", "_TOKEN", "_API_KEY"))
    ]
    for service in ["retrieval", "triton", "qdrant", "mlflow"]:
        result = subprocess.run(
            ["docker", "logs", "ledgerlight-" + service + "-1"], capture_output=True, text=True
        )
        content = result.stdout + result.stderr
        for value in secrets:
            content = content.replace(value, "[REDACTED]")
        (directory / (service + ".log")).write_text(content)
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{json .State}}", "ledgerlight-retrieval-1"],
        capture_output=True,
        text=True,
    )
    (directory / "retrieval-state.json").write_text(result.stdout)


def parity(only=None):
    lifecycle("stop", OUT / "parity/stop")
    for engine in ["torch", "ort", "openvino", "openvino-gpu", "triton-ort", "triton-openvino"]:
        if only and engine != only:
            continue
        destination = OUT / "parity" / (engine + ".json")
        if destination.exists():
            result = json.loads(destination.read_text())
        else:
            directory = OUT / "parity" / engine
            directory.mkdir(parents=True, exist_ok=True)
            try:
                if engine.startswith("triton-"):
                    import importlib.util

                    spec = importlib.util.spec_from_file_location(
                        "ledgerlight_site", ROOT / "scripts/site.py"
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    config = module.configure(
                        Namespace(
                            search=engine[7:],
                            triton=True,
                            gpu_layers="0",
                            cpu=False,
                            force=True,
                            public=False,
                        )
                    )
                    with (directory / "server.log").open("w") as stream:
                        subprocess.run(
                            module.compose(config)
                            + ["up", "-d", "--no-build", "--pull", "never", "triton"],
                            check=True,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                        )
                    module.healthy("http://127.0.0.1:8002/v2/health/ready", time.monotonic() + 120)
                with (directory / "measure.log").open("w") as stream:
                    subprocess.run(
                        [sys.executable, "scripts/checks/backend_parity.py", "--engine", engine],
                        cwd=ROOT,
                        check=True,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=1200,
                    )
                result = json.loads(destination.read_text())
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                result = {"engine": engine, "passed": False, "technical_error": str(exc)}
                save(directory / "failure.json", result)
            finally:
                if engine.startswith("triton-"):
                    lifecycle("stop", directory)
        summary(
            "parity",
            engine,
            {k: v for k, v in result.items() if k not in {"rows", "query_vectors"}},
        )
        print(json.dumps({"parity": engine, "passed": result["passed"]}), flush=True)


def threads():
    results = []
    for count in [4, 8, 12]:
        directory = OUT / "threads" / str(count)
        target = directory / "result.json"
        if target.exists():
            result = json.loads(target.read_text())
        else:
            monitor = Sampler()
            monitor.start()
            rows = []
            try:
                cold = lifecycle("start", directory, "openvino", "0", threads=count)
                for number, q in enumerate(dev_queries()[:3]):
                    rows.append(ask_stream(q, number + 1))
                    print(
                        json.dumps(
                            {
                                "threads": count,
                                "query": number + 1,
                                "elapsed": rows[-1]["elapsed"],
                                "error": rows[-1]["error"],
                            }
                        ),
                        flush=True,
                    )
                result = {
                    **summarize(rows),
                    "threads": count,
                    "cold_start_seconds": cold,
                    **monitor.finish(),
                }
                save(directory / "rows.json", rows)
                save(directory / "samples.json", monitor.samples)
            finally:
                if monitor.thread.is_alive():
                    monitor.finish()
                lifecycle("stop", directory)
            save(target, result)
        results.append(result)
        summary("thread_sweep", str(count), result)
    eligible = [r for r in results if r["completed"] == 3]
    if not eligible:
        raise RuntimeError("All CPU thread sweep configurations failed")
    selected = min(eligible, key=lambda r: r["e2e_seconds_all_offered"]["mean"])["threads"]
    settings = read_json("config/site.json")
    settings["llm_threads"] = selected
    save(ROOT / "config/site.json", settings)
    summary(
        "thread_sweep",
        "selection",
        {
            "threads": selected,
            "rule": "Minimum mean E2E over the same three DEV questions, LLM CPU-only",
        },
    )


def search_measure(engine, directory):
    target = directory / "search.json"
    if target.exists():
        return json.loads(target.read_text())
    questions = dev_queries()
    rows = []
    rankings = {}
    expected = read_json("data/evaluation/dev_rankings.json")["CatBoost"]
    with httpx.Client(timeout=300) as client:
        for q in questions[:5]:
            response = client.post(
                "http://127.0.0.1:8001/search", json={"question": q["question"], "limit": 20}
            )
            response.raise_for_status()
        for number, q in enumerate(questions):
            clock = time.monotonic()
            response = client.post(
                "http://127.0.0.1:8001/search", json={"question": q["question"], "limit": 20}
            )
            if response.is_error:
                save(
                    directory / "search-failure.json",
                    {
                        "engine": engine,
                        "qid": q["id"],
                        "completed_before_failure": len(rows),
                        "http_status": response.status_code,
                        "response": response.json(),
                    },
                )
            response.raise_for_status()
            payload = response.json()
            rankings[q["id"]] = [p["passage_id"] for p in payload["results"]]
            rows.append(
                {
                    "qid": q["id"],
                    "search_seconds": payload["latencies"]["total"],
                    "e2e_seconds": time.monotonic() - clock,
                    "ranking": rankings[q["id"]],
                    "equals_frozen": rankings[q["id"]] == expected[q["id"]][:20],
                    "latencies": payload["latencies"],
                }
            )
            save(directory / "search-partial.json", rows)
            if (number + 1) % 10 == 0:
                print(json.dumps({"search": engine, "completed": number + 1}), flush=True)
    labels = read_json("data/processed/labels.json")
    metrics, _ = evaluate(rankings, {q: labels[q] for q in rankings})
    result = {
        "engine": engine,
        "warmups_discarded": 5,
        "queries": len(rows),
        "search_seconds": latency_summary([r["search_seconds"] for r in rows]),
        "e2e_seconds": latency_summary([r["e2e_seconds"] for r in rows]),
        "equals_frozen_top20": sum(r["equals_frozen"] for r in rows),
        "metrics": metrics,
    }
    save(directory / "search-rows.json", rows)
    save(target, result)
    summary("search_latency", engine, result)
    return result


def search_retry(engine, reason):
    directory = OUT / "search-retry" / engine
    if (directory / "attempt.json").exists():
        raise RuntimeError("Only one technical search retry is permitted per variant")
    target = OUT / "matrix" / engine / "all"
    if (target / "search.json").exists():
        raise RuntimeError("A successful search measurement must not be repeated")
    directory.mkdir(parents=True, exist_ok=True)
    result = {"engine": engine, "reason": reason, "status": "PENDING"}
    save(directory / "attempt.json", result)
    monitor = Sampler()
    monitor.start()
    try:
        lifecycle(
            "start",
            directory,
            engine[7:] if engine.startswith("triton-") else engine,
            "all",
            engine.startswith("triton-"),
        )
        result["measurement"] = search_measure(engine, target)
        result["status"] = "DONE"
    except (OSError, RuntimeError, subprocess.SubprocessError, httpx.HTTPError) as exc:
        result.update(status="FAILED", cause=str(exc))
    finally:
        result["search_pass_memory"] = monitor.finish()
        save(directory / "samples.json", monitor.samples)
        capture_services(directory)
        lifecycle("stop", directory)
        result["stop"] = read_json("data/runtime/stop.json")
        save(directory / "attempt.json", result)
        summary("search_retries", engine, result)


def matrix():
    preparation = OUT / "matrix/preparation"
    preparation.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/checks/data_quality.py",
            "--output",
            str(preparation / "quality"),
        ],
        check=True,
        env=dict(os.environ, DVC_SITE_CACHE_DIR=str(ROOT / ".cache/dvc")),
    )
    with (preparation / "build.log").open("w") as stream:
        subprocess.run(
            ["docker", "compose", "build", "retrieval", "forwarder"],
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    settings = read_json("config/site.json")
    total = settings["total_layers"]
    if not total:
        raise RuntimeError("Read and record total LLM layer count before matrix")
    for engine in VARIANTS:
        search = engine[7:] if engine.startswith("triton-") else engine
        for label, layers in [("all", "all"), ("half", str(total // 2)), ("0", "0")]:
            directory = OUT / "matrix" / engine / label
            target = directory / "result.json"
            if target.exists():
                continue
            monitor = Sampler()
            monitor.start()
            rows = []
            try:
                cold = lifecycle("start", directory, search, layers, engine.startswith("triton-"))
                for number, q in enumerate(dev_queries()[:8]):
                    rows.append(ask_stream(q, number + 1))
                    save(directory / "rows.json", rows)
                    print(
                        json.dumps(
                            {
                                "cell": engine + "/" + label,
                                "query": number + 1,
                                "elapsed": rows[-1]["elapsed"],
                                "error": rows[-1]["error"],
                            }
                        ),
                        flush=True,
                    )
                result = {
                    "status": "DONE",
                    "engine": engine,
                    "layers": layers,
                    "layer_label": label,
                    "cold_start_seconds": cold,
                    **summarize(rows),
                    **monitor.finish(),
                }
                if result["completed"] != 8:
                    result["status"] = "FAILED"
                    result["technical_error"] = (
                        f"Only {result['completed']}/8 streamed requests completed: {result['errors']}"
                    )
                save(directory / "samples.json", monitor.samples)
                save(target, result)
                if label == "all":
                    try:
                        search_measure(engine, directory)
                        result["search_pass_status"] = "DONE"
                    except httpx.HTTPError as exc:
                        result["search_pass_status"] = "FAILED"
                        result["search_pass_cause"] = str(exc)
            except (OSError, RuntimeError, subprocess.SubprocessError, httpx.HTTPError) as exc:
                result = {
                    "status": "FAILED",
                    "engine": engine,
                    "layers": layers,
                    "layer_label": label,
                    "technical_error": str(exc),
                    **summarize(rows),
                    **monitor.finish(),
                }
            finally:
                if monitor.thread.is_alive():
                    monitor.finish()
                capture_services(directory / "services")
                try:
                    lifecycle("stop", directory)
                    stop = read_json("data/runtime/stop.json")
                except (RuntimeError, subprocess.SubprocessError) as exc:
                    stop = {"failed": str(exc)}
            result["stop"] = stop
            save(target, result)
            summary("matrix", engine + "/" + label, result)


def choose_defaults(excluded=()):
    document = read_json("data/analysis/stage2/summary.json")
    searches = {
        engine: result
        for engine, result in document.get("search_latency", {}).items()
        if result.get("queries") == 118 and "search_seconds" in result
    }
    parities = document.get("parity", {})
    cells = document.get("matrix", {})
    candidates = ["torch", "ort", "openvino", "triton-ort"]
    eligible = [
        e
        for e in candidates
        if e not in excluded
        and parities.get(e, {}).get("passed")
        and e in searches
        and cells.get(e + "/all", {}).get("completed") == 8
        and not cells[e + "/all"].get("errors")
        and cells[e + "/all"].get("peak_gpu_mib", 10**9) <= 3700
    ]
    if not eligible:
        raise RuntimeError("No passing measured search engine")
    fastest = min(searches[e]["search_seconds"]["p95"] for e in eligible)
    near = [e for e in eligible if searches[e]["search_seconds"]["p95"] <= 1.1 * fastest]
    minimum_vram = min(cells[e + "/all"]["peak_gpu_mib"] for e in near)
    tied = [e for e in near if cells[e + "/all"]["peak_gpu_mib"] <= minimum_vram + 100]
    priority = ["triton-ort", "ort", "torch", "openvino"]
    selected = min(tied, key=priority.index)
    engine = selected.removeprefix("triton-")
    use_triton = selected.startswith("triton-")
    settings = read_json("config/site.json")
    table = {}
    for key, cell in cells.items():
        if cell.get("status") == "DONE" and cell.get("completed") == 8:
            name = cell["engine"]
            key = name[7:] + "-triton" if name.startswith("triton-") else name
            table.setdefault(key, {})[str(cell["layers"])] = cell["peak_gpu_mib"]
    settings.update(
        search=engine, triton=use_triton, gpu_layers="all", vram_mib=table, measured=True
    )
    save(ROOT / "config/site.json", settings)
    summary(
        "defaults",
        "selection",
        {
            "settings": settings,
            "selected": selected,
            "eligible": eligible,
            "excluded": list(excluded),
            "p_min": fastest,
            "p95_limit": 1.1 * fastest,
            "shortlist": near,
            "minimum_vram_mib": minimum_vram,
            "within_100_mib": tied,
            "priority": priority,
            "inputs": {
                e: {
                    "p95": searches[e]["search_seconds"]["p95"],
                    "peak_gpu_mib": cells[e + "/all"]["peak_gpu_mib"],
                }
                for e in eligible
            },
        },
    )


def replay():
    directory = OUT / "replay"
    target = directory / "result.json"
    if target.exists():
        raise RuntimeError("Completed DEV118 replay must not be repeated")
    settings = read_json("config/site.json")
    monitor = Sampler()
    monitor.start()
    rows = []
    active = read_json("data/runtime/active.json")
    if (active["search"], active["triton"], active["gpu_layers"]) != (
        settings["search"],
        settings["triton"],
        "all",
    ):
        raise RuntimeError("Start the selected defaults before the replay")
    httpx.get("http://127.0.0.1:8000/health", timeout=30).raise_for_status()
    cold = active["cold_start_seconds"]
    online_rankings = {}
    for number, q in enumerate(dev_queries()):
        rows.append(ask_stream(q, number + 1))
        save(directory / "rows.json", rows)
        with httpx.Client(timeout=300) as client:
            response = client.post(
                "http://127.0.0.1:8001/search", json={"question": q["question"], "limit": 20}
            )
            response.raise_for_status()
        online_rankings[q["id"]] = [r["passage_id"] for r in response.json()["results"]]
        save(directory / "online-rankings.json", online_rankings)
        print(
            json.dumps(
                {"replay": number + 1, "elapsed": rows[-1]["elapsed"], "error": rows[-1]["error"]}
            ),
            flush=True,
        )
    result = {**summarize(rows), "cold_start_seconds": cold, **monitor.finish()}
    labels = read_json("data/processed/labels.json")
    result["citing_positive_parent_overlap_proxy"] = sum(
        bool(
            {c["passage_id"] for c in r["response"].get("citations", [])}
            & set(labels[r["qid"]]["positives"])
        )
        for r in rows
        if r["response"]
    )
    frozen = read_json("data/evaluation/dev_rankings.json")["CatBoost"]
    metrics, _ = evaluate(online_rankings, {q: labels[q] for q in online_rankings})
    result["search_parity"] = {
        "metrics": metrics,
        "queries": len(online_rankings),
        "equals_frozen_top20": sum(
            ranking == frozen[q][:20] for q, ranking in online_rankings.items()
        ),
    }
    result["passed"] = (
        result["completed"] == 118
        and not result["errors"]
        and abs(result["search_parity"]["metrics"]["ndcg@10"] - 0.4367276797012511) <= 0.005
    )
    save(directory / "samples.json", monitor.samples)
    save(target, result)
    summary("replay", "dev118", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["parity", "threads", "matrix", "replay", "defaults", "search-retry"]
    )
    parser.add_argument("--engine", choices=VARIANTS + ["openvino-gpu"])
    parser.add_argument("--reason", default="Technical failure")
    parser.add_argument("--exclude", action="append", default=[], choices=VARIANTS)
    args = parser.parse_args()
    {
        "parity": lambda: parity(args.engine),
        "threads": threads,
        "matrix": matrix,
        "replay": replay,
        "defaults": lambda: choose_defaults(args.exclude),
        "search-retry": lambda: search_retry(args.engine, args.reason),
    }[args.stage]()
