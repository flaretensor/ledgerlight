import argparse
import json
import subprocess
import threading
import time

import httpx

from ledgerlight.common import digest, path, read_json, write_json
from ledgerlight.posthoc import summarize_replay
from ledgerlight.split_guard import read_queries


def dev_queries():
    queries = {q["id"]: q["question"] for q in read_queries()}
    return {q: queries[q] for q in sorted(read_json("data/processed/splits.json")["dev"])}


def registered_hashes():
    from ledgerlight.registry import PIPELINE_NAME, client, manifest

    c = client()
    champion = c.get_model_version_by_alias(PIPELINE_NAME, "champion")
    config = manifest(champion.version)
    selected = read_json("models/production/config.json")
    hashes = {}
    for role, key in [
        ("encoder", "encoder_path"),
        ("cross_encoder", "cross_encoder_path"),
        ("ranker", "ltr_path"),
    ]:
        suffix = "" if role == "ranker" else "/model.safetensors"
        actual = digest(path(config[key] + suffix))
        expected = digest(path(selected[key] + suffix))
        assert actual == expected, role + " registered weight mismatch"
        hashes[role] = actual
    return {"pipeline": str(champion.version), "components": config["components"], "sha256": hashes}


def registry():
    from ledgerlight.registry import client

    c = client()
    aliases = {
        name: c.get_registered_model(name).aliases
        for name in [
            "ledgerlight-encoder",
            "ledgerlight-cross-encoder",
            "ledgerlight-ranker",
            "ledgerlight-pipeline",
        ]
    }
    result = {"registered": registered_hashes(), "aliases": aliases}
    write_json("data/analysis/runtime/registry.json", result)
    print(json.dumps(result, indent=2))


def search():
    expected = read_json("data/evaluation/dev_rankings.json")["CatBoost"]
    rows = []
    output = path("data/analysis/runtime/search_dev118.jsonl")
    with output.open("x") as stream, httpx.Client(timeout=180) as c:
        for qid, question in dev_queries().items():
            started = time.monotonic()
            response = c.post(
                "http://127.0.0.1:8001/search", json={"question": question, "limit": 20}
            )
            payload = response.json()
            ids = [r["passage_id"] for r in payload.get("results", [])]
            row = {
                "qid": qid,
                "http_status": response.status_code,
                "elapsed": time.monotonic() - started,
                "exact_top20": ids == expected[qid][:20],
                "response": payload,
                "expected": expected[qid][:20],
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            rows.append(row)
    result = {
        "queries": len(rows),
        "exact_top20": sum(r["exact_top20"] for r in rows),
        "mismatches": [
            {k: r[k] for k in ["qid", "http_status", "expected", "response"]}
            for r in rows
            if not r["exact_top20"]
        ],
    }
    write_json("data/analysis/runtime/search_parity.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "mismatches"}, indent=2))


def ask():
    output = path("data/analysis/ask_dev118")
    output.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()
    samples = []
    started = time.monotonic()

    def sample():
        while not stopped.is_set():
            value = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            samples.append({"seconds": time.monotonic() - started, "mib": int(value.strip())})
            stopped.wait(1)

    monitor = threading.Thread(target=sample)
    rows = []
    with (output / "responses.jsonl").open("x") as stream, httpx.Client(timeout=300) as c:
        monitor.start()
        try:
            for qid, question in dev_queries().items():
                clock = time.monotonic()
                row = {
                    "qid": qid,
                    "question": question,
                    "http_status": None,
                    "response": {},
                    "error": None,
                }
                try:
                    response = c.post("http://127.0.0.1:8000/ask", json={"question": question})
                    row["http_status"] = response.status_code
                    row["response"] = response.json()
                    if response.status_code != 200:
                        row["error"] = str(row["response"].get("detail"))
                except (httpx.HTTPError, ValueError) as exc:
                    row["error"] = type(exc).__name__ + ": " + str(exc)
                row["elapsed"] = time.monotonic() - clock
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "dev_ask": len(rows),
                            "total": 118,
                            "http_status": row["http_status"],
                            "elapsed": row["elapsed"],
                        }
                    ),
                    flush=True,
                )
        finally:
            stopped.set()
            monitor.join()
            write_json("data/analysis/ask_dev118/vram.json", samples)
    result = summarize_replay(rows, read_json("data/processed/labels.json"))
    result["peak_sampled_vram_mib"] = max(r["mib"] for r in samples)
    result["wall_seconds"] = time.monotonic() - started
    write_json("data/analysis/ask_dev118/result.json", result)
    print(json.dumps(result, indent=2))


def rollback():
    from ledgerlight.registry import PIPELINE_NAME, client
    from ledgerlight.registry import rollback as switch_previous

    c = client()
    names = [
        "ledgerlight-encoder",
        "ledgerlight-cross-encoder",
        "ledgerlight-ranker",
        PIPELINE_NAME,
    ]
    before = {name: dict(c.get_registered_model(name).aliases) for name in names}
    if "previous" not in before[PIPELINE_NAME]:
        write_json(
            "data/analysis/runtime/rollback.json",
            {"demonstrated": False, "reason": "rollback not demonstrated: previous alias missing"},
        )
        return
    qid, question = next(iter(dev_queries().items()))
    records = []

    def capture(label):
        response = httpx.post(
            "http://127.0.0.1:8001/search", json={"question": question, "limit": 20}, timeout=180
        )
        response.raise_for_status()
        payload = response.json()
        record = {
            "step": label,
            "qid": qid,
            "http_status": response.status_code,
            "versions": payload["versions"],
        }
        records.append(record)
        return str(payload["versions"]["pipeline"])

    try:
        assert capture("champion") == before[PIPELINE_NAME]["champion"]
        switch_previous()
        assert capture("previous") == before[PIPELINE_NAME]["previous"]
        switch_previous()
        assert capture("restored") == before[PIPELINE_NAME]["champion"]
    finally:
        for name, aliases in before.items():
            current = dict(c.get_registered_model(name).aliases)
            for alias in current.keys() - aliases.keys():
                c.delete_registered_model_alias(name, alias)
            for alias, version in aliases.items():
                if current.get(alias) != version:
                    c.set_registered_model_alias(name, alias, version)
    result = {
        "demonstrated": True,
        "records": records,
        "restored_hashes": registered_hashes(),
        "aliases_restored": all(
            dict(c.get_registered_model(name).aliases) == aliases
            for name, aliases in before.items()
        ),
    }
    write_json("data/analysis/runtime/rollback.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["registry", "search", "ask", "rollback"])
    {"registry": registry, "search": search, "ask": ask, "rollback": rollback}[
        parser.parse_args().action
    ]()
