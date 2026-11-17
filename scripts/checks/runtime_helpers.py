import fcntl
import json
import os
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

import httpx

from ledgerlight.common import ROOT, read_json, read_jsonl
from ledgerlight.posthoc import latency_summary

OUT = ROOT / "data/analysis/serving-benchmarks"


def save(target, value):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summary(section, name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "summary.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = OUT / "summary.json"
        document = json.loads(target.read_text()) if target.exists() else {}
        document.setdefault(section, {})[name] = value
        save(target, document)


def dev_queries():
    ids = set(read_json("data/processed/splits.json")["dev"])
    return [q for q in read_jsonl("data/processed/queries.jsonl") if q["id"] in ids]


def lifecycle(action, directory, search=None, layers=None, triton=False, threads=None):
    command = [os.sys.executable, "scripts/site.py", action]
    if action == "start":
        command += [
            "--search",
            search,
            "--gpu-layers",
            str(layers),
            "--triton" if triton else "--no-triton",
            "--force",
        ]
    env = dict(os.environ, LEDGERLIGHT_REQUEST_DEADLINE="300")
    if threads is not None:
        env["LEDGERLIGHT_LLM_THREADS"] = str(threads)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (action + ".log")).open("w") as stream:
        clock = time.monotonic()
        result = subprocess.run(
            command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=700
        )
    if result.returncode:
        raise RuntimeError(action + " failed; see " + str(directory / (action + ".log")))
    return time.monotonic() - clock


def process_tree(pid):
    found = set()
    if not pid:
        return found
    found.add(pid)
    try:
        children = (Path("/proc") / str(pid) / "task" / str(pid) / "children").read_text().split()
        for child in children:
            found.update(process_tree(int(child)))
    except OSError:
        pass
    return found


class Sampler:
    def __init__(self, project="ledgerlight"):
        self.project = project
        self.done = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.sample)

    def sample(self):
        next_sample = time.monotonic()
        while not self.done.is_set():
            row = {"time": time.monotonic(), "rss_mib": {}}
            try:
                gpu = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                )
                row["gpu_mib"] = int(gpu.strip().splitlines()[0])
                pids = set()
                service_pids = {}
                for service in ["retrieval", "llama", "triton"]:
                    result = subprocess.run(
                        [
                            "docker",
                            "inspect",
                            "--format",
                            "{{.State.Pid}}",
                            f"{self.project}-{service}-1",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    tree = (
                        process_tree(int(result.stdout.strip()))
                        if result.returncode == 0 and result.stdout.strip().isdigit()
                        else set()
                    )
                    pids.update(tree)
                    service_pids[service] = tree
                    rss = 0
                    for pid in tree:
                        try:
                            for line in (
                                (Path("/proc") / str(pid) / "status").read_text().splitlines()
                            ):
                                if line.startswith("VmRSS:"):
                                    rss += int(line.split()[1]) / 1024
                        except OSError:
                            pass
                    row["rss_mib"][service] = rss
                compute = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                allocations = {
                    int(line.split(",")[0].strip()): int(line.split(",")[1].strip())
                    for line in compute.splitlines()
                    if line.split(",")[0].strip().isdigit()
                }
                row["project_gpu_mib"] = sum(allocations.get(pid, 0) for pid in pids)
                row["process_gpu_mib"] = {
                    service: sum(allocations.get(pid, 0) for pid in tree)
                    for service, tree in service_pids.items()
                }
                self.samples.append(row)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            next_sample += 1
            self.done.wait(max(0, next_sample - time.monotonic()))

    def start(self):
        self.thread.start()

    def finish(self):
        self.done.set()
        self.thread.join()
        return summarize_samples(self.samples)


def summarize_samples(samples):
    at_peak = max(samples, key=lambda s: s["project_gpu_mib"], default={})
    return {
        "samples": len(samples),
        "peak_gpu_mib": max((s["gpu_mib"] for s in samples), default=0),
        "peak_project_gpu_mib": max((s["project_gpu_mib"] for s in samples), default=0),
        "process_gpu_mib_at_project_peak": at_peak.get("process_gpu_mib"),
        "peak_process_gpu_mib": {
            name: max(
                (s["process_gpu_mib"][name] for s in samples if "process_gpu_mib" in s),
                default=None,
            )
            for name in ["retrieval", "llama", "triton"]
        },
        "peak_rss_mib": {
            name: max((s["rss_mib"].get(name, 0) for s in samples), default=0)
            for name in ["retrieval", "llama", "triton"]
        },
    }


def ask_stream(query, client_number=1):
    clock = time.monotonic()
    row = {
        "qid": query["id"],
        "http_status": None,
        "response": {},
        "error": None,
        "first_text_seconds": None,
        "delta_count": 0,
        "reset_count": 0,
    }
    try:
        with httpx.stream(
            "POST",
            "http://127.0.0.1:8000/ask/stream",
            json={"question": query["question"]},
            headers={
                "CF-Connecting-IP": f"198.18.{client_number // 250}.{client_number % 250 + 1}"
            },
            timeout=httpx.Timeout(300, connect=5),
        ) as response:
            row["http_status"] = response.status_code
            if response.status_code != 200:
                response.read()
                row["error"] = response.json().get("detail", {}).get("code", "error")
            else:
                event = None
                for line in response.iter_lines():
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        value = json.loads(line[5:])
                        if event == "delta" and value.get("text"):
                            row["delta_count"] += 1
                            if row["first_text_seconds"] is None:
                                row["first_text_seconds"] = time.monotonic() - clock
                            if "[[SRC_" in value["text"]:
                                raise RuntimeError("Raw citation marker was emitted")
                        elif event == "reset":
                            row["reset_count"] += 1
                        elif event == "final":
                            row["response"] = value
                        elif event == "error":
                            row["error"] = value.get("code", "error")
                if not row["response"] and not row["error"]:
                    row["error"] = "missing_final"
    except httpx.TimeoutException:
        row["error"] = "timeout"
    except (httpx.HTTPError, ValueError) as exc:
        row["error"] = type(exc).__name__
    row["elapsed"] = time.monotonic() - clock
    return row


def summarize(rows):
    valid = [r for r in rows if r.get("response") and not r.get("error")]
    payloads = [r["response"] for r in valid]
    return {
        "offered": len(rows),
        "completed": len(valid),
        "http_statuses": dict(Counter(str(r["http_status"]) for r in rows)),
        "cited": sum(bool(p.get("citations")) for p in payloads),
        "abstained": sum(p.get("abstained", False) for p in payloads),
        "errors": dict(Counter(r["error"] for r in rows if r.get("error"))),
        "length_retries": sum(
            p.get("generation", {}).get("retried_without_thinking", False) for p in payloads
        ),
        "context_clipped": sum(
            p.get("generation", {}).get("context_clipped", False) for p in payloads
        ),
        "e2e_seconds_all_offered": latency_summary([r["elapsed"] for r in rows]),
        "e2e_seconds_successes": latency_summary([r["elapsed"] for r in valid]),
        "first_text_seconds": latency_summary(
            [r["first_text_seconds"] for r in valid if r["first_text_seconds"] is not None]
        ),
        "search_seconds": latency_summary([p["latencies"]["retrieval"] for p in payloads]),
        "generation_tokens_per_second": latency_summary(
            [
                p.get("generation", {}).get("timings", {})["predicted_per_second"]
                for p in payloads
                if "predicted_per_second" in p.get("generation", {}).get("timings", {})
            ]
        ),
    }
