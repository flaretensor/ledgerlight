import itertools
import json
import re
import time

import gevent
import httpx
from bench_runtime import OUT, Sampler, dev_queries, mark_step, save, summarize_samples, summary
from locust import HttpUser, task
from locust.env import Environment
from locust.exception import StopUser

from ledgerlight.analysis import latency_summary


class DemoUser(HttpUser):
    host = "http://127.0.0.1:8000"

    def on_start(self):
        self.number = next(self.environment.numbers)

    @task
    def measure(self):
        specification = self.environment.specification
        for repetition in range(specification["requests"]):
            started = time.monotonic()
            row = {
                "started_monotonic": started,
                "http_status": None,
                "outcome": "error",
                "first_text_seconds": None,
                "expected_disconnect": False,
            }
            client_number = self.number if specification.get("distinct") else 1
            with self.client.post(
                "/ask/stream",
                json={"question": self.environment.question},
                headers={"CF-Connecting-IP": f"198.19.{specification['group']}.{client_number}"},
                stream=True,
                catch_response=True,
                timeout=300,
                name="/ask/stream",
            ) as response:
                row["http_status"] = response.status_code
                try:
                    if response.status_code == 429:
                        row["outcome"] = response.json()["detail"]["code"]
                        response.failure(row["outcome"])
                    elif response.status_code == 200:
                        event = None
                        for line in response.iter_lines(chunk_size=64, decode_unicode=True):
                            if isinstance(line, bytes):
                                line = line.decode()
                            if line.startswith("event:"):
                                event = line[6:].strip()
                            elif line.startswith("data:"):
                                value = json.loads(line[5:])
                                if event == "thinking" and specification.get("disconnect"):
                                    deadline = time.monotonic() + 5
                                    active = None
                                    while time.monotonic() < deadline:
                                        metrics = httpx.get(
                                            "http://127.0.0.1:8091/metrics", timeout=3
                                        ).text
                                        active = re.search(
                                            r"^llamacpp:requests_processing\s+([\d.]+)",
                                            metrics,
                                            re.M,
                                        )
                                        if active and float(active[1]) > 0:
                                            break
                                        gevent.sleep(0.05)
                                    row["llama_processing_before_close"] = (
                                        float(active[1]) if active else None
                                    )
                                    row.update(
                                        outcome="disconnected",
                                        expected_disconnect=True,
                                        cancelled_phase="thinking",
                                    )
                                    response.close()
                                    response.success()
                                    break
                                if event == "delta" and value.get("text"):
                                    if row["first_text_seconds"] is None:
                                        row["first_text_seconds"] = time.monotonic() - started
                                    if specification.get("disconnect"):
                                        row.update(outcome="disconnected", expected_disconnect=True)
                                        response.close()
                                        response.success()
                                        break
                                elif event == "final":
                                    row["outcome"] = "abstained" if value["abstained"] else "ok"
                                    response.success()
                                elif event == "error":
                                    row["outcome"] = value.get("code", "error")
                                    response.failure(row["outcome"])
                    else:
                        response.failure("error")
                except Exception as exc:
                    row["outcome"] = (
                        "timeout"
                        if "timeout" in type(exc).__name__.casefold()
                        or "timed out" in str(exc).casefold()
                        else "error"
                    )
                    response.failure(row["outcome"])
                row["elapsed"] = time.monotonic() - started
                response.request_meta["response_time"] = row["elapsed"] * 1000
            self.environment.rows.append(row)
        raise StopUser()


def run_case(name, users, requests, group, distinct=False, disconnect=False, monitor=None):
    began = time.monotonic()
    environment = Environment(user_classes=[DemoUser])
    environment.rows = []
    environment.numbers = itertools.count(1)
    environment.question = dev_queries()[0]["question"]
    environment.specification = {
        "requests": requests,
        "group": group,
        "distinct": distinct,
        "disconnect": disconnect,
    }
    runner = environment.create_local_runner()
    runner.start(users, spawn_rate=users * 100)
    runner.spawning_greenlet.join()
    deadline = time.monotonic() + 1200
    while runner.user_count and time.monotonic() < deadline:
        gevent.sleep(0.1)
    runner.quit()
    rows = environment.rows
    successful = [r for r in rows if r["outcome"] in {"ok", "abstained"}]
    ended = time.monotonic()
    result = {
        "case": name,
        "offered": users * requests,
        "recorded": len(rows),
        "completed_final": len(successful),
        "http_completed": sum(bool(r["http_status"]) for r in rows),
        "http_429_busy": sum(r["outcome"] == "busy" for r in rows),
        "http_429_rate": sum(r["outcome"] == "rate" for r in rows),
        "timeouts": sum(r["outcome"] == "timeout" for r in rows),
        "errors": sum(r["outcome"] == "error" for r in rows),
        "expected_disconnects": sum(r["expected_disconnect"] for r in rows),
        "e2e_seconds_all_offered": latency_summary([r["elapsed"] for r in rows]),
        "e2e_seconds_successes": latency_summary([r["elapsed"] for r in successful]),
        "locust_requests": environment.stats.total.num_requests,
        "locust_failures": environment.stats.total.num_failures,
        "request_start_span_seconds": max(r["started_monotonic"] for r in rows)
        - min(r["started_monotonic"] for r in rows),
        "rows": rows,
    }
    if monitor:
        samples = [s for s in monitor.samples if began <= s["time"] <= ended]
        result["memory"] = summarize_samples(samples) if samples else {"samples": 0}
    save(OUT / "load" / f"{name}.json", result)
    summary("load", name, {k: v for k, v in result.items() if k != "rows"})
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
    return result


def main():
    if (OUT / "load/result.json").exists():
        raise RuntimeError("Completed Locust measurements must not be repeated")
    monitor = Sampler()
    monitor.start()
    results = []
    try:
        for specification in [
            ("baseline", 1, 10, 10, False, False),
            ("burst12", 12, 1, 11, True, False),
            ("same_client10", 10, 1, 12, False, False),
            ("recovery", 1, 1, 13, False, False),
            ("disconnect", 1, 1, 14, False, True),
        ]:
            results.append(run_case(*specification, monitor=monitor))
        released = time.monotonic()
        processing = None
        with httpx.Client(timeout=3) as client:
            for _ in range(200):
                capacity = client.get("http://127.0.0.1:8000/capacity").json()
                metrics = client.get("http://127.0.0.1:8091/metrics").text
                active = re.search(r"^llamacpp:requests_processing\s+([\d.]+)", metrics, re.M)
                processing = float(active[1]) if active else None
                if not capacity["active"] and processing == 0:
                    break
                gevent.sleep(0.025)
        release_seconds = time.monotonic() - released
        results.append(run_case("after_disconnect", 1, 1, 15, monitor=monitor))
        idle = httpx.get("http://127.0.0.1:8000/capacity", timeout=3).json()
        result = {
            "cases": [{k: v for k, v in r.items() if k != "rows"} for r in results],
            "disconnect_slot_release_seconds": release_seconds,
            "cancellation": {
                "gateway_active_before_next_request": capacity["active"],
                "gateway_waiting_before_next_request": capacity["waiting"],
                "llama_processing_before_next_request": processing,
                "passed": not capacity["active"] and capacity["waiting"] == 0 and processing == 0,
            },
            "gateway_after_final_request": idle,
            **monitor.finish(),
        }
        save(OUT / "load/samples.json", monitor.samples)
        save(OUT / "load/result.json", result)
        summary("load", "aggregate", result)
        mark_step(8, "8.4", "DONE", artifacts=["data/analysis/stage2/load/result.json"])
    finally:
        if monitor.thread.is_alive():
            monitor.finish()


if __name__ == "__main__":
    main()
