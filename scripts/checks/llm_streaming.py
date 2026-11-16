import json
import subprocess
import threading
import time

import httpx
import numpy as np

from ledgerlight.citations import issue
from ledgerlight.common import params, read_json, read_jsonl, write_json
from ledgerlight.evaluation import tracking
from ledgerlight.gateway import SYSTEM


def quantiles(values):
    return {"p50": float(np.quantile(values, 0.5)), "p95": float(np.quantile(values, 0.95))}


subprocess.run(["docker", "compose", "restart", "llama"], check=True)
deadline = time.monotonic() + 180
with httpx.Client(timeout=180) as client:
    while True:
        try:
            ready = client.get("http://127.0.0.1:8091/health", timeout=2)
            ready.raise_for_status()
            break
        except httpx.HTTPError:
            if time.monotonic() >= deadline:
                raise RuntimeError("LLM did not become ready") from None
            time.sleep(1)
    queries = {q["id"]: q["question"] for q in read_jsonl("data/processed/queries.jsonl")}
    qid = read_json("data/processed/splits.json")["dev"][0]
    question = queries[qid]
    retrieved = client.post("http://127.0.0.1:8001/search", json={"question": question})
    retrieved.raise_for_status()
    issued = issue(retrieved.json()["results"])
    cfg = params()["serving"]
    excerpts = []
    for marker, item in issued.items():
        sources = "; ".join(
            f"{s['ticker']} / {s['form']} / {s['year']} / page {s['page']}" for s in item["sources"]
        )
        excerpts.append(marker + " " + sources + "\n" + "\n".join(item["windows"]))
    user_message = "Question: " + question + "\n\nSource excerpts:\n\n" + "\n\n".join(excerpts)
    counted = client.post(
        "http://127.0.0.1:8091/tokenize",
        json={"content": SYSTEM + user_message, "add_special": True},
    )
    counted.raise_for_status()
    assert len(counted.json()["tokens"]) + cfg["max_tokens"] + 256 <= cfg["context"]
    body = {
        "model": "ledgerlight-qwen",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": cfg["max_tokens"],
        "temperature": 0.2,
        "top_p": 0.8,
        "seed": 42,
        "cache_prompt": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    stop = threading.Event()
    memory = []

    def sample_memory():
        while not stop.is_set():
            value = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            memory.append(float(value.strip()))
            stop.wait(0.25)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()

    def attempt(thinking):
        started = time.perf_counter()
        first, first_answer, finish, usage, timings = None, None, None, {}, {}
        answer, reasoning = [], []
        request = {**body, "chat_template_kwargs": {"enable_thinking": thinking}}
        with client.stream(
            "POST", "http://127.0.0.1:8091/v1/chat/completions", json=request
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                payload = json.loads(line[5:])
                usage = payload.get("usage") or usage
                timings = payload.get("timings") or timings
                for choice in payload.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    elapsed = time.perf_counter() - started
                    if (content or thought) and first is None:
                        first = elapsed
                    if content and first_answer is None:
                        first_answer = elapsed
                    answer.append(content)
                    reasoning.append(thought)
                    finish = choice.get("finish_reason") or finish
        elapsed = time.perf_counter() - started
        assert first is not None, "Streaming returned no text tokens"
        thoughts = "".join(reasoning)
        tokenized = client.post(
            "http://127.0.0.1:8091/tokenize", json={"content": thoughts, "add_special": False}
        )
        tokenized.raise_for_status()
        thought_tokens = len(tokenized.json()["tokens"])
        if thinking:
            assert thought_tokens <= 512
        return {
            "thinking": thinking,
            "ttft_any_token_seconds": first,
            "first_answer_token_seconds": first_answer,
            "e2e_seconds": elapsed,
            "finish_reason": finish,
            "usage": usage,
            "timings": timings,
            "decode_tokens_per_second": timings.get("predicted_per_second"),
            "reasoning_tokens_retokenized": thought_tokens,
            "answer": "".join(answer),
        }

    rows = []
    try:
        for i in range(6):
            attempts = [attempt(True)]
            if attempts[0]["finish_reason"] == "length":
                attempts.append(attempt(False))
            rows.append(
                {
                    "phase": "cold" if i == 0 else "warm",
                    "attempts": attempts,
                    "retried": len(attempts) == 2,
                    "e2e_seconds": sum(a["e2e_seconds"] for a in attempts),
                }
            )
            print(
                json.dumps(
                    {
                        "phase": rows[-1]["phase"],
                        "completed": len(rows),
                        "seconds": rows[-1]["e2e_seconds"],
                        "retry": rows[-1]["retried"],
                    }
                ),
                flush=True,
            )
    finally:
        stop.set()
        sampler.join()
warm = rows[1:]
decodes = [
    a["decode_tokens_per_second"]
    for row in rows
    for a in row["attempts"]
    if a["decode_tokens_per_second"] is not None
]
result = {
    "query_id": qid,
    "split": "dev",
    "requests": len(rows),
    "cold": rows[0],
    "warm": {
        "requests": len(warm),
        "e2e_seconds": quantiles([r["e2e_seconds"] for r in warm]),
        "ttft_any_token_seconds": quantiles(
            [r["attempts"][0]["ttft_any_token_seconds"] for r in warm]
        ),
        "first_answer_token_seconds": quantiles(
            [
                r["attempts"][0]["first_answer_token_seconds"]
                for r in warm
                if r["attempts"][0]["first_answer_token_seconds"] is not None
            ]
        ),
    },
    "completion_limit_and_retry_rate": sum(r["retried"] for r in rows) / len(rows),
    "decode_tokens_per_second": quantiles(decodes) if decodes else None,
    "vram_peak_mib": max(memory),
    "vram_samples": len(memory),
    "rows": rows,
    "protocol": "LLM-only HTTP streaming on one fixed DEV question; one process-cold request after manual container restart, five warm requests, OS page cache warm, prompt cache disabled. TTFT includes reasoning; first answer token reported separately. Existing gateway truncation retry policy. Not a /ask TTFT measurement.",
}
write_json("data/serving/llm_streaming.json", result)
tracking(
    "llm-dev-streaming",
    {
        "warm_ttft_p50": result["warm"]["ttft_any_token_seconds"]["p50"],
        "warm_e2e_p95": result["warm"]["e2e_seconds"]["p95"],
        "retry_rate": result["completion_limit_and_retry_rate"],
        "vram_peak_mib": result["vram_peak_mib"],
    },
    {"split": "dev", "warm_requests": 5},
    result,
)
print(
    json.dumps(
        {key: value for key, value in result.items() if key not in ["rows", "cold"]}, indent=2
    )
)
