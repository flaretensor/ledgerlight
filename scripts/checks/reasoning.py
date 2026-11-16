import json
import time

import httpx

from ledgerlight.common import read_json, write_json
from ledgerlight.evaluation import tracking

lock = read_json("services.lock.json")
prompt = "Analyze at least 100 distinct steps internally before giving a short answer: explain how an analyst should reconcile segment revenue, operating profit, intersegment eliminations and total consolidated revenue in an annual report."
with httpx.Client(timeout=180) as client:
    started = time.perf_counter()
    response = client.post(
        "http://127.0.0.1:8091/v1/chat/completions",
        json={
            "model": "ledgerlight-qwen",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.2,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": True},
            "cache_prompt": False,
        },
    )
    response.raise_for_status()
    payload = response.json()
    message = payload["choices"][0]["message"]
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    counted = client.post(
        "http://127.0.0.1:8091/tokenize", json={"content": reasoning, "add_special": False}
    )
    counted.raise_for_status()
    count = len(counted.json()["tokens"])
    runtime_count = (
        payload.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens")
    )
    result = {
        "build": lock["llama"]["revision"],
        "reasoning_budget_flag": 512,
        "observed_reasoning_tokens_retokenized": count,
        "runtime_reasoning_tokens": runtime_count,
        "finish_reason": payload["choices"][0].get("finish_reason"),
        "completion_tokens": payload.get("usage", {}).get("completion_tokens"),
        "usage": payload.get("usage", {}),
        "seconds": time.perf_counter() - started,
        "reasoning_present": bool(reasoning),
        "cap_observed": bool(reasoning)
        and (runtime_count if runtime_count is not None else count) <= 512,
        "answer_present": bool(message.get("content")),
        "caveat": "Retokenized reasoning excludes template boundary tokens; runtime flag and server log provide enforcement evidence",
    }
write_json("data/serving/reasoning_cap.json", result)
write_json("data/serving/reasoning_response.json", payload)
print(json.dumps(result, indent=2))
if (runtime_count if runtime_count is not None else count) > 512:
    raise SystemExit("Observed reasoning exceeded the requested cap")


tracking(
    "qwen-reasoning-cap",
    {
        "reasoning_tokens": runtime_count if runtime_count is not None else count,
        "cap_observed": int(result["cap_observed"]),
        "seconds": result["seconds"],
    },
    {"budget": 512, "runtime_build": lock["llama"]["revision"]},
    result,
)
