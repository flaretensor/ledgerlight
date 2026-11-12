import asyncio
import json
import subprocess
import time

import httpx
import numpy as np
from transformers import AutoTokenizer

from ledgerlight.common import params, path, read_json, read_jsonl, write_json
from ledgerlight.evaluation import tracking
from ledgerlight.onnx_backend import session

URL = "http://127.0.0.1:8002"


def payload(arrays):
    return {
        "inputs": [
            {
                "name": name,
                "shape": list(value.shape),
                "datatype": "INT64",
                "data": value.reshape(-1).tolist(),
            }
            for name, value in arrays.items()
        ],
        "outputs": [{"name": "output"}],
    }


def result(response):
    response.raise_for_status()
    value = response.json()["outputs"][0]
    return np.asarray(value["data"], dtype=np.float32).reshape(value["shape"])


async def concurrent(name, body):
    async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=32)) as client:

        async def one():
            started = time.perf_counter()
            response = await client.post(URL + f"/v2/models/{name}/infer", json=body)
            value = result(response)
            return time.perf_counter() - started, value

        return await asyncio.gather(*(one() for _ in range(32)))


configuration = read_json("models/production/config.json")
sample = next(
    w["text"] for w in read_jsonl("indices/base/windows.jsonl") if len(w["input_ids"]) == 510
)
queries = {q["id"]: q["question"] for q in read_jsonl("data/processed/queries.jsonl")}
question = queries[read_json("data/processed/splits.json")["dev"][0]]
rows, coalescing = [], []
with httpx.Client(timeout=120) as client:
    health = client.get(URL + "/v2/health/ready")
    health.raise_for_status()
    for name, key in [("encoder", "encoder"), ("cross_encoder", "cross_encoder")]:
        source = configuration["encoder_path" if key == "encoder" else "cross_encoder_path"]
        tokenizer = AutoTokenizer.from_pretrained(
            str(path(source)) if source else params()[key]["name"],
            revision=None if source else params()[key]["revision"],
            local_files_only=True,
        )
        reference = session(path(f"models/onnx/{name}/model.onnx"), "CPUExecutionProvider")
        model_config = client.get(URL + f"/v2/models/{name}/config")
        model_config.raise_for_status()
        assert model_config.json()["max_batch_size"] == 8
        assert model_config.json().get("dynamic_batching")
        for batch in [1, 8]:
            tokens = (
                tokenizer(
                    [sample] * batch,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    return_tensors="np",
                )
                if name == "encoder"
                else tokenizer(
                    [question] * batch,
                    [sample] * batch,
                    padding="max_length",
                    max_length=512,
                    truncation="only_second",
                    return_tensors="np",
                )
            )
            arrays = {
                key: tokens[key].astype(np.int64)
                for key in ["input_ids", "attention_mask", "token_type_ids"]
            }
            expected = reference.run(None, arrays)[0].reshape(-1)
            body = payload(arrays)
            for _ in range(3):
                actual = result(client.post(URL + f"/v2/models/{name}/infer", json=body))
            times = []
            for _ in range(20):
                started = time.perf_counter()
                actual = result(client.post(URL + f"/v2/models/{name}/infer", json=body))
                times.append((time.perf_counter() - started) * 1000)
            error = float(np.max(np.abs(actual.reshape(-1) - expected)))
            assert np.allclose(actual.reshape(-1), expected, atol=1e-4, rtol=1e-3)
            rows.append(
                {
                    "model": name,
                    "batch": batch,
                    "sequence": 512,
                    "p50_ms": float(np.median(times)),
                    "p95_ms": float(np.quantile(times, 0.95)),
                    "max_abs_error_vs_ort_cpu": error,
                    "repeats": 20,
                }
            )
            if batch == 1:
                single_body = body
                single_expected = expected
        before = client.get(URL + f"/v2/models/{name}/stats").json()["model_stats"][0]
        responses = asyncio.run(concurrent(name, single_body))
        after = client.get(URL + f"/v2/models/{name}/stats").json()["model_stats"][0]
        inferences = int(after["inference_count"]) - int(before["inference_count"])
        executions = int(after["execution_count"]) - int(before["execution_count"])
        assert inferences == 32 and 0 < executions < inferences
        assert all(
            np.allclose(value.reshape(-1), single_expected, atol=1e-4, rtol=1e-3)
            for _, value in responses
        )
        coalescing.append(
            {
                "model": name,
                "concurrent_batch1_requests": 32,
                "inferences": inferences,
                "executions": executions,
                "mean_execution_batch": inferences / executions,
                "p50_seconds": float(np.median([seconds for seconds, _ in responses])),
                "p95_seconds": float(np.quantile([seconds for seconds, _ in responses], 0.95)),
                "config": model_config.json(),
                "statistics_after": after,
            }
        )
memory = float(
    subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    ).strip()
)
report = {
    "runtime_verified": True,
    "backend": "onnxruntime_cuda",
    "benchmarks": rows,
    "dynamic_batching": coalescing,
    "vram_after_mib": memory,
    "frozen_configuration": configuration,
    "experiment_split": "dev_only_post_freeze_backend_validation",
    "protocol": "Isolated Triton GPU container; llama and retrieval stopped. 3 warmups, 20 timed HTTP round trips at batch 1/8, tokenization excluded; 32 concurrent batch-1 requests confirm coalescing through model statistics. Not comparable to model-only latency without HTTP overhead.",
    "statistics_documentation": "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_statistics.html",
}
write_json("data/serving/triton.json", report)
tracking(
    "triton-dev-backend",
    {
        "vram_after_mib": memory,
        **{row["model"] + "_mean_batch": row["mean_execution_batch"] for row in coalescing},
    },
    {"backend": "onnxruntime", "split": "dev", "queue_delay_us": 2000},
    report,
)
print(json.dumps(report, indent=2))
