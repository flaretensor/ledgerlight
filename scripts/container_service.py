import os
import sys
import time
from pathlib import Path

import httpx

from ledgerlight.common import database_url, digest, path, read_json

kind = sys.argv[1]
if kind == "mlflow":
    os.environ["MLFLOW_BACKEND_STORE_URI"] = database_url()
    args = [
        "mlflow",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        "5000",
        "--workers",
        "2",
        "--serve-artifacts",
        "--artifacts-destination",
        str(path("mlruns/artifacts")),
        "--allowed-hosts",
        "127.0.0.1:5000,localhost:5000,127.0.0.1,localhost",
    ]
elif kind == "llama":
    executable = next(p for p in path(".tools/llama").rglob("llama-server") if p.is_file())
    libs = list(Path("/opt/venv/lib/python3.12/site-packages/nvidia").glob("*/lib"))
    libs.append(executable.parent)
    os.environ["LD_LIBRARY_PATH"] = ":".join(map(str, libs))
    pin = read_json("services.lock.json")["qwen"]
    model = path("models/qwen") / pin["name"]
    if digest(model) != pin["digest"].split(":")[-1]:
        raise ValueError("Local Qwen checksum mismatch")
    args = [
        str(executable),
        "--model",
        str(model),
        "--alias",
        "ledgerlight-qwen",
        "--host",
        "127.0.0.1",
        "--port",
        "8091",
        "--ctx-size",
        "8192",
        "--parallel",
        "1",
        "--n-gpu-layers",
        "99",
        "--batch-size",
        "256",
        "--ubatch-size",
        "128",
        "--flash-attn",
        "on",
        "--metrics",
        "--reasoning",
        "on",
        "--reasoning-format",
        "deepseek",
        "--reasoning-budget",
        "512",
    ]
elif kind == "retrieval":
    deadline = time.monotonic() + 180
    while True:
        try:
            response = httpx.get("http://127.0.0.1:6333/readyz", timeout=2)
            response.raise_for_status()
            break
        except httpx.HTTPError:
            if time.monotonic() >= deadline:
                raise RuntimeError("Qdrant did not become ready") from None
            time.sleep(1)
    from ledgerlight.vector_store import upload

    upload()
    args = [
        "python",
        "-m",
        "uvicorn",
        "ledgerlight.retrieval_api:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
    ]
else:
    raise SystemExit("Unknown container service")
os.execvpe(args[0], args, os.environ)
