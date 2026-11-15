import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from ledgerlight.common import database_url, digest, path, read_json


def supervise(command):
    process = subprocess.Popen(command, env=os.environ, start_new_session=True)
    stopping = {"requested": False}

    def forward(number, frame):
        stopping["requested"] = True
        try:
            os.killpg(process.pid, number)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    code = process.wait()
    if stopping["requested"]:
        deadline = time.monotonic() + 25
        while True:
            try:
                while os.waitpid(-1, os.WNOHANG)[0]:
                    pass
            except ChildProcessError:
                pass
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                print("MLflow process group stopped cleanly", flush=True)
                return 0
            if time.monotonic() >= deadline:
                raise RuntimeError("MLflow process group did not stop within 25 seconds")
            time.sleep(0.1)
    return code if code >= 0 else 128 - code


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
        os.environ.get("LEDGERLIGHT_GPU_LAYERS", "99").replace("all", "99"),
        "--threads",
        os.environ.get("LEDGERLIGHT_LLM_THREADS", "8"),
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
if kind == "mlflow":
    raise SystemExit(supervise(args))
os.execvpe(args[0], args, os.environ)
