import argparse
import os
import re
import signal
import subprocess
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


def serve_llama(command):
    process = subprocess.Popen(
        command,
        env=os.environ,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def forward(number, frame):
        try:
            os.killpg(process.pid, number)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    loading = True
    for line in process.stdout:
        if loading:
            match = re.search(r"\bn_layer\s*=\s*(\d+)", line)
            if match:
                print("llama.cpp load: n_layer=" + match[1], flush=True)
            match = re.search(r"offloaded (\d+)/(\d+) layers", line)
            if match:
                print("llama.cpp load: offloaded=" + match[1] + "/" + match[2], flush=True)
            if "model loaded" in line:
                print("llama.cpp ready", flush=True)
                loading = False
    code = process.wait()
    print("llama.cpp exited: " + str(code), flush=True)
    return code if code >= 0 else 128 - code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["mlflow", "llama", "retrieval"])
    kind = parser.parse_args().kind
    bind = os.environ.get("LEDGERLIGHT_BIND_ADDRESS", "127.0.0.1")
    if kind == "mlflow":
        os.environ["MLFLOW_BACKEND_STORE_URI"] = database_url()
        args = [
            "mlflow",
            "server",
            "--host",
            bind,
            "--port",
            "5000",
            "--workers",
            "2",
            "--serve-artifacts",
            "--artifacts-destination",
            str(path("mlruns/artifacts")),
            "--allowed-hosts",
            "127.0.0.1:5000,localhost:5000,127.0.0.1,localhost,mlflow:5000,mlflow",
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
            bind,
            "--port",
            "8091",
            "--ctx-size",
            "8192",
            "--cache-ram",
            "2048",
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
            "-lv",
            "4",
        ]
    elif kind == "retrieval":
        deadline = time.monotonic() + 180
        while True:
            try:
                response = httpx.get(
                    os.environ.get("LEDGERLIGHT_QDRANT_URL", "http://127.0.0.1:6333") + "/readyz",
                    timeout=2,
                )
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
            bind,
            "--port",
            "8001",
        ]
    else:
        raise SystemExit("Unknown container service")
    if kind == "mlflow":
        raise SystemExit(supervise(args))
    if kind == "llama":
        raise SystemExit(serve_llama(args))
    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()
