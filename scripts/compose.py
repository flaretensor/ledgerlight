import subprocess
import sys
import time

import httpx

from ledgerlight.common import ROOT, path

action = sys.argv[1]
if action == "start":
    subprocess.run(["bash", "scripts/storage.sh", "runtime-start"], cwd=ROOT, check=True)
    path(".tmp").mkdir(exist_ok=True)
    subprocess.run(["docker", "compose", "up", "-d", "--build"], cwd=ROOT, check=True)
    deadline = time.monotonic() + 300
    for url in ["http://127.0.0.1:8000/health", "http://127.0.0.1:8501/_stcore/health"]:
        while True:
            try:
                response = httpx.get(url, timeout=20)
                response.raise_for_status()
                break
            except httpx.HTTPError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Compose startup timed out; inspect docker compose logs"
                    ) from None
                time.sleep(1)
    print("Ledgerlight ready at http://127.0.0.1:8501")
elif action == "stop":
    command = [
        "docker",
        "compose",
        "-f",
        "compose.yaml",
        "-f",
        "compose.triton.yaml",
        "--profile",
        "triton",
    ]
    for services in [
        [
            "ui",
            "gateway",
            "llama",
            "retrieval",
            "triton",
            "qdrant",
            "prometheus",
            "grafana",
            "mlflow",
        ],
        ["postgres"],
    ]:
        identifiers = subprocess.check_output(
            command + ["ps", "--status", "running", "-q", *services], cwd=ROOT, text=True
        ).split()
        if identifiers:
            subprocess.run(["docker", "kill", "--signal", "SIGTERM", *identifiers], check=True)
            subprocess.run(["docker", "wait", *identifiers], check=True, timeout=60)
else:
    raise SystemExit("Expected start or stop")
