import json
import os
import signal
import subprocess
import sys
import time

import httpx

from ledgerlight.common import ROOT, database_url, path

RUNTIME = path(".runtime")
RUNTIME.mkdir(exist_ok=True)
path("logs").mkdir(exist_ok=True)


def start(name, command, extra_env=None):
    pid_file = RUNTIME / f"{name}.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text())
            os.kill(pid, 0)
            if path(f"/proc/{pid}/stat").read_text().split()[2] != "Z":
                return
        except ProcessLookupError:
            pass
    env = os.environ.copy()
    env.update(extra_env or {})
    log = path(f"logs/{name}.log").open("a")
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True
    )
    pid_file.write_text(str(process.pid))
    print(json.dumps({"service": name, "pid": process.pid}))


def stop(name):
    pid_file = RUNTIME / f"{name}.pid"
    if not pid_file.exists():
        return
    pid = int(pid_file.read_text())
    try:
        if not str(os.readlink(f"/proc/{pid}/cwd")).startswith(str(ROOT)):
            raise RuntimeError("Refusing to stop an unowned process")
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def wait(url, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                print(json.dumps({"ready": url}))
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"Not ready: {url}")


def tracking():
    start(
        "mlflow",
        [
            str(path(".venv/bin/mlflow")),
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
            "--workers",
            "2",
            "--backend-store-uri",
            database_url(),
            "--serve-artifacts",
            "--artifacts-destination",
            str(path("mlruns/artifacts")),
            "--allowed-hosts",
            "127.0.0.1:5000,localhost:5000,127.0.0.1,localhost",
        ],
        {"MLFLOW_BACKEND_STORE_URI": database_url()},
    )
    wait("http://127.0.0.1:5000/health")


if __name__ == "__main__":
    {"tracking": tracking}[sys.argv[1]]()
