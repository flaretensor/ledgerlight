import json
import os
import signal
import subprocess
import sys
import time

import httpx

from ledgerlight.common import ROOT, database_url, digest, path, read_json

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
        except (ProcessLookupError, FileNotFoundError):
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
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                if path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
                    break
                os.kill(pid, 0)
            except (ProcessLookupError, FileNotFoundError):
                break
            time.sleep(0.1)
    except (ProcessLookupError, FileNotFoundError):
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
            "--serve-artifacts",
            "--artifacts-destination",
            str(path("mlruns/artifacts")),
            "--allowed-hosts",
            "127.0.0.1:5000,localhost:5000,127.0.0.1,localhost",
        ],
        {"MLFLOW_BACKEND_STORE_URI": database_url()},
    )
    wait("http://127.0.0.1:5000/health")


def binary(component, name):
    matches = [
        p for p in path(".tools/" + component).rglob(name) if p.is_file() and os.access(p, os.X_OK)
    ]
    if not matches:
        raise FileNotFoundError(f"Run python scripts/fetch_services.py {component}")
    return str(matches[0])


def postgres():
    import psycopg
    from psycopg import sql

    bindir = path(".tools/postgres/usr/lib/postgresql/18/bin")
    extra = str(path(".tools/postgres/usr/lib/x86_64-linux-gnu"))
    env = {**os.environ, "LD_LIBRARY_PATH": extra + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
    cluster = path("data/postgres")
    if not (cluster / "PG_VERSION").exists():
        password = path(".tmp/pgpass")
        password.parent.mkdir(exist_ok=True)
        password.write_text(os.environ["POSTGRES_PASSWORD"])
        password.chmod(0o600)
        subprocess.run(
            [
                str(bindir / "initdb"),
                "-D",
                str(cluster),
                "-U",
                "ledgerlight",
                "--pwfile=" + str(password),
                "--auth-host=scram-sha-256",
                "--auth-local=trust",
                "-L",
                str(path(".tools/postgres/usr/share/postgresql/18")),
            ],
            env=env,
            check=True,
        )
    status = subprocess.run(
        [str(bindir / "pg_ctl"), "-D", str(cluster), "status"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if status.returncode:
        options = f"-h 127.0.0.1 -p 5433 -k {RUNTIME} -c dynamic_library_path={bindir.parent}/lib"
        subprocess.run(
            [
                str(bindir / "pg_ctl"),
                "-D",
                str(cluster),
                "-l",
                str(path("logs/postgres.log")),
                "-o",
                options,
                "start",
            ],
            env=env,
            check=True,
        )
    with psycopg.connect(
        database_url().rsplit("/", 1)[0] + "/postgres", autocommit=True
    ) as connection:
        if not connection.execute("SELECT 1 FROM pg_database WHERE datname='ledgerlight'").fetchone():
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier("ledgerlight")))


def qdrant():
    start(
        "qdrant",
        [binary("qdrant", "qdrant"), "--config-path", str(path("config/qdrant.yaml"))],
        {"QDRANT__TELEMETRY_DISABLED": "true"},
    )
    wait("http://127.0.0.1:6333/readyz")


def cuda_libraries():
    root = path(".venv/lib/python3.12/site-packages/nvidia")
    return (
        ":".join(str(p) for p in root.glob("*/lib"))
        + ":"
        + str(path(".venv/lib/python3.12/site-packages/torch/lib"))
    )


def llama():
    pin = read_json("services.lock.json")["qwen"]
    model = path("models/qwen") / pin["name"]
    if digest(model) != pin["digest"].split(":")[-1]:
        raise ValueError("Local Qwen checksum mismatch")
    start(
        "llama",
        [
            binary("llama", "llama-server"),
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
            "--threads",
            "8",
            "--threads-batch",
            "8",
            "--metrics",
            "--reasoning",
            "on",
            "--reasoning-format",
            "deepseek",
            "--reasoning-budget",
            "512",
        ],
        {"LD_LIBRARY_PATH": cuda_libraries() + ":" + os.environ.get("LD_LIBRARY_PATH", "")},
    )
    wait("http://127.0.0.1:8091/health")


def app_services():
    python = str(path(".venv/bin/python"))
    start(
        "retrieval",
        [
            python,
            "-m",
            "uvicorn",
            "ledgerlight.retrieval_api:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8001",
        ],
    )
    wait("http://127.0.0.1:8001/health", timeout=180)
    llama()
    start(
        "gateway",
        [
            python,
            "-m",
            "uvicorn",
            "ledgerlight.gateway:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
    )
    wait("http://127.0.0.1:8000/health")
    start(
        "ui",
        [
            python,
            "-m",
            "streamlit",
            "run",
            "src/ledgerlight/ui.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501",
            "--browser.gatherUsageStats",
            "false",
        ],
    )
    wait("http://127.0.0.1:8501/_stcore/health")


def observability():
    start(
        "prometheus",
        [
            binary("prometheus", "prometheus"),
            "--config.file=" + str(path("config/prometheus.yaml")),
            "--storage.tsdb.path=" + str(path("data/prometheus")),
            "--storage.tsdb.retention.time=7d",
            "--web.listen-address=127.0.0.1:9090",
        ],
    )
    wait("http://127.0.0.1:9090/-/ready")
    home = next(p.parent for p in path(".tools/grafana").rglob("bin") if (p / "grafana").exists())
    start(
        "grafana",
        [
            str(home / "bin/grafana"),
            "server",
            "--homepath",
            str(home),
            "--config",
            str(path("config/grafana.ini")),
        ],
        {
            "GF_SECURITY_ADMIN_USER": "ledgerlight",
            "GF_SECURITY_ADMIN_PASSWORD": os.environ["GRAFANA_PASSWORD"],
            "GF_PATHS_DATA": str(path("data/grafana")),
            "GF_PATHS_LOGS": str(path("logs/grafana")),
            "GF_PATHS_PLUGINS": str(path(".cache/grafana-plugins")),
            "GF_PATHS_PROVISIONING": str(path("config/grafana/provisioning")),
        },
    )
    wait("http://127.0.0.1:3000/api/health")


def start_all():
    postgres()
    tracking()
    qdrant()
    from ledgerlight.vector_store import upload

    upload()
    app_services()


def stop_all():
    for name in [
        "ui",
        "gateway",
        "retrieval",
        "llama",
        "grafana",
        "prometheus",
        "mlflow",
        "qdrant",
        "triton",
    ]:
        stop(name)
    bindir = path(".tools/postgres/usr/lib/postgresql/18/bin")
    env = {**os.environ, "LD_LIBRARY_PATH": str(path(".tools/postgres/usr/lib/x86_64-linux-gnu"))}
    subprocess.run(
        [str(bindir / "pg_ctl"), "-D", str(path("data/postgres")), "stop", "-m", "fast"],
        env=env,
        check=False,
    )


if __name__ == "__main__":
    {
        "tracking": tracking,
        "postgres": postgres,
        "qdrant": qdrant,
        "llama": llama,
        "app": app_services,
        "start": start_all,
        "stop": stop_all,
        "observability": observability,
    }[sys.argv[1]]()
