import os
import subprocess
import sys
import time

import httpx
import yaml

from ledgerlight.common import ROOT, namespace


def command():
    config = {"services": {}}
    for service in ["mlflow", "retrieval", "llama", "gateway", "ui"]:
        config["services"][service] = {
            "image": "ledgerlight-rebuild:local",
            "environment": {
                "LEDGERLIGHT_RUN_MODE": "canonical",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "LEDGERLIGHT_BACKEND": "torch",
            },
        }
    override = ROOT / ".tmp" / (namespace() + ".compose.yaml")
    override.parent.mkdir(exist_ok=True)
    override.write_text(yaml.safe_dump(config, sort_keys=False))
    return [
        "docker",
        "compose",
        "--project-directory",
        str(ROOT),
        "-p",
        namespace(),
        "-f",
        str(ROOT / "compose.yaml"),
        "-f",
        str(override),
    ]


def main(action):
    cmd = command()
    if action == "stop":
        subprocess.run(
            cmd
            + [
                "stop",
                "ui",
                "gateway",
                "retrieval",
                "llama",
                "grafana",
                "prometheus",
                "qdrant",
                "mlflow",
            ],
            check=True,
        )
        subprocess.run(cmd + ["stop", "postgres"], check=True)
        return
    services = ["postgres", "mlflow"] if action == "tracking" else []
    subprocess.run(cmd + ["up", "-d", "--no-build", "--pull", "never", *services], check=True)
    urls = (
        ["http://127.0.0.1:5000/health"]
        if services
        else [
            "http://127.0.0.1:5000/health",
            "http://127.0.0.1:6333/readyz",
            "http://127.0.0.1:8001/health",
            "http://127.0.0.1:8091/health",
            "http://127.0.0.1:8000/health",
            "http://127.0.0.1:8501/_stcore/health",
            "http://127.0.0.1:9090/-/ready",
            "http://127.0.0.1:3000/api/health",
        ]
    )
    for url in urls:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            try:
                httpx.get(url, timeout=4).raise_for_status()
                break
            except httpx.HTTPError:
                time.sleep(1)
        else:
            raise RuntimeError("Service did not become ready: " + url)
    print("Ready: " + namespace(), flush=True)


if __name__ == "__main__":
    if (
        len(sys.argv) > 1
        and sys.argv[1] in {"start", "stop", "status"}
    ):
        os.execv(sys.executable, [sys.executable, str(ROOT / "scripts/site.py"), *sys.argv[1:]])
    if len(sys.argv) != 2 or sys.argv[1] not in {"tracking", "start", "stop"}:
        raise SystemExit("Expected tracking, start or stop")
    main(sys.argv[1])
