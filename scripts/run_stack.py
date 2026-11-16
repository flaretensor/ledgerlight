import os
import subprocess
import sys
import time

import httpx
import yaml

from ledgerlight.common import ROOT, namespace


def command():
    config = {"services": {"mlflow": {"volumes": ["./mlruns:/workspace/mlruns"]}}}
    override = ROOT / ".tmp" / (namespace() + ".tracking.compose.yaml")
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


def main():
    subprocess.run(
        command() + ["up", "-d", "--build", "--pull", "never", "postgres", "mlflow"], check=True
    )
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        try:
            httpx.get("http://127.0.0.1:5000/health", timeout=4).raise_for_status()
            break
        except httpx.HTTPError:
            time.sleep(1)
    else:
        raise RuntimeError("Service did not become ready: http://127.0.0.1:5000/health")
    print("Ready: " + namespace(), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"start", "stop", "status"}:
        os.execv(sys.executable, [sys.executable, str(ROOT / "scripts/site.py"), *sys.argv[1:]])
    if len(sys.argv) != 2 or sys.argv[1] != "tracking":
        raise SystemExit("Expected tracking, start, stop or status")
    main()
