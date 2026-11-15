import argparse
import json
import os
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

from ledgerlight.common import ROOT, read_json

PORTS = [8501, 8000, 8001, 8091, 6333, 6334, 5433, 5000, 9090, 3000, 8002, 8003, 8004]
PROJECT = os.environ.get("LEDGERLIGHT_COMPOSE_PROJECT", "ledgerlight")
ACTIVE = ROOT / "data/runtime/active.json"
OVERRIDE = ROOT / "data/runtime/active.compose.yaml"
TOKEN = ROOT / ".secrets/cloudflared/token"


def public_token():
    if not TOKEN.is_file() or TOKEN.is_symlink() or TOKEN.stat().st_size == 0:
        raise ValueError("Public tunnel token is absent or empty: .secrets/cloudflared/token")
    if (
        stat.S_IMODE(TOKEN.parent.stat().st_mode) != 0o700
        or stat.S_IMODE(TOKEN.stat().st_mode) != 0o600
    ):
        raise ValueError("Tunnel token requires directory mode 0700 and file mode 0600")
    return TOKEN.stat()


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def container_ids(running=False):
    return subprocess.check_output(
        [
            "docker",
            "ps",
            "-q" if running else "-aq",
            "--filter",
            "label=com.docker.compose.project=" + PROJECT,
        ],
        text=True,
    ).split()


def inspect_containers():
    ids = container_ids()
    return json.loads(subprocess.check_output(["docker", "inspect", *ids])) if ids else []


def project_pids():
    pids = set()
    for identifier in container_ids(True):
        result = subprocess.run(
            ["docker", "top", identifier, "-eo", "pid"], capture_output=True, text=True
        )
        pids.update(
            int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()
        )
    return pids


def gpu_processes(pids):
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [s.strip() for s in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) in pids:
            rows.append({"pid": int(fields[0]), "mib": int(fields[1])})
    return rows


def compose(config=None):
    cmd = [
        "docker",
        "compose",
        "--project-directory",
        str(ROOT),
        "-p",
        PROJECT,
        "-f",
        str(ROOT / "compose.yaml"),
    ]
    if OVERRIDE.exists():
        cmd += ["-f", str(OVERRIDE)]
    if config and config["triton"]:
        cmd += ["--profile", "triton"]
    if config and config.get("public"):
        cmd += ["--profile", "public"]
    return cmd


def vram_requirement(config, settings):
    key = config["search"] + ("-triton" if config["triton"] else "")
    table = settings["vram_mib"].get(key, {})
    layers = str(config["gpu_layers"])
    if layers in table:
        return float(table[layers]), False
    total = settings.get("total_layers")
    if layers.isdigit() and total and "0" in table and "all" in table:
        points = sorted(
            (total if key == "all" else int(key), float(value)) for key, value in table.items()
        )
        requested = min(int(layers), total)
        for (lower, low_value), (upper, high_value) in zip(points, points[1:]):
            if lower <= requested <= upper:
                return max(low_value, high_value), True
    return None, True


def configure(args):
    settings = read_json("config/site.json")
    config = {key: settings[key] for key in ["search", "triton", "gpu_layers", "llm_threads"]}
    config["llm_threads"] = int(os.environ.get("LEDGERLIGHT_LLM_THREADS", config["llm_threads"]))
    for key in ["search", "triton", "gpu_layers"]:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.cpu:
        config.update(search="openvino", triton=False, gpu_layers="0")
    layers = str(config["gpu_layers"])
    if layers != "all" and (not layers.isdigit() or int(layers) < 0):
        raise ValueError("GPU layers must be all or a nonnegative integer")
    if config["triton"] and config["search"] == "torch":
        raise ValueError("Triton requires --search ort or --search openvino")
    config.update(gpu_layers=layers, public=args.public)
    token_stat = public_token() if args.public else None
    required = [
        "models/encoder/trial-2/model.safetensors",
        "models/cross_encoder/trial-3/model.safetensors",
        "models/ltr/trial-37.cbm",
        "models/qwen/" + read_json("services.lock.json")["qwen"]["name"],
        "data/processed/corpus.jsonl",
        "indices/base/windows.jsonl",
        "indices/trained/vectors.npy",
        "data/registry/versions.json",
        "data/postgres/PG_VERSION",
    ]
    if config["search"] != "torch":
        required += [
            f"data/runtime/onnx/{role}/2/model.onnx" for role in ["encoder", "cross_encoder"]
        ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise ValueError("Missing runtime artifacts: " + ", ".join(missing))
    peak, estimated = vram_requirement(config, settings)
    config.update(peak_vram_mib=peak, vram_estimate=estimated)
    free = int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
        ).splitlines()[0]
    )
    cpu_only = config["search"] == "openvino" and layers == "0"
    if not cpu_only and not args.force and (peak is None or free < peak + 150):
        raise ValueError(
            f"VRAM fit unverified or insufficient: free={free} MiB, peak={peak}; use --force to skip"
        )
    services = {}
    if token_stat is not None:
        services["tunnel"] = {"user": f"{token_stat.st_uid}:{token_stat.st_gid}"}
        protocol = settings.get("tunnel_protocol")
        if protocol:
            if protocol not in {"auto", "quic", "http2"}:
                raise ValueError("Tunnel protocol must be auto, quic, or http2")
            services["tunnel"]["command"] = [
                "tunnel",
                "--no-autoupdate",
                "run",
                "--protocol",
                protocol,
                "--token-file",
                "/run/secrets/tunnel-token",
            ]
    for name in ["retrieval", "llama", "triton"]:
        use_gpu = (
            name == "retrieval"
            and not config["triton"]
            and config["search"] in {"torch", "ort"}
            or name == "llama"
            and layers != "0"
            or name == "triton"
            and config["triton"]
            and config["search"] == "ort"
        )
        services[name] = {"environment": {"NVIDIA_VISIBLE_DEVICES": "all" if use_gpu else "void"}}
        if use_gpu:
            services[name]["gpus"] = "all"
    services["retrieval"]["environment"].update(
        LEDGERLIGHT_BACKEND=config["search"],
        LEDGERLIGHT_TRITON="1" if config["triton"] else "0",
        LEDGERLIGHT_DEVICE="cuda"
        if config["search"] in {"torch", "ort"} and not config["triton"]
        else "cpu",
    )
    services["llama"]["environment"].update(
        LEDGERLIGHT_GPU_LAYERS=layers, LEDGERLIGHT_LLM_THREADS=str(config["llm_threads"])
    )
    services["triton"]["command"] = [
        "tritonserver",
        "--model-repository=/serving/triton/" + config["search"],
        "--model-control-mode=explicit",
        "--load-model=encoder_2",
        "--load-model=cross_encoder_2",
        "--http-port=8000",
        "--grpc-port=8001",
        "--metrics-port=8002",
        "--allow-gpu-metrics=false",
        "--cuda-memory-pool-byte-size=0:67108864",
        "--pinned-memory-pool-byte-size=67108864",
    ]
    if config["triton"]:
        services["prometheus"] = {
            "volumes": ["./config/prometheus.triton.yaml:/etc/prometheus/prometheus.yml:ro"]
        }
    OVERRIDE.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDE.write_text(yaml.safe_dump({"services": services}, sort_keys=False))
    ACTIVE.write_text(json.dumps(config, indent=2) + "\n")
    return config


def healthy(url, deadline):
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=5)
            response.raise_for_status()
            return
        except httpx.HTTPError:
            time.sleep(0.5)
    raise RuntimeError("Service health deadline exceeded: " + url)


def start(args):
    if args.public:
        public_token()
    if container_ids(True):
        stop()
    config = configure(args)
    clock = time.monotonic()
    cmd = compose(config)
    if config["triton"]:
        run(cmd + ["up", "-d", "--no-build", "--pull", "never", "triton"])
        healthy("http://127.0.0.1:8002/v2/health/ready", time.monotonic() + 180)
    run(cmd + ["up", "-d", "--no-build", "--pull", "never"], timeout=600)
    for port, endpoint in [
        (8000, "/health"),
        (8501, "/_stcore/health"),
        (9090, "/-/ready"),
        (3000, "/api/health"),
    ]:
        healthy(f"http://127.0.0.1:{port}{endpoint}", time.monotonic() + 120)
    config["cold_start_seconds"] = time.monotonic() - clock
    ACTIVE.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"url": "http://127.0.0.1:8501", "configuration": config}, indent=2))
    if str(config["gpu_layers"]) == "0" and config["search"] == "openvino" and not config["triton"]:
        processes = gpu_processes(project_pids())
        if processes:
            raise RuntimeError("CPU configuration still owns GPU memory")
        print("PASS: stack GPU memory is 0 MiB")


def stop():
    (ROOT / "data/runtime").mkdir(parents=True, exist_ok=True)
    pids = project_pids()
    containers = inspect_containers()
    running = [c for c in containers if c["State"]["Running"]]
    for group in [
        [
            c["Id"]
            for c in running
            if c["Config"]["Labels"].get("com.docker.compose.service") != "postgres"
        ],
        [
            c["Id"]
            for c in running
            if c["Config"]["Labels"].get("com.docker.compose.service") == "postgres"
        ],
    ]:
        if group:
            run(["docker", "stop", "--time", "30", *group], timeout=90)
    tunnel_ids = [
        c["Id"]
        for c in containers
        if c["Config"]["Labels"].get("com.docker.compose.service") == "tunnel"
    ]
    if tunnel_ids:
        run(["docker", "rm", *tunnel_ids])
    listening = []
    for port in PORTS:
        with socket.socket() as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                listening.append(port)
    tunnels = []
    for directory in Path("/proc").glob("[0-9]*"):
        try:
            command = (directory / "cmdline").read_bytes()
            executable = Path(os.fsdecode(command.split(b"\0")[0])).name
            if executable == "cloudflared" and (
                str(ROOT).encode() in command or b"ledgerlight.flaretensor.uk" in command
            ):
                tunnels.append(int(directory.name))
        except (OSError, PermissionError):
            pass
    result = {
        "no_running_project_containers": not container_ids(True),
        "no_project_tunnel": not tunnels
        and not any(
            c["Config"]["Labels"].get("com.docker.compose.service") == "tunnel"
            for c in inspect_containers()
        ),
        "no_stack_port_listening": not listening,
        "no_stack_gpu_process": not gpu_processes(pids),
        "listening_ports": listening,
        "project_tunnel_pids": tunnels,
    }
    for key, value in result.items():
        if isinstance(value, bool):
            print(("PASS" if value else "FAIL") + ": " + key)
    (ROOT / "data/runtime/stop.json").write_text(json.dumps(result, indent=2) + "\n")
    if not all(value for value in result.values() if isinstance(value, bool)):
        raise RuntimeError("Stack stop verification failed")


def status():
    rows = []
    for c in inspect_containers():
        rows.append(
            {
                "service": c["Config"]["Labels"].get("com.docker.compose.service"),
                "running": c["State"]["Running"],
                "health": c["State"].get("Health", {}).get("Status"),
            }
        )
    print(
        json.dumps(
            {
                "containers": rows,
                "configuration": json.loads(ACTIVE.read_text()) if ACTIVE.exists() else None,
                "stack_gpu": gpu_processes(project_pids()),
                "tunnel_up": any(r["service"] == "tunnel" and r["running"] for r in rows),
                "tunnel_state": next(
                    (
                        "running" if r["running"] else "stopped"
                        for r in rows
                        if r["service"] == "tunnel"
                    ),
                    "absent",
                ),
                "url": "http://127.0.0.1:8501",
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Manual Ledgerlight lifecycle; no background startup. Example: python scripts/site.py start --cpu"
    )
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("--search", choices=["torch", "ort", "openvino"])
    parser.add_argument("--triton", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gpu-layers")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    if subprocess.run(
        ["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode:
        print("Docker is not running: sudo systemctl start docker")
        return 1
    try:
        {"start": lambda: start(args), "stop": stop, "status": status}[args.action]()
    except (ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
