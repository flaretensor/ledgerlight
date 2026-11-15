import argparse
import json
import subprocess
from pathlib import Path

PORTS = {
    "gateway": [8000],
    "retrieval": [8001],
    "llama": [8091],
    "qdrant": [6333, 6334],
    "postgres": [5433],
    "mlflow": [5000],
    "triton": [8000, 8001, 8002],
    "prometheus": [9090],
    "grafana": [3000],
    "forwarder": [8501],
}
PROBE = """import json,os,socket,sys
targets=json.loads(sys.argv[1]);rows=[]
for target in targets:
 with socket.socket() as probe:
  probe.settimeout(0.25)
  try:reachable=probe.connect_ex((target["address"],target["port"]))==0
  except OSError:reachable=False
 rows.append({**target,"reachable":reachable})
dns={}
for name in ["gateway","retrieval","llama","qdrant","postgres","mlflow","triton","prometheus","grafana","forwarder"]:
 try:dns[name]=sorted({v[4][0] for v in socket.getaddrinfo(name,None)})
 except socket.gaierror:dns[name]=[]
try:
 with open("/ledgerlight-isolation-probe","x") as stream:stream.write("probe")
 writable=True;os.unlink("/ledgerlight-isolation-probe")
except OSError:writable=False
caps=next(line.split(":")[1].strip() for line in open("/proc/self/status") if line.startswith("CapEff:"))
print(json.dumps({"tcp":rows,"dns":dns,"uid":os.geteuid(),"effective_caps":caps,"root_writable":writable,"paths":{p:os.path.exists(p) for p in ["/home","/data","/var/run/docker.sock"]},"routes":open("/proc/net/route").read()}))"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", default="ledgerlight")
    args = parser.parse_args()
    ids = subprocess.check_output(
        ["docker", "ps", "-q", "--filter", "label=com.docker.compose.project=" + args.project],
        text=True,
    ).split()
    containers = json.loads(subprocess.check_output(["docker", "inspect", *ids])) if ids else []
    services = {c["Config"]["Labels"]["com.docker.compose.service"]: c for c in containers}
    if not {"ui", "gateway"} <= set(services):
        raise RuntimeError("UI and gateway must be running")
    targets = []
    for service, c in services.items():
        for network, connection in c["NetworkSettings"]["Networks"].items():
            for port in PORTS.get(service, []):
                targets.append(
                    {
                        "label": service + "/" + network,
                        "service": service,
                        "address": connection["IPAddress"],
                        "port": port,
                    }
                )
    networks = json.loads(
        subprocess.check_output(
            [
                "docker",
                "network",
                "inspect",
                *subprocess.check_output(["docker", "network", "ls", "-q"], text=True).split(),
            ]
        )
    )
    for network in networks:
        if network["Driver"] != "bridge":
            continue
        for ipam in network["IPAM"].get("Config", []):
            address = ipam.get("Gateway")
            if not address:
                continue
            for port in [
                22,
                8501,
                8000,
                8001,
                8091,
                6333,
                6334,
                5433,
                5000,
                9090,
                3000,
                8002,
                8003,
                8004,
            ]:
                targets.append(
                    {
                        "label": "host/" + network["Name"],
                        "service": "host",
                        "address": address,
                        "port": port,
                    }
                )
    targets.append({"label": "internet", "service": "internet", "address": "1.1.1.1", "port": 443})
    result = {}
    for name in ["ui", "gateway"]:
        response = subprocess.check_output(
            ["docker", "exec", services[name]["Id"], "python", "-c", PROBE, json.dumps(targets)],
            text=True,
        )
        result[name] = json.loads(response)
    ui = services["ui"]
    host = ui["HostConfig"]
    result["configuration"] = {
        "nonroot": result["ui"]["uid"] != 0,
        "readonly_rootfs": host["ReadonlyRootfs"],
        "cap_drop_all": "ALL" in host.get("CapDrop", []),
        "zero_effective_caps": int(result["ui"]["effective_caps"], 16) == 0,
        "no_new_privileges": any("no-new-privileges" in v for v in host.get("SecurityOpt", [])),
        "no_bind_mounts": not any(m["Type"] == "bind" for m in ui["Mounts"]),
        "not_privileged": not host["Privileged"],
        "no_devices": not host.get("Devices") and not host.get("DeviceRequests"),
        "no_host_network": host["NetworkMode"] != "host",
        "memory_limit": host["Memory"] > 0,
        "pids_limit": bool(host["PidsLimit"] and host["PidsLimit"] > 0),
        "no_secret_environment": not any(
            v.split("=", 1)[0].endswith(("_PASSWORD", "_TOKEN", "_API_KEY"))
            for v in ui["Config"]["Env"]
        ),
        "no_sensitive_paths": not any(result["ui"]["paths"].values()),
        "root_write_denied": not result["ui"]["root_writable"],
    }
    reachable = [r for r in result["ui"]["tcp"] if r["reachable"]]
    result["ui_only_gateway_reachable"] = bool(reachable) and all(
        r["service"] == "gateway" and r["port"] == 8000 for r in reachable
    )
    result["positive_control_backend"] = all(
        any(r["reachable"] and r["service"] == name for r in result["gateway"]["tcp"])
        for name in ["retrieval", "llama", "qdrant", "postgres", "mlflow", "prometheus", "grafana"]
    )
    result["published_localhost_only"] = all(
        binding["HostIp"] == "127.0.0.1"
        for c in containers
        for bindings in c["NetworkSettings"]["Ports"].values()
        for binding in bindings or []
    )
    listeners = subprocess.check_output(["ss", "-H", "-ltn"], text=True).splitlines()
    result["host_wildcard_listeners"] = [
        line.split()[3]
        for line in listeners
        if line.split()[3].startswith(("0.0.0.0:", "[::]:", "*:"))
    ]
    result["host_no_wildcard_listeners"] = not result["host_wildcard_listeners"]
    result["container_boundary_pass"] = (
        all(result["configuration"].values())
        and result["ui_only_gateway_reachable"]
        and result["positive_control_backend"]
        and result["published_localhost_only"]
    )
    result["passed"] = result["container_boundary_pass"] and result["host_no_wildcard_listeners"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"ui", "gateway"}}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
