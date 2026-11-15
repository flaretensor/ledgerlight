import argparse
import ipaddress
import json
import re
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
 with socket.socket(socket.AF_INET6 if ":" in target["address"] else socket.AF_INET) as probe:
  probe.settimeout(0.25)
  try:code=probe.connect_ex((target["address"],target["port"]));reachable=code==0
  except OSError as exc:code=exc.errno;reachable=False
 rows.append({**target,"reachable":reachable,"connect_errno":code})
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


def tunnel_probe(project, services, targets):
    configuration = json.loads(
        subprocess.check_output(
            [
                "docker",
                "compose",
                "-p",
                project,
                "--profile",
                "public",
                "config",
                "--format",
                "json",
            ],
            text=True,
        )
    )
    tunnel = configuration["services"]["tunnel"]
    network_keys = set(tunnel["networks"])
    names = {configuration["networks"][n]["name"] for n in network_keys}
    for key in network_keys:
        name = configuration["networks"][key]["name"]
        inspected = subprocess.run(
            ["docker", "network", "inspect", name], capture_output=True, text=True
        )
        if inspected.returncode:
            subprocess.run(
                [
                    "docker",
                    "network",
                    "create",
                    "--label",
                    "com.docker.compose.project=" + project,
                    "--label",
                    "com.docker.compose.network=" + key,
                    name,
                ],
                check=True,
                capture_output=True,
            )
        elif (
            json.loads(inspected.stdout)[0].get("Labels", {}).get("com.docker.compose.project")
            != project
        ):
            raise RuntimeError("Tunnel network belongs to a different project")
    public = configuration["networks"]["public_ui"]["name"]
    egress = configuration["networks"]["tunnel_egress"]["name"]
    address = services["ui"]["NetworkSettings"]["Networks"][public]["IPAddress"]
    probe_targets = [
        *targets,
        {"label": "ui/public_ui", "service": "ui", "address": address, "port": 8501},
    ]
    name = project + "-tunnel-probe"
    identifier = subprocess.check_output(
        [
            "docker",
            "create",
            "--name",
            name,
            "--network",
            public,
            "--user",
            "10001:10001",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--memory",
            "128m",
            "--pids-limit",
            "32",
            "--entrypoint",
            "python",
            services["ui"]["Config"]["Image"],
            "-c",
            PROBE,
            json.dumps(probe_targets),
        ],
        text=True,
    ).strip()
    try:
        subprocess.run(
            ["docker", "network", "connect", egress, identifier], check=True, capture_output=True
        )
        attached = json.loads(
            subprocess.check_output(["docker", "inspect", identifier], text=True)
        )[0]
        response = subprocess.check_output(["docker", "start", "--attach", identifier], text=True)
        result = json.loads(response)
        result["exact_networks"] = set(attached["NetworkSettings"]["Networks"]) == names
    finally:
        subprocess.run(["docker", "rm", "--force", identifier], check=True, capture_output=True)
    mounts = tunnel.get("volumes", [])
    result["configuration"] = {
        "public_profile": "public" in tunnel.get("profiles", []),
        "restart_no": tunnel.get("restart") == "no",
        "readonly": tunnel.get("read_only") is True,
        "cap_drop_all": "ALL" in tunnel.get("cap_drop", []),
        "no_new_privileges": any("no-new-privileges" in x for x in tunnel.get("security_opt", [])),
        "no_ports": not tunnel.get("ports"),
        "limited": bool(tunnel.get("mem_limit") and tunnel.get("pids_limit")),
        "single_readonly_token_mount": len(mounts) == 1
        and mounts[0].get("read_only")
        and mounts[0]["target"] == "/run/secrets/tunnel-token",
        "dedicated_networks": network_keys == {"public_ui", "tunnel_egress"},
        "ui_only_peer": all(
            not (set(c["NetworkSettings"]["Networks"]) & names)
            for service, c in services.items()
            if service != "ui"
        ),
    }
    result["ui_reachable"] = any(r["service"] == "ui" and r["reachable"] for r in result["tcp"])
    result["forbidden_unreachable"] = not any(
        r["reachable"] for r in result["tcp"] if r["service"] not in {"ui", "internet"}
    )
    result["internet_egress_expected"] = any(
        r["service"] == "internet" and r["reachable"] for r in result["tcp"]
    )
    result["passed"] = (
        result["exact_networks"]
        and all(result["configuration"].values())
        and result["ui_reachable"]
        and result["forbidden_unreachable"]
    )
    return result


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
    host_interfaces = json.loads(subprocess.check_output(["ip", "-j", "address", "show"]))
    host_addresses = []
    host_ports = [22, 8501, 8000, 8001, 8091, 6333, 6334, 5433, 5000, 9090, 3000, 8002, 8003, 8004]
    for interface in host_interfaces:
        for address in interface.get("addr_info", []):
            value = address["local"]
            if ipaddress.ip_address(value).is_loopback:
                continue
            host_addresses.append({"interface": interface["ifname"], "address": value})
            for port in host_ports:
                targets.append(
                    {
                        "label": "host-interface/" + interface["ifname"],
                        "service": "host",
                        "address": value,
                        "port": port,
                    }
                )
    result = {}
    result["host_interface_addresses"] = host_addresses
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
    ui_targets = [
        {"label": "ui/" + network, "service": "ui", "address": info["IPAddress"], "port": 8501}
        for network, info in ui["NetworkSettings"]["Networks"].items()
    ]
    reverse = json.loads(
        subprocess.check_output(
            [
                "docker",
                "exec",
                services["gateway"]["Id"],
                "python",
                "-c",
                PROBE,
                json.dumps(ui_targets),
            ],
            text=True,
        )
    )
    result["gateway_to_ui"] = reverse["tcp"]
    result["gateway_cannot_reach_ui"] = not any(r["reachable"] for r in reverse["tcp"])
    result["published_localhost_only"] = all(
        binding["HostIp"] == "127.0.0.1"
        for c in containers
        for bindings in c["NetworkSettings"]["Ports"].values()
        for binding in bindings or []
    )
    listeners = subprocess.check_output(["ss", "-H", "-ltnp"], text=True).splitlines()
    project_pids = set()
    for container in containers:
        top = subprocess.run(
            ["docker", "top", container["Id"], "-eo", "pid"], capture_output=True, text=True
        )
        project_pids.update(int(v) for v in top.stdout.split() if v.isdigit())
    stack_ports = set(host_ports) - {22}
    result["host_stack_listeners"] = []
    result["host_nonstack_listeners_info"] = []
    for line in listeners:
        endpoint = line.split()[3]
        address, port = endpoint.rsplit(":", 1)
        address = address.strip("[]")
        processes = re.findall(r'\("([^"]+)",pid=(\d+)', line)
        owned = int(port) in stack_ports or any(int(pid) in project_pids for _, pid in processes)
        row = {
            "address": address,
            "port": int(port),
            "bind": "wildcard" if address in {"0.0.0.0", "::", "*"} else "specific",
            "processes": [name for name, _ in processes]
            or (
                ["sshd (user attribution; owner PID not exposed)"]
                if int(port) == 22
                else ["owner not exposed to this uid"]
            ),
        }
        result["host_stack_listeners" if owned else "host_nonstack_listeners_info"].append(row)
    result["stack_owned_listeners_localhost_only"] = all(
        row["address"] == "127.0.0.1" for row in result["host_stack_listeners"]
    )
    result["ui_cannot_reach_host"] = not any(
        row["reachable"] for row in result["ui"]["tcp"] if row["service"] == "host"
    )
    result["container_boundary_pass"] = (
        all(result["configuration"].values())
        and result["ui_only_gateway_reachable"]
        and result["positive_control_backend"]
        and result["published_localhost_only"]
        and result["gateway_cannot_reach_ui"]
        and result["ui_cannot_reach_host"]
    )
    result["passed"] = (
        result["container_boundary_pass"] and result["stack_owned_listeners_localhost_only"]
    )
    result["tunnel"] = tunnel_probe(args.project, services, targets)
    result["passed"] = result["passed"] and result["tunnel"]["passed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"ui", "gateway", "tunnel"}}, indent=2
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
