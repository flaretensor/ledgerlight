import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ledgerlight.common import ROOT, write_json

identifiers = subprocess.check_output(["docker", "compose", "ps", "-q"], text=True).split()
containers = []
for identifier in identifiers:
    value = subprocess.check_output(
        [
            "docker",
            "inspect",
            "--format",
            "{{json .Name}} {{json .State.Pid}} {{json .HostConfig.RestartPolicy.Name}} {{json .HostConfig.NetworkMode}}",
            identifier,
        ],
        text=True,
    ).split()
    containers.append(
        {
            "name": json.loads(value[0]),
            "pid": int(value[1]),
            "restart": json.loads(value[2]),
            "network": json.loads(value[3]),
        }
    )
pids = {item["pid"] for item in containers}
parents = {}
for directory in Path("/proc").iterdir():
    if not directory.name.isdigit():
        continue
    try:
        for line in (directory / "status").read_text().splitlines():
            if line.startswith("PPid:"):
                parents[int(directory.name)] = int(line.split()[1])
        cwd = os.readlink(directory / "cwd")
        if cwd == str(ROOT) or cwd.startswith(str(ROOT) + "/"):
            pids.add(int(directory.name))
    except (OSError, ProcessLookupError):
        pass
while True:
    expanded = pids | {pid for pid, parent in parents.items() if parent in pids}
    if expanded == pids:
        break
    pids = expanded
open_tmp = []
for pid in sorted(pids):
    try:
        for descriptor in Path(f"/proc/{pid}/fd").iterdir():
            try:
                target = os.readlink(descriptor)
                if target.startswith(("/tmp/", "/var/tmp/")):
                    open_tmp.append({"pid": pid, "target": target})
            except OSError:
                pass
    except OSError:
        pass
entries = []
for pattern in ["skills-*", "ledgerlight*", "ledgerlight*"]:
    for entry in Path("/tmp").glob(pattern):
        if entry.stat().st_uid == os.getuid():
            entries.append(str(entry))
result = {
    "time_utc": datetime.now(UTC).isoformat(),
    "project_named_tmp_entries": entries,
    "project_process_open_tmp_files": open_tmp,
    "processes_checked": len(pids),
    "containers": containers,
    "all_restart_policies_no": all(item["restart"] == "no" for item in containers),
    "scope": "Known project-named entries plus open temporary files of native project processes and Compose container descendants; unrelated temporary files not read or deleted",
}
write_json("data/runtime/audit.json", result)
print(json.dumps(result, indent=2))
