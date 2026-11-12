import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import yaml

from ledgerlight.common import write_json

legacy = yaml.safe_load(Path("compose.yaml").read_text())["name"]
pattern = re.compile(re.escape(legacy), re.I)
paths = subprocess.check_output(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], text=True
).split("\0")
rows = []
for name in sorted(set(paths)):
    file = Path(name)
    if not file.is_file() or name == "reports/rename/name_inventory.json":
        continue
    try:
        text = file.read_text()
    except UnicodeDecodeError:
        continue
    if name == "docs/BUILD_LOG.md":
        text = text.split("## Remaining name inventory")[0]
    for line_number, line in enumerate(text.splitlines(), 1):
        if not pattern.search(line):
            continue
        if name.startswith("reports/rename/"):
            category = "frozen artifact receipt"
        elif name == "docs/BUILD_LOG.md":
            category = "retained identifier documentation"
        elif name in {"pyproject.toml", "scripts/registered_load_check.py"}:
            category = "frozen artifact namespace compatibility"
        elif name == "scripts/runtime_audit.py":
            category = "frozen historical temporary-path audit"
        elif name == "src/ledgerlight/gateway.py" and "You are " in line:
            category = "frozen prompt"
        elif any(
            value in line.lower()
            for value in [
                legacy + "-",
                legacy + "_",
                '"' + legacy + '"',
                "'" + legacy + "'",
                "name: " + legacy,
                "folder: " + legacy,
                "user=" + legacy,
                "db=" + legacy,
                "-u " + legacy,
                "postgres_user=" + legacy,
                "postgres_db=" + legacy,
                "gf_security_admin_user: " + legacy,
            ]
        ):
            category = "retained store or telemetry identifier"
        else:
            raise ValueError(f"Unclassified legacy-name occurrence: {name}:{line_number}")
        rows.append({"path": name, "line": line_number, "category": category})
for name in sorted(set(paths)):
    if Path(name).is_file() and pattern.search(name):
        rows.append(
            {"path": name, "line": 0, "category": "frozen artifact namespace compatibility"}
        )
counts = Counter(row["category"] for row in rows)
write_json(
    "reports/rename/name_inventory.json",
    {
        "scope": "All tracked and nonignored candidate text files; generated inventory does not inventory itself. Build-log inventory section omitted to avoid self-reference. Immutable stores, dependencies, and historical logs are inventoried separately by path only.",
        "occurrences": rows,
        "counts": dict(counts),
        "bugs_remaining": 0,
    },
)
print(json.dumps(dict(counts), indent=2))
