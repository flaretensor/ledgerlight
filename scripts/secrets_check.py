import json
import os
import re
import stat
import subprocess
import time
from pathlib import Path

import psycopg
from dotenv import dotenv_values

from ledgerlight.common import ROOT, database_url, path, read_json, write_json

configuration = dotenv_values(path(".env"))
secrets = {
    value.encode()
    for key, value in configuration.items()
    if value and len(value) >= 16 and any(word in key for word in ["PASSWORD", "TOKEN", "SECRET"])
}
retired = set()
for relative in [".tmp/pgpass", ".runtime/pgpass"]:
    target = path(relative)
    if target.exists():
        value = target.read_text().strip().rsplit(":", 1)[-1]
        if re.fullmatch(r"[0-9a-f]{32,64}", value) and value != configuration["POSTGRES_PASSWORD"]:
            retired.add(value)
            secrets.add(value.encode())
            target.write_text("retired credential removed\n")
            target.chmod(0o600)
longest = max(map(len, secrets))
findings = []
scanned_bytes = {}
token_pattern = re.compile(rb"(?:hf_|gh[pousr]_|github_pat_|sk-(?:proj-)?)[A-Za-z0-9_-]{24,}")
database_pattern = re.compile(rb"postgresql(?:\+\w+)?://[^:\s/@]+:([^@/\s]+)@")


def scan(stream, category, name):
    tail = b""
    total = 0
    while chunk := stream.read(4 * 1024 * 1024):
        total += len(chunk)
        value = tail + chunk
        if any(secret in value for secret in secrets):
            findings.append({"category": category, "path": name, "kind": "known credential"})
        if category == "process_arguments" or Path(name).suffix in {
            ".py",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".txt",
            ".log",
            ".sh",
            ".lock",
        }:
            if token_pattern.search(value):
                findings.append({"category": category, "path": name, "kind": "API token pattern"})
            for match in database_pattern.finditer(value):
                if not any(character in match.group(1) for character in [b"{", b"}"]):
                    findings.append(
                        {
                            "category": category,
                            "path": name,
                            "kind": "literal database URI credential",
                        }
                    )
        tail = value[-longest:]
    scanned_bytes[category] = scanned_bytes.get(category, 0) + total


revisions = subprocess.check_output(["git", "rev-list", "--all"], text=True).split()
blobs = {}
for revision in revisions:
    tree = subprocess.check_output(["git", "ls-tree", "-rz", revision])
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        header, filename = entry.split(b"\t", 1)
        assert filename != b".env" and not filename.startswith(b"data/backups/")
        _, kind, identifier = header.split()
        if kind == b"blob":
            blobs[identifier.decode()] = filename.decode()
for identifier, filename in blobs.items():
    process = subprocess.Popen(["git", "cat-file", "blob", identifier], stdout=subprocess.PIPE)
    scan(process.stdout, "git_history", filename)
    assert process.wait() == 0
for relative, category in [
    (".dvc/cache", "dvc_cache"),
    ("mlruns/artifacts", "mlflow_artifacts"),
    ("logs", "logs"),
]:
    for target in path(relative).rglob("*"):
        if target.is_file():
            with target.open("rb") as stream:
                scan(stream, category, str(target.relative_to(ROOT)))
for target in [
    path("dvc.yaml"),
    path("dvc.lock"),
    *path(".dvc").glob("config*"),
    *ROOT.glob("*.dvc"),
    *path("models").glob("*.dvc"),
]:
    if target.is_file():
        with target.open("rb") as stream:
            scan(stream, "dvc_metadata", str(target.relative_to(ROOT)))
audit = read_json("data/runtime/audit.json")
owned = {item["pid"] for item in audit["containers"]}
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
            owned.add(int(directory.name))
    except OSError:
        pass
while True:
    expanded = owned | {pid for pid, parent in parents.items() if parent in owned}
    if expanded == owned:
        break
    owned = expanded
for pid in owned:
    try:
        with Path(f"/proc/{pid}/cmdline").open("rb") as stream:
            scan(stream, "process_arguments", f"pid:{pid}")
    except OSError:
        pass
with psycopg.connect(database_url()) as connection:
    assert connection.execute("SELECT current_user").fetchone()[0] == "ledgerlight"
for value in retired:
    try:
        connection = psycopg.connect(
            host="127.0.0.1",
            port=5433,
            user="ledgerlight",
            dbname="ledgerlight",
            password=value,
            connect_timeout=3,
        )
    except psycopg.OperationalError:
        continue
    connection.close()
    raise RuntimeError("A retired credential still authenticates")
backup = path("data/backups/tracking-before-compose.dump")
assert subprocess.run(["git", "check-ignore", "-q", str(backup)]).returncode == 0
assert stat.S_IMODE(backup.stat().st_mode) == 0o600
result = {
    "git_revisions_checked": len(revisions),
    "git_blobs_checked": len(blobs),
    "scanned_bytes": scanned_bytes,
    "findings": findings,
    "active_database_credential_works": True,
    "retired_credentials_rejected": len(retired),
    "backup_gitignored": True,
    "backup_mode": "0600",
    "scope": "Exact active/retired local secrets plus API-token and literal database-URI patterns in text; all Git blobs, DVC cache/metadata, MLflow artifact bytes, logs, and owned process arguments",
    "historical_deviation": "A retired database password appeared once in tool output; it has been rotated and is rejected by PostgreSQL. Tool conversation history cannot be rewritten from this workspace.",
}
if path("data/audits/secrets.json").exists():
    previous = read_json("data/audits/secrets.json")
    write_json(f"data/audits/secrets-{time.time_ns()}.json", previous)
    result["previous_retired_credentials_rejected"] = previous.get(
        "retired_credentials_rejected", 0
    )
write_json("data/audits/secrets.json", result)
print(json.dumps(result, indent=2))
if findings:
    raise SystemExit("Credential findings require remediation; values deliberately not printed")
