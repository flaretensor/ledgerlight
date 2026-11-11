import json
import os
import secrets
import subprocess
import time

import httpx
import psycopg
from mlflow import MlflowClient
from psycopg import sql
from services import stop

from ledgerlight.common import ROOT, database_url, path, write_json


def inventory():
    client = MlflowClient("http://127.0.0.1:5000")
    experiments = client.search_experiments()
    return {
        "experiments": sorted((e.experiment_id, e.name) for e in experiments),
        "runs": sorted(
            [
                {
                    "id": run.info.run_id,
                    "status": run.info.status,
                    "metrics": run.data.metrics,
                    "artifact_uri": run.info.artifact_uri,
                }
                for run in client.search_runs(
                    [e.experiment_id for e in experiments], max_results=10000
                )
            ],
            key=lambda run: run["id"],
        ),
        "models": sorted(
            [
                {
                    "name": model.name,
                    "aliases": model.aliases,
                    "versions": sorted(
                        (version.version, version.run_id, version.source)
                        for version in client.search_model_versions(f"name='{model.name}'")
                    ),
                }
                for model in client.search_registered_models()
            ],
            key=lambda model: model["name"],
        ),
    }


before = inventory()
write_json("data/runtime/tracking_before_compose.json", before)
with psycopg.connect(database_url()) as connection:
    native_version = connection.execute("SELECT version()").fetchone()[0]
subprocess.run(["docker", "compose", "pull", "postgres"], cwd=ROOT, check=True)
bindir = path(".tools/postgres/usr/lib/postgresql/18/bin")
env = {
    **os.environ,
    "LD_LIBRARY_PATH": str(path(".tools/postgres/usr/lib/x86_64-linux-gnu")),
    "PGPASSWORD": os.environ["POSTGRES_PASSWORD"],
}
backup = path("data/backups/tracking-before-compose.dump")
backup.parent.mkdir(parents=True, exist_ok=True)
if backup.exists():
    raise RuntimeError("Migration backup already exists; inspect the recorded migration state")
subprocess.run(
    [
        str(bindir / "pg_dump"),
        "-h",
        "127.0.0.1",
        "-p",
        "5433",
        "-U",
        "ledgerlight",
        "-Fc",
        "-f",
        str(backup),
        "ledgerlight",
    ],
    env=env,
    check=True,
)
backup.chmod(0o600)
stop("mlflow")
password = secrets.token_hex(24)
with psycopg.connect(database_url(), autocommit=True) as connection:
    connection.execute(
        sql.SQL("ALTER ROLE {} PASSWORD {}").format(
            sql.Identifier("ledgerlight"), sql.Literal(password)
        )
    )
configuration = path(".env")
configuration.write_text(
    "\n".join(
        "POSTGRES_PASSWORD=" + password if line.startswith("POSTGRES_PASSWORD=") else line
        for line in configuration.read_text().splitlines()
    )
    + "\n"
)
configuration.chmod(0o600)
os.environ["POSTGRES_PASSWORD"] = password
subprocess.run(
    [str(bindir / "pg_ctl"), "-D", str(path("data/postgres")), "stop", "-m", "fast", "-w"],
    env=env,
    check=True,
)
subprocess.run(["docker", "compose", "up", "-d", "postgres", "mlflow"], cwd=ROOT, check=True)
deadline = time.monotonic() + 180
while True:
    try:
        response = httpx.get("http://127.0.0.1:5000/health", timeout=2)
        response.raise_for_status()
        break
    except httpx.HTTPError:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Compose MLflow did not become ready; data and backup retained"
            ) from None
        time.sleep(1)
after = inventory()
write_json("data/runtime/tracking_after_compose.json", after)
if before != after:
    raise RuntimeError("Tracking inventory changed during migration; compare recorded snapshots")
with psycopg.connect(database_url()) as connection:
    compose_version = connection.execute("SELECT version()").fetchone()[0]
result = {
    "preserved": before == after,
    "runs": len(after["runs"]),
    "registered_models": len(after["models"]),
    "registered_versions": sum(len(model["versions"]) for model in after["models"]),
    "native_postgres": native_version,
    "compose_postgres": compose_version,
    "backup": str(backup.relative_to(ROOT)),
    "backup_bytes": backup.stat().st_size,
    "local_database_credential_rotated": True,
}
write_json("data/runtime/tracking_migration.json", result)
print(json.dumps(result, indent=2))
