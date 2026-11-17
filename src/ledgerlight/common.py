import hashlib
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(os.environ.get("LEDGERLIGHT_ROOT", Path.cwd())).resolve()
load_dotenv(ROOT / ".env", override=False)


def path(value):
    return ROOT / value


def namespace():
    return "ledgerlight"


def read_json(value):
    return json.loads(path(value).read_text())


def write_json(value, obj):
    target = path(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(value):
    if value == "data/processed/queries.jsonl":
        import re

        from ledgerlight.split_guard import held_out_allowed

        splits = read_json("data/processed/splits.json")
        allowed = set(splits["train"]) | set(splits["dev"])
        rows = []
        with path(value).open("rb") as stream:
            for line in stream:
                if not line.strip():
                    continue
                qid = json.loads(re.match(rb'\s*\{\s*"id"\s*:\s*("[^"]+")', line)[1])
                rows.append(
                    json.loads(line)
                    if qid in allowed or held_out_allowed()
                    else {"id": qid, "question": ""}
                )
        return rows
    return [json.loads(line) for line in path(value).read_text().splitlines() if line.strip()]


def write_jsonl(value, rows):
    target = path(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def params():
    return yaml.safe_load(path("params.yaml").read_text())


def digest(target):
    h = hashlib.sha256()
    with Path(target).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def database_url():
    password = os.environ["POSTGRES_PASSWORD"]
    user = os.environ.get("POSTGRES_USER", "ledgerlight")
    port = os.environ.get("POSTGRES_PORT", "5433")
    db = os.environ.get("POSTGRES_DB", "ledgerlight")
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"
