import hashlib
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(os.environ.get("LEDGERLIGHT_ROOT", Path.cwd())).resolve()
load_dotenv(ROOT / ".env", override=False)


def path(value):
    if os.environ.get("LEDGERLIGHT_RUN_MODE") == "pilot":
        isolated = (
            "indices/",
            "data/baselines/",
            "data/training/",
            "data/evaluation/",
            "data/registry/",
            "data/serving/",
            "models/encoder/",
            "models/cross_encoder/",
            "models/ltr/",
            "models/production/",
            "models/onnx/",
            "data/optuna.db",
        )
        name = str(value)
        if any(name == prefix.rstrip("/") or name.startswith(prefix) for prefix in isolated):
            return ROOT / "data/pilot" / name
    return ROOT / value


def namespace():
    return (
        "ledgerlight-pilot" if os.environ.get("LEDGERLIGHT_RUN_MODE") == "pilot" else "ledgerlight"
    )


def read_json(value):
    return json.loads(path(value).read_text())


def write_json(value, obj):
    target = path(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(value):
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
    return f"postgresql://{user}:{password}@127.0.0.1:{port}/{db}"
