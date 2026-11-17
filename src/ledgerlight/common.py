import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote

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
    password = quote(os.environ["POSTGRES_PASSWORD"], safe="")
    user = quote(os.environ.get("POSTGRES_USER", "ledgerlight"), safe="")
    port = os.environ.get("POSTGRES_PORT", "5433")
    db = os.environ.get("POSTGRES_DB", "ledgerlight")
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"
