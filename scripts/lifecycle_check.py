import json
import subprocess

import httpx

from ledgerlight.common import read_json, read_jsonl, write_json
from ledgerlight.registry import PIPELINE_NAME, client

c = client()
versions = read_json("data/registry/versions.json")
selected = versions["selected"]
base = versions["pipelines"]["BM25"]
records = []
qid = read_json("data/processed/splits.json")["dev"][0]
question = next(q["question"] for q in read_jsonl("data/processed/queries.jsonl") if q["id"] == qid)


def capture(label):
    response = httpx.get("http://127.0.0.1:8001/health", timeout=180)
    response.raise_for_status()
    payload = response.json()
    served = httpx.post("http://127.0.0.1:8001/search", json={"question": question}, timeout=180)
    served.raise_for_status()
    assert served.json()["versions"]["pipeline"] == payload["versions"]["pipeline"]
    assert served.json()["results"]
    record = {"step": label, "health": payload, "search": served.json()}
    if label in {"promoted", "rolled_back"}:
        answer = httpx.post("http://127.0.0.1:8000/ask", json={"question": question}, timeout=300)
        answer.raise_for_status()
        assert answer.json()["ranking"]["versions"]["pipeline"] == payload["versions"]["pipeline"]
        record["ask"] = answer.json()
    records.append(record)
    return payload["versions"]["pipeline"]


current = c.get_model_version_by_alias(PIPELINE_NAME, "champion")
if str(current.version) != str(base):
    subprocess.run(["ledgerlight", "rollback"], check=True)
assert capture("baseline_before_promote") == base
subprocess.run(["ledgerlight", "promote", selected], check=True)
assert capture("promoted") == selected
subprocess.run(["ledgerlight", "rollback"], check=True)
assert capture("rolled_back") == base
subprocess.run(["ledgerlight", "promote", selected], check=True)
assert capture("restored_selected") == selected
rejected = subprocess.run(["ledgerlight", "promote", base], capture_output=True, text=True)
assert rejected.returncode != 0
assert str(c.get_model_version_by_alias(PIPELINE_NAME, "champion").version) == selected
records.append(
    {
        "step": "regression_rejected",
        "candidate": base,
        "exit_code": rejected.returncode,
        "stdout": rejected.stdout,
        "stderr": rejected.stderr,
    }
)
write_json("data/serving/lifecycle.json", records)
print(json.dumps(records, indent=2))
