import gc
import json
import sys

import mlflow.pyfunc
import pandas as pd
import torch

import ledgerlight
from ledgerlight.common import read_json, read_jsonl, write_json

version = sys.argv[1]
assert version in {"1", "7"}
qid = read_json("data/processed/splits.json")["dev"][0]
question = next(q["question"] for q in read_jsonl("data/processed/queries.jsonl") if q["id"] == qid)
model = mlflow.pyfunc.load_model(f"models:/ledgerlight-pipeline/{version}")
result = model.predict(pd.DataFrame({"question": [question]}))[0]
baseline = read_json("reports/rename/baseline.json")
registered = next(m for m in baseline["models"] if m["name"] == "ledgerlight-pipeline")
metadata = next(v for v in registered["versions"] if v["version"] == version)
assert result["versions"]["method"] == metadata["tags"]["method"]
assert result["versions"]["backend"] == "ort-cuda"
assert result["results"]
receipt = {
    "registered_name": "ledgerlight-pipeline",
    "version": version,
    "versions": result["versions"],
    "query_split": "dev",
    "query_id": qid,
    "results": [row["passage_id"] for row in result["results"]],
    "legacy_namespace": ledgerlight.__name__,
    "legacy_namespace_target": "src/ledgerlight",
    "serialized_manifest_note": "The immutable stored manifest has no registry_version field. Direct pyfunc output reports null; the live service supplies the registry-resolved version separately.",
    "new_runs_created": False,
}
write_json(f"reports/rename/registered_load_{version}.json", receipt)
print(json.dumps(receipt, indent=2))
del model
gc.collect()
torch.cuda.empty_cache()
