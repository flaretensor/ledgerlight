import gc
import json
import shutil

import numpy as np
import torch

from ledgerlight.common import digest, params, path, read_json, read_jsonl, write_json
from ledgerlight.onnx_backend import OnnxCrossEncoder, OnnxEncoder
from ledgerlight.registry import manifest
from ledgerlight.retrieval import load_model

versions = read_json("data/registry/versions.json")
query_id = read_json("data/processed/splits.json")["dev"][0]
question = next(
    q["question"] for q in read_jsonl("data/processed/queries.jsonl") if q["id"] == query_id
)
window = read_jsonl("indices/base/windows.jsonl")[0]["text"]
selected = read_json("models/production/config.json")
records = []
seen = set()
for method in ["CE-zero", "CE-ft"]:
    configuration = manifest(versions["pipelines"][method])
    for role, source_key, adapter in [
        ("encoder", "encoder_path", OnnxEncoder),
        ("cross_encoder", "cross_encoder_path", OnnxCrossEncoder),
    ]:
        version = configuration["components"][role]
        if (role, version) in seen:
            continue
        seen.add((role, version))
        destination = path(f"models/onnx/components/{role}/{version}/model.onnx")
        destination.parent.mkdir(parents=True, exist_ok=True)
        original = path(selected[source_key])
        source = path(configuration[source_key])
        if (
            original.exists()
            and digest(original / "model.safetensors") == digest(source / "model.safetensors")
            and not destination.exists()
        ):
            shutil.copyfile(path(f"models/onnx/{role}/model.onnx"), destination)
        runtime = adapter(source, version, "CPUExecutionProvider")
        model = load_model(role, source, device="cpu")
        if role == "encoder":
            text = params()["encoder"]["prefix"] + question
            expected = model.encode([text], normalize_embeddings=True)
            actual = runtime.encode([text])
        else:
            expected = model.predict(
                [(question, window)], activation_fn=torch.nn.Identity(), show_progress_bar=False
            )
            actual = runtime.predict([(question, window)])
        assert np.allclose(actual, expected, atol=1e-4, rtol=1e-3)
        records.append(
            {
                "role": role,
                "registry_version": version,
                "provider": runtime.runtime.get_providers()[0],
                "numeric_max_abs_error": float(np.max(np.abs(actual - expected))),
                "graph_sha256": digest(destination),
            }
        )
        del runtime, model
        gc.collect()
write_json(
    "data/serving/onnx_components.json",
    {"components": records, "query_split": "dev_only", "query_id": query_id},
)
print(json.dumps(records, indent=2))
