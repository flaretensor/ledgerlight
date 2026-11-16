import hashlib
import json

from ledgerlight.common import ROOT, digest, params, path, read_json


def resolved_parameters():
    import torch

    return {
        "precision": "bfloat16" if torch.cuda.is_bf16_supported() else "float32",
        "params": params(),
        "encoder": {
            "trials": 6,
            "lr": [5e-6, 4e-5],
            "epochs": [1, 2],
            "batch": [16, 32],
            "scale": [10.0, 20.0],
            "negatives": 2,
            "sampler_seed": 42,
            "mini_batch": 4,
        },
        "cross_encoder": {
            "trials": 6,
            "lr": [1e-6, 3e-5],
            "epochs": [1, 2],
            "batch": 8,
            "negative_windows": 4,
            "sampler_seed": 43,
            "precision": "bfloat16_if_supported",
        },
        "ranker": {
            "trials": params()["training"]["catboost_trials"],
            "iterations": [150, 1500, 50],
            "depth": [3, 7],
            "learning_rate": [0.02, 0.3],
            "l2_leaf_reg": [1, 20],
            "loss": "YetiRank",
            "sampler_seed": 44,
            "threads": 8,
        },
        "optimizer": {"name": "AdamW", "weight_decay": 0.01, "clip_norm": 1.0},
        "seeds": [42, 123, 2026],
    }


def selected_weights(stage):
    names = {
        "encoder": "encoder_selected.json",
        "cross_encoder": "ce_selected.json",
        "ranker": "ltr_selected.json",
    }
    selected = read_json("data/training/" + names[stage])
    target = path(selected["path"])
    files = (
        [target]
        if target.is_file()
        else sorted(
            p
            for p in target.rglob("*")
            if p.is_file() and p.suffix in {".safetensors", ".bin", ".cbm"}
        )
    )
    if not files:
        raise FileNotFoundError(f"No selected {stage} weights")
    return {
        p.name if target.is_file() else p.relative_to(target).as_posix(): digest(p) for p in files
    }


def training_fingerprint(stage="pipeline"):
    upstream = {
        "encoder": [],
        "cross_encoder": ["encoder"],
        "ranker": ["encoder", "cross_encoder"],
        "pipeline": ["encoder", "cross_encoder", "ranker"],
    }[stage]
    sources = sorted((ROOT / "src/ledgerlight").glob("*.py"))
    included = {
        "common.py",
        "data.py",
        "retrieval.py",
        "features.py",
        "training.py",
        "evaluation.py",
        "final_evaluation.py",
        "pairs.py",
        "cross_encoder.py",
        "training_identity.py",
        "test_access.py",
        "run_training.py",
    }
    values = {
        "stage": stage,
        "source": {p.name: digest(p) for p in sources if p.name in included},
        "aliases": digest(path("config/company_aliases.json")),
        "parameters": resolved_parameters(),
        "dataset_sha256": digest(path("data/raw/FinRank.jsonl")),
        "splits": read_json("data/processed/splits.json"),
        "upstream_weights": {name: selected_weights(name) for name in upstream},
    }
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
