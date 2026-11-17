import argparse
import json
import os
import shutil
import subprocess
import uuid

import numpy as np

from ledgerlight.common import ROOT, path, read_json, read_jsonl, write_json, write_jsonl


def ingest_fixture():
    rows = []
    for i in range(8):
        ticker = ["ALFA", "BRAVO", "CHARLIE", "DELTA"][i % 4]
        text = f"{ticker} segment {i} produces equipment for market {i}. Revenue in 2024 was {i + 10} million."
        negative = f"Unrelated company ZULU operates a distinct segment {i + 100}."
        rows.append(
            {
                "id": f"q{i}",
                "question_id": f"fixture-{i}",
                "ticker": ticker,
                "year": "2024",
                "doc_type": "10-K",
                "question": f"What does {ticker} segment {i} produce?",
                "answer": "fixture-only",
                "passages": [{"text": text, "page_number": str(i + 1)}],
                "hard_negatives": [
                    {
                        "text": negative,
                        "page_number": "99",
                        "ticker": "ZULU",
                        "year": "2024",
                        "doc_type": "10-K",
                    }
                ],
            }
        )
    write_json(
        "config/company_aliases.json",
        {name: [name] for name in ["ALFA", "BRAVO", "CHARLIE", "DELTA", "ZULU"]},
    )
    write_jsonl("data/raw/FinRank.jsonl", rows)
    write_json(
        "data/raw/splits.json",
        {
            "random": {
                "train": [f"q{i}" for i in range(4)],
                "dev": ["q4", "q5"],
                "test": ["q6", "q7"],
            }
        },
    )


def fit_fixture():
    from catboost import CatBoostRanker, Pool

    corpus = read_jsonl("data/processed/corpus.jsonl")
    queries = read_jsonl("data/processed/queries.jsonl")
    labels = read_json("data/processed/labels.json")
    train = set(read_json("data/processed/splits.json")["train"])
    scores = np.load(path("indices/bm25/scores.npy"))
    x, y, groups = [], [], []
    for i, query in enumerate(queries):
        if query["id"] not in train:
            continue
        x.extend([[float(score), len(corpus[j]["text"])] for j, score in enumerate(scores[i])])
        y.extend(int(p["id"] in labels[query["id"]]["positives"]) for p in corpus)
        groups.extend([i] * len(corpus))
    model = CatBoostRanker(
        iterations=8,
        depth=2,
        loss_function="YetiRank",
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
        random_seed=42,
    )
    model.fit(Pool(x, y, group_id=groups))
    path("models").mkdir(exist_ok=True)
    model.save_model(str(path("models/fixture.cbm")))


def evaluate_fixture():
    from catboost import CatBoostRanker

    from ledgerlight.evaluation import evaluate
    from ledgerlight.retrieval import order

    corpus = read_jsonl("data/processed/corpus.jsonl")
    queries = read_jsonl("data/processed/queries.jsonl")
    splits = read_json("data/processed/splits.json")
    scores = np.load(path("indices/bm25/scores.npy"))
    labels = read_json("data/processed/labels.json")
    model = CatBoostRanker()
    model.load_model(str(path("models/fixture.cbm")))
    rankings = {}
    for i, query in enumerate(queries):
        if query["id"] not in splits["dev"]:
            continue
        values = model.predict(
            [[float(score), len(corpus[j]["text"])] for j, score in enumerate(scores[i])]
        )
        rankings[query["id"]] = [corpus[j]["id"] for j in order(values)]
    metrics, _ = evaluate(rankings, labels)
    if not all(np.isfinite(list(metrics.values()))):
        raise ValueError("Non-finite fixture metrics")
    write_json(
        "data/metrics.json",
        {
            "fixture_only": True,
            "queries": len(queries),
            "passages": len(corpus),
            "dev": metrics,
            "gpu": False,
        },
    )


def run_ci():
    import yaml

    fixture_root = path(".tmp/ci-" + uuid.uuid4().hex)
    fixture_root.mkdir(parents=True)
    env = {**os.environ, "LEDGERLIGHT_ROOT": str(fixture_root), "CUDA_VISIBLE_DEVICES": ""}
    stages = {
        "ingest": {"cmd": "python -m ledgerlight.fixture ingest", "outs": ["data/raw"]},
        "validate": {
            "cmd": "python -m ledgerlight.data validate",
            "deps": ["data/raw"],
            "outs": ["data/processed"],
        },
        "index": {
            "cmd": "python -m ledgerlight.retrieval index_bm25",
            "deps": ["data/processed"],
            "outs": ["indices/bm25"],
        },
        "train": {
            "cmd": "python -m ledgerlight.fixture train",
            "deps": ["indices/bm25"],
            "outs": ["models/fixture.cbm"],
        },
        "evaluate": {
            "cmd": "python -m ledgerlight.fixture evaluate",
            "deps": ["models/fixture.cbm"],
            "outs": ["data/metrics.json"],
        },
    }
    (fixture_root / "dvc.yaml").write_text(yaml.safe_dump({"stages": stages}, sort_keys=False))
    for command in [
        ["dvc", "init", "--no-scm"],
        ["dvc", "config", "core.analytics", "false"],
        ["dvc", "repro"],
        ["dvc", "status"],
    ]:
        subprocess.run(command, cwd=fixture_root, env=env, check=True)
    result = json.loads((fixture_root / "data/metrics.json").read_text())
    write_json("data/ci/" + fixture_root.name + ".json", result)
    print(json.dumps({"fixture_root": str(fixture_root.relative_to(ROOT)), **result}))
    if os.environ.get("LEDGERLIGHT_CI_CLEANUP") == "1":
        shutil.rmtree(fixture_root)


if __name__ == "__main__":
    commands = {
        "ingest": ingest_fixture,
        "train": fit_fixture,
        "evaluate": evaluate_fixture,
        "ci": run_ci,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=commands)
    commands[parser.parse_args().action]()
