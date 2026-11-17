import argparse
import json
import os
import subprocess
import threading
import time

from ledgerlight.common import digest, params, path, read_json, write_json


def run_stage(name, function, outputs):
    checkpoint = f"data/training/checkpoints/{name}.json"
    if path(checkpoint).exists():
        previous = read_json(checkpoint)
        if all(
            path(item).is_file() and digest(path(item)) == sha
            for item, sha in previous["outputs"].items()
        ):
            print(json.dumps({"stage": name, "resume": "completed"}), flush=True)
            return previous
    started = time.monotonic()
    function()
    result = {
        "stage": name,
        "seconds": time.monotonic() - started,
        "outputs": {item: digest(path(item)) for item in outputs},
    }
    write_json(checkpoint, result)
    print(json.dumps(result), flush=True)
    return result


def run():
    from ledgerlight.evaluation import baseline
    from ledgerlight.final_evaluation import evaluate_dev
    from ledgerlight.registry import register_all
    from ledgerlight.retrieval import embed, lexical_index
    from ledgerlight.training import neural_cross_encoder, neural_encoder, prepare, train_ltr

    if not path("data/processed/splits.json").exists():
        raise FileNotFoundError("Download and validate the pinned dataset before training")
    if digest(path("data/raw/FinRank.jsonl")) != params()["data"]["sha256"]:
        raise ValueError("Pinned dataset checksum mismatch")
    os.environ["LEDGERLIGHT_RESUME"] = "1"
    peak, stop = [0], threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                sample = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                )
                peak[0] = max(peak[0], int(sample.strip().splitlines()[0]))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            stop.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        stages = [
            ("lexical", lexical_index, ["indices/bm25/scores.npy"]),
            (
                "base_embeddings",
                embed,
                [
                    "indices/base/vectors.npy",
                    "indices/base/queries.npy",
                    "indices/base/scores.npy",
                    "indices/base/windows.jsonl",
                ],
            ),
            (
                "baselines",
                baseline,
                [
                    "data/baselines/metrics.json",
                    "data/baselines/rankings.json",
                    "data/baselines/ce_scores.json",
                ],
            ),
            (
                "partition",
                prepare,
                ["data/training/split.json", "data/training/examples.json"],
            ),
            (
                "encoder",
                neural_encoder,
                ["data/training/encoder_selected.json", "indices/trained/vectors.npy"],
            ),
            (
                "cross_encoder",
                neural_cross_encoder,
                ["data/training/ce_selected.json", "data/training/ce_trials.json"],
            ),
            (
                "ranker",
                train_ltr,
                ["data/training/ltr_selected.json", "data/training/ltr_scores.json"],
            ),
            (
                "dev_report",
                evaluate_dev,
                ["data/evaluation/dev_metrics.json", "data/evaluation/freeze.json"],
            ),
            (
                "registry",
                register_all,
                ["data/registry/versions.json"],
            ),
        ]
        records = [run_stage(*stage) for stage in stages]
        split = read_json("data/training/split.json")
        if (len(split["A"]), len(split["B"])) != (712, 236):
            raise ValueError("Unexpected TRAIN partition")
        report = {
            "seconds": time.monotonic() - started,
            "peak_vram_mib": peak[0],
            "stages": records,
            "dev_report": str(path("data/evaluation/dev_metrics.json")),
            "test_evaluated": False,
        }
        for family, filename in [
            ("encoder", "encoder_trials.json"),
            ("cross_encoder", "ce_trials.json"),
            ("ranker", "ltr_trials.json"),
        ]:
            trials = read_json("data/training/" + filename)
            report[family] = [
                {
                    "trial": item["trial"],
                    "seconds": item["training_seconds"],
                    "steps": item.get("steps"),
                    "steps_per_second": item.get("steps", 0) / item["training_seconds"],
                }
                for item in trials
            ]
        write_json("data/training/run.json", report)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        stop.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
