import gc
import json
import os
import subprocess
import time

import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from ledgerlight.common import digest, params, path, read_json, read_jsonl, write_json
from ledgerlight.final_evaluation import paired_bootstrap
from ledgerlight.posthoc import metrics
from ledgerlight.retrieval import encode_queries, encode_windows, fuse, order, parent_scores
from ledgerlight.split_guard import allow_test


def main():
    started = time.monotonic()
    target = path("data/analysis/bge_large")
    target.mkdir(parents=True, exist_ok=True)
    if (target / "started.json").exists():
        raise FileExistsError("Baseline already started; explicit review required before rerun")
    write_json(
        "data/analysis/bge_large/started.json", {"configuration": "fixed, zero-shot, float32"}
    )
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True
    ).strip()
    if busy:
        raise RuntimeError("GPU compute workload already present: " + busy)
    repo = "BAAI/bge-large-en-v1.5"
    info = HfApi().model_info(repo)
    revision = info.sha
    snapshot = snapshot_download(
        repo,
        revision=revision,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.safetensors",
            "1_Pooling/*",
            "README.md",
            "LICENSE*",
        ],
        ignore_patterns=["onnx/*", "openvino/*"],
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from pathlib import Path

    files = {
        str(p.relative_to(snapshot)): digest(p)
        for p in sorted(Path(snapshot).rglob("*"))
        if p.is_file()
    }
    model = SentenceTransformer(
        snapshot,
        device="cuda",
        trust_remote_code=False,
        local_files_only=True,
        model_kwargs={"attn_implementation": "eager"},
    )
    small = AutoTokenizer.from_pretrained(
        params()["encoder"]["name"], revision=params()["encoder"]["revision"], local_files_only=True
    )
    windows = read_jsonl("indices/base/windows.jsonl")
    assert len(windows) == 12604
    vocabulary_equal = small.get_vocab() == model.tokenizer.get_vocab()
    assert vocabulary_equal, "Tokenizer vocabulary differs"
    window_lengths = [
        len(model.tokenizer.encode(w["text"], add_special_tokens=True, truncation=False))
        for w in windows
    ]
    assert all(
        model.tokenizer.encode(w["text"], add_special_tokens=False) == w["input_ids"]
        for w in windows
    )
    result = {
        "note": "baseline added after TEST unblinding; no selection or tuning",
        "repository": repo,
        "revision": revision,
        "license": info.card_data.get("license") if info.card_data else None,
        "files_sha256": files,
        "tokenizer_vocabulary_equal": vocabulary_equal,
        "windows": len(windows),
        "windows_over512_with_special_tokens": sum(n > 512 for n in window_lengths),
        "max_window_tokens_with_special_tokens": max(window_lengths),
        "query_prefix": params()["encoder"]["prefix"],
        "pooling": "CLS, L2 normalized float32; maximum window cosine per parent",
        "rrf_k": read_json("data/baselines/metrics.json")["rrf_k"],
        "rrf_depth": 100,
        "precision": "float32",
        "fallbacks": [],
    }
    write_json("data/analysis/bge_large/provenance.json", result)
    torch.cuda.reset_peak_memory_stats()
    peak_samples = []
    import threading

    stopped = threading.Event()

    def sample():
        while not stopped.is_set():
            try:
                value = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                )
                peak_samples.append(
                    {"seconds": time.monotonic() - started, "mib": int(value.strip())}
                )
            except (OSError, ValueError, subprocess.CalledProcessError):
                pass
            stopped.wait(1)

    monitor = threading.Thread(target=sample)
    monitor.start()
    batch_size = 16
    chunks = []
    offset = 0
    inference_started = time.monotonic()
    try:
        while offset < len(windows):
            try:
                chunk = encode_windows(model, windows[offset : offset + batch_size], batch_size)
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if batch_size > 1:
                    batch_size //= 2
                elif next(model.parameters()).dtype == torch.float32:
                    model.half()
                    result["precision"] = "float16, output float32"
                else:
                    model.cpu().float()
                    result["precision"] = "float32 CPU"
                result["fallbacks"].append(
                    {"offset": offset, "batch_size": batch_size, "precision": result["precision"]}
                )
                continue
            chunks.append(chunk)
            offset += len(chunk)
            if offset % 512 == 0 or offset == len(windows):
                print(
                    json.dumps(
                        {
                            "windows": offset,
                            "total": len(windows),
                            "seconds": time.monotonic() - inference_started,
                        }
                    ),
                    flush=True,
                )
        vectors = np.concatenate(chunks)
        np.save(target / "vectors.npy", vectors)
        del chunks
        with allow_test(True):
            all_queries = read_jsonl("data/processed/queries.jsonl")
        splits = read_json("data/processed/splits.json")
        wanted = set(splits["dev"] + splits["test"])
        queries = sorted((q for q in all_queries if q["id"] in wanted), key=lambda q: q["id"])
        qvectors = np.concatenate(
            [
                encode_queries(model, queries[i : i + batch_size])
                for i in range(0, len(queries), batch_size)
            ]
        )
        np.save(target / "queries.npy", qvectors)
        result["inference_seconds"] = time.monotonic() - inference_started
        result["peak_torch_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        result["peak_torch_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
    finally:
        stopped.set()
        monitor.join()
        write_json("data/analysis/bge_large/vram.json", peak_samples)
    result["peak_sampled_vram_mib"] = max((r["mib"] for r in peak_samples), default=None)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    corpus = read_jsonl("data/processed/corpus.jsonl")
    parents = np.asarray([w["passage_index"] for w in windows])
    dense = parent_scores(qvectors, vectors, parents, len(corpus))
    np.save(target / "scores.npy", dense)
    import re

    import bm25s

    bm25 = bm25s.BM25.load(str(path("indices/bm25")), load_corpus=False)
    rankings = {"BGE-large": {}, "RRF-large": {}}
    for i, q in enumerate(queries):
        lexical = bm25.get_scores(re.findall(r"\w+", q["question"].lower()))
        rankings["BGE-large"][q["id"]] = [corpus[p]["id"] for p in order(dense[i])]
        rrf = fuse(lexical, dense[i], result["rrf_k"])
        rankings["RRF-large"][q["id"]] = [corpus[p]["id"] for p in order(rrf) if rrf[p] > 0]
    result["metrics"], result["bootstrap"] = {}, {}
    labels = read_json("data/processed/labels.json")
    for split in ["dev", "test"]:
        saved = read_json(
            "data/evaluation/" + ("dev_per_query.json" if split == "dev" else "per_query.json")
        )
        values = {}
        result["metrics"][split] = {}
        for method, ranking in rankings.items():
            selected = {q: ranking[q] for q in sorted(splits[split])}
            result["metrics"][split][method], values[method] = metrics(selected, labels)
        result["bootstrap"][split] = {}
        for left, right in [
            ("BGE-large", "BGE"),
            ("BGE-ft-dense", "BGE-large"),
            ("RRF-ft", "RRF-large"),
            ("CatBoost", "RRF-large"),
        ]:
            combined = {**saved, **values}
            result["bootstrap"][split][f"{left}-minus-{right}"] = paired_bootstrap(
                combined[left]["ndcg@10"], combined[right]["ndcg@10"]
            )
    result["seconds"] = time.monotonic() - started
    write_json("data/analysis/bge_large/rankings.json", rankings)
    write_json("data/analysis/bge_large/result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "files_sha256"}, indent=2))


if __name__ == "__main__":
    main()
