import json
import re
import sys
import time

import numpy as np

from ledgerlight.common import params, path, read_jsonl, write_json, write_jsonl


def order(scores):
    return np.argsort(-np.asarray(scores), kind="stable")


def fuse(lexical, dense, k=60, depth=100):
    scores = np.zeros(len(lexical), dtype=np.float64)
    for values in (lexical, dense):
        indices = order(values)[:depth]
        scores[indices] += 1.0 / (k + np.arange(1, len(indices) + 1))
    return scores


def load_model(kind="encoder", model_path=None, device="cuda"):
    from sentence_transformers import CrossEncoder, SentenceTransformer

    config = params()["encoder" if kind == "encoder" else "cross_encoder"]
    cls = SentenceTransformer if kind == "encoder" else CrossEncoder
    kw = {
        "device": device,
        "trust_remote_code": False,
        "model_kwargs": {"attn_implementation": "eager"},
    }
    if model_path is None:
        kw["revision"] = config["revision"]
    if kind != "encoder":
        kw["max_length"] = 512
    return cls(str(model_path) if model_path else config["name"], **kw)


def lexical_index():
    import bm25s

    corpus = read_jsonl("data/processed/corpus.jsonl")
    queries = read_jsonl("data/processed/queries.jsonl")
    token_lists = [re.findall(r"\w+", p["text"].lower()) for p in corpus]
    index = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
    index.index(token_lists, show_progress=False)
    out = path("indices/bm25")
    out.mkdir(parents=True, exist_ok=True)
    index.save(str(out))
    scores = np.asarray(
        [index.get_scores(re.findall(r"\w+", q["question"].lower())) for q in queries],
        dtype=np.float32,
    )
    np.save(out / "scores.npy", scores)
    write_json(
        "indices/bm25/config.json",
        {"method": "lucene", "k1": 1.5, "b": 0.75, "tokenizer": r"lowercase \w+"},
    )
    print(json.dumps({"queries": len(queries), "passages": len(corpus)}))


def make_windows(tokenizer, corpus):
    cfg = params()["encoder"]
    windows = []
    for index, passage in enumerate(corpus):
        ids = tokenizer.encode(passage["text"], add_special_tokens=False)
        start = 0
        while True:
            end = min(start + cfg["content_tokens"], len(ids))
            selected = ids[start:end]
            windows.append(
                {
                    "id": len(windows),
                    "passage_index": index,
                    "start": start,
                    "end": end,
                    "input_ids": selected,
                    "text": tokenizer.decode(selected, skip_special_tokens=True),
                }
            )
            if end == len(ids):
                break
            start += cfg["stride"]
    return windows


def encode_windows(model, windows, batch_size=16):
    import torch
    import torch.nn.functional as F

    output = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(windows), batch_size):
            items = [
                model.tokenizer.prepare_for_model(
                    w["input_ids"], add_special_tokens=True, return_attention_mask=True
                )
                for w in windows[offset : offset + batch_size]
            ]
            batch = model.tokenizer.pad(items, padding=True, return_tensors="pt")
            batch = {k: v.to(model.device) for k, v in batch.items()}
            result = model(batch)["sentence_embedding"]
            output.append(F.normalize(result, p=2, dim=1).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def encode_queries(model, queries):
    prefix = params()["encoder"]["prefix"]
    return model.encode(
        [prefix + q["question"] for q in queries],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)


def parent_scores(query_vectors, vectors, parents, count):
    result = np.full((len(query_vectors), count), -1.0, dtype=np.float32)
    for i, query in enumerate(query_vectors):
        np.maximum.at(result[i], parents, vectors @ query)
    return result


def embed(model_path=None, destination="indices/base"):
    started = time.monotonic()
    model = load_model(model_path=model_path)
    corpus = read_jsonl("data/processed/corpus.jsonl")
    queries = read_jsonl("data/processed/queries.jsonl")
    windows = make_windows(model.tokenizer, corpus)
    vectors = encode_windows(model, windows, params()["encoder"]["batch_size"])
    qvectors = encode_queries(model, queries)
    parents = np.asarray([w["passage_index"] for w in windows], dtype=np.int32)
    scores = parent_scores(qvectors, vectors, parents, len(corpus))
    dest = path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "vectors.npy", vectors)
    np.save(dest / "queries.npy", qvectors)
    np.save(dest / "parents.npy", parents)
    np.save(dest / "scores.npy", scores)
    if destination == "indices/base":
        write_jsonl("indices/base/windows.jsonl", windows)
    write_json(
        f"{destination}/config.json",
        {
            "model_path": str(model_path) if model_path else None,
            "revision": params()["encoder"]["revision"],
            "windows": len(windows),
            "seconds": time.monotonic() - started,
            "dtype": "float32",
            "aggregation": "max",
        },
    )
    print(json.dumps({"windows": len(windows), "seconds": time.monotonic() - started}))


def best_windows(query_vector, vectors, windows, candidates, count=1):
    scores = vectors @ query_vector
    parent_groups = {}
    for i, window in enumerate(windows):
        parent_groups.setdefault(window["passage_index"], []).append(i)
    chosen = []
    for pid in candidates:
        ids = np.asarray(parent_groups[int(pid)])
        chosen.append(ids[order(scores[ids])[:count]].tolist())
    return chosen


def cross_scores(model, question, candidates, qvector, vectors, windows, count=1):
    selected = best_windows(qvector, vectors, windows, candidates, count)
    pairs = [(question, windows[wid]["text"]) for group in selected for wid in group]
    values = model.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False,
        activation_fn=__import__("torch").nn.Identity(),
    )
    result, offset = [], 0
    for group in selected:
        result.append(float(np.max(values[offset : offset + len(group)])))
        offset += len(group)
    return np.asarray(result, dtype=np.float32)


if __name__ == "__main__":
    {"index_bm25": lexical_index, "embed": embed}[sys.argv[1]]()
