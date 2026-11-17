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
    from sentence_transformers import SentenceTransformer

    from ledgerlight.cross_encoder import PolicyCrossEncoder

    config = params()["encoder" if kind == "encoder" else "cross_encoder"]
    cls = SentenceTransformer if kind == "encoder" else PolicyCrossEncoder
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


def active_queries(queries):
    from ledgerlight.common import read_json
    from ledgerlight.split_guard import held_out_allowed

    if held_out_allowed():
        return list(enumerate(queries))
    splits = read_json("data/processed/splits.json")
    allowed = set(splits["train"]) | set(splits["dev"])
    return [(i, q) for i, q in enumerate(queries) if q["id"] in allowed]


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
    active = active_queries(queries)
    scores = np.full((len(queries), len(corpus)), np.nan, dtype=np.float32)
    for i, query in active:
        scores[i] = index.get_scores(re.findall(r"\w+", query["question"].lower()))
    np.save(out / "scores.npy", scores)
    write_json(
        "indices/bm25/config.json",
        {"method": "lucene", "k1": 1.5, "b": 0.75, "tokenizer": r"lowercase \w+"},
    )
    print(json.dumps({"queries": len(queries), "passages": len(corpus)}))


def make_windows(tokenizer, corpus, content_tokens=None, stride=None):
    from bisect import bisect_right

    cfg = params()["encoder"]
    limit = cfg["content_tokens"] if content_tokens is None else content_tokens
    step = cfg["stride"] if stride is None else stride
    if not tokenizer.is_fast or not 0 < step <= limit:
        raise ValueError("Fast tokenizer and valid window step required")
    windows = []
    for index, passage in enumerate(corpus):
        text = passage["text"]
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        words = encoded.word_ids()
        boundaries = [0] + [i for i in range(1, len(ids)) if words[i] != words[i - 1]] + [len(ids)]
        start = 0
        while start < len(ids):
            end = boundaries[bisect_right(boundaries, start + limit) - 1]
            if end <= start:
                raise ValueError("A pre-token exceeds the window capacity")
            char_start, char_end = offsets[start][0], offsets[end - 1][1]
            selected = ids[start:end]
            window_text = text[char_start:char_end]
            if tokenizer.encode(window_text, add_special_tokens=False) != selected:
                raise ValueError(f"Window token round-trip failed for {passage['id']}")
            if not window_text or end - start > limit:
                raise ValueError("Invalid window size")
            windows.append(
                {
                    "id": len(windows),
                    "passage_index": index,
                    "start": start,
                    "end": end,
                    "char_start": char_start,
                    "char_end": char_end,
                    "input_ids": selected,
                    "text": window_text,
                }
            )
            if end == len(ids):
                if text[char_end:].strip():
                    raise ValueError("Passage tail not covered")
                break
            next_start = boundaries[bisect_right(boundaries, start + step) - 1]
            if not start < next_start <= end:
                raise ValueError("Window step leaves a gap or fails to progress")
            start = next_start
        if not ids:
            raise ValueError("Passage has no tokens")
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
    active = active_queries(queries)
    qvectors = np.full((len(queries), vectors.shape[1]), np.nan, dtype=np.float32)
    qvectors[[i for i, _ in active]] = encode_queries(model, [q for _, q in active])
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
    from ledgerlight.pairs import torch_pair_scores

    values = (
        model.predict(pairs, batch_size=8)
        if hasattr(model, "runtime")
        else torch_pair_scores(model, pairs)
    )
    result, offset = [], 0
    for group in selected:
        result.append(float(np.max(values[offset : offset + len(group)])))
        offset += len(group)
    return np.asarray(result, dtype=np.float32)


if __name__ == "__main__":
    {"index_bm25": lexical_index, "embed": embed}[sys.argv[1]]()
