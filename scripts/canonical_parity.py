import gc
import json
import time

import bm25s
import numpy as np
import torch
from catboost import CatBoostRanker

from ledgerlight.common import digest, params, path, read_json, read_jsonl, write_json
from ledgerlight.features import features
from ledgerlight.onnx_backend import (
    CrossExport,
    EncoderExport,
    OnnxCrossEncoder,
    OnnxEncoder,
    export,
    session,
)
from ledgerlight.retrieval import best_windows, encode_queries, fuse, load_model, order


def main():
    started = time.monotonic()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = read_json("models/production/config.json")
    out = path("data/analysis/runtime")
    out.mkdir(parents=True, exist_ok=True)
    encoder = load_model(model_path=path(config["encoder_path"]), device="cpu")
    cross = load_model("cross_encoder", path(config["cross_encoder_path"]), device="cpu")
    for role, model, wrapper in [
        ("encoder", encoder, EncoderExport),
        ("cross_encoder", cross, CrossExport),
    ]:
        export(wrapper(model), model.tokenizer, out / (role + ".onnx"))
    encoder.cuda()
    cross.model.cuda()
    ort_encoder = object.__new__(OnnxEncoder)
    ort_encoder.tokenizer = encoder.tokenizer
    ort_encoder.runtime = session(out / "encoder.onnx", "CUDAExecutionProvider")
    ort_cross = object.__new__(OnnxCrossEncoder)
    ort_cross.tokenizer = cross.tokenizer
    ort_cross.runtime = session(out / "cross_encoder.onnx", "CUDAExecutionProvider")
    queries = {q["id"]: q for q in read_jsonl("data/processed/queries.jsonl")}
    ids = sorted(read_json("data/processed/splits.json")["dev"])
    qtorch = encode_queries(encoder, [queries[q] for q in ids])
    qort = np.concatenate(
        [
            ort_encoder.encode(
                [params()["encoder"]["prefix"] + queries[q]["question"] for q in ids[i : i + 16]]
            )
            for i in range(0, len(ids), 16)
        ]
    )
    encoder_error = float(np.max(np.abs(qtorch - qort)))
    windows = read_jsonl("indices/base/windows.jsonl")
    corpus = read_jsonl("data/processed/corpus.jsonl")
    vectors = np.load(path("indices/trained/vectors.npy"))
    bm25 = bm25s.BM25.load(str(path("indices/bm25")), load_corpus=False)
    import re

    lengths, counts = np.zeros(len(corpus)), np.zeros(len(corpus))
    parent_ids = np.asarray([w["passage_index"] for w in windows])
    for w in windows:
        p = w["passage_index"]
        lengths[p] = max(lengths[p], w["end"])
        counts[p] += 1
    ranker = CatBoostRanker()
    ranker.load_model(str(path(config["ltr_path"])))
    frozen = read_json("data/evaluation/dev_rankings.json")["CatBoost"]
    rows = []
    max_ce_error = 0.0
    for number, qid in enumerate(ids):
        question = queries[qid]["question"]
        lexical = bm25.get_scores(re.findall(r"\w+", question.lower()))
        rankings, selections = {}, {}
        reference_pairs, reference_logits = None, None
        pair_error = None
        for backend, qvector, model in [
            ("torch", qtorch[number], cross),
            ("ort", qort[number], ort_cross),
        ]:
            dense = np.full(len(corpus), -1.0, dtype=np.float32)
            np.maximum.at(dense, parent_ids, vectors @ qvector)
            fusion = fuse(lexical, dense, config["rrf_k"])
            candidates = order(fusion)[:100]
            chosen = best_windows(qvector, vectors, windows, candidates)
            pairs = [(question, windows[group[0]]["text"]) for group in chosen]
            logits = model.predict(pairs, batch_size=4)
            if backend == "torch":
                reference_pairs, reference_logits = pairs, logits
            else:
                matched = (
                    logits
                    if pairs == reference_pairs
                    else model.predict(reference_pairs, batch_size=4)
                )
                pair_error = float(np.max(np.abs(matched - reference_logits)))
                max_ce_error = max(max_ce_error, pair_error)
            matrix = features(
                question, candidates, lexical, dense, fusion, logits, corpus, lengths, counts
            )
            rankings[backend] = [corpus[p]["id"] for p in candidates[order(ranker.predict(matrix))]]
            selections[backend] = [group[0] for group in chosen]
        rows.append(
            {
                "qid": qid,
                "top20_equal": rankings["torch"][:20] == rankings["ort"][:20],
                "torch_equals_frozen": rankings["torch"][:20] == frozen[qid][:20],
                "ort_equals_frozen": rankings["ort"][:20] == frozen[qid][:20],
                "selected_window_ids_equal": selections["torch"] == selections["ort"],
                "ce_max_abs": pair_error,
                "torch": rankings["torch"][:20],
                "ort": rankings["ort"][:20],
            }
        )
        if number % 10 == 0:
            print(
                json.dumps(
                    {
                        "dev_queries": number + 1,
                        "total": len(ids),
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    result = {
        "encoder_max_abs": encoder_error,
        "ce_max_abs": max_ce_error,
        "top20_equal_queries": sum(r["top20_equal"] for r in rows),
        "torch_equals_frozen_queries": sum(r["torch_equals_frozen"] for r in rows),
        "ort_equals_frozen_queries": sum(r["ort_equals_frozen"] for r in rows),
        "queries": len(ids),
        "numeric_inputs": "118 prefixed DEV questions; all 11800 Torch-selected CE pairs, batch 4",
        "index": "Frozen trained window vectors shared by both backends, matching serving",
        "providers": {
            "encoder": ort_encoder.runtime.get_providers(),
            "cross_encoder": ort_cross.runtime.get_providers(),
        },
        "graph_sha256": {
            role: digest(out / (role + ".onnx")) for role in ["encoder", "cross_encoder"]
        },
        "source_sha256": {
            "encoder": digest(path(config["encoder_path"]) / "model.safetensors"),
            "cross_encoder": digest(path(config["cross_encoder_path"]) / "model.safetensors"),
            "ranker": digest(path(config["ltr_path"])),
        },
        "seconds": time.monotonic() - started,
        "rows": rows,
    }
    result["gate_passed"] = (
        encoder_error <= 1e-4 and max_ce_error <= 1e-4 and result["top20_equal_queries"] == 118
    )
    result["recommended_backend"] = "ort-cuda" if result["gate_passed"] else "torch"
    write_json("data/analysis/runtime/parity.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    del encoder, cross, ort_encoder, ort_cross
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
