import argparse
import gc
import json
import os
import re
import time

import bm25s
import numpy as np
import torch
from catboost import CatBoostRanker

from ledgerlight.common import params, path, read_json, read_jsonl, write_json
from ledgerlight.evaluation import evaluate
from ledgerlight.features import features
from ledgerlight.inference import RuntimeCrossEncoder, RuntimeEncoder
from ledgerlight.onnx_backend import OnnxCrossEncoder, OnnxEncoder
from ledgerlight.retrieval import best_windows, encode_queries, fuse, load_model, order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        choices=["torch", "ort", "openvino", "triton-ort", "triton-openvino", "openvino-gpu"],
        required=True,
    )
    args = parser.parse_args()
    target = "data/analysis/stage2/parity/" + args.engine + ".json"
    if path(target).exists():
        raise RuntimeError("Parity result already exists; do not repeat a completed measurement")
    clock = time.monotonic()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = read_json("models/production/config.json")
    ids = sorted(read_json("data/processed/splits.json")["dev"])
    questions = {q["id"]: q for q in read_jsonl("data/processed/queries.jsonl") if q["id"] in ids}
    corpus = read_jsonl("data/processed/corpus.jsonl")
    windows = read_jsonl("indices/base/windows.jsonl")
    vectors = np.load(path("indices/trained/vectors.npy"))
    parents = np.asarray([w["passage_index"] for w in windows])
    lengths, counts = np.zeros(len(corpus)), np.zeros(len(corpus))
    for w in windows:
        p = w["passage_index"]
        lengths[p] = max(lengths[p], w["end"])
        counts[p] += 1
    lexical = bm25s.BM25.load(str(path("indices/bm25")), load_corpus=False)
    ranker = CatBoostRanker()
    ranker.load_model(str(path(config["ltr_path"])))
    encoder_source, ce_source = path(config["encoder_path"]), path(config["cross_encoder_path"])
    engine = args.engine
    if engine == "torch":
        encoder = load_model("encoder", encoder_source)
        ce = load_model("cross_encoder", ce_source)
        qvectors = encode_queries(encoder, [questions[q] for q in ids])
        reference = None
    else:
        reference = read_json("data/analysis/stage2/parity/torch.json")
        os.environ["LEDGERLIGHT_TRITON"] = "1" if engine.startswith("triton-") else "0"
        os.environ["LEDGERLIGHT_TRITON_URL"] = "127.0.0.1:8002"
        os.environ["LEDGERLIGHT_OPENVINO_DEVICE"] = "GPU" if engine == "openvino-gpu" else "CPU"
        if engine == "ort":
            encoder = OnnxEncoder(encoder_source, "2", "CUDAExecutionProvider")
            ce = OnnxCrossEncoder(ce_source, "2", "CUDAExecutionProvider")
        else:
            encoder = RuntimeEncoder(encoder_source, "2", engine)
            ce = RuntimeCrossEncoder(ce_source, "2", engine)
        qvectors = np.concatenate(
            [
                encoder.encode(
                    [
                        params()["encoder"]["prefix"] + questions[q]["question"]
                        for q in ids[i : i + 16]
                    ]
                )
                for i in range(0, len(ids), 16)
            ]
        )
    frozen = read_json("data/evaluation/dev_rankings.json")["CatBoost"]
    rows, rankings = [], {}
    encoder_error = (
        0
        if reference is None
        else float(
            np.max(np.abs(qvectors - np.asarray(reference["query_vectors"], dtype=np.float32)))
        )
    )
    ce_error = 0.0
    for number, qid in enumerate(ids):
        question = questions[qid]["question"]
        qvector = qvectors[number]
        bm25 = lexical.get_scores(re.findall(r"\w+", question.lower()))
        dense = np.full(len(corpus), -1, dtype=np.float32)
        np.maximum.at(dense, parents, vectors @ qvector)
        fusion = fuse(bm25, dense, config["rrf_k"])
        candidates = order(fusion)[:100]
        selected = [v[0] for v in best_windows(qvector, vectors, windows, candidates)]
        pairs = [(question, windows[v]["text"]) for v in selected]
        logits = ce.predict(pairs, batch_size=4)
        numeric_logits = logits
        if reference and selected != reference["rows"][number]["selected_windows"]:
            numeric_logits = ce.predict(
                [
                    (question, windows[v]["text"])
                    for v in reference["rows"][number]["selected_windows"]
                ],
                batch_size=4,
            )
        pair_error = (
            0
            if reference is None
            else float(
                np.max(
                    np.abs(
                        numeric_logits
                        - np.asarray(reference["rows"][number]["logits"], dtype=np.float32)
                    )
                )
            )
        )
        ce_error = max(ce_error, pair_error)
        matrix = features(
            question, candidates, bm25, dense, fusion, logits, corpus, lengths, counts
        )
        predictions = np.asarray(ranker.predict(matrix))
        positions = order(predictions)
        ranking = [corpus[p]["id"] for p in candidates[positions]]
        scores = predictions[positions].tolist()
        rankings[qid] = ranking
        ref = frozen[qid][:20] if reference is None else reference["rows"][number]["ranking"][:20]
        differing = [i for i in range(20) if ranking[i] != ref[i]]
        rows.append(
            {
                "qid": qid,
                "ranking": ranking,
                "scores": scores,
                "selected_windows": selected,
                "logits": logits.tolist(),
                "ce_max_abs": pair_error,
                "equals_frozen_top20": ranking[:20] == frozen[qid][:20],
                "differs_reference_positions": [i + 1 for i in differing],
                "score_gaps": [
                    {
                        "position": i + 1,
                        "passage": ranking[i],
                        "next_gap": scores[i] - scores[i + 1],
                    }
                    for i in differing
                ],
            }
        )
        if number % 20 == 0:
            print(json.dumps({"engine": engine, "queries": number + 1}), flush=True)
    metrics, _ = evaluate(
        rankings, {qid: read_json("data/processed/labels.json")[qid] for qid in ids}
    )
    result = {
        "engine": engine,
        "queries": len(ids),
        "encoder_max_abs": encoder_error,
        "ce_max_abs": ce_error,
        "metrics": metrics,
        "equals_frozen_top20": sum(r["equals_frozen_top20"] for r in rows),
        "equals_torch_top20": sum(not r["differs_reference_positions"] for r in rows)
        if reference
        else len(rows),
        "query_vectors": qvectors.tolist(),
        "rows": rows,
        "seconds": time.monotonic() - clock,
        "numeric_protocol": "Same batched DEV inputs and 11800 reference CE pairs as canonical_parity.py; frozen corpus vectors",
    }
    result["passed"] = (
        encoder_error <= 1e-4
        and ce_error <= 1e-4
        and abs(metrics["ndcg@10"] - 0.4367276797012511) <= 0.005
    )
    write_json(target, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"query_vectors", "rows"}}, indent=2
        )
    )
    del encoder, ce
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
