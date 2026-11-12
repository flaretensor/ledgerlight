import gc
import json
import time

import numpy as np
import onnxruntime as ort
import torch
from catboost import CatBoostRanker

from ledgerlight.common import digest, params, path, read_json, read_jsonl, write_json
from ledgerlight.evaluation import evaluate
from ledgerlight.features import features
from ledgerlight.onnx_backend import CrossExport, EncoderExport, export, session
from ledgerlight.retrieval import best_windows, fuse, load_model, order, parent_scores


def benchmark(wrapper, tokenizer, kind, destination):
    cache = (
        read_json("data/serving/backend_timing_cache.json")
        if path("data/serving/backend_timing_cache.json").exists()
        else {"frozen_metrics_sha256": digest(path("data/evaluation/metrics.json")), "rows": []}
    )
    assert cache["frozen_metrics_sha256"] == digest(path("data/evaluation/metrics.json"))
    completed = [row for row in cache["rows"] if row["model"] == kind]
    if len(completed) == 8:
        print(
            json.dumps({"reused_completed_timing_cells": len(completed), "model": kind}), flush=True
        )
        return completed
    names = ["input_ids", "attention_mask", "token_type_ids"]
    wrapper.eval().cpu()
    export(wrapper, tokenizer, destination)
    cpu = session(destination, "CPUExecutionProvider")
    cuda = session(destination, "CUDAExecutionProvider")
    results = []
    for batch in [1, 8]:
        sample = next(
            w["text"]
            for w in read_jsonl("indices/base/windows.jsonl")
            if len(w["input_ids"]) == 510
        )
        question = read_jsonl("data/processed/queries.jsonl")[0]["question"]
        if kind == "encoder":
            tokens = tokenizer(
                [sample] * batch,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
        else:
            tokens = tokenizer(
                [question] * batch,
                [sample] * batch,
                padding="max_length",
                max_length=512,
                truncation="only_second",
                return_tensors="pt",
            )
        arrays = {k: tokens[k].numpy() for k in names}
        with torch.inference_mode():
            expected = wrapper(*(tokens[k] for k in names)).numpy()
        for backend in ["torch-cpu", "ort-cpu", "torch-cuda", "ort-cuda"]:
            if backend.startswith("torch"):
                device = "cuda" if backend.endswith("cuda") else "cpu"
                wrapper.to(device)

                def run():
                    local = tuple(tokens[k].to(device) for k in names)
                    with torch.inference_mode():
                        result = wrapper(*local)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    return result.detach().cpu().numpy()
            else:
                selected = cuda if backend == "ort-cuda" else cpu

                def run():
                    return selected.run(None, arrays)[0]

            for _ in range(3):
                actual = run()
            values = []
            for _ in range(20):
                started = time.perf_counter()
                actual = run()
                values.append((time.perf_counter() - started) * 1000)
            max_error = float(np.max(np.abs(actual - expected)))
            if not np.allclose(actual, expected, atol=1e-4, rtol=1e-3):
                raise ValueError(f"Numeric parity failed: {kind}/{backend}: {max_error}")
            results.append(
                {
                    "model": kind,
                    "batch": batch,
                    "sequence": 512,
                    "real_tokens_per_input": int(
                        tokens["attention_mask"].sum(dim=1).float().mean()
                    ),
                    "backend": backend,
                    "median_ms": float(np.median(values)),
                    "p95_ms": float(np.quantile(values, 0.95)),
                    "max_abs_error": max_error,
                    "repeats": 20,
                }
            )
            print(json.dumps(results[-1]), flush=True)
            cache["rows"].append(results[-1])
            write_json("data/serving/backend_timing_cache.json", cache)
        wrapper.cpu()
    return results


def run_encoder(session, tokenizer, windows):
    results = []
    for offset in range(0, len(windows), 16):
        items = [
            tokenizer.prepare_for_model(
                w["input_ids"], add_special_tokens=True, return_attention_mask=True
            )
            for w in windows[offset : offset + 16]
        ]
        batch = tokenizer.pad(items, padding=True, return_tensors="np")
        arrays = {
            key: batch[key].astype(np.int64)
            for key in ["input_ids", "attention_mask", "token_type_ids"]
        }
        results.append(session.run(None, arrays)[0])
        if offset % 1600 == 0:
            print(
                json.dumps(
                    {
                        "encoder_windows": min(offset + 16, len(windows)),
                        "total": len(windows),
                        "provider": session.get_providers()[0],
                    }
                ),
                flush=True,
            )
    return np.concatenate(results)


def relevance(encoder_session, ce_session, tokenizer, ce_tokenizer, config, verify_windows=False):
    windows = read_jsonl("indices/base/windows.jsonl")
    corpus = read_jsonl("data/processed/corpus.jsonl")
    all_queries = read_jsonl("data/processed/queries.jsonl")
    dev = set(read_json("data/processed/splits.json")["dev"])
    queries = [q for q in all_queries if q["id"] in dev]
    saved = np.load(path(f"indices/{config['index']}/vectors.npy"))
    vector_error = None
    if verify_windows:
        cache_file = path(
            ".cache/backend-vectors/" + digest(path("models/onnx/encoder/model.onnx")) + ".npy"
        )
        if cache_file.exists():
            encoded = np.load(cache_file)
        else:
            encoded = run_encoder(encoder_session, tokenizer, windows)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, encoded)
        vector_error = float(np.max(np.abs(saved - encoded)))
    vectors = saved
    qarrays = tokenizer(
        [params()["encoder"]["prefix"] + q["question"] for q in queries],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )
    qvectors = encoder_session.run(
        None,
        {k: qarrays[k].astype(np.int64) for k in ["input_ids", "attention_mask", "token_type_ids"]},
    )[0]
    parents = np.asarray([w["passage_index"] for w in windows])
    dense = parent_scores(qvectors, vectors, parents, len(corpus))
    query_lookup = {q["id"]: i for i, q in enumerate(all_queries)}
    lexical = np.load(path("indices/bm25/scores.npy"))
    lengths, counts = np.zeros(len(corpus)), np.zeros(len(corpus))
    for w in windows:
        lengths[w["passage_index"]] = max(lengths[w["passage_index"]], w["end"])
        counts[w["passage_index"]] += 1
    ranker = None
    if config["ltr_path"]:
        ranker = CatBoostRanker()
        ranker.load_model(str(path(config["ltr_path"])))
    reference_rankings = None
    if config["mode"] == "ltr":
        from ledgerlight.training import candidate_context, rank_ltr

        reference_rankings = rank_ltr(
            ranker,
            read_json("data/training/ltr_scores.json")["dev"],
            candidate_context(config["index"]),
        )
    rankings = {}
    for number, query in enumerate(queries):
        lex = lexical[query_lookup[query["id"]]]
        fusion = fuse(lex, dense[number], config["rrf_k"])
        candidates = order(fusion)[:100]
        mode = config["mode"]
        if mode == "bm25":
            ranked = order(lex)
        elif mode == "dense":
            ranked = order(dense[number])
        elif mode == "rrf":
            ranked = np.asarray([p for p in order(fusion) if fusion[p] > 0])
        else:
            chosen = best_windows(qvectors[number], vectors, windows, candidates)
            scores = []
            pairs = [(query["question"], windows[group[0]]["text"]) for group in chosen]
            for start in range(0, len(pairs), 8):
                subset = pairs[start : start + 8]
                batch = ce_tokenizer(
                    [p[0] for p in subset],
                    [p[1] for p in subset],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="np",
                )
                scores.extend(
                    ce_session.run(
                        None,
                        {
                            k: batch[k].astype(np.int64)
                            for k in ["input_ids", "attention_mask", "token_type_ids"]
                        },
                    )[0].tolist()
                )
            ce = np.asarray(scores)
            if mode == "ltr":
                matrix = features(
                    query["question"],
                    candidates,
                    lex,
                    dense[number],
                    fusion,
                    ce,
                    corpus,
                    lengths,
                    counts,
                )
                ranked = candidates[order(ranker.predict(matrix))]
            else:
                depth = config["ce_depth"]
                ranked = np.concatenate([candidates[:depth][order(ce[:depth])], candidates[depth:]])
        rankings[query["id"]] = [corpus[p]["id"] for p in ranked]
        if number % 25 == 0:
            print(
                json.dumps(
                    {
                        "dev_parity_queries": number + 1,
                        "total": len(queries),
                        "provider": encoder_session.get_providers()[0],
                    }
                ),
                flush=True,
            )
    metrics, _ = evaluate(rankings, read_json("data/processed/labels.json"))
    reference = read_json("data/evaluation/metrics.json")["dev"][config["method"]]
    return {
        "dev": metrics,
        "torch_dev": reference,
        "delta": {k: metrics[k] - reference[k] for k in metrics},
        "window_vector_max_abs_error": vector_error,
        "ranking_index": "Frozen Torch window vectors, as used by live serving",
        "queries": len(queries),
        "exact_top20_queries": sum(
            rankings[qid][:20] == reference_rankings[qid][:20] for qid in rankings
        )
        if reference_rankings
        else None,
        "mean_top20_overlap": float(
            np.mean(
                [
                    len(set(ranks[:20]) & set(reference_rankings[qid][:20])) / 20
                    for qid, ranks in rankings.items()
                ]
            )
        )
        if reference_rankings
        else None,
    }


def main():
    frozen_digest = digest(path("data/evaluation/metrics.json"))
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    ort.preload_dlls(directory="")
    config = read_json("models/production/config.json")
    root = path("models/onnx")
    (root / "encoder").mkdir(parents=True, exist_ok=True)
    (root / "cross_encoder").mkdir(exist_ok=True)
    encoder = load_model(
        model_path=path(config["encoder_path"]) if config["encoder_path"] else None, device="cpu"
    )
    cross = load_model(
        "cross_encoder",
        path(config["cross_encoder_path"]) if config["cross_encoder_path"] else None,
        device="cpu",
    )
    results = benchmark(
        EncoderExport(encoder), encoder.tokenizer, "encoder", root / "encoder/model.onnx"
    )
    results += benchmark(
        CrossExport(cross), cross.tokenizer, "cross_encoder", root / "cross_encoder/model.onnx"
    )
    gc.collect()
    torch.cuda.empty_cache()
    parity = relevance(
        session(root / "encoder/model.onnx", "CUDAExecutionProvider"),
        session(root / "cross_encoder/model.onnx", "CUDAExecutionProvider"),
        encoder.tokenizer,
        cross.tokenizer,
        config,
        verify_windows=True,
    )
    cpu_parity = relevance(
        session(root / "encoder/model.onnx", "CPUExecutionProvider"),
        session(root / "cross_encoder/model.onnx", "CPUExecutionProvider"),
        encoder.tokenizer,
        cross.tokenizer,
        config,
    )
    assert digest(path("data/evaluation/metrics.json")) == frozen_digest
    result = {
        "benchmarks": results,
        "retrieval_parity": parity,
        "cpu_retrieval_parity": cpu_parity,
        "experiment_split": "dev_only_post_freeze_backend_validation",
        "frozen_test_untouched": True,
        "frozen_test_metrics_sha256": frozen_digest,
        "config": config,
        "protocol": "Full financial windows at 512 tokens; 3 warmups, 20 timed repetitions; transfer included, tokenization excluded, TF32 disabled",
        "ranking_protocol": "Both providers evaluate all 118 DEV queries against the frozen serving index. CUDA also checks every window embedding. CPU window numerical parity uses the completed batch-1/8 checks. CE inference capped at 8 after the retained 1 GiB arena failure at batch 32.",
    }
    write_json("data/serving/backends.json", result)
    from ledgerlight.evaluation import tracking

    tracking(
        "onnx-backend-parity",
        {
            "window_vector_max_abs_error": parity["window_vector_max_abs_error"],
            "dev_ndcg_delta": parity["delta"]["ndcg@10"],
        },
        {"precision": "float32", "providers": "CPU,CUDA"},
        result,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
