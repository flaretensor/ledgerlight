import argparse
import gc
import json
import os
import re
import shutil
import time

import numpy as np
import torch

from ledgerlight.common import params, path, read_json, read_jsonl, write_json, write_jsonl
from ledgerlight.evaluation import baseline, evaluate
from ledgerlight.features import query_tickers, query_years
from ledgerlight.retrieval import (
    best_windows,
    cross_scores,
    encode_queries,
    encode_windows,
    fuse,
    lexical_index,
    load_model,
    make_windows,
    order,
    parent_scores,
)


def emit(event, **values):
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


def release(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def paired(old, new):
    delta = np.asarray(new) - np.asarray(old)
    rng = np.random.default_rng(42)
    means = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    return {"delta": float(delta.mean()), "ci95": np.quantile(means, [0.025, 0.975]).tolist()}


def feature_changes(queries, splits, corpus):
    tickers = {s["ticker"] for p in corpus for s in p["sources"]}
    years = {s["year"] for p in corpus for s in p["sources"]}
    result = {}
    for split in ["train", "dev"]:
        selected = set(splits[split])
        changed_ticker = changed_year = 0
        for query in queries:
            if query["id"] not in selected:
                continue
            tokens = set(re.findall(r"\w+", query["question"].casefold()))
            old_tickers = {t for t in tickers if t.casefold() in tokens}
            changed_ticker += old_tickers != query_tickers(query["question"]) | old_tickers
            changed_year += years.intersection(tokens) != years.intersection(
                query_years(query["question"])
            )
        result[split] = {
            "ticker_match_changed_questions": changed_ticker,
            "year_match_changed_questions": changed_year,
            "questions": len(selected),
        }
    return result


def compare():
    started = time.monotonic()
    corpus = read_jsonl("data/processed/corpus.jsonl")
    queries = read_jsonl("data/processed/queries.jsonl")
    splits = read_json("data/processed/splits.json")
    assert {key: len(value) for key, value in splits.items()} == {
        "train": 948,
        "dev": 118,
        "test": 119,
    }
    allowed = set(splits["train"]) | set(splits["dev"])
    active = [(i, q) for i, q in enumerate(queries) if q["id"] in allowed]
    labels = {
        qid: value
        for qid, value in read_json("data/processed/labels.json").items()
        if qid in allowed
    }
    assert len(active) == 1066
    lexical_index()
    lexical = np.load(path("indices/bm25/scores.npy"))
    docids = [p["id"] for p in corpus]
    bm25 = {
        q["id"]: [docids[p] for p in order(lexical[i])]
        for i, q in active
        if q["id"] in set(splits["dev"])
    }
    bm_metrics, _ = evaluate(bm25, labels)
    previous = read_json("docs/previous_iteration.json")["dev"]["BM25"]
    for metric in ["ndcg@10", "recall@100"]:
        assert abs(bm_metrics[metric] - previous[metric]) < 1e-10, (
            metric,
            bm_metrics[metric],
            previous[metric],
        )
    emit("bm25_dev_parity", metrics=bm_metrics)
    encoder = load_model()
    active_vectors = encode_queries(encoder, [q for _, q in active])
    qvectors = np.full((len(queries), active_vectors.shape[1]), np.nan, dtype=np.float32)
    qvectors[[i for i, _ in active]] = active_vectors
    times = {}
    variants = {}
    for name, limit, step in [("old", 510, 384), ("new", 448, 320)]:
        begin = time.monotonic()
        windows = make_windows(encoder.tokenizer, corpus, limit, step)
        vectors = encode_windows(encoder, windows, params()["encoder"]["batch_size"])
        parents = np.asarray([w["passage_index"] for w in windows], dtype=np.int32)
        scores = parent_scores(qvectors, vectors, parents, len(corpus))
        destination = "indices/window-" + name
        directory = path(destination)
        directory.mkdir(parents=True, exist_ok=True)
        for filename, array in [
            ("vectors", vectors),
            ("queries", qvectors),
            ("parents", parents),
            ("scores", scores),
        ]:
            np.save(directory / (filename + ".npy"), array)
        write_jsonl(destination + "/windows.jsonl", windows)
        variants[name] = {
            "scores": scores,
            "vectors": vectors,
            "windows": windows,
            "destination": destination,
        }
        times[name + "_index_seconds"] = time.monotonic() - begin
        emit(
            "window_invariants",
            variant=name,
            windows=len(windows),
            parents=len({w["passage_index"] for w in windows}),
            max_tokens=max(len(w["input_ids"]) for w in windows),
            roundtrip_failures=0,
            tail_failures=0,
            seconds=times[name + "_index_seconds"],
        )
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    ce = load_model("cross_encoder")
    all_metrics, all_per_query, overflow = {}, {}, {}
    for name, item in variants.items():
        begin = time.monotonic()
        dense_rankings, ce_rankings = {}, {}
        dev_total = dev_over = 0
        for number, (index, query) in enumerate(active):
            qid = query["id"]
            dense_rankings[qid] = [docids[p] for p in order(item["scores"][index])]
            fusion = fuse(lexical[index], item["scores"][index], 10)
            candidates = order(fusion)[:100]
            if qid in set(splits["dev"]):
                selected = best_windows(
                    qvectors[index], item["vectors"], item["windows"], candidates
                )
                qlength = len(ce.tokenizer.encode(query["question"], add_special_tokens=False))
                for group in selected:
                    length = (
                        qlength
                        + len(
                            ce.tokenizer.encode(
                                item["windows"][group[0]]["text"], add_special_tokens=False
                            )
                        )
                        + 3
                    )
                    dev_over += int(length > 512)
                    dev_total += 1
            scores = cross_scores(
                ce,
                query["question"],
                candidates[:50],
                qvectors[index],
                item["vectors"],
                item["windows"],
            )
            ranked = np.concatenate([candidates[:50][order(scores)], candidates[50:]])
            ce_rankings[qid] = [docids[p] for p in ranked]
            if number % 100 == 0:
                emit(
                    "window_ce_progress",
                    variant=name,
                    completed=number + 1,
                    total=len(active),
                    seconds=time.monotonic() - begin,
                )
        all_metrics[name], all_per_query[name] = {}, {}
        for partition, ids in [
            ("train", splits["train"]),
            ("dev", splits["dev"]),
            ("combined", list(allowed)),
        ]:
            all_metrics[name][partition], all_per_query[name][partition] = {}, {}
            for method, rankings in [("BGE", dense_rankings), ("CE-zero", ce_rankings)]:
                selected = {qid: rankings[qid] for qid in ids}
                metrics, per_query = evaluate(selected, labels)
                all_metrics[name][partition][method] = metrics
                all_per_query[name][partition][method] = per_query
        overflow[name] = {"pairs": dev_total, "overflow": dev_over, "rate": dev_over / dev_total}
        times[name + "_ce_seconds"] = time.monotonic() - begin
    del ce
    gc.collect()
    torch.cuda.empty_cache()
    comparisons = {}
    for partition in ["train", "dev", "combined"]:
        comparisons[partition] = {}
        for method, metrics in [
            ("BGE", ["recall@20", "recall@100", "ndcg@10"]),
            ("CE-zero", ["ndcg@10"]),
        ]:
            comparisons[partition][method] = {
                metric: paired(
                    all_per_query["old"][partition][method][metric],
                    all_per_query["new"][partition][method][metric],
                )
                for metric in metrics
            }
    passed = all(
        comparisons["combined"]["BGE"][metric]["ci95"][1] >= 0
        for metric in ["recall@100", "ndcg@10"]
    )
    changes = feature_changes(queries, splits, corpus)
    result = {
        "metrics": all_metrics,
        "comparisons": comparisons,
        "dev_pair_overflow": overflow,
        "feature_changes": changes,
        "times": times,
        "seconds": time.monotonic() - started,
        "gate_passed": passed,
        "rrf_k": 10,
        "ce_depth": 50,
        "test_evaluated": False,
        "construction": "same original-text/pre-token-boundary builder and shared CE policy in both arms",
    }
    write_json("data/baselines/window_comparison.json", result)
    emit("window_comparison", **result)
    if not passed:
        return False
    base = path("indices/base")
    base.mkdir(parents=True, exist_ok=True)
    for p in path("indices/window-new").iterdir():
        shutil.copy2(p, base / p.name)
    write_json(
        "indices/base/config.json",
        {
            "revision": params()["encoder"]["revision"],
            "windows": len(variants["new"]["windows"]),
            "aggregation": "max",
            "dtype": "float32",
        },
    )
    return True


def parity_and_registry():
    import mlflow

    from ledgerlight.onnx_backend import CrossExport, EncoderExport, export, session
    from ledgerlight.pairs import pair_inputs, torch_pair_scores
    from ledgerlight.registry import CrossEncoderAdapter, EncoderAdapter, RankerAdapter

    encoder_selection = read_json("data/training/encoder_selected.json")
    ce_selection = read_json("data/training/ce_selected.json")
    ltr = read_json("data/training/ltr_selected.json")
    splits = read_json("data/processed/splits.json")
    queries = {q["id"]: q["question"] for q in read_jsonl("data/processed/queries.jsonl")}
    windows = read_jsonl("indices/base/windows.jsonl")
    pairs = [(queries[qid], windows[i]["text"]) for i, qid in enumerate(splits["dev"][:12])]
    encoder = load_model(model_path=path(encoder_selection["path"]), device="cpu")
    ce = load_model("cross_encoder", path(ce_selection["path"]), device="cpu")
    encoder_target = path("models/onnx/encoder.onnx")
    ce_target = path("models/onnx/cross_encoder.onnx")
    export(EncoderExport(encoder), encoder.tokenizer, encoder_target)
    export(CrossExport(ce), ce.tokenizer, ce_target)
    ce_inputs = pair_inputs(ce.tokenizer, pairs, return_tensors="np")
    enc_text = [params()["encoder"]["prefix"] + question for question, _ in pairs]
    enc_inputs = encoder.tokenizer(
        enc_text, padding=True, truncation=True, max_length=512, return_tensors="np"
    )
    ce_values = torch_pair_scores(ce, pairs)
    enc_values = encoder.encode(enc_text, normalize_embeddings=True)
    ce_session = session(ce_target, "CPUExecutionProvider")
    enc_session = session(encoder_target, "CPUExecutionProvider")
    ce_ort = ce_session.run(
        None,
        {
            k: ce_inputs[k].astype(np.int64)
            for k in ["input_ids", "attention_mask", "token_type_ids"]
        },
    )[0].reshape(-1)
    enc_ort = enc_session.run(
        None,
        {
            k: enc_inputs[k].astype(np.int64)
            for k in ["input_ids", "attention_mask", "token_type_ids"]
        },
    )[0]
    errors = {
        "encoder_max_abs": float(np.max(np.abs(enc_ort - enc_values))),
        "ce_max_abs": float(np.max(np.abs(ce_ort - ce_values))),
        "pairs": len(pairs),
        "provider": "CPUExecutionProvider",
    }
    assert errors["encoder_max_abs"] < 1e-4 and errors["ce_max_abs"] < 1e-3
    mlflow.set_experiment("ledgerlight-pilot")
    registered = {}
    for kind, adapter, source, requirements in [
        (
            "encoder",
            EncoderAdapter(),
            encoder_selection["path"],
            ["sentence-transformers==5.2.3", "torch==2.8.0"],
        ),
        (
            "cross-encoder",
            CrossEncoderAdapter(),
            ce_selection["path"],
            ["sentence-transformers==5.2.3", "torch==2.8.0"],
        ),
        ("ranker", RankerAdapter(), ltr["path"], ["catboost==1.2.10"]),
    ]:
        with mlflow.start_run(run_name="pilot-register-" + kind):
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=adapter,
                artifacts={"weights": str(path(source))},
                code_paths=["src/ledgerlight"],
                registered_model_name="ledgerlight-pilot-" + kind,
                pip_requirements=requirements,
            )
        registered[kind] = str(info.registered_model_version)
    emit("parity_registry", parity=errors, registered=registered)


def train():
    from catboost import CatBoostRanker

    from ledgerlight.training import (
        candidate_context,
        ce_rankings,
        neural_cross_encoder,
        neural_encoder,
        prepare,
        rank_ltr,
        train_ltr,
    )

    gate = read_json("data/baselines/window_comparison.json")
    if not gate["gate_passed"]:
        raise RuntimeError("Window comparison stopped the pilot")
    times = {}
    for name, function in [
        ("baselines", baseline),
        ("prepare", prepare),
        ("encoder", neural_encoder),
        ("cross_encoder", neural_cross_encoder),
        ("ltr", train_ltr),
    ]:
        started = time.monotonic()
        function()
        times[name] = time.monotonic() - started
        emit("pilot_stage", stage=name, seconds=times[name])
        if name == "prepare":
            split = read_json("data/training/split.json")
            assert len(split["A"]) == 712 and len(split["B"]) == 236
            emit("neural_partition", A=len(split["A"]), B=len(split["B"]))
    labels = read_json("data/processed/labels.json")
    baseline_result = read_json("data/baselines/metrics.json")
    ladder = dict(baseline_result["dev"])
    context = candidate_context()
    cross = read_json("data/training/ce_selected.json")
    ladder["BGE-ft"] = read_json("data/training/ft_encoder_zero_ce.json")["metrics"]
    ladder["CE-ft"], _ = evaluate(
        ce_rankings(read_json(f"data/training/ce_trial_{cross['trial']}.json"), context), labels
    )
    ltr = read_json("data/training/ltr_selected.json")
    ranker = CatBoostRanker()
    ranker.load_model(str(path(ltr["path"])))
    ladder["CatBoost"], _ = evaluate(
        rank_ltr(
            ranker,
            read_json("data/training/ltr_scores.json")["dev"],
            candidate_context(ltr["index"]),
        ),
        labels,
    )
    dev = read_json("data/processed/splits.json")["dev"]
    for name in ["BGE-ft-dense", "RRF-ft"]:
        rankings = {}
        for qid in dev:
            index = context["lookup"][qid]
            scores = (
                context["dense"][index]
                if name == "BGE-ft-dense"
                else fuse(context["lexical"][index], context["dense"][index], context["k"])
            )
            rankings[qid] = [
                context["corpus"][p]["id"]
                for p in order(scores)
                if name != "RRF-ft" or scores[p] > 0
            ]
        ladder[name], _ = evaluate(rankings, labels)
    started = time.monotonic()
    parity_and_registry()
    times["parity_registry"] = time.monotonic() - started
    emit(
        "pilot_finished",
        dev_ladder=ladder,
        label_audit=ltr["label_audit"],
        times=times,
        interpretation="Truncated smoke fits; not final performance",
        test_evaluated=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["compare", "train", "all"], default="all")
    args = parser.parse_args()
    if os.environ.get("LEDGERLIGHT_RUN_MODE") != "pilot":
        raise SystemExit("Set LEDGERLIGHT_RUN_MODE=pilot")
    if args.phase in {"compare", "all"} and not compare():
        emit("pilot_stopped", reason="Negative upper confidence bound for the window gate")
        return
    if args.phase in {"train", "all"}:
        train()


if __name__ == "__main__":
    main()
