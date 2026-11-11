import gc
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import optuna
import torch

from ledgerlight.common import params, path, read_json, read_jsonl, write_json
from ledgerlight.data import normalize
from ledgerlight.evaluation import evaluate, tracking
from ledgerlight.retrieval import (
    embed,
    encode_queries,
    encode_windows,
    load_model,
    order,
    parent_scores,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare():
    queries = read_jsonl("data/processed/queries.jsonl")
    corpus = read_jsonl("data/processed/corpus.jsonl")
    labels = read_json("data/processed/labels.json")
    splits = read_json("data/processed/splits.json")
    lookup = {q["id"]: i for i, q in enumerate(queries)}
    groups = defaultdict(list)
    for qid in splits["train"]:
        groups[normalize(queries[lookup[qid]]["question"])].append(qid)
    keys = sorted(groups)
    random.Random(params()["training"]["seed"]).shuffle(keys)
    cut = round(len(keys) * params()["training"]["neural_fraction"])
    a = [qid for key in keys[:cut] for qid in groups[key]]
    b = [qid for key in keys[cut:] for qid in groups[key]]
    write_json("data/training/split.json", {"A": a, "B": b, "group_counts": [cut, len(keys) - cut]})
    lexical = np.load(path("indices/bm25/scores.npy"))
    dense = np.load(path("indices/base/scores.npy"))
    qvectors = np.load(path("indices/base/queries.npy"))
    vectors = np.load(path("indices/base/vectors.npy"))
    windows = read_jsonl("indices/base/windows.jsonl")
    doc_lookup = {p["id"]: i for i, p in enumerate(corpus)}
    canonical = {}
    text_owner = {}
    for i, passage in enumerate(corpus):
        key = " ".join(passage["text"].split())
        canonical[i] = text_owner.setdefault(key, i)
    parent_groups = defaultdict(list)
    for wid, w in enumerate(windows):
        parent_groups[w["passage_index"]].append(wid)
    examples = []
    for qid in a:
        index = lookup[qid]
        relevant = {doc_lookup[p] for p in labels[qid]["positives"]}
        positive_texts = {" ".join(corpus[p]["text"].split()) for p in relevant}
        pool = [doc_lookup[p] for p in labels[qid]["hard_negatives"]]
        for scores in (lexical[index], dense[index]):
            pool.extend(order(scores)[4:50].tolist())
        negatives = list(
            dict.fromkeys(
                p
                for p in pool
                if p not in relevant and " ".join(corpus[p]["text"].split()) not in positive_texts
            )
        )
        if not negatives:
            continue
        scores = vectors @ qvectors[index]
        chosen = {}
        for pid in relevant | set(negatives):
            ids = np.asarray(parent_groups[pid])
            chosen[pid] = int(ids[order(scores[ids])[0]])
        for positive in sorted(relevant):
            examples.append(
                {
                    "qid": qid,
                    "question": queries[index]["question"],
                    "positive": positive,
                    "positive_key": canonical[positive],
                    "positive_window": chosen[positive],
                    "negative_windows": [chosen[p] for p in negatives],
                    "negative_passages": negatives,
                    "all_positives": sorted({canonical[p] for p in relevant}),
                    "negative_keys": [canonical[p] for p in negatives],
                }
            )
    write_json("data/training/examples.json", examples)
    print(json.dumps({"A": len(a), "B": len(b), "neural_pairs": len(examples)}), flush=True)


def safe_batches(examples, batch_size, rng):
    shuffled = list(examples)
    rng.shuffle(shuffled)
    batch, all_documents, all_positive_texts = [], set(), set()
    for item in shuffled:
        positive = set(item["all_positives"])
        docs = {item["positive_key"], *item["selected_negative_keys"]}
        if batch and (
            len(batch) >= batch_size or docs & all_positive_texts or positive & all_documents
        ):
            yield batch
            batch, all_documents, all_positive_texts = [], set(), set()
        batch.append(item)
        all_documents |= docs
        all_positive_texts |= positive
    if batch:
        yield batch


def fit_encoder(config, seed, destination):
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss

    seed_everything(seed)
    model = load_model()
    model.train()
    examples = read_json("data/training/examples.json")
    windows = read_jsonl("indices/base/windows.jsonl")
    rng = random.Random(seed)
    loss_fn = CachedMultipleNegativesRankingLoss(model, scale=config["scale"], mini_batch_size=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    started = time.monotonic()
    losses, steps = [], 0
    for epoch in range(config["epochs"]):
        selected = []
        for example in examples:
            positions = rng.sample(
                range(len(example["negative_windows"])),
                min(config["negatives"], len(example["negative_windows"])),
            )
            while len(positions) < config["negatives"]:
                positions.append(positions[0])
            selected.append(
                {
                    **example,
                    "selected_negatives": [example["negative_passages"][i] for i in positions],
                    "selected_negative_keys": [example["negative_keys"][i] for i in positions],
                    "selected_windows": [example["negative_windows"][i] for i in positions],
                }
            )
        for batch in safe_batches(selected, config["batch"], rng):
            columns = [
                [params()["encoder"]["prefix"] + e["question"] for e in batch],
                [windows[e["positive_window"]]["text"] for e in batch],
            ]
            for number in range(config["negatives"]):
                columns.append([windows[e["selected_windows"][number]]["text"] for e in batch])
            tokenized = [{k: v.to("cuda") for k, v in model.tokenize(c).items()} for c in columns]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(tokenized, None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
            if steps % 50 == 0:
                print(
                    json.dumps(
                        {
                            "encoder_step": steps,
                            "loss": losses[-1],
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    model.save_pretrained(str(path(destination)), create_model_card=False)
    seconds = time.monotonic() - started
    del optimizer, loss_fn
    model.eval()
    return model, {"training_seconds": seconds, "mean_loss": float(np.mean(losses)), "steps": steps}


def encoder_metrics(model):
    windows = read_jsonl("indices/base/windows.jsonl")
    corpus = read_jsonl("data/processed/corpus.jsonl")
    queries = read_jsonl("data/processed/queries.jsonl")
    dev = set(read_json("data/processed/splits.json")["dev"])
    selected = [q for q in queries if q["id"] in dev]
    vectors = encode_windows(model, windows)
    qvectors = encode_queries(model, selected)
    parents = np.asarray([w["passage_index"] for w in windows])
    scores = parent_scores(qvectors, vectors, parents, len(corpus))
    rankings = {
        q["id"]: [corpus[p]["id"] for p in order(scores[i])] for i, q in enumerate(selected)
    }
    metrics, _ = evaluate(rankings, read_json("data/processed/labels.json"))
    return metrics


def neural_encoder():
    resume = os.environ.get("LEDGERLIGHT_RESUME") == "1"
    history = (
        read_json("data/training/encoder_trials.json")
        if resume and path("data/training/encoder_trials.json").exists()
        else []
    )
    best_score, best_trial, best_config = -1, None, None
    if history:
        entry = max(history, key=lambda row: row["metrics"]["ndcg@10"])
        best_score, best_trial, best_config = (
            entry["metrics"]["ndcg@10"],
            entry["trial"],
            entry["config"],
        )
    storage = "sqlite:///" + str(path("data/optuna.db"))
    study = optuna.create_study(
        direction="maximize",
        study_name="encoder",
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )
    trial_limit = 6
    if history and history[0]["training_seconds"] > 900:
        trial_limit = 3 if history[0]["training_seconds"] <= 3600 else 1
    for number in range(len(history), trial_limit):
        trial = study.ask()
        config = {
            "lr": trial.suggest_float("lr", 5e-6, 4e-5, log=True),
            "epochs": trial.suggest_int("epochs", 1, 2),
            "batch": trial.suggest_categorical("batch", [16, 32]),
            "scale": trial.suggest_categorical("scale", [10.0, 20.0]),
            "negatives": 2,
        }
        model, timing = fit_encoder(config, 42, f"models/encoder/trial-{trial.number}")
        metrics = encoder_metrics(model)
        run_id = tracking(
            f"encoder-trial-{trial.number}",
            {**metrics, **timing},
            {**config, "seed": 42, "split": "dev"},
            metrics,
        )
        study.tell(trial, metrics["ndcg@10"])
        history.append(
            {
                "trial": trial.number,
                "config": config,
                "metrics": metrics,
                **timing,
                "run_id": run_id,
            }
        )
        if metrics["ndcg@10"] > best_score:
            best_score, best_trial, best_config = metrics["ndcg@10"], trial.number, config
        del model
        gc.collect()
        torch.cuda.empty_cache()
        write_json("data/training/encoder_trials.json", history)
        if number == 0:
            if timing["training_seconds"] > 3600:
                break
            if timing["training_seconds"] > 900:
                trial_limit = 3
        if number + 1 >= trial_limit:
            break
    best_path = f"models/encoder/trial-{best_trial}"
    seed_metrics = [
        {"seed": 42, **history[[h["trial"] for h in history].index(best_trial)]["metrics"]}
    ]
    best_time = history[[h["trial"] for h in history].index(best_trial)]["training_seconds"]
    if best_time < 600:
        for seed in [123, 2026]:
            model, timing = fit_encoder(best_config, seed, f"models/encoder/seed-{seed}")
            metrics = encoder_metrics(model)
            tracking(f"encoder-seed-{seed}", {**metrics, **timing}, {**best_config, "seed": seed})
            seed_metrics.append({"seed": seed, **metrics})
            del model
            gc.collect()
            torch.cuda.empty_cache()
    result = {
        "path": best_path,
        "config": best_config,
        "seeds": seed_metrics,
        "dev_ndcg_mean": float(np.mean([m["ndcg@10"] for m in seed_metrics])),
        "dev_ndcg_range": [
            min(m["ndcg@10"] for m in seed_metrics),
            max(m["ndcg@10"] for m in seed_metrics),
        ],
    }
    write_json("data/training/encoder_selected.json", result)
    embed(path(best_path), "indices/trained")
    return result


def fit_cross_encoder(config, seed, destination):
    seed_everything(seed)
    model = load_model("cross_encoder")
    examples = read_json("data/training/examples.json")
    windows = read_jsonl("indices/base/windows.jsonl")
    rng = random.Random(seed)
    pairs = []
    for item in examples:
        pairs.append((item["question"], windows[item["positive_window"]]["text"], 1.0))
        ids = rng.sample(item["negative_windows"], min(4, len(item["negative_windows"])))
        pairs.extend((item["question"], windows[wid]["text"], 0.0) for wid in ids)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=config["lr"], weight_decay=0.01)
    model.model.train()
    losses = []
    started = time.monotonic()
    for epoch in range(config["epochs"]):
        rng.shuffle(pairs)
        for offset in range(0, len(pairs), 8):
            batch = pairs[offset : offset + 8]
            inputs = model.tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                truncation="only_second",
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            target = torch.tensor([p[2] for p in batch], device="cuda")
            optimizer.zero_grad(set_to_none=True)
            use_bf16 = config.get("precision") == "bfloat16"
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model.model(**inputs).logits.reshape(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if len(losses) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "ce_step": len(losses),
                            "loss": losses[-1],
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    model.model.save_pretrained(str(path(destination)))
    model.tokenizer.save_pretrained(str(path(destination)))
    return model, {
        "training_seconds": time.monotonic() - started,
        "mean_loss": float(np.mean(losses)),
        "pairs": len(pairs),
    }


def candidate_context(index_name="trained"):
    queries = read_jsonl("data/processed/queries.jsonl")
    return {
        "queries": queries,
        "lookup": {q["id"]: i for i, q in enumerate(queries)},
        "corpus": read_jsonl("data/processed/corpus.jsonl"),
        "windows": read_jsonl("indices/base/windows.jsonl"),
        "lexical": np.load(path("indices/bm25/scores.npy")),
        "dense": np.load(path(f"indices/{index_name}/scores.npy")),
        "vectors": np.load(path(f"indices/{index_name}/vectors.npy")),
        "qvectors": np.load(path(f"indices/{index_name}/queries.npy")),
        "k": read_json("data/baselines/metrics.json")["rrf_k"],
        "depth": read_json("data/baselines/metrics.json")["ce_depth"],
    }


def score_ce(model, ids, context):
    from ledgerlight.retrieval import cross_scores, fuse, order

    output = {}
    for number, qid in enumerate(ids):
        index = context["lookup"][qid]
        fusion = fuse(context["lexical"][index], context["dense"][index], context["k"])
        candidates = order(fusion)[:100]
        scores = cross_scores(
            model,
            context["queries"][index]["question"],
            candidates,
            context["qvectors"][index],
            context["vectors"],
            context["windows"],
        )
        output[qid] = {"candidates": candidates.tolist(), "scores": scores.tolist()}
        if number % 25 == 0:
            print(json.dumps({"ce_queries": number + 1, "total": len(ids)}), flush=True)
    return output


def ce_rankings(scored, context):
    from ledgerlight.retrieval import order

    out = {}
    for qid, item in scored.items():
        candidates = np.asarray(item["candidates"])
        n = context["depth"]
        ranked = np.concatenate(
            [candidates[:n][order(np.asarray(item["scores"])[:n])], candidates[n:]]
        )
        out[qid] = [context["corpus"][i]["id"] for i in ranked]
    return out


def neural_cross_encoder():
    from ledgerlight.retrieval import load_model

    context = candidate_context()
    dev = read_json("data/processed/splits.json")["dev"]
    labels = read_json("data/processed/labels.json")
    base = load_model("cross_encoder")
    zero_scores = score_ce(base, dev, context)
    zero_metrics, _ = evaluate(ce_rankings(zero_scores, context), labels)
    tracking("trained-encoder-zero-CE-dev", zero_metrics, {"split": "dev"})
    write_json(
        "data/training/ft_encoder_zero_ce.json", {"scores": zero_scores, "metrics": zero_metrics}
    )
    del base
    gc.collect()
    torch.cuda.empty_cache()
    study = optuna.create_study(
        direction="maximize",
        study_name="cross_encoder",
        storage="sqlite:///" + str(path("data/optuna.db")),
        sampler=optuna.samplers.TPESampler(seed=43),
        load_if_exists=True,
    )
    history, best_score, selected_trial, selected_config = [], -1, None, None
    trial_limit = 6
    for number in range(trial_limit):
        trial = study.ask()
        config = {
            "lr": trial.suggest_float("lr", 1e-6, 3e-5, log=True),
            "epochs": trial.suggest_int("epochs", 1, 2),
            "precision": "bfloat16" if torch.cuda.is_bf16_supported() else "float32",
        }
        model, timing = fit_cross_encoder(config, 42, f"models/cross_encoder/trial-{trial.number}")
        scored = score_ce(model, dev, context)
        metrics, _ = evaluate(ce_rankings(scored, context), labels)
        study.tell(trial, metrics["ndcg@10"])
        run_id = tracking(
            f"cross-encoder-trial-{trial.number}",
            {**metrics, **timing},
            {**config, "seed": 42, "split": "dev"},
        )
        history.append(
            {
                "trial": trial.number,
                "config": config,
                "metrics": metrics,
                **timing,
                "run_id": run_id,
            }
        )
        write_json(f"data/training/ce_trial_{trial.number}.json", scored)
        if metrics["ndcg@10"] > best_score:
            best_score, selected_trial, selected_config = metrics["ndcg@10"], trial.number, config
        del model
        gc.collect()
        torch.cuda.empty_cache()
        write_json("data/training/ce_trials.json", history)
        if number == 0:
            if timing["training_seconds"] > 3600:
                break
            if timing["training_seconds"] > 900:
                trial_limit = 3
        if number + 1 >= trial_limit:
            break
    best_path = f"models/cross_encoder/trial-{selected_trial}"
    best_entry = next(h for h in history if h["trial"] == selected_trial)
    seed_metrics = [{"seed": 42, **best_entry["metrics"]}]
    if best_entry["training_seconds"] < 600:
        for seed in [123, 2026]:
            model, timing = fit_cross_encoder(
                selected_config, seed, f"models/cross_encoder/seed-{seed}"
            )
            scored = score_ce(model, dev, context)
            metrics, _ = evaluate(ce_rankings(scored, context), labels)
            tracking(
                f"cross-encoder-seed-{seed}",
                {**metrics, **timing},
                {**selected_config, "seed": seed},
            )
            seed_metrics.append({"seed": seed, **metrics})
            write_json(f"data/training/ce_seed_{seed}.json", scored)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    result = {
        "path": best_path,
        "config": selected_config,
        "trial": selected_trial,
        "seeds": seed_metrics,
        "dev_ndcg_mean": float(np.mean([m["ndcg@10"] for m in seed_metrics])),
        "dev_ndcg_range": [
            min(m["ndcg@10"] for m in seed_metrics),
            max(m["ndcg@10"] for m in seed_metrics),
        ],
    }
    write_json("data/training/ce_selected.json", result)
    return result


def ltr_inputs(scored, context):
    from ledgerlight.features import features
    from ledgerlight.retrieval import fuse

    labels = read_json("data/processed/labels.json")
    windows = context["windows"]
    count = len(context["corpus"])
    lengths, counts = np.zeros(count), np.zeros(count)
    for window in windows:
        pid = window["passage_index"]
        lengths[pid] = max(lengths[pid], window["end"])
        counts[pid] += 1
    matrices, targets, groups, ids = [], [], [], []
    for group, (qid, item) in enumerate(scored.items()):
        index = context["lookup"][qid]
        candidates = item["candidates"]
        fusion = fuse(context["lexical"][index], context["dense"][index], context["k"])
        matrix = features(
            context["queries"][index]["question"],
            candidates,
            context["lexical"][index],
            context["dense"][index],
            fusion,
            np.asarray(item["scores"]),
            context["corpus"],
            lengths,
            counts,
        )
        positives = set(labels[qid]["positives"])
        y = [float(context["corpus"][p]["id"] in positives) for p in candidates]
        matrices.append(matrix)
        targets.extend(y)
        groups.extend([group] * len(candidates))
        ids.append(qid)
    return np.concatenate(matrices), np.asarray(targets), np.asarray(groups), ids


def rank_ltr(model, scored, context):
    from catboost import Pool

    data, _, groups, ids = ltr_inputs(scored, context)
    predictions = model.predict(Pool(data, group_id=groups))
    result = {}
    from ledgerlight.retrieval import order

    for group, qid in enumerate(ids):
        candidates = np.asarray(scored[qid]["candidates"])
        ranked = candidates[order(predictions[groups == group])]
        result[qid] = [context["corpus"][p]["id"] for p in ranked]
    return result


def train_ltr():
    from catboost import CatBoostRanker, Pool

    from ledgerlight.features import FEATURE_NAMES
    from ledgerlight.retrieval import load_model

    context = candidate_context()
    baseline = read_json("data/baselines/metrics.json")
    zero = read_json("data/training/ft_encoder_zero_ce.json")
    selected = read_json("data/training/ce_selected.json")
    tuned_scores = read_json(f"data/training/ce_trial_{selected['trial']}.json")
    tuned_metrics, _ = evaluate(
        ce_rankings(tuned_scores, context), read_json("data/processed/labels.json")
    )
    upstream = [
        (baseline["dev"]["CE-zero"]["ndcg@10"], "base", None, "CE-zero"),
        (zero["metrics"]["ndcg@10"], "trained", None, "BGE-ft"),
        (tuned_metrics["ndcg@10"], "trained", selected["path"], "CE-ft"),
    ]
    _, index_name, cross_path, upstream_name = max(upstream, key=lambda x: x[0])
    context = candidate_context(index_name)
    split = read_json("data/training/split.json")
    dev = read_json("data/processed/splits.json")["dev"]
    model_ce = load_model("cross_encoder", path(cross_path) if cross_path else None)
    train_scored = score_ce(model_ce, split["B"], context)
    if upstream_name == "CE-ft":
        dev_scored = tuned_scores
    elif upstream_name == "BGE-ft":
        dev_scored = zero["scores"]
    else:
        base_scores = read_json("data/baselines/ce_scores.json")
        dev_scored = {
            qid: {
                "candidates": base_scores["candidates"][qid],
                "scores": base_scores["scores"][qid],
            }
            for qid in dev
        }
    write_json("data/training/ltr_scores.json", {"train": train_scored, "dev": dev_scored})
    del model_ce
    gc.collect()
    torch.cuda.empty_cache()
    x, y, groups, _ = ltr_inputs(train_scored, context)
    train_pool = Pool(x, y, group_id=groups, feature_names=FEATURE_NAMES)
    labels = read_json("data/processed/labels.json")
    study = optuna.create_study(
        direction="maximize",
        study_name="catboost",
        storage="sqlite:///" + str(path("data/optuna.db")),
        sampler=optuna.samplers.TPESampler(seed=44),
        load_if_exists=True,
    )
    history = []
    for number in range(params()["training"]["catboost_trials"]):
        trial = study.ask()
        config = {
            "iterations": trial.suggest_int("iterations", 150, 500, step=50),
            "depth": trial.suggest_int("depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 20, log=True),
        }
        model = CatBoostRanker(
            **config,
            loss_function="YetiRank",
            random_seed=42,
            thread_count=8,
            verbose=False,
            allow_writing_files=False,
        )
        started = time.monotonic()
        model.fit(train_pool)
        seconds = time.monotonic() - started
        rankings = rank_ltr(model, dev_scored, context)
        metrics, _ = evaluate(rankings, labels)
        study.tell(trial, metrics["ndcg@10"])
        target = path(f"models/ltr/trial-{trial.number}.cbm")
        target.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(target))
        run_id = tracking(
            f"catboost-trial-{trial.number}",
            {**metrics, "training_seconds": seconds},
            {**config, "seed": 42, "upstream": upstream_name, "split": "dev"},
        )
        history.append(
            {
                "trial": trial.number,
                "config": config,
                "metrics": metrics,
                "training_seconds": seconds,
                "run_id": run_id,
            }
        )
        write_json("data/training/ltr_trials.json", history)
        print(json.dumps({"ltr_trial": trial.number, **metrics, "seconds": seconds}), flush=True)
    best = max(history, key=lambda h: h["metrics"]["ndcg@10"])
    seeds = [{"seed": 42, **best["metrics"]}]
    if best["training_seconds"] < 600:
        for seed in [123, 2026]:
            model = CatBoostRanker(
                **best["config"],
                loss_function="YetiRank",
                random_seed=seed,
                thread_count=8,
                verbose=False,
                allow_writing_files=False,
            )
            started = time.monotonic()
            model.fit(train_pool)
            seconds = time.monotonic() - started
            rankings = rank_ltr(model, dev_scored, context)
            metrics, _ = evaluate(rankings, labels)
            model.save_model(str(path(f"models/ltr/seed-{seed}.cbm")))
            tracking(
                f"catboost-seed-{seed}",
                {**metrics, "training_seconds": seconds},
                {**best["config"], "seed": seed, "upstream": upstream_name},
            )
            seeds.append({"seed": seed, **metrics})
    selection = {
        "path": f"models/ltr/trial-{best['trial']}.cbm",
        "config": best["config"],
        "index": index_name,
        "cross_encoder_path": cross_path,
        "upstream": upstream_name,
        "seeds": seeds,
        "dev_ndcg_mean": float(np.mean([m["ndcg@10"] for m in seeds])),
        "dev_ndcg_range": [min(m["ndcg@10"] for m in seeds), max(m["ndcg@10"] for m in seeds)],
    }
    write_json("data/training/ltr_selected.json", selection)
    return selection


def train():
    resume = os.environ.get("LEDGERLIGHT_RESUME") == "1"
    if not resume or not path("data/training/examples.json").exists():
        prepare()
    if not resume or not path("data/training/encoder_selected.json").exists():
        neural_encoder()
    if not resume or not path("data/training/ce_selected.json").exists():
        neural_cross_encoder()
    if not resume or not path("data/training/ltr_selected.json").exists():
        train_ltr()


if __name__ == "__main__":
    train()
