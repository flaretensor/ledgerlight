import os
import re
import time

import bm25s
import numpy as np
from qdrant_client import QdrantClient, models

from ledgerlight.common import digest, namespace, params, path, read_jsonl
from ledgerlight.features import features, ranks
from ledgerlight.retrieval import best_windows, cross_scores, load_model, order


def source_matches(source, filters):
    return all(
        str(source.get(key, "")).casefold() == str(value).casefold()
        for key, value in filters.items()
        if value is not None and value != ""
    )


def payload_filter(filters):
    fields = [
        models.FieldCondition(key=key, match=models.MatchValue(value=str(value)))
        for key, value in filters.items()
        if value is not None and value != ""
    ]
    if not fields:
        return None
    return models.Filter(
        must=[
            models.NestedCondition(
                nested=models.Nested(key="sources", filter=models.Filter(must=fields))
            )
        ]
    )


class SearchEngine:
    def __init__(self, config, device="cuda", qdrant_url="http://127.0.0.1:6333"):
        self.config = config
        if digest(path("data/processed/corpus.jsonl")) != config["corpus_sha256"]:
            raise ValueError("Corpus does not match the registered pipeline")
        self.corpus = read_jsonl("data/processed/corpus.jsonl")
        self.windows = read_jsonl("indices/base/windows.jsonl")
        self.index = bm25s.BM25.load(str(path("indices/bm25")), load_corpus=False)
        self.vectors = np.load(path(f"indices/{config['index']}/vectors.npy"))
        self.parents = np.load(path(f"indices/{config['index']}/parents.npy"))
        self.lengths = np.zeros(len(self.corpus))
        self.counts = np.zeros(len(self.corpus))
        for window in self.windows:
            pid = window["passage_index"]
            self.lengths[pid] = max(self.lengths[pid], window["end"])
            self.counts[pid] += 1
        self.backend = os.environ.get("LEDGERLIGHT_BACKEND", "torch")
        if self.backend not in {"torch", "ort-cpu", "ort-cuda"}:
            raise ValueError("Supported backends: torch, ort-cpu, ort-cuda")
        source = path(config["encoder_path"]) if config["encoder_path"] else None
        ce_source = path(config["cross_encoder_path"]) if config["cross_encoder_path"] else None
        self.cross_encoder = None
        if self.backend.startswith("ort-"):
            from ledgerlight.onnx_backend import OnnxCrossEncoder, OnnxEncoder

            provider = (
                "CUDAExecutionProvider" if self.backend == "ort-cuda" else "CPUExecutionProvider"
            )
            components = config.get("components", {})
            self.encoder = OnnxEncoder(source, components.get("encoder", config["index"]), provider)
            if config["mode"] in {"ce", "ltr"}:
                self.cross_encoder = OnnxCrossEncoder(
                    ce_source, components.get("cross_encoder", "base"), provider
                )
        elif self.backend == "torch":
            self.encoder = load_model(model_path=source, device=device)
            if config["mode"] in {"ce", "ltr"}:
                self.cross_encoder = load_model("cross_encoder", ce_source, device=device)
        else:
            raise ValueError("Supported backends: torch, ort-cpu, ort-cuda")
        self.ranker = None
        if config["ltr_path"]:
            from catboost import CatBoostRanker

            self.ranker = CatBoostRanker()
            self.ranker.load_model(str(path(config["ltr_path"])))
        self.qdrant = QdrantClient(qdrant_url, timeout=30, check_compatibility=False)
        version = config.get("components", {}).get("encoder", config["index"])
        self.collection = namespace() + "-windows-encoder-" + str(version)

    def search(self, question, filters=None, limit=5):
        started = time.perf_counter()
        filters = {k: v for k, v in (filters or {}).items() if v is not None and v != ""}
        eligible = np.asarray(
            [any(source_matches(s, filters) for s in p["sources"]) for p in self.corpus]
        )
        if not eligible.any():
            return {
                "results": [],
                "trace": [],
                "latencies": {"total": time.perf_counter() - started},
                "versions": self.versions(),
                "filters": filters,
            }
        timings = {}
        clock = time.perf_counter()
        lexical = self.index.get_scores(re.findall(r"\w+", question.lower())).astype(np.float64)
        lexical[~eligible] = -np.inf
        timings["bm25"] = time.perf_counter() - clock
        clock = time.perf_counter()
        qvector = self.encoder.encode(
            params()["encoder"]["prefix"] + question, normalize_embeddings=True
        )
        timings["embed"] = time.perf_counter() - clock
        clock = time.perf_counter()
        groups = self.qdrant.query_points_groups(
            collection_name=self.collection,
            query=qvector.tolist(),
            group_by="passage_id",
            group_size=1,
            limit=min(100, int(eligible.sum())),
            query_filter=payload_filter(filters),
            search_params=models.SearchParams(hnsw_ef=256, exact=True),
            with_payload=True,
        )
        qdrant_scores = {
            int(g.hits[0].payload["passage_index"]): float(g.hits[0].score) for g in groups.groups
        }
        dense = np.full(len(self.corpus), -np.inf)
        window_scores = self.vectors @ qvector
        np.maximum.at(dense, self.parents, window_scores)
        dense[~eligible] = -np.inf
        dense_order = sorted(qdrant_scores, key=lambda i: (-qdrant_scores[i], i))
        fusion = np.zeros(len(self.corpus), dtype=np.float64)
        for indices in [order(lexical)[: min(100, int(eligible.sum()))], dense_order]:
            indices = np.asarray(indices, dtype=np.int64)
            fusion[indices] += 1 / (self.config["rrf_k"] + np.arange(1, len(indices) + 1))
        fusion[~eligible] = -np.inf
        timings["qdrant"] = time.perf_counter() - clock
        candidates = order(fusion)[: min(100, int(eligible.sum()))]
        ce = np.zeros(len(candidates), dtype=np.float32)
        ltr = np.zeros(len(candidates), dtype=np.float32)
        mode = self.config["mode"]
        if mode == "bm25":
            ranked = order(lexical)[: min(100, int(eligible.sum()))]
            candidates = ranked
        elif mode == "dense":
            ranked = np.asarray(dense_order)
            candidates = ranked
        elif mode == "rrf":
            ranked = candidates
        else:
            clock = time.perf_counter()
            ce = cross_scores(
                self.cross_encoder, question, candidates, qvector, self.vectors, self.windows
            )
            timings["ce"] = time.perf_counter() - clock
            if mode == "ltr":
                clock = time.perf_counter()
                matrix = features(
                    question,
                    candidates,
                    lexical,
                    dense,
                    fusion,
                    ce,
                    self.corpus,
                    self.lengths,
                    self.counts,
                )
                ltr = np.asarray(self.ranker.predict(matrix))
                ranked = candidates[order(ltr)]
                timings["ltr"] = time.perf_counter() - clock
            else:
                depth = min(self.config["ce_depth"], len(candidates))
                ranked = np.concatenate([candidates[:depth][order(ce[:depth])], candidates[depth:]])
        stage_candidates = {
            "bm25": [
                {"passage_id": self.corpus[int(i)]["id"], "score": float(lexical[i])}
                for i in order(lexical)[: min(100, int(eligible.sum()))]
            ],
            "dense": [
                {"passage_id": self.corpus[int(i)]["id"], "score": qdrant_scores[int(i)]}
                for i in dense_order
            ],
            "rrf": [
                {"passage_id": self.corpus[int(i)]["id"], "score": float(fusion[i])}
                for i in order(fusion)[: min(100, int(eligible.sum()))]
            ],
        }
        lex_rank, dense_rank, fusion_rank = ranks(lexical), ranks(dense), ranks(fusion)
        ce_rank, ltr_rank = ranks(ce), ranks(ltr)
        traces = []
        for pos, pid in enumerate(candidates):
            traces.append(
                {
                    "passage_id": self.corpus[pid]["id"],
                    "bm25_rank": int(lex_rank[pid]),
                    "dense_rank": int(dense_rank[pid]),
                    "rrf_rank": int(fusion_rank[pid]),
                    "ce_rank": int(ce_rank[pos]) if mode in {"ce", "ltr"} else None,
                    "ltr_rank": int(ltr_rank[pos]) if mode == "ltr" else None,
                    "bm25": float(lexical[pid]),
                    "dense": float(dense[pid]),
                    "rrf": float(fusion[pid]),
                    "ce": float(ce[pos]) if mode in {"ce", "ltr"} else None,
                    "ltr": float(ltr[pos]) if mode == "ltr" else None,
                }
            )
        selected_windows = best_windows(
            qvector, self.vectors, self.windows, ranked[:limit], count=1
        )
        results = []
        trace_map = {t["passage_id"]: t for t in traces}
        for pid, window_ids in zip(ranked[:limit], selected_windows):
            passage = self.corpus[pid]
            results.append(
                {
                    "passage_id": passage["id"],
                    "sources": [s for s in passage["sources"] if source_matches(s, filters)],
                    "all_sources": passage["sources"],
                    "windows": [self.windows[wid]["text"] for wid in window_ids],
                    "text": passage["text"],
                    "scores": trace_map.get(passage["id"], {}),
                }
            )
        timings["total"] = time.perf_counter() - started
        return {
            "results": results,
            "trace": traces,
            "stage_candidates": stage_candidates,
            "latencies": timings,
            "versions": self.versions(),
            "filters": filters,
        }

    def versions(self):
        return {
            "pipeline": self.config.get("registry_version"),
            "method": self.config["method"],
            "components": self.config.get("components", {}),
            "collection": self.collection,
            "index": self.config["index"],
            "backend": self.backend,
        }
