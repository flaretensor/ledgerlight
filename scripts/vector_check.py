import json
import time

import numpy as np
from qdrant_client import QdrantClient, models

from ledgerlight.common import path, read_json, read_jsonl, write_json
from ledgerlight.engine import payload_filter
from ledgerlight.evaluation import tracking

config = read_json("models/production/config.json")
versions = read_json("data/registry/versions.json")["components"]["encoder"]
collection = "ledgerlight-windows-encoder-" + str(versions[int(config["index"] == "trained")])
client = QdrantClient("http://127.0.0.1:6333", timeout=30, check_compatibility=False)
info = client.get_collection(collection)
queries = read_jsonl("data/processed/queries.jsonl")
ids = set(read_json("data/processed/splits.json")["dev"][:25])
vectors = np.load(path(f"indices/{config['index']}/queries.npy"))
corpus = read_jsonl("data/processed/corpus.jsonl")
source = corpus[0]["sources"][0]
cases = {
    "unfiltered": {},
    "form": {"form": source["form"]},
    "ticker": {"ticker": source["ticker"]},
    "ticker_year": {"ticker": source["ticker"], "year": source["year"]},
    "early_year": {"year": "2022"},
}
rows = []
for name, filters in cases.items():
    latencies = {True: [], False: []}
    recalls = []
    eligible = client.count(collection, count_filter=payload_filter(filters), exact=True).count
    for i, query in enumerate(queries):
        if query["id"] not in ids:
            continue
        results = {}
        for exact in [True, False]:
            started = time.perf_counter()
            groups = client.query_points_groups(
                collection,
                query=vectors[i].tolist(),
                group_by="passage_id",
                group_size=1,
                limit=20,
                query_filter=payload_filter(filters),
                search_params=models.SearchParams(exact=exact, hnsw_ef=64),
                with_payload=True,
            )
            latencies[exact].append((time.perf_counter() - started) * 1000)
            results[exact] = {g.hits[0].payload["passage_id"] for g in groups.groups}
        if results[True]:
            recalls.append(len(results[True] & results[False]) / len(results[True]))
    rows.append(
        {
            "filter": name,
            "values": filters,
            "eligible_windows": eligible,
            "selectivity": eligible / info.points_count,
            "queries": len(recalls),
            "parent_recall20_vs_exact": float(np.mean(recalls)) if recalls else None,
            "exact_p50_ms": float(np.median(latencies[True])),
            "exact_p95_ms": float(np.quantile(latencies[True], 0.95)),
            "approx_p50_ms": float(np.median(latencies[False])),
            "approx_p95_ms": float(np.quantile(latencies[False], 0.95)),
        }
    )
result = {
    "collection": collection,
    "points": info.points_count,
    "indexed_vectors": info.indexed_vectors_count,
    "hnsw_config": info.config.hnsw_config.model_dump(),
    "rows": rows,
    "protocol": "25 fixed DEV queries; grouped passage max; exact flag vs approximate query planner at ef=64. Selective filters may use full scan.",
}
write_json("data/serving/vector_search.json", result)
print(json.dumps(result, indent=2))


tracking(
    "qdrant-exact-approx",
    {
        r["filter"] + "_recall20": r["parent_recall20_vs_exact"]
        for r in rows
        if r["parent_recall20_vs_exact"] is not None
    },
    {"collection": collection, "queries": 25},
    result,
)
