import json
import time

import numpy as np
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from ledgerlight.common import database_url, path, read_json, read_jsonl, write_json
from ledgerlight.evaluation import tracking

configuration = read_json("models/production/config.json")
windows = read_jsonl("indices/base/windows.jsonl")
corpus = read_jsonl("data/processed/corpus.jsonl")
vectors = np.load(path(f"indices/{configuration['index']}/vectors.npy"))
qvectors = np.load(path(f"indices/{configuration['index']}/queries.npy"))
queries = read_jsonl("data/processed/queries.jsonl")
dev = set(read_json("data/processed/splits.json")["dev"][:25])
source = corpus[0]["sources"][0]
cases = {
    "unfiltered": {},
    "form": {"form": source["form"]},
    "ticker": {"ticker": source["ticker"]},
    "ticker_year": {"ticker": source["ticker"], "year": source["year"]},
    "early_year": {"year": "2022"},
}
rows = []
with psycopg.connect(database_url(), autocommit=True) as connection:
    connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    connection.execute("""CREATE TABLE IF NOT EXISTS vector_windows (
        window_id INTEGER PRIMARY KEY, passage_id TEXT NOT NULL, sources JSONB NOT NULL,
        embedding vector(384) NOT NULL)""")
    count = connection.execute("SELECT count(*) FROM vector_windows").fetchone()[0]
    if count == 0:
        with connection.cursor().copy(
            "COPY vector_windows (window_id,passage_id,sources,embedding) FROM STDIN"
        ) as copy:
            for w in windows:
                p = corpus[w["passage_index"]]
                vector = "[" + ",".join(map(str, vectors[w["id"]].tolist())) + "]"
                copy.write_row((w["id"], p["id"], Jsonb(p["sources"]), vector))
    if connection.execute("SELECT count(*) FROM vector_windows").fetchone()[0] != len(windows):
        raise ValueError("Incomplete pgvector corpus")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS vector_windows_hnsw ON vector_windows USING hnsw (embedding vector_cosine_ops) WITH (m=16,ef_construction=100)"
    )
    connection.execute("ANALYZE vector_windows")
    connection.execute("SET hnsw.ef_search=128")
    connection.execute("SET hnsw.iterative_scan='strict_order'")
    connection.execute("SET enable_seqscan=off")
    plan = connection.execute(
        "EXPLAIN SELECT passage_id FROM vector_windows ORDER BY embedding <=> %s::vector LIMIT 400",
        ("[" + ",".join(map(str, qvectors[0].tolist())) + "]",),
    ).fetchall()
    for name, filters in cases.items():
        predicates = [
            sql.SQL("s->>{} = {}").format(sql.Literal(key), sql.Literal(value))
            for key, value in filters.items()
        ]
        predicate = (
            sql.SQL("TRUE")
            if not predicates
            else sql.SQL("EXISTS (SELECT 1 FROM jsonb_array_elements(sources) s WHERE {})").format(
                sql.SQL(" AND ").join(predicates)
            )
        )
        query = sql.SQL("""WITH nearest AS MATERIALIZED (
          SELECT passage_id, embedding <=> %s::vector AS distance
          FROM vector_windows WHERE {} ORDER BY embedding <=> %s::vector LIMIT 400
        ) SELECT passage_id,min(distance) FROM nearest GROUP BY passage_id ORDER BY min(distance),passage_id LIMIT 20""").format(
            predicate
        )
        exact_query = sql.SQL("""SELECT passage_id,min(embedding <=> %s::vector) AS distance
          FROM vector_windows WHERE {} GROUP BY passage_id ORDER BY distance,passage_id LIMIT 20""").format(
            predicate
        )
        latencies, recalls = {True: [], False: []}, []
        for i, q in enumerate(queries):
            if q["id"] not in dev:
                continue
            vector = "[" + ",".join(map(str, qvectors[i].tolist())) + "]"
            results = {}
            for exact in [True, False]:
                connection.execute("SET enable_indexscan=" + ("off" if exact else "on"))
                connection.execute("SET enable_seqscan=" + ("on" if exact else "off"))
                started = time.perf_counter()
                selected_query = exact_query if exact else query
                parameters = (vector,) if exact else (vector, vector)
                results[exact] = {
                    r[0] for r in connection.execute(selected_query, parameters).fetchall()
                }
                latencies[exact].append((time.perf_counter() - started) * 1000)
            if results[True]:
                recalls.append(len(results[True] & results[False]) / len(results[True]))
        rows.append(
            {
                "filter": name,
                "parent_recall20_vs_exact": float(np.mean(recalls)) if recalls else None,
                "queries_with_exact_results": len(recalls),
                "exact_p50_ms": float(np.median(latencies[True])),
                "exact_p95_ms": float(np.quantile(latencies[True], 0.95)),
                "hnsw_p50_ms": float(np.median(latencies[False])),
                "hnsw_p95_ms": float(np.quantile(latencies[False], 0.95)),
            }
        )
result = {
    "points": len(windows),
    "rows": rows,
    "plan": [r[0] for r in plan],
    "protocol": "Full-scan exact parent minimum cosine distance versus HNSW top-400 window oversampling grouped to parent top-20; strict iterative scan; fixed 25 DEV queries",
}
write_json("data/serving/pgvector.json", result)
print(json.dumps(result, indent=2))


tracking(
    "pgvector-exact-hnsw",
    {
        r["filter"] + "_recall20": r["parent_recall20_vs_exact"]
        for r in rows
        if r["parent_recall20_vs_exact"] is not None
    },
    {"queries": 25, "oversampling": 400},
    result,
)
