import json
import os

import numpy as np
from qdrant_client import QdrantClient, models

from ledgerlight.common import digest, namespace, path, read_json, read_jsonl, write_json


def upload():
    client = QdrantClient(
        os.environ.get("LEDGERLIGHT_QDRANT_URL", "http://127.0.0.1:6333"),
        timeout=120,
        check_compatibility=False,
    )
    corpus = read_jsonl("data/processed/corpus.jsonl")
    windows = read_jsonl("indices/base/windows.jsonl")
    versions = read_json("data/registry/versions.json")["components"]["encoder"]
    registry = []
    indices = ["base", "trained"] if len(versions) == 2 else ["trained"]
    for index_name, version in zip(indices, versions, strict=True):
        collection = namespace() + "-windows-encoder-" + str(version)
        vectors = np.load(path(f"indices/{index_name}/vectors.npy"))
        if not client.collection_exists(collection):
            client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=1000),
                on_disk_payload=True,
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1000),
            )
            for key in [
                "passage_id",
                "sources[].ticker",
                "sources[].year",
                "sources[].form",
                "sources[].page",
            ]:
                client.create_payload_index(
                    collection, key, models.PayloadSchemaType.KEYWORD, wait=True
                )
            for start in range(0, len(windows), 256):
                points = []
                for window in windows[start : start + 256]:
                    pid = window["passage_index"]
                    points.append(
                        models.PointStruct(
                            id=window["id"],
                            vector=vectors[window["id"]].tolist(),
                            payload={
                                "passage_id": corpus[pid]["id"],
                                "passage_index": pid,
                                "window_id": window["id"],
                                "sources": corpus[pid]["sources"],
                            },
                        )
                    )
                client.upsert(collection, points=points, wait=True)
        count = client.count(collection, exact=True).count
        if count != len(windows):
            raise ValueError(f"Incomplete Qdrant collection {collection}: {count}")
        registry.append(
            {
                "collection": collection,
                "encoder_version": version,
                "points": count,
                "vectors_sha256": digest(path(f"indices/{index_name}/vectors.npy")),
            }
        )
    write_json("data/serving/collections.json", registry)
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    upload()
