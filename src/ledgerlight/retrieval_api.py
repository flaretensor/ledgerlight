import gc
import os
import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest
from qdrant_client import models

from ledgerlight.api_models import SearchRequest
from ledgerlight.engine import SearchEngine
from ledgerlight.observability import ERRORS, REQUESTS, STAGES
from ledgerlight.registry import PIPELINE_NAME, client, manifest

LOCK = threading.RLock()
ENGINE = None
GPU_MEMORY = Gauge("ledgerlight_gpu_memory_used_bytes", "Total GPU memory in use")


def refresh():
    global ENGINE

    current = client().get_model_version_by_alias(PIPELINE_NAME, "champion")
    if ENGINE is None or ENGINE.config["registry_version"] != str(current.version):
        config = manifest(current.version)
        ENGINE = None
        gc.collect()
        torch.cuda.empty_cache()
        ENGINE = SearchEngine(
            config,
            device=os.environ.get("LEDGERLIGHT_DEVICE", "cuda"),
            qdrant_url=os.environ.get("LEDGERLIGHT_QDRANT_URL", "http://127.0.0.1:6333"),
        )
        aliases = ENGINE.qdrant.get_aliases().aliases
        if any(
            a.alias_name == "ledgerlight-active" and a.collection_name == ENGINE.collection
            for a in aliases
        ):
            return ENGINE
        operations = []
        if any(a.alias_name == "ledgerlight-active" for a in aliases):
            operations.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name="ledgerlight-active")
                )
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=ENGINE.collection, alias_name="ledgerlight-active"
                )
            )
        )
        ENGINE.qdrant.update_collection_aliases(change_aliases_operations=operations)
    return ENGINE


@asynccontextmanager
async def lifespan(app):
    with LOCK:
        refresh()
    yield


app = FastAPI(title="Ledgerlight retrieval", lifespan=lifespan)


@app.get("/health")
def health():
    with LOCK:
        try:
            engine = refresh()
        except Exception as exc:
            raise HTTPException(503, "Registry or retrieval engine unavailable") from exc
        options = {
            key: sorted({s[key] for p in engine.corpus for s in p["sources"]})
            for key in ["ticker", "year", "form"]
        }
        return {"status": "ready", "versions": engine.versions(), "filters": options}


@app.post("/search")
def search(request: SearchRequest):
    started = time.perf_counter()
    acquired = LOCK.acquire(timeout=60)
    if not acquired:
        REQUESTS.labels("retrieval", "busy").inc()
        raise HTTPException(429, "Retrieval queue is full")
    try:
        queue_seconds = time.perf_counter() - started
        registry_started = time.perf_counter()
        engine = refresh()
        registry_seconds = time.perf_counter() - registry_started
        result = engine.search(
            request.question, request.filters.model_dump(exclude_none=True), request.limit
        )
        result["latencies"]["queue"] = queue_seconds
        result["latencies"]["registry"] = registry_seconds
        result["latencies"]["e2e"] = time.perf_counter() - started
        for name, value in result["latencies"].items():
            STAGES.labels("retrieval", name).observe(max(0, value))
        REQUESTS.labels("retrieval", "ok").inc()
        return result
    except Exception as exc:
        ERRORS.labels("retrieval", type(exc).__name__).inc()
        REQUESTS.labels("retrieval", "error").inc()
        raise HTTPException(503, "Retrieval unavailable; inspect the local service log") from exc
    finally:
        LOCK.release()


@app.get("/metrics")
def metrics():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        GPU_MEMORY.set(total - free)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
