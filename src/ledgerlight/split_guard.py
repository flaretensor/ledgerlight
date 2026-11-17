import json
import re
from contextlib import contextmanager
from contextvars import ContextVar

from ledgerlight.common import path, read_json

_ALLOWED = ContextVar("held_out_metrics_allowed", default=False)


def held_out_allowed():
    return _ALLOWED.get()


def guard_queries(query_ids):
    if path("data/processed/splits.json").exists():
        test = set(read_json("data/processed/splits.json")["test"])
        if test.intersection(query_ids) and not _ALLOWED.get():
            raise PermissionError("TEST metrics require explicit --allow-test")


@contextmanager
def allow_test(enabled):
    if not enabled:
        raise PermissionError("TEST metrics require explicit --allow-test")
    token = _ALLOWED.set(True)
    try:
        yield
    finally:
        _ALLOWED.reset(token)


def read_queries():
    splits = read_json("data/processed/splits.json")
    allowed = set(splits["train"]) | set(splits["dev"])
    rows = []
    with path("data/processed/queries.jsonl").open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            qid = json.loads(re.match(rb'\s*\{\s*"id"\s*:\s*("[^"]+")', line)[1])
            rows.append(
                json.loads(line)
                if qid in allowed or held_out_allowed()
                else {"id": qid, "question": ""}
            )
    return rows
