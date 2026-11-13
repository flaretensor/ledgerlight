import os
from contextlib import contextmanager
from contextvars import ContextVar

from ledgerlight.common import path, read_json

_ALLOWED = ContextVar("held_out_metrics_allowed", default=False)


def guard_queries(query_ids):
    if path("data/processed/splits.json").exists():
        test = set(read_json("data/processed/splits.json")["test"])
        if test.intersection(query_ids) and not _ALLOWED.get():
            raise PermissionError("TEST metrics require explicit --allow-test outside pilot mode")


@contextmanager
def allow_test(enabled):
    if not enabled or os.environ.get("LEDGERLIGHT_RUN_MODE") == "pilot":
        raise PermissionError("TEST metrics require explicit --allow-test outside pilot mode")
    token = _ALLOWED.set(True)
    try:
        yield
    finally:
        _ALLOWED.reset(token)
