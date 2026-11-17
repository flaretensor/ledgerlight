from typer.testing import CliRunner

from ledgerlight.cli import app


def test_cli_version():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == "0.1.0"


def test_retrieval_metric_denominators():
    import math

    from ledgerlight.evaluation import evaluate

    metrics, _ = evaluate(
        {"q": ["wrong", "relevant"]}, {"q": {"positives": ["relevant", "missing"]}}
    )
    assert metrics["recall@10"] == 0.5
    assert metrics["mrr"] == 0.5
    assert abs(metrics["ndcg@10"] - (1 / math.log2(3)) / (1 + 1 / math.log2(3))) < 1e-10


def test_rrf_has_stable_ties_and_bounded_candidates():
    import numpy as np

    from ledgerlight.retrieval import fuse, order

    values = fuse(np.array([3.0, 2.0, 1.0]), np.array([2.0, 3.0, 1.0]), depth=2)
    assert order(values).tolist() == [0, 1, 2]
    assert values[2] == 0


def test_promotion_gate_rejects_regression_and_nonfinite_metrics():
    from ledgerlight.registry import acceptable

    assert acceptable(0.3, 0.3)
    assert acceptable(0.31, 0.3)
    assert acceptable(0.1, None)
    assert not acceptable(0.29, 0.3)
    assert not acceptable(float("nan"), 0.3)


def test_neural_batches_exclude_known_positive_collisions():
    import random

    from ledgerlight.training import safe_batches

    items = [
        {"positive_key": 1, "all_positives": [1], "selected_negative_keys": [2]},
        {"positive_key": 2, "all_positives": [2], "selected_negative_keys": [3]},
    ]
    assert all(len(batch) == 1 for batch in safe_batches(items, 16, random.Random(42)))
