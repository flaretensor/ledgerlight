from typer.testing import CliRunner

from ledgerlight.cli import app


def test_cli_version():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.output


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
