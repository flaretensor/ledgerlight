import numpy as np

from ledgerlight.engine import eligible_passages, fusion_scores, lexical_scores
from ledgerlight.retrieval import best_windows, order


def test_eligibility_requires_one_matching_source_record():
    corpus = [
        {"sources": [{"ticker": "GM", "year": "2024"}, {"ticker": "F", "year": "2025"}]},
        {"sources": [{"ticker": "gm", "year": "2025"}]},
    ]
    assert eligible_passages(corpus, {"ticker": "GM", "year": "2025"}).tolist() == [False, True]
    assert eligible_passages(corpus, {}).tolist() == [True, True]
    assert eligible_passages(corpus, {"ticker": "unknown"}).tolist() == [False, False]


def test_lexical_scores_mask_ineligible_passages_without_mutating_index():
    class Index:
        values = np.asarray([1, 3, 2], dtype=np.float32)

        def get_scores(self, tokens):
            assert tokens == ["revenue", "2024"]
            return self.values

    index = Index()
    scores = lexical_scores(index, "Revenue 2024?", np.asarray([True, False, True]))
    assert scores.dtype == np.float64
    assert order(scores).tolist() == [2, 0, 1]
    assert index.values.tolist() == [1, 3, 2]


def test_fusion_preserves_stable_ties_and_excludes_ineligible_passages():
    eligible = np.asarray([True, True, False])
    values = fusion_scores(np.asarray([1.0, 2.0, -np.inf]), [0, 1], eligible, 10)
    assert values[0] == values[1] == 1 / 11 + 1 / 12
    assert np.isneginf(values[2])
    assert order(values).tolist() == [0, 1, 2]


def test_best_window_selection_uses_similarity_and_stable_ties():
    windows = [{"passage_index": p} for p in [0, 0, 1, 1]]
    vectors = np.asarray([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    assert best_windows(np.asarray([1.0, 0.0]), vectors, windows, [0, 1], count=1) == [[1], [2]]
