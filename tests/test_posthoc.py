import pytest

from ledgerlight.posthoc import decompose, metrics, overlap_groups, strata_report, summarize_replay


def test_decomposition_counts_multiple_positives_once():
    rankings = {"a": ["x"], "b": ["x"] * 5 + ["p"], "c": ["p", "z"]}
    labels = {q: {"positives": ["p", "p", "z"]} for q in rankings}
    result = decompose(rankings, labels)
    assert list(result["queries"].values()) == [1, 1, 1]
    assert list(result["positives"].values()) == [3, 0, 3]


def test_text_overlap_uses_all_train_positives_and_normalization():
    labels = {"train": {"positives": ["p"]}, "a": {"positives": ["q", "r"]}}
    texts = {"p": "one two", "q": "one  two", "r": "different"}
    exact = overlap_groups(["a"], labels, texts, ["train"])
    normalized = overlap_groups(["a"], labels, texts, ["train"], True)
    assert exact["none_seen"] == ["a"]
    assert normalized["any_seen"] == ["a"]
    assert normalized["not_all_seen"] == ["a"]


def test_metrics_preserve_sorted_query_pairing_and_macro_recall():
    labels = {"z": {"positives": ["p", "q"]}, "a": {"positives": ["q"]}}
    means, values = metrics({"z": ["p"], "a": ["x"]}, labels)
    assert values["ndcg@10"][0] == 0
    assert means["recall@5"] == 0.25
    assert means["hit_rate@5"] == 0.5
    assert means["ndcg@5"] == pytest.approx(0.5 / (1 + 1 / 1.584962500721156))


def test_strata_align_by_query_id_and_handle_empty_group():
    values = {
        "CatBoost": {"ndcg@10": [0.2, 0.8], "recall@10": [0.4, 1]},
        "BM25": {"ndcg@10": [0.3, 0.1], "recall@10": [0.5, 0.5]},
    }
    result = strata_report({"chosen": ["b"], "empty": []}, values, ["a", "b"])
    assert result["chosen"]["bootstrap"]["delta"] == pytest.approx(0.7)
    assert result["empty"]["bootstrap"] is None


def test_replay_summary_includes_failed_request_latency_and_separates_overlap():
    rows = [
        {
            "qid": "a",
            "http_status": 200,
            "elapsed": 2.0,
            "response": {"citations": [{"passage_id": "p"}], "latencies": {"retrieval": 0.5}},
        },
        {"qid": "b", "http_status": None, "elapsed": 20.0, "response": {}, "error": "ReadTimeout"},
    ]
    result = summarize_replay(rows, {"a": {"positives": ["other"]}})
    assert result["cited_answers"] == 1
    assert result["citing_positive_parent_overlap_proxy"] == 0
    assert result["e2e_seconds_all_offered"]["mean"] == 11.0
    assert result["timeouts"] == 1
