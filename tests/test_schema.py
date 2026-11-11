import copy

import pytest


@pytest.mark.parametrize("invalid", ["duplicate_id", "empty_question", "no_positive"])
def test_pandera_rejects_invalid_finrank_rows(tmp_path, monkeypatch, invalid):
    import pandera.pandas as pa

    import ledgerlight.common as common
    from ledgerlight.data import validate

    monkeypatch.setattr(common, "ROOT", tmp_path)
    row = {
        "id": "q0",
        "question": "What did ALFA sell?",
        "ticker": "ALFA",
        "year": "2024",
        "doc_type": "10-K",
        "passages": [{"text": "ALFA sold equipment.", "page_number": "1"}],
        "hard_negatives": [],
    }
    rows = [row]
    if invalid == "duplicate_id":
        rows.append(copy.deepcopy(row))
    elif invalid == "empty_question":
        row["question"] = ""
    else:
        row["passages"] = []
    common.write_jsonl("data/raw/FinRank.jsonl", rows)
    with pytest.raises(pa.errors.SchemaErrors):
        validate()
