import json

from ledgerlight.common import read_json, read_jsonl, write_json

corpus = read_jsonl("data/processed/corpus.jsonl")
generation = read_jsonl("data/generation/responses.jsonl")
cited = sum(bool(row["response"].get("citations")) for row in generation if row["status"] == 200)
hits = sum(bool(row["positive_citation_hit"]) for row in generation)
source_records = [len(row["sources"]) for row in corpus]
facts = {
    "passages": len(corpus),
    "passages_with_multiple_source_records": sum(count > 1 for count in source_records),
    "total_source_records": sum(source_records),
    "extra_source_records": sum(count - 1 for count in source_records),
    "source_records_in_multiprovenance_passages": sum(
        count for count in source_records if count > 1
    ),
    "passages_with_multiple_document_metadata": sum(
        len({(source["ticker"], source["year"], source["form"]) for source in row["sources"]}) > 1
        for row in corpus
    ),
    "saved_generation_requests": len(generation),
    "accepted_citation_answers": cited,
    "saved_positive_citation_hits": hits,
    "positive_hit_among_cited": hits / cited,
    "proxy_interpretation": "Lower-bound qrel-membership proxy with incomplete qrels; not answer correctness or entailment",
    "learned_reranker_candidate_ceiling": {
        split: {
            method: read_json("data/evaluation/metrics.json")[split][method]["recall@100"]
            for method in ["BGE-ft", "CE-ft", "CatBoost"]
        }
        for split in ["dev", "test"]
    },
    "seed_summary": {
        "seed42_ndcg10": read_json("data/training/ltr_selected.json")["seeds"][0]["ndcg@10"],
        "mean": read_json("data/training/ltr_selected.json")["dev_ndcg_mean"],
        "range": read_json("data/training/ltr_selected.json")["dev_ndcg_range"],
    },
    "inputs": [
        "data/processed/corpus.jsonl",
        "data/generation/responses.jsonl",
        "data/evaluation/metrics.json",
        "data/training/ltr_selected.json",
    ],
    "new_test_predictions": False,
}
assert facts["saved_generation_requests"] == 119 and cited == 51 and hits == 20
write_json("reports/docs_facts.json", facts)
print(json.dumps(facts, indent=2))
