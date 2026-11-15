import json
import re

from ledgerlight.common import read_json, read_jsonl, write_json
from ledgerlight.examples import numeric_check


def main():
    rows = read_jsonl("data/analysis/ask_dev118/responses.jsonl")
    labels = read_json("data/processed/labels.json")
    aliases = read_json("config/company_aliases.json")
    sectors = {
        **dict.fromkeys(["F", "GM", "TSLA"], "Automotive"),
        **dict.fromkeys(["ABBV", "AMGN", "LLY", "JNJ", "MRK", "MRNA", "PFE"], "Healthcare"),
    }
    candidates = []
    for row in rows:
        result = row["response"]
        citations = result.get("citations", [])
        if not {c["passage_id"] for c in citations} & set(labels[row["qid"]]["positives"]):
            continue
        question = row["question"]
        mentioned = {
            ticker.upper()
            for ticker, names in aliases.items()
            if any(name.casefold() in question.casefold() for name in names)
            or re.search(r"\b" + ticker.upper() + r"\b", question)
        }
        sources = [s for c in citations for s in c["sources"]]
        companies = {s["ticker"].upper() for s in sources} & mentioned
        if not companies:
            continue
        candidates.append(
            {
                "qid": row["qid"],
                "question": question,
                "answer": result["answer"],
                "cited_source_labels": [
                    f"{s['ticker']} / {s['form']} / {s['year']} / p. {s['page']}" for s in sources
                ],
                "numeric_check": numeric_check(result["answer"], citations),
                "companies": sorted(companies),
                "sectors": sorted({sectors.get(c, "Energy") for c in companies}),
                "forms": sorted({s["form"] for s in sources}),
            }
        )
    passing = [item for item in candidates if item["numeric_check"]["passed"]]
    if (
        len(passing) >= 15
        and len({company for item in passing for company in item["companies"]}) >= 10
        and {sector for item in passing for sector in item["sectors"]}
        == {"Automotive", "Energy", "Healthcare"}
        and {"10-K", "10-Q"} <= {form for item in passing for form in item["forms"]}
    ):
        candidates = passing
    selected, companies, groups, forms = [], set(), set(), set()
    while candidates and len(selected) < 15:

        def priority(item):
            return (
                20 * len(set(item["sectors"]) - groups)
                + 10 * len(set(item["companies"]) - companies)
                + 3 * len(set(item["forms"]) - forms)
                + int(item["numeric_check"]["passed"]),
                -len(item["answer"]),
            )

        item = max(candidates, key=priority)
        selected.append(item)
        candidates.remove(item)
        companies.update(item["companies"])
        groups.update(item["sectors"])
        forms.update(item["forms"])
    assert len(selected) == 15 and len(companies) >= 10 and len(groups) == 3
    assert {"10-K", "10-Q"} <= forms
    write_json("data/analysis/stage2/examples/candidates.json", selected)
    summary = {
        "selected": len(selected),
        "companies": len(companies),
        "sectors": sorted(groups),
        "forms": sorted(forms),
        "numeric_pass": sum(x["numeric_check"]["passed"] for x in selected),
    }
    write_json("data/analysis/stage2/examples/summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
