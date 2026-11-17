import argparse
import json
import urllib.request

from ledgerlight.common import digest, params, path, write_json


def ingest():
    cfg = params()["data"]
    root = path("data/raw")
    root.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/datanxt/FinRank/{cfg['revision']}/"
    for name in ("FinRank.jsonl", "baselines/splits.json"):
        target = root / name.split("/")[-1]
        if not target.exists():
            urllib.request.urlretrieve(base + name, target)
    actual = digest(root / "FinRank.jsonl")
    if actual != cfg["sha256"]:
        raise ValueError(f"Dataset checksum mismatch: {actual}")
    write_json(
        "data/raw/provenance.json",
        {
            "revision": cfg["revision"],
            "sha256": actual,
            "bytes": (root / "FinRank.jsonl").stat().st_size,
            "license": "CC-BY-NC-4.0",
            "url": base + "FinRank.jsonl",
            "splits_sha256": digest(root / "splits.json"),
        },
    )
    print(json.dumps({"sha256": actual, "bytes": (root / "FinRank.jsonl").stat().st_size}))


def normalize(text):
    return " ".join(text.split()).casefold()


def validate():
    import hashlib
    import warnings
    from collections import Counter, defaultdict

    import pandas as pd
    import pandera.pandas as pa

    from ledgerlight.common import read_json, read_jsonl, write_jsonl

    rows = read_jsonl("data/raw/FinRank.jsonl")
    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(str, unique=True),
            "question": pa.Column(str, checks=pa.Check.str_length(min_value=1)),
            "passages": pa.Column(object, checks=pa.Check(lambda x: len(x) > 0, element_wise=True)),
        }
    )
    schema.validate(pd.DataFrame(rows), lazy=True)
    lookup, corpus, queries, labels = {}, [], [], {}
    sha_mismatches = 0
    positives_by_text = set()
    negative_texts = []
    for row in rows:
        qid = row["id"]
        queries.append({"id": qid, "question": row["question"]})
        labels[qid] = {"positives": [], "hard_negatives": []}
        for field, role in (("passages", "positives"), ("hard_negatives", "hard_negatives")):
            for item in row[field]:
                text = item["text"]
                if not text.strip():
                    raise ValueError("Empty passage")
                if text not in lookup:
                    lookup[text] = len(corpus)
                    corpus.append({"id": f"p{len(corpus):05d}", "text": text, "sources": []})
                index = lookup[text]
                source = {
                    "ticker": str(item.get("ticker", row["ticker"])),
                    "year": str(item.get("year", row["year"])),
                    "form": str(item.get("doc_type", row["doc_type"])).upper(),
                    "page": str(item["page_number"]),
                }
                if source not in corpus[index]["sources"]:
                    corpus[index]["sources"].append(source)
                labels[qid][role].append(corpus[index]["id"])
                stored = item.get("text_sha1")
                if stored and stored != hashlib.sha1(text.strip().encode()).hexdigest():
                    sha_mismatches += 1
                normalized = " ".join(text.split())
                if role == "positives":
                    positives_by_text.add(normalized)
                else:
                    negative_texts.append(normalized)
    pa.DataFrameSchema(
        {
            "id": pa.Column(str, unique=True),
            "text": pa.Column(str),
            "sources": pa.Column(object),
        },
        strict=True,
    ).validate(pd.DataFrame(corpus), lazy=True)
    pa.DataFrameSchema(
        {
            "id": pa.Column(str, unique=True),
            "question": pa.Column(str),
        },
        strict=True,
    ).validate(pd.DataFrame(queries), lazy=True)
    splits = read_json("data/raw/splits.json")["random"]
    sets = [set(ids) for ids in splits.values()]
    if set.union(*sets) != set(labels) or sum(map(len, sets)) != len(labels):
        raise ValueError("Invalid official split coverage/disjointness")
    by_question = defaultdict(list)
    query_lookup = {q["id"]: q for q in queries}
    for split, ids in splits.items():
        for qid in ids:
            by_question[normalize(query_lookup[qid]["question"])].append(split)
    crossings = {q: v for q, v in by_question.items() if len(set(v)) > 1}
    for question, locations in crossings.items():
        warnings.warn(f"Cross-split question: {question!r} in {locations}", stacklevel=2)
    counts = Counter(normalize(q["question"]) for q in queries)
    audit = {
        "queries": len(rows),
        "passages": len(corpus),
        "whitespace_unique_passages": len({" ".join(p["text"].split()) for p in corpus}),
        "positive_occurrences": sum(len(q["positives"]) for q in labels.values()),
        "hard_negative_occurrences": len(negative_texts),
        "hard_negatives_positive_elsewhere": sum(t in positives_by_text for t in negative_texts),
        "stored_sha1_mismatches": sha_mismatches,
        "multi_provenance_passages": sum(len(p["sources"]) > 1 for p in corpus),
        "multiple_document_metadata_passages": sum(
            len({(s["ticker"], s["year"], s["form"]) for s in p["sources"]}) > 1 for p in corpus
        ),
        "duplicate_question_groups": sum(v > 1 for v in counts.values()),
        "cross_split_duplicate_questions": crossings,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "metadata_fields": ["ticker", "year", "form", "page"],
        "company_names": False,
        "source_urls": False,
        "query_input_fields": ["id", "question"],
    }
    aliases = read_json("config/company_aliases.json")
    tickers = {source["ticker"] for passage in corpus for source in passage["sources"]}
    if set(aliases) != tickers or any(not names for names in aliases.values()):
        raise ValueError("Company alias keys must exactly cover the original corpus tickers")
    for ticker, names in aliases.items():
        for name in names:
            if not any(
                name.casefold() in passage["text"].casefold()
                and any(source["ticker"] == ticker for source in passage["sources"])
                for passage in corpus
            ):
                raise ValueError(f"Company alias not found in ticker passages: {ticker}")
    write_jsonl("data/processed/corpus.jsonl", corpus)
    write_jsonl("data/processed/queries.jsonl", queries)
    write_json("data/processed/labels.json", labels)
    write_json("data/processed/splits.json", splits)
    write_json("data/processed/validation.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    commands = {"ingest": ingest, "validate": validate}
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=commands)
    commands[parser.parse_args().action]()
