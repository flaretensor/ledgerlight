import re
from functools import lru_cache

import numpy as np

from ledgerlight.retrieval import order

FEATURE_NAMES = [
    "bm25",
    "dense",
    "rrf",
    "ce",
    "bm25_rank",
    "dense_rank",
    "rrf_rank",
    "ce_rank",
    "passage_tokens",
    "window_count",
    "query_tokens",
    "ticker_match",
    "year_match",
    "form_match",
]


@lru_cache(maxsize=1)
def company_aliases():
    from ledgerlight.common import read_json

    return read_json("config/company_aliases.json")


def query_years(question):
    years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", question))
    years.update(re.findall(r"(?i)\b(?:FY|Q[1-4])\s*(?:[-/]\s*)?((?:19|20)\d{2})\b", question))
    years.update("20" + year for year in re.findall(r"(?i)\bFY\s*(?:[-/]\s*)?(\d{2})\b", question))
    return years


def query_tickers(question):
    tokens = set(re.findall(r"\w+", question))
    ford_alias_question = re.sub(r"(?i)(?<!\w)Eagle\s+Ford(?!\w)", "", question)
    return {
        ticker
        for ticker, aliases in company_aliases().items()
        if ticker.upper() in tokens
        or any(
            re.search(
                r"(?<!\w)" + re.escape(alias) + r"(?!\w)",
                ford_alias_question if ticker == "F" else question,
                re.I,
            )
            for alias in aliases
        )
    }


def ranks(values):
    output = np.empty(len(values), dtype=np.float32)
    output[order(values)] = np.arange(1, len(values) + 1)
    return output


def features(question, candidates, lexical, dense, fusion, ce, corpus, lengths, counts):
    candidates = np.asarray(candidates)
    tickers, years = query_tickers(question), query_years(question)
    compact_question = re.sub(r"[^a-z0-9]", "", question.casefold())
    base = np.stack(
        [
            lexical[candidates],
            dense[candidates],
            fusion[candidates],
            ce,
            ranks(lexical)[candidates],
            ranks(dense)[candidates],
            ranks(fusion)[candidates],
            ranks(ce),
            lengths[candidates],
            counts[candidates],
            np.full(len(candidates), len(re.findall(r"\w+", question))),
        ],
        axis=1,
    )
    metadata = []
    for candidate in candidates:
        sources = corpus[int(candidate)]["sources"]
        metadata.append(
            [
                float(any(s["ticker"] in tickers for s in sources)),
                float(any(s["year"] in years for s in sources)),
                float(
                    any(
                        re.sub(r"[^a-z0-9]", "", s["form"].casefold()) in compact_question
                        for s in sources
                    )
                ),
            ]
        )
    return np.column_stack([base, np.asarray(metadata, dtype=np.float32)]).astype(np.float32)
