import re

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


def ranks(values):
    output = np.empty(len(values), dtype=np.float32)
    output[order(values)] = np.arange(1, len(values) + 1)
    return output


def features(question, candidates, lexical, dense, fusion, ce, corpus, lengths, counts):
    candidates = np.asarray(candidates)
    qtokens = set(re.findall(r"\w+", question.casefold()))
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
                float(any(s["ticker"].casefold() in qtokens for s in sources)),
                float(any(s["year"] in qtokens for s in sources)),
                float(
                    any(
                        re.sub(r"[^a-z0-9]", "", s["form"].casefold()) in compact_question
                        for s in sources
                    )
                ),
            ]
        )
    return np.column_stack([base, np.asarray(metadata, dtype=np.float32)]).astype(np.float32)
