---
license: cc-by-nc-4.0
language:
- en
base_model:
- BAAI/bge-small-en-v1.5
- cross-encoder/ms-marco-MiniLM-L6-v2
tags:
- information-retrieval
- finance
---

# Ledgerlight models

Retrieval models behind [Ledgerlight](https://github.com/flaretensor/ledgerlight), retrieval-augmented question answering over SEC filing excerpts from [FinRank](https://github.com/datanxt/FinRank).

`BM25 + BGE-small → reciprocal rank fusion → MiniLM cross-encoder → CatBoost ranker → Qwen3.5-2B`

| Path | Model | Role |
|---|---|---|
| `models/encoder/trial-2` | BGE-small-en-v1.5, fine-tuned on FinRank | Dense retrieval over passage windows |
| `models/cross_encoder/trial-3` | ms-marco-MiniLM-L6-v2, fine-tuned on FinRank | Question–passage scoring |
| `models/ltr/trial-37.cbm` | CatBoost, YetiRank | Final ranking over 14 features |
| `data/runtime/onnx` | ONNX exports of both neural models | Serving through Triton |

## Results

119 held-out FinRank test questions:

| Method | nDCG@10 |
|---|---|
| BM25 | 22.79% |
| Full pipeline | 38.58% |

The gain is +15.79 points (paired bootstrap, 95% CI 10.27–21.44). The test questions are new and are searched against the same passages as in training. The metric measures retrieval; the generated answers were not scored.

## Use

The bootstrap script described in the [repository README](https://github.com/flaretensor/ledgerlight#readme) downloads these models from a pinned commit of this repository. The answer model, Qwen3.5-2B (Apache-2.0), is not included; the code fetches it from its upstream repository.

## License

CC BY-NC 4.0, because the models were trained on FinRank annotations (CC BY-NC 4.0). Base models: BGE-small-en-v1.5 (MIT) and ms-marco-MiniLM-L6-v2 (Apache-2.0).
