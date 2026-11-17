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

Retrieval models behind [Ledgerlight](https://github.com/flaretensor/ledgerlight), a local question-answering demo over SEC filing excerpts. They are trained on [FinRank](https://github.com/datanxt/FinRank).

| Path | Contents |
|---|---|
| `models/encoder/trial-2` | BGE-small encoder fine-tuned on FinRank |
| `models/cross_encoder/trial-3` | MiniLM-L6 cross-encoder fine-tuned on FinRank |
| `models/ltr/trial-37.cbm` | CatBoost ranker over 14 retrieval features |
| `data/runtime/onnx` | ONNX exports used for serving |

## Results

On the 119 held-out test questions of FinRank, the full pipeline reaches nDCG@10 of 38.58%, against 22.79% for BM25 (+15.79 points, 95% CI 10.27–21.44). The test questions are new and are searched against the same passages as in training. The metric measures retrieval; the generated answers were not scored.

## Use

The models are downloaded by the bootstrap script described in the [repository README](https://github.com/flaretensor/ledgerlight#readme), pinned to a commit of this repository. The answer model, Qwen3.5-2B (Apache-2.0), is not included; the code fetches it from its upstream repository.

## License

CC BY-NC 4.0, because the models were trained on FinRank annotations (CC BY-NC 4.0). Base models: BGE-small-en-v1.5 (MIT) and ms-marco-MiniLM-L6-v2 (Apache-2.0).
