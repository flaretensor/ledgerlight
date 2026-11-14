# Ledgerlight

English financial-document retrieval and local cited answers over 5,230 curated FinRank passages, rather than complete SEC filings. This workspace contains a revised pipeline and an isolated TRAIN/DEV pilot. Full retraining, current TEST results, a browser audit and fresh-user deployment verification remain pending.

## Pipeline

BM25 and normalized BGE window embeddings feed RRF, a MiniLM cross-encoder and a 14-feature CatBoost ranker. Five ranked passages contribute one selected window each to local Qwen3.5-2B Q4_K_M. The model returns issued source markers or abstains; citation membership does not establish answer truth.

Windows preserve original case and punctuation, record token IDs and character offsets, and end at tokenizer pre-token boundaries. Their 448-token limit reserves 64 of the cross-encoder's 512 positions; up to 61 question tokens plus three pair special tokens fit without truncation. Longer questions still require the shared only_second truncation policy. The 128-token overlap follows the common long-context QA convention; this is a heuristic, not a tuned optimum. Boundary snapping can change the exact overlap. BGE indexing/training, CE Torch/ORT/training and Qwen consume the same window text.

The ranker retains its exact 14-feature order. Year features support FY2024, Q42024 and FY24 (interpreted as 2024). Static company aliases are checked against passages of each original ticker. The source anomaly ABbV remains a distinct metadata value. Year metadata represents the upstream document label and is not independently reconciled to fiscal periods or SEC accessions.

Only whitespace-equivalent copies of known positive passages are removed from negative TRAIN candidates. Other unjudged candidates retain label zero: sparse qrels remain a training limitation. Official evaluation qrels are unchanged.

Explicit ticker tokens must be uppercase; lowercase tickers are not recognised (use the ticker filter). ABBV and ABbV match the same company, while raw source metadata is retained. Company aliases remain case-insensitive.

## Local setup

Use Python 3.12 and uv. Source scripts/env.sh, then uv sync --frozen. All host caches and temporary files stay inside this workspace. Docker uses project ledgerlight, image ledgerlight-rebuild:local, localhost bindings and restart:no. Its dependency_cache build context is the populated local .cache/uv directory and its python_runtime context is .python, allowing an offline dependency installation. Populate fresh ignored local database/Grafana credentials before services start; never copy credentials or persistent stores from another workspace.

The pinned llama.cpp b12284 /apply-template and /tokenize endpoints count the actual chat prompt with the same thinking option as generation. Excerpts are trimmed by token budget at whitespace boundaries. Sampling, the 512-token thinking cap, 1024 total output-token cap, SYSTEM text and 8192-token context are unchanged. Live Qwen endpoint verification is separate from the training-only pilot.

## Pilot and held-out protection

Run source scripts/env.sh followed by LEDGERLIGHT_RUN_MODE=pilot python scripts/pilot.py. This uses data/pilot for indices, baseline/training/evaluation/model outputs and Optuna storage; studies and the MLflow experiment/models have pilot-specific names. It evaluates TRAIN/DEV only. Canonical full-run model outputs and studies remain empty. The pilot uses one truncated BGE fit, one truncated CE fit and three CatBoost trials; its metrics establish execution, not final quality.

The paired window comparison uses pretrained models and the same original-text window builder and CE input policy for both 510/384 and 448/320 settings. It is not a historical binary reproduction. It reports TRAIN and DEV separately and their combined paired bootstrap. A negative upper 95% CI for BGE recall@100 or nDCG@10 in the combined comparison stops training.

Final TEST evaluation requires python -m ledgerlight.final_evaluation --allow-test outside pilot mode. The default DVC evaluate stage deliberately refuses until this explicit action is chosen. scripts/generation_check.py --help is read-only; generation diagnostics default to DEV and also require --allow-test for TEST.

## Historical evidence and limits

[Previous iteration](docs/previous_iteration.json) retains the prior DEV/TEST retrieval ladders and metric hashes. Those numbers describe a different pipeline and are not results of this workspace. No current frozen TEST has run. Shared-document native random splits, sparse qrels and one duplicated TRAIN/TEST question limit interpretation. No independent semantic answer-accuracy or production claim is made.
