# Ledgerlight

English financial-document retrieval and local cited answers over 5,230 curated FinRank passages from 22 companies. These are filing excerpts, not complete SEC reports.

## Pipeline

BM25 and normalized BGE window embeddings feed reciprocal rank fusion (RRF), a MiniLM cross-encoder and a 14-feature CatBoost learning-to-rank model. Five ranked passages contribute one selected window each to local Qwen3.5-2B Q4_K_M through llama.cpp. Issued source markers become numbered citations; the model can abstain. Citation membership does not establish factual correctness.

Original-text windows preserve case, punctuation and character offsets. The 448-token limit and 320-token step reserve space in the cross-encoder's 512-token input; long questions use the shared `only_second` truncation policy. CE training and serving select windows using the same selected encoder. The LLM uses an 8,192-token context, a 512-token thinking budget and a 1,024-token total generation limit.

Training uses the native random split: 948 TRAIN queries partitioned into 712 neural-training queries and 236 LTR-training queries; 118 DEV queries select configurations. Independent Optuna studies run 6 BGE, 6 cross-encoder and 50 CatBoost trials. Source, parameter, dataset and upstream-weight fingerprints protect resume and cache reuse. Sparse qrels leave unjudged candidates with zero targets; only whitespace-equivalent copies of known positives are excluded from training negatives. Documents are shared across splits, so evaluation measures new questions about known documents.

Explicit ticker tokens must be uppercase; company aliases are case-insensitive. ABBV and ABbV match AbbVie while raw metadata is retained. `year_match` compares question years with filing-year metadata exactly; filing year can differ from the fiscal year covered.

## Local setup

Use Python 3.12, uv, Docker Compose and a CUDA-capable GPU. Run `source scripts/env.sh` and `uv sync --frozen`. Caches and temporary files use this workspace. Create local database/Grafana credentials from `.env.example`; keep `.env` private. Download the pinned dataset with `python -m ledgerlight.data ingest`, validate it with `python -m ledgerlight.data validate`, and populate the pinned base-model and service caches before offline training. Trained model artifacts are not included in Git.

The Compose app image is `ledgerlight-rebuild:local`. Its `dependency_cache` build context uses `.cache/uv`, and `python_runtime` uses `.python`; populate both before an offline image build. Services bind to localhost and use `restart: no`.

Run `bash scripts/train.sh` for resumable training, DEV selection and registry registration. The nine-method DEV ladder and paired query bootstraps (10,000 resamples, seed 42) are written to `data/evaluation/dev_metrics.json`. TEST questions are withheld from indexing and training. A separate `python -m ledgerlight.final_evaluation --allow-test` requires frozen DEV selection and refuses an existing or previously started TEST run.

Use `python scripts/run_stack.py start` to serve canonical champions at `http://127.0.0.1:8501`; `python scripts/run_stack.py stop` stops PostgreSQL last. The isolated rehearsal uses `LEDGERLIGHT_RUN_MODE=pilot LEDGERLIGHT_EXPERIMENT=rehearsal LEDGERLIGHT_STUDY_SUFFIX=-rehearsal bash scripts/train.sh` and budgets of 24 BGE steps, 32 CE steps and three CatBoost trials. Its artifacts stay under `data/pilot/rehearsal`; its results are execution checks.

`bash scripts/ci.sh` runs lint, formatting, unit tests, leakage checks and an isolated synthetic CPU pipeline. Local MLflow, Qdrant, PostgreSQL, Prometheus and Grafana support versioning, retrieval, request logging and diagnostics; this is not a production deployment.

## Results

Results will be added after the final training run.

## License

Code is licensed under [Apache-2.0](LICENSE). The FinRank dataset is CC BY-NC 4.0, is not redistributed, and is downloaded at pinned commit `916dc27e77b90e294ea1b3a2cc5b0abcf07392d1` with a verified SHA256. Trained weights, when published, will carry CC BY-NC 4.0 metadata. Upstream models and dependencies retain their own licenses.
