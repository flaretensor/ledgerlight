# Ledgerlight

Ledgerlight answers questions about the financial reports of 22 U.S. companies and cites the passages it used. When the retrieved passages do not support an answer, the model is instructed to reply `not found in sources`.

The whole system runs on one laptop with a 4 GB GPU: hybrid retrieval, two fine-tuned neural rankers, a learning-to-rank model, a local LLM, a held-out evaluation, and an isolated public demo.

**Demo:** <https://ledgerlight.flaretensor.uk>. It is served from a single laptop and is online only while that laptop is running.

**Headline result:** on the held-out TEST questions the full ranking pipeline reaches nDCG@10 of 38.58%, against 22.79% for BM25 (+15.79 points, 95% CI 10.27 to 21.44). Answer correctness is not measured; see [Limitations](#limitations).

## Corpus

The corpus is the passage collection of [FinRank](https://github.com/datanxt/FinRank/tree/916dc27e77b90e294ea1b3a2cc5b0abcf07392d1) ([paper](https://arxiv.org/abs/2608.07400)): 5,230 curated excerpts from 10-K and 10-Q filings of 22 companies in energy, healthcare and automotive. These are excerpts, not complete SEC reports, and the dataset carries no links to the original filings.

FinRank also provides 1,185 human-written questions with passage-level relevance labels. Ledgerlight uses its native split: 948 TRAIN, 118 DEV and 119 TEST questions.

## How a question is answered

1. **Filters.** The user may restrict the search to a company, a form type (10-K or 10-Q) and a filing year. Filters apply before retrieval. They are never filled in from the relevance labels.
2. **First-stage retrieval.** BM25 (`bm25s`) and a fine-tuned `bge-small-en-v1.5` encoder (384 dimensions, stored in Qdrant) each return 100 candidates.
3. **Fusion.** Reciprocal rank fusion (k = 10) merges the two lists.
4. **Cross-encoder.** A fine-tuned MiniLM-L6 cross-encoder rescores the top 100 fused candidates.
5. **Learning to rank.** A CatBoost YetiRank model reorders the 100 candidates using 14 features: the BM25, dense, fusion and cross-encoder scores and ranks, passage and question lengths, the window count, and whether the question's ticker, year and form match the passage metadata.
6. **Generation.** The five best passages each contribute one window to Qwen3.5-2B (Q4_K_M GGUF, served by llama.cpp) with an 8,192-token context, a 512-token thinking budget and a 1,024-token generation limit.
7. **Validation.** Each request issues its own source markers. The gateway converts valid markers to numbered citations and rejects invented markers, URLs and HTML. Answers stream to the browser as they are generated; the model's reasoning is hidden, and the final text passes the same validation as a non-streamed answer.

### Long passages

3,246 of the 5,230 passages are longer than the encoder's 512-token limit, so each passage is split into overlapping windows of 448 tokens with a step of 320, 12,604 windows in total. A passage is scored by its best window. Windows are cut on character offsets of the original text, so figures such as `$31.9 billion` reach the rankers and the LLM exactly as written. The 448-token limit leaves room for the question in the cross-encoder's 512-token input.

## Evaluation protocol

- TRAIN is split into 712 questions for the neural models and 236 for the ranker, so the ranker learns from scores the neural models did not train on.
- DEV selects every configuration: 6 Optuna trials for the encoder, 6 for the cross-encoder and 50 for CatBoost.
- TEST was scored once, after the selection was frozen. The evaluation command refuses to run a second time.
- Metrics are averaged over questions. Confidence intervals come from a paired bootstrap over questions (10,000 resamples, seed 42).
- Training targets treat unlabeled candidates as non-relevant, except whitespace-equivalent copies of known positives, which are excluded.

## Results

TEST, 119 questions, all values in percent. The last column gives DEV nDCG@10 for comparison.

| Method | Recall@10 | Recall@100 | MRR | nDCG@10 | DEV nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| BM25 | 37.04 | 65.62 | 22.97 | 22.79 | 24.18 |
| BGE, pretrained | 33.40 | 67.37 | 19.81 | 19.96 | 23.57 |
| BM25 + BGE, fused | 41.25 | 70.03 | 22.96 | 24.43 | 27.32 |
| BM25 + BGE, fused, pretrained cross-encoder | 39.22 | 70.03 | 26.75 | 25.02 | 26.89 |
| BGE, fine-tuned | 43.98 | 73.39 | 30.38 | 28.96 | 32.57 |
| BM25 + fine-tuned BGE, fused | 49.51 | 73.32 | 30.41 | 31.23 | 32.19 |
| The same with the pretrained cross-encoder | 42.58 | 73.32 | 29.07 | 27.17 | 30.21 |
| The same with the fine-tuned cross-encoder | 51.89 | 73.32 | 34.02 | 33.79 | 34.77 |
| Fine-tuned cross-encoder and CatBoost ranker (selected) | 55.25 | 73.32 | 38.06 | 38.58 | 43.67 |

Paired differences in TEST nDCG@10, in points, with 95% confidence intervals:

| Comparison | Difference | 95% CI |
| --- | --- | --- |
| Selected pipeline vs. BM25 | +15.79 | 10.27 to 21.44 |
| CatBoost ranker vs. fine-tuned cross-encoder | +4.79 | 1.57 to 8.18 |
| Fine-tuned vs. pretrained cross-encoder | +6.62 | 3.06 to 10.46 |
| Fine-tuned vs. pretrained BGE, dense only | +9.00 | 5.25 to 12.93 |

Only the first comparison was planned before TEST was scored. The others, and everything below, were computed afterwards from the saved rankings and are descriptive.

**DEV to TEST gap.** The selected pipeline scores 43.67% on DEV and 38.58% on TEST. DEV was used for selection, so the TEST figure is the one to quote.

**Candidate coverage.** Recall@100 of 73.32% caps what any reranker can reach. For 22 of the 119 TEST questions no labeled passage is among the 100 candidates; for 27 one is among the candidates but not in the top five; for 70 at least one is in the five passages sent to the LLM.

**Feature groups.** Retraining the ranker without one group of features lowers DEV nDCG@10 by 5.51 points without the metadata matches (CI 2.62 to 8.69), 5.26 without the first-stage scores (2.05 to 8.42) and 3.94 without the cross-encoder (0.98 to 7.02). Removing the length features costs 2.67 points, with an interval that includes zero.

**A larger encoder without training.** `bge-large-en-v1.5`, used as published, reaches 24.54% dense and 25.44% fused TEST nDCG@10. The fine-tuned small encoder is 4.42 points ahead of it with an interval that includes zero (−0.84 to 9.55). The full pipeline is 13.14 points ahead of the fused large-encoder baseline (7.58 to 19.02).

**Known passages.** For 70 of the 119 TEST questions at least one relevant passage is also labeled relevant for some TRAIN question. The gain over BM25 is 12.98 points on those questions and 19.81 points on the other 49, so it is positive in both groups. All splits share one corpus, and normalized text matching raises the number of such questions to 78, so this does not show how the pipeline behaves on documents it has not seen.

## Serving

Measured on the reference machine: Intel Core i5-12500H, 64 GB RAM, NVIDIA RTX 3050 Laptop with 4 GB of memory. The encoder and cross-encoder can run on five backends without retraining. Latency is the search stage over the 118 DEV questions. Peak memory is the whole GPU while the LLM, fully on the GPU, answers eight questions.

| Search backend | Search p95, s | Peak GPU memory, MiB | Status |
| --- | --- | --- | --- |
| Triton Inference Server, ONNX Runtime | 0.911 | 2,424 | default |
| PyTorch | 0.944 | 3,130 | available |
| OpenVINO on CPU | 5.359 | 1,461 | available; used by the CPU mode |
| ONNX Runtime CUDA, in process | not completed | 2,626 | excluded |

Every backend reproduces the PyTorch scores within 1e-4 and the DEV nDCG@10 within 0.005. In-process ONNX Runtime answers single questions but ran out of its CUDA memory arena partway through the 118-question pass while sharing the GPU with the LLM, so it is excluded. Triton with the OpenVINO backend also runs and is never a default.

The default is chosen by a fixed rule from these measurements: among backends that pass every check and stay under 3,700 MiB, keep those within 10% of the fastest p95, then take the one with the lowest peak memory.

With the default backend, a sequential replay of the 118 DEV questions through the streaming endpoint returned 118 answers with no errors or timeouts: 88 with citations and 30 abstentions. The first visible text, observed in 102 of the 118 requests, arrived after 7.3 s (median) and 7.5 s (p95); the full answer took 9.7 s (median) and 19.9 s (p95). The LLM generates about 90 tokens per second on the GPU.

**CPU mode.** `--cpu` runs retrieval on OpenVINO and the LLM entirely on the CPU, leaving the GPU free for other work. One measured answer took 57.6 s, with the first text after 43.8 s, at about 20 tokens per second.

**Model registry.** MLflow registers all nine pipeline variants. The serving stack loads the `champion` alias and keeps a `previous` alias; switching champion to previous and back has been exercised end to end.

**Monitoring.** Prometheus and a Grafana dashboard track request outcomes, queue depth, stage latencies and time to first answer text. Both listen on localhost only.

## The public demo

- **Capacity.** One answer is generated at a time. Up to three questions wait in a queue for at most 60 seconds; beyond that the page says the demo is busy. Each client may ask six questions per minute, and a question may be 500 characters long.
- **Privacy.** Questions, IP addresses and visitor identities are not stored. The service keeps aggregate counters and an in-memory rate limiter that forgets idle clients.
- **Isolation.** The web page runs in an unprivileged, read-only container with no host mounts, no GPU and no internet access. It can reach the answer gateway and nothing else: not the models, the databases, the monitoring, or the host. Of the stack's own services, the Cloudflare Tunnel container can reach only the web page; its other connection is the outbound one to Cloudflare, which is how visitors reach the site. Every port the stack publishes is bound to `127.0.0.1`. These properties are checked by `scripts/isolation_check.py`.
- **Manual operation.** Nothing starts on boot. The site exists only between a manual start and a manual stop.

## Running it

Requirements: Python 3.12, uv and Docker Compose. GPU serving needs an NVIDIA GPU with the NVIDIA container runtime. CPU mode uses no GPU memory, but the start command currently still expects `nvidia-smi` to be present.

```sh
source scripts/env.sh
uv sync --frozen
cp .env.example .env    # then set local database and Grafana passwords; keep .env private
docker compose --profile triton pull --ignore-buildable
```

Trained weights are not stored in Git. They are packaged as a model bundle of 25 files (674 MB): the fine-tuned encoder, cross-encoder and ranker with tokenizers and configurations, their ONNX exports, `SHA256SUMS` and `provenance.json`. The bootstrap command verifies the checksums, downloads the pinned FinRank file and the pinned Qwen GGUF, and rebuilds the windows, the BM25 and Qdrant indexes and the registry:

```sh
python scripts/release_models.py bootstrap --bundle PATH
# or, from a Hugging Face model repository pinned to a full commit hash:
python scripts/release_models.py bootstrap --repo REPOSITORY --revision COMMIT
```

On the reference machine the bootstrap took about 14 minutes with downloads already cached.

```sh
python scripts/site.py start          # default backend, LLM on the GPU
python scripts/site.py start --cpu    # no GPU memory used
python scripts/site.py status
python scripts/site.py stop
```

The page is served at `http://127.0.0.1:8501`. A cold start takes about a minute and a half. Stop the stack before switching modes. `start --public` additionally starts the tunnel container and refuses to run without a token file at `.secrets/cloudflared/token`.

### Training and evaluation

```sh
python -m ledgerlight.data ingest      # download the pinned dataset
python -m ledgerlight.data validate
bash scripts/train.sh                  # resumable training, DEV selection, registration
bash scripts/ci.sh                     # lint, format, tests, leakage checks, synthetic CPU pipeline
```

Training took about two and a half hours on the reference machine. Fingerprints over source code, parameters, dataset and upstream weights guard resumption and cache reuse. `python -m ledgerlight.final_evaluation --allow-test` scores TEST only when a frozen DEV selection exists and no TEST run has been started before.

GitHub Actions runs `scripts/ci.sh` on every push. It needs neither a GPU nor Docker.

## Limitations

- **Answer correctness is not measured.** FinRank has no reference answers. A valid citation shows that the model pointed at a retrieved passage, not that the passage supports the statement or that a figure is right. Of the 88 cited DEV answers, 40 cite at least one passage labeled relevant; this is an overlap count, not an accuracy.
- **Relevance labels are incomplete.** Passages nobody judged count as non-relevant in every metric, and in the ranker's training targets.
- **TEST contains new questions about known documents.** All splits share one corpus. The results say nothing about documents or companies outside it.
- **Filing year is not fiscal year.** The year metadata is the filing year, sometimes imputed by the dataset authors. A question about fiscal 2024 is often answered by a filing from 2025, so a year filter can hide the right passage.
- **The corpus is small.** 22 companies, excerpts only, no original SEC documents.
- **This is not a production deployment.** The registry, rollback, monitoring and load tests are local exercises on one machine.

## License

Code is licensed under [Apache-2.0](LICENSE).

The FinRank annotations are CC BY-NC 4.0. They are not redistributed here; the pinned commit `916dc27e77b90e294ea1b3a2cc5b0abcf07392d1` is downloaded and checked against a SHA256. The underlying SEC filings are separate from that license. Trained weights, when published, carry CC BY-NC 4.0 metadata because they were trained on those annotations.

Base models keep their own licenses: `BAAI/bge-small-en-v1.5` (MIT), `cross-encoder/ms-marco-MiniLM-L6-v2` (Apache-2.0) and Qwen3.5-2B (Apache-2.0).
