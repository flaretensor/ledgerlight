# Post-hoc DEV feature ablation

post-hoc exploratory fixed-hyperparameter DEV-only feature ablation

Frozen trial 26, tuned for all 14 features; no retuning for ablations

Three fixed seeds per variant, TRAIN-B only, DEV118 only. No TEST scoring, registry update, alias change, or serving change.

| Variant | Features | DEV nDCG@10 mean [range] (%) | MRR mean [range] (%) |
|---|---:|---:|---:|
| full_14 | 14 | 39.824 [39.112, 40.264] | 42.269 [41.010, 42.945] |
| without_metadata | 11 | 35.448 [35.279, 35.693] | 38.485 [37.491, 39.329] |
| without_ce | 12 | 38.175 [37.807, 38.577] | 41.219 [40.641, 41.771] |
| without_ranks | 10 | 39.379 [38.816, 39.830] | 42.715 [41.761, 43.508] |
| without_length_count | 11 | 38.010 [37.749, 38.443] | 39.926 [39.273, 41.000] |
| retrieval_only | 6 | 34.111 [34.018, 34.166] | 36.176 [36.027, 36.296] |

## Frozen model importance

PredictionValuesChange, descriptive model importance; not causal attribution.

| Feature | Importance |
|---|---:|
| passage_tokens | 15.7909 |
| ce_rank | 13.4736 |
| ticker_match | 11.4261 |
| dense_rank | 9.9573 |
| dense | 9.9080 |
| ce | 9.0239 |
| bm25_rank | 8.0185 |
| bm25 | 6.2519 |
| rrf | 5.3030 |
| query_tokens | 4.9500 |
| rrf_rank | 4.7008 |
| year_match | 0.8748 |
| window_count | 0.3178 |
| form_match | 0.0034 |
