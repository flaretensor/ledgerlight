# Post-hoc paired bootstrap

post-hoc exploratory paired bootstrap of saved TEST query scores only

Three unadjusted primary comparisons; secondary MRR intervals are also exploratory. No selection change or new TEST inference.

| CatBoost minus | Metric | Difference (pp) | 95% CI (pp) |
|---|---|---:|---:|
| RRF-ft | ndcg@10 | 5.139 | [0.616, 9.624] |
| RRF-ft | mrr | 6.029 | [-0.139, 12.168] |
| CE-ft | ndcg@10 | 6.448 | [2.457, 10.513] |
| CE-ft | mrr | 9.634 | [3.576, 15.704] |
| BGE-ft-dense | ndcg@10 | 6.044 | [2.028, 10.063] |
| BGE-ft-dense | mrr | 7.117 | [1.427, 12.881] |
