## Table 1. Replication of Misyris et al. (2019): ours vs. the paper

Forward: 40 boundary measurements, 8000 collocation points, L-BFGS; mean ± sd over seeds. Identification: clean-data runs of sweep S6.

| experiment | metric | ours | paper | train_time |
|---|---|---|---|---|
| Forward, PINN | rel. L2 of δ, full grid | 1.82e-02 ± 1.4e-02 (n=3 seeds) | 1.34e-2 | 1398 s |
| Forward, Data-only | rel. L2 of δ, full grid | 4.06e-01 ± 2.9e-01 (n=3 seeds) | not reported | 54 s |
| Identification, PINN | error in m, 10 pairs x 3 seeds, clean data | mean 1.28%, median 0.51%; 1 of 30 runs above 10%; 0 flagged failed | 0.74% | 202 s |
| Identification, PINN | error in d, 10 pairs x 3 seeds, clean data | mean 1.19%, median 0.55%; 0 of 30 runs above 10%; 0 flagged failed | 1.28% | 202 s |
| Identification, Two-stage | error in m, 10 pairs x 3 seeds, clean data | mean 14.97%, median 15.01%; 25 of 30 runs above 10%; 0 flagged failed | not reported | 56 s |
| Identification, Two-stage | error in d, 10 pairs x 3 seeds, clean data | mean 15.14%, median 12.77%; 20 of 30 runs above 10%; 0 flagged failed | not reported | 56 s |
