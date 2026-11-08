Paired two-sided Wilcoxon signed-rank tests (same seed = same data). `median_ratio` = median of (baseline error / PINN error); above 1 means the PINN is better.

| sweep | noise | metric | n_pairs | median_pinn | median_two_stage | frac_pinn_better | median_ratio | p_wilcoxon |
|---|---|---|---|---|---|---|---|---|
| S6_identification | 0 | err_m_pct | 30 | 0.513 | 15 | 1 | 30.3 | 1.86e-09 |
| S6_identification | 0.05 | err_m_pct | 30 | 13.7 | 93.4 | 0.967 | 6.25 | 5.59e-09 |
| S6_identification | 0.15 | err_m_pct | 30 | 84.6 | 100 | 0.933 | 1.19 | 4.66e-08 |
| S6_identification | 0.3 | err_m_pct | 30 | 87.8 | 100 | 0.967 | 1.16 | 3.73e-09 |
| S6_identification | 0 | err_d_pct | 30 | 0.551 | 12.8 | 0.967 | 22 | 1.86e-08 |
| S6_identification | 0.05 | err_d_pct | 30 | 16 | 46.2 | 0.8 | 2.54 | 0.00073 |
| S6_identification | 0.15 | err_d_pct | 30 | 91.9 | 88.9 | 0.367 | 0.96 | 0.184 |
| S6_identification | 0.3 | err_d_pct | 30 | 94.3 | 98.1 | 0.6 | 1.03 | 0.503 |
