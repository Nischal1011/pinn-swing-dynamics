Paired two-sided Wilcoxon signed-rank tests (same seed = same data). `median_ratio` = median of (baseline error / PINN error); above 1 means the PINN is better.

| sweep | N_u | noise | metric | n_pairs | median_pinn | median_data_only | frac_pinn_better | median_ratio | p_wilcoxon | median_two_stage |
|---|---|---|---|---|---|---|---|---|---|---|
| S1_size_noise_boundary | 10 | 0 | rel_l2_delta | 10 | 0.0456 | 0.54 | 1 | 12.6 | 0.00195 | nan |
| S1_size_noise_boundary | 10 | 0.05 | rel_l2_delta | 10 | 0.062 | 0.552 | 1 | 14.6 | 0.00195 | nan |
| S1_size_noise_boundary | 10 | 0.15 | rel_l2_delta | 10 | 0.102 | 0.646 | 1 | 7.19 | 0.00195 | nan |
| S1_size_noise_boundary | 10 | 0.3 | rel_l2_delta | 10 | 0.111 | 0.733 | 1 | 9.26 | 0.00195 | nan |
| S1_size_noise_boundary | 20 | 0 | rel_l2_delta | 10 | 0.0228 | 0.363 | 1 | 17.1 | 0.00195 | nan |
| S1_size_noise_boundary | 20 | 0.05 | rel_l2_delta | 10 | 0.0375 | 0.587 | 1 | 17.6 | 0.00195 | nan |
| S1_size_noise_boundary | 20 | 0.15 | rel_l2_delta | 10 | 0.0748 | 1.37 | 1 | 16.9 | 0.00195 | nan |
| S1_size_noise_boundary | 20 | 0.3 | rel_l2_delta | 10 | 0.134 | 1.78 | 1 | 13.2 | 0.00195 | nan |
| S1_size_noise_boundary | 40 | 0 | rel_l2_delta | 10 | 0.0146 | 0.477 | 1 | 34.5 | 0.00195 | nan |
| S1_size_noise_boundary | 40 | 0.05 | rel_l2_delta | 10 | 0.0584 | 1.06 | 1 | 14.8 | 0.00195 | nan |
| S1_size_noise_boundary | 40 | 0.15 | rel_l2_delta | 10 | 0.0717 | 2.02 | 1 | 30.6 | 0.00195 | nan |
| S1_size_noise_boundary | 40 | 0.3 | rel_l2_delta | 10 | 0.119 | 5.75 | 1 | 46.4 | 0.00195 | nan |
| S1_size_noise_boundary | 100 | 0 | rel_l2_delta | 10 | 0.0135 | 0.281 | 1 | 18.9 | 0.00195 | nan |
| S1_size_noise_boundary | 100 | 0.05 | rel_l2_delta | 10 | 0.0464 | 0.785 | 1 | 38.8 | 0.00195 | nan |
| S1_size_noise_boundary | 100 | 0.15 | rel_l2_delta | 10 | 0.0995 | 3.93 | 1 | 41.8 | 0.00195 | nan |
| S1_size_noise_boundary | 100 | 0.3 | rel_l2_delta | 10 | 0.137 | 2.92 | 1 | 21.8 | 0.00195 | nan |
| S6_identification | nan | 0 | err_m_pct | 30 | 0.513 | nan | 1 | 30.3 | 1.86e-09 | 15 |
| S6_identification | nan | 0.05 | err_m_pct | 30 | 13.7 | nan | 0.967 | 6.25 | 5.59e-09 | 93.4 |
| S6_identification | nan | 0.15 | err_m_pct | 30 | 84.6 | nan | 0.933 | 1.19 | 4.66e-08 | 100 |
| S6_identification | nan | 0.3 | err_m_pct | 30 | 87.8 | nan | 0.967 | 1.16 | 3.73e-09 | 100 |
| S6_identification | nan | 0 | err_d_pct | 30 | 0.551 | nan | 0.967 | 22 | 1.86e-08 | 12.8 |
| S6_identification | nan | 0.05 | err_d_pct | 30 | 16 | nan | 0.8 | 2.54 | 0.00073 | 46.2 |
| S6_identification | nan | 0.15 | err_d_pct | 30 | 91.9 | nan | 0.367 | 0.96 | 0.184 | 88.9 |
| S6_identification | nan | 0.3 | err_d_pct | 30 | 94.3 | nan | 0.6 | 1.03 | 0.503 | 98.1 |
