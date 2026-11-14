## Table 3. Design choices: physics weight λ and collocation points N_f

N_u = 40 clean measurements, boundary sampling, 10 seeds per row. Error: test relative L2 of δ, mean [95% bootstrap CI]. Defaults (λ = 1, N_f = 8000) come from sweep S1.

| setting | value | rel. L2 error of δ | residual RMS | failed / runs | time per run (s) |
|---|---|---|---|---|---|
| λ | 0 | 3.82e-01 [2.6e-01, 4.9e-01] | 6.2e-02 | 1 / 10 | 39 |
| λ | 0.01 | 7.55e-02 [6.5e-02, 8.8e-02] | 9.4e-03 | 0 / 10 | 2176 |
| λ | 0.1 | 3.22e-02 [2.8e-02, 3.7e-02] | 2.9e-03 | 0 / 10 | 2259 |
| λ | 1 | 1.60e-02 [1.3e-02, 2.0e-02] | 9.2e-04 | 0 / 10 | 904 |
| λ | 10 | 1.38e-02 [8.5e-03, 2.1e-02] | 3.3e-04 | 0 / 10 | 1880 |
| N_f (λ = 1) | 500 | 2.16e-02 [1.8e-02, 2.6e-02] | 1.2e-03 | 0 / 10 | 211 |
| N_f (λ = 1) | 2000 | 2.07e-02 [1.2e-02, 3.2e-02] | 1.0e-03 | 0 / 10 | 419 |
| N_f (λ = 1) | 8000 | 1.60e-02 [1.3e-02, 2.0e-02] | 9.2e-04 | 0 / 10 | 904 |
