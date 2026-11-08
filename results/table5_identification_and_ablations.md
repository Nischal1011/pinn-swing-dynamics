### Identification of m and d (median over 10 pairs of each pair's median over 3 seeds)

| method | noise | error m (%) | error d (%) | failed / runs | time per run (s) |
|---|---|---|---|---|---|
| pinn | 0% | 0.51 | 0.53 | 0 / 30 | 202 |
| pinn | 5% | 17.26 | 15.25 | 0 / 30 | 206 |
| pinn | 15% | 83.36 | 91.94 | 3 / 30 | 207 |
| pinn | 30% | 89.75 | 94.27 | 8 / 30 | 198 |
| two_stage | 0% | 15.41 | 12.77 | 0 / 30 | 56 |
| two_stage | 5% | 92.11 | 43.43 | 0 / 30 | 57 |
| two_stage | 15% | 100.03 | 89.27 | 3 / 30 | 59 |
| two_stage | 30% | 100.02 | 98.38 | 5 / 30 | 57 |

### Physics weight λ (N_u = 40, no noise, boundary sampling)

| λ | rel. L2 error of δ (mean [95% CI]) | residual RMS | failed / runs | time per run (s) |
|---|---|---|---|---|
| 0.01 | 7.55e-02 [6.5e-02, 8.8e-02] | 9.4e-03 | 0 / 10 | 2176 |
| 0.1 | 3.22e-02 [2.8e-02, 3.7e-02] | 2.9e-03 | 0 / 10 | 2259 |
| 10 | 1.38e-02 [8.5e-03, 2.1e-02] | 3.3e-04 | 0 / 10 | 1880 |

### Collocation points N_f (PINN, N_u = 40, no noise)

| N_f | rel. L2 error of δ (mean [95% CI]) | failed / runs | time per run (s) |
|---|---|---|---|
| 500 | 2.16e-02 [1.8e-02, 2.6e-02] | 0 / 10 | 211 |
| 2000 | 2.07e-02 [1.2e-02, 3.2e-02] | 0 / 10 | 419 |
