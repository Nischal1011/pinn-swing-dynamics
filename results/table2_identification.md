## Table 2. Identifying inertia m and damping d: PINN vs. two-stage baseline

Relative error of the estimates, median [interquartile range] over 10 (m, d) pairs x 3 seeds, 100 measurements each. Two-stage: fit a data-only network, differentiate it, solve least squares.

| noise | method | error m (%) | error d (%) | failed / runs | time per run (s) |
|---|---|---|---|---|---|
| 0% | PINN | 0.5 [0.1, 0.8] | 0.6 [0.3, 1.2] | 0 / 30 | 202 |
| 0% | Two-stage | 15.0 [11.5, 18.9] | 12.8 [8.1, 19.4] | 0 / 30 | 56 |
| 5% | PINN | 13.7 [9.1, 40.8] | 16.0 [10.0, 32.9] | 0 / 30 | 206 |
| 5% | Two-stage | 93.4 [80.8, 99.5] | 46.2 [27.3, 64.0] | 0 / 30 | 57 |
| 15% | PINN | 84.6 [76.0, 93.6] | 91.9 [87.0, 95.9] | 3 / 30 | 207 |
| 15% | Two-stage | 100.0 [99.6, 100.3] | 88.9 [80.1, 95.6] | 3 / 30 | 59 |
| 30% | PINN | 87.8 [80.4, 93.8] | 94.3 [88.6, 98.4] | 8 / 30 | 198 |
| 30% | Two-stage | 100.0 [99.6, 100.1] | 98.1 [91.3, 99.5] | 5 / 30 | 57 |
