# How much does physics help? A PINN study for SMIB swing dynamics

CISC 889 final project. We replicate Misyris, Venzke & Chatzivasileiadis, *Physics-Informed Neural Networks for Power
Systems* (arXiv 1911.03737), then ablate the swing-equation loss: same network, data, optimizer, iteration budget and
seeds, with and without the physics term. The axes are data size, measurement noise, extrapolation window, and
parameter identification under noise.

```
m * delta'' + d * delta' + B * V1 * V2 * sin(delta) - P = 0
```

**Problem.** After a disturbance, a generator's rotor angle δ swings before it settles (or the machine loses
synchronism). The swing equation above describes a single machine on an infinite bus (SMIB): input power P,
inertia m, damping d. Solving it for every operating point is slow, and m and d are often unknown.

**Method.** A tanh network maps (P, t) to δ(P, t). The loss is `MSE_u + λ · MSE_f`: the error on measured δ plus
λ times the mean squared swing-equation residual, computed with autodiff at collocation points that need no
measurements. λ = 0 gives the same network trained on data alone, our baseline. For identification, m and d are
trainable parameters; the baseline there is a two-stage method (fit a data-only network, differentiate it, solve
least squares for m and d). Every comparison is paired: identical network, data, L-BFGS optimizer, 50k-iteration
budget and random seed; only the physics term changes. 960 training runs in total, 10 seeds per setting
(3 per identification pair).

## Results

The project reports five results. Each has a script that rebuilds it from the committed CSVs (see Reproduce).

| | Result | File |
|---|---|---|
| Table 1 | Replication: forward error 1.8e-2 vs. the paper's 1.3e-2; identification error in m / d 0.5% / 0.6% vs. 0.74% / 1.28% | [results/table1_replication.md](results/table1_replication.md) |
| Fig. 1 | Data size × noise, two measurement layouts. PINN error lower in 314 of 320 seed-matched pairs: median gap 7–46× with measurements on the domain boundary, 2–8× with interior measurements. Data-only error exceeds 1 in 101 of 320 runs (it memorises noise, so more noisy data makes it worse); the PINN never does | [figures/fig1_error_vs_data_noise.png](figures/fig1_error_vs_data_noise.png) |
| Fig. 2 | Extrapolation, measurements only on t ≤ 10 s, error over 0–20 s (median): PINN with physics on 0–20 s 0.015; physics only on 0–10 s 0.19; data-only 0.53. Extrapolation comes from enforcing the equation where there is no data | [figures/fig2_extrapolation.png](figures/fig2_extrapolation.png) |
| Table 2 | Identifying m and d (median error): clean data PINN 0.5% / 0.6% vs. two-stage 15% / 13%; 5% noise 14% / 16% vs. 93% / 46%; at 15–30% noise both methods fail (> 80%) | [results/table2_identification.md](results/table2_identification.md) |
| Table 3 | Design choices: error falls as λ grows (0.075 at λ = 0.01 to 0.014 at λ = 10); 500 collocation points are nearly as accurate as 8,000 (0.022 vs. 0.016) at a quarter of the training time | [results/table3_design_choices.md](results/table3_design_choices.md) |

Paired two-sided Wilcoxon signed-rank tests for every setting: [results/stats_paired_tests.md](results/stats_paired_tests.md).
Per-run and per-setting CSVs (mean, median, 95% bootstrap CI, failures) for each sweep: `results/S*_runs.csv`,
`results/S*_summary.csv`.

**Takeaways.** The physics term acts as a regulariser: it helps most when data is scarce, noisy, or does not cover
the domain, and it is what makes extrapolation work. It does not undo heavy noise: above ~15% noise, neither method
identifies m and d.


## Layout

```
configs/
  forward.yaml               forward problem (authors' settings recovered from their code)
  identification.yaml        inverse problem: 10 stable (m, d) pairs, network, optimizer
  experiments/S1..S6.yaml    one file per sweep (960 runs in total)
src/pinn_swing/
  sim.py                     reference solver (SciPy DOP853), verified against the authors' data
  data.py                    forward grid, trajectory split, boundary/interior sampling, noise, collocation
  models.py                  tanh MLP (P, t) -> delta; optional trainable m, d
  physics.py                 swing-equation residual via autodiff
  train.py                   L-BFGS, loss = MSE_u + lambda * MSE_f (same code path for lambda = 0)
  eval.py                    relative L2 errors (delta, omega), extrapolation error, residual, error over time
  identification.py          inverse PINN and the two-stage baseline
scripts/
  reproduce.sh               one command for all five results (see below)
  run_forward.py             one forward run -> JSON
  run_identification.py      one identification run -> JSON
  run_sweep.py               a whole sweep, restart-safe, shardable, with a free-RAM guard
  aggregate.py               raw JSON -> per-run and per-setting CSVs (mean, 95% bootstrap CI, failures)
  stats.py                   paired Wilcoxon tests (PINN vs. data-only, PINN vs. two-stage)
  summarize_replication.py   Table 1
  make_figures.py            Fig. 1, Fig. 2, Table 2, Table 3
  animate_overview.py        presentation animation, scene 1 (the swing)
  record_animation_runs.py   records predictions during training for scenes 2-3
  animate_training.py        scenes 2 (with vs. without physics) and 3 (identifying m, d)
tests/                       simulator, data, physics, training, identification and sweep tooling (pytest)
reference/                   authors' released .mat data, used to verify the simulator (read-only)
results/                     tables, per-run CSVs, replication runs (raw sweep JSON is not committed)
figures/                     Fig. 1 and Fig. 2
```

| Sweep | What varies | Runs |
|---|---|---|
| S1, S2 | N_u ∈ {10, 20, 40, 100} × noise ∈ {0, 5, 15, 30}% × λ ∈ {0, 1}; boundary (S1) or interior (S2) measurements | 320 + 320 |
| S3 | λ ∈ {0.01, 0.1, 10} (λ = 0 and 1 come from S1) | 30 |
| S4 | N_f ∈ {500, 2000} (8000 comes from S1) | 20 |
| S5 | measurements on t ≤ 10 s; data-only, physics on t ≤ 10 s, physics on t ≤ 20 s | 30 |
| S6 | 10 (m, d) pairs × {PINN, two-stage} × noise ∈ {0, 5, 15, 30}% | 240 |

## Reproduce

Tested with Python 3.12, PyTorch 2.14, NumPy 2.5, SciPy 1.18, pandas 3.0, matplotlib 3.11, on macOS (CPU, float64).

```bash
pip install -r requirements.txt
pytest                                   # tests
sh scripts/reproduce.sh                  # rebuild all tables, figures and tests from the committed CSVs (~1 min)
sh scripts/reproduce.sh --full           # re-run all 966 trainings first, then rebuild
```

**Data.** Nothing to download. Every run simulates its own trajectories with SciPy's DOP853 solver
(`src/pinn_swing/sim.py`, settings in `configs/*.yaml`), so the data are regenerated exactly from the config and
seed. The simulator matches the authors' released data (`reference/`) to a relative L2 error of 2.4e-4.

**Compute.** A full-budget PINN run takes ~4–40 min on one CPU thread, depending on N_f and λ; a data-only run ~1 min. The full set takes
several days one run at a time; `WORKERS=4 sh scripts/reproduce.sh --full` runs four at once (~400 MB RAM each),
and `run_sweep.py --shard i/n` splits a sweep across machines. Single runs:
`python scripts/run_forward.py --lam 1 --N_u 40 --noise 0.05 --seed 0` and
`python scripts/run_identification.py --pair 1 --method pinn --seed 0`.

## Limitations and known issues

- Classical single-machine model, simulated data, small networks trained with L-BFGS; results may not transfer to
  multi-machine systems or measured data.
- Identification uses only (m, d) pairs whose trajectories stay synchronous, which limits damping to
  d ≈ 0.085–0.15.
- A run counts as failed if it crashes or gives NaN, if its relative L2 error exceeds 1, or if a physics run is
  stuck (final residual > 100× its setting's median). Fig. 1 and Table 2 use medians over all runs, failures
  included; Table 3 reports means with bootstrap CIs.
- L-BFGS occasionally stalls in a poor local minimum: e.g. one S2 seed at N_u = 10 ends near 0.52 error with or
  without physics.
- The raw per-run JSON (results/raw/) is not committed because of its size; Fig. 2's per-time curves are kept in
  `results/S5_extrapolation_error_by_time.csv` so the figure rebuilds without it.
- Runs are deterministic per seed on one machine, but PyTorch and BLAS versions can change the last digits.
