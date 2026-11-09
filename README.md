# How much does physics help? A PINN study for SMIB swing dynamics

CISC 889 final project. We replicate Misyris, Venzke & Chatzivasileiadis, *Physics-Informed Neural Networks for Power
Systems* (arXiv 1911.03737), then ablate the swing-equation loss: same network, data, optimizer, iteration budget and
seeds, with and without the physics term. The axes are data size, measurement noise, extrapolation window, and
parameter identification under noise.

```
m * delta'' + d * delta' + B * V1 * V2 * sin(delta) - P = 0
```

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
  run_forward.py             one forward run -> JSON
  run_identification.py      one identification run -> JSON
  run_sweep.py               a whole sweep in parallel, restart-safe, shardable across machines
  aggregate.py               raw JSON -> per-run and per-setting CSVs (mean, 95% bootstrap CI, failures)
  stats.py                   paired Wilcoxon tests (PINN vs. data-only, PINN vs. two-stage)
  summarize_replication.py   replication table, ours vs. the paper
  make_figures.py            result figures and table 5
  animate_overview.py        presentation animation, scene 1 (the swing)
  record_animation_runs.py   records predictions during training for scenes 2-3
  animate_training.py        scenes 2 (with vs. without physics) and 3 (identifying m, d)
tests/                       simulator, data, physics, training, identification and sweep tooling (pytest)
reference/                   authors' original TF1 code and .mat data (read-only)
notebooks/                   early prototype
results/                     summaries, tables and replication runs (raw sweep output is not committed)
figures/                     generated figures
```

## Reproduce

```bash
pip install -r requirements.txt
pytest                                                    # all tests
python scripts/run_sweep.py configs/experiments/S1_size_noise_boundary.yaml --workers 8
python scripts/aggregate.py S1_size_noise_boundary        # repeat for S2..S6
python scripts/summarize_replication.py
python scripts/stats.py
python scripts/make_figures.py
```

A full-budget PINN run takes ~20–35 min on one CPU thread; all sweeps take ~21 h with 8 parallel workers. Use
`--shard i/n` to split a sweep across machines.

## Status

See [TRACKER.md](TRACKER.md) for the detailed plan.

- [x] Simulator matches the authors' released data (rel. L2 2.4e-4)
- [x] Forward replication: PINN 1.0e-2–3.4e-2 (3 seeds) vs. paper 1.34e-2
- [x] Sweeps S3 (physics weight) and S4 (collocation points)
- [ ] Sweeps S1, S2 (data size × noise), S5 (extrapolation), S6 (identification) running
- [ ] Figures, statistics, slides

## Limitations

Classical single-machine model, simulated data, small networks trained with L-BFGS. Identification uses only pairs
whose trajectories stay synchronous, which limits the damping range to d ≈ 0.085–0.15.

## Generative AI disclosure

Claude (Anthropic) was used to read the baseline code, scaffold this repository, and write code and tests.
All generated code and results are verified by the authors.
