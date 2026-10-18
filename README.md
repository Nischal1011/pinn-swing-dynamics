# How much does physics help? A PINN study for SMIB swing dynamics

CISC 889 final project. Replicates Misyris, Venzke & Chatzivasileiadis, *Physics-Informed Neural Networks for Power
Systems* (arXiv 1911.03737). Then it ablates the swing-equation loss: same network, data, optimizer, budget, and seeds,
with the physics weight lambda in {0, 0.01, 0.1, 1, 10}. The axes are data size, noise, and extrapolation window.

```
m * delta'' + d * delta' + B * V1 * V2 * sin(delta) - P = 0
```

## Layout

```
configs/                 forward.yaml, identification.yaml, experiments/*.yaml (one file per sweep)
src/pinn_swing/
  sim.py                 reference solver (SciPy DOP853)            [done, verified]
  data.py                grids, train/val/test trajectory split, N_u sampling (boundary | interior), noise
  models.py              MLP (P, t) -> delta; trainable m, d for identification
  physics.py             swing residual via autodiff
  train.py               L-BFGS loop, loss = MSE_u + lambda * MSE_f, logging
  eval.py                rel. L2 (delta, omega), extrapolation error, residual, timing, param error
  viz.py                 the five figures
scripts/
  make_data.py           regenerate data/ from configs
  run_sweep.py           run an experiment config over seeds in parallel -> results/raw/*.json
  aggregate.py           results/raw -> results/*.csv (mean, 95% CI, failed-run counts)
  reproduce.sh           end to end
tests/                   simulator sanity checks (pytest)
reference/               authors' original TF1 code + .mat data (read-only, for comparison)
notebooks/               prototypes and exploration
data/ results/ figures/  generated outputs
```

## Quick start

```bash
pip install -r requirements.txt
pytest                       # simulator sanity checks
```

## Status

- [x] Simulator matches the authors' released data (rel. L2 2.4e-4) and passes sanity checks
- [x] Prototype forward PINN: 2.85e-2 vs paper 1.34e-2; data-only 2.8e-1 (`notebooks/prototype_forward_pinn.py`)
- [ ] Replication: forward and identification
- [ ] Ablation sweeps
- [ ] Figures, slides

## Limitations

Classical single-machine model, simulated data, small networks.

