# Project tracker

This file is the team's single source of truth for what is done, what is in progress and what is next. Update it in the same commit as the work.

**Key dates:** proposal Oct 18 (submitted) · replication done by **Nov 1** · sweeps done by **Nov 18–22** · presentation and repo due **Dec 8**.

Status key: `[x]` done · `[~]` in progress · `[ ]` not started. Fill in an owner when you pick a task up.

---

## Phase 0: Setup (done)

| Status | Task | Owner | Notes |
|---|---|---|---|
| [x] | Repo layout, configs, requirements | Nischal | |
| [x] | SMIB simulator `src/pinn_swing/sim.py` | Nischal | Matches the authors' data to rel. L2 2.4e-4 |
| [x] | Simulator tests `tests/test_sim.py` | Nischal | Equilibrium, tolerance, energy, released data |
| [x] | Prototype forward PINN `notebooks/prototype_forward_pinn.py` | Nischal | PINN 2.86e-2 vs data-only 2.8e-1 (one seed) |
| [x] | Compute benchmark | Nischal | ~41 ms per L-BFGS iteration per run with 8 runs in parallel |

## Phase 1: Core pipeline (done Oct 18)

| Status | Task | Owner | Notes |
|---|---|---|---|
| [x] | `data.py`: forward grid, trajectory split, boundary and interior sampling, noise, collocation | Nischal | Edge trajectories always in train |
| [x] | `models.py`: MLP (P, t) → δ, Xavier init, input scaling, optional trainable m and d | Nischal | float64 |
| [x] | `physics.py`: swing residual via autodiff | Nischal | Checked against a hand derivation |
| [x] | `train.py`: L-BFGS loop, loss = MSE_u + λ·MSE_f, history, timing | Nischal | Same code path for λ = 0; snapshot hook for the animation |
| [x] | `eval.py`: rel. L2 (δ, ω by autodiff and finite difference), extrapolation error, residual RMS | Nischal | Test trajectories and full grid |
| [x] | `scripts/run_forward.py`: one config + seed → JSON result | Nischal | Smoke run, 2k iters, seed 0: PINN 3.2e-2 vs data-only 9.6e-2 (test) |
| [x] | Unit tests for data, physics and train (smoke test) | Nischal | `tests/test_pipeline.py`; 14 tests pass in total |

## Phase 2: Replication (target Nov 1)

| Status | Task | Owner | Notes |
|---|---|---|---|
| [ ] | Forward replication, faithful settings (full grid, boundary sampling) | | Paper: 1.34e-2 |
| [ ] | Generate the 10 (m, d) identification datasets | | m ∈ [0.1, 0.4], d ∈ [0.05, 0.15] |
| [ ] | Inverse PINN (trainable m, d) | | Paper: 0.74% (m), 1.28% (d) |
| [ ] | Two-stage baseline: data-only fit, then least squares for (m, d) | | |
| [ ] | Replication table (results figure/table 2) | | |

## Phase 3: Sweep infrastructure (target Nov 1)

| Status | Task | Owner | Notes |
|---|---|---|---|
| [ ] | `configs/experiments/*.yaml`, one per sweep | | |
| [ ] | `scripts/run_sweep.py`: expand the grid × seeds, run in parallel (1 thread each), skip finished runs | | Must survive restarts |
| [ ] | `scripts/aggregate.py`: raw JSON → CSV with mean, 95% bootstrap CI, failed-run count | | |

## Phase 4: Sweeps (Nov 2 – Nov 18/22)

| Status | Sweep | Runs | Owner | Notes |
|---|---|---|---|---|
| [ ] | S1 Data size × noise, boundary sampling | 320 | | N_u {10,20,40,100} × noise {0,5,15,30}% × λ {0,1} × 10 seeds |
| [ ] | S2 Data size × noise, interior sampling | 320 | | Same grid |
| [ ] | S3 Physics weight λ ∈ {0.01, 0.1, 10} | 30 | | Can cut to 5 seeds |
| [ ] | S4 Collocation N_f ∈ {500, 2000} | 20 | | Can cut to 5 seeds |
| [ ] | S5 Extrapolation (train on t ≤ 10 s) | 30 | | Data-only; PINN colloc [0,10]; PINN colloc [0,20] |
| [ ] | S6 Identification × noise | 80 | | 10 pairs × {PINN, two-stage} × 4 noise levels |

## Phase 5: Analysis (Nov 19 – Dec 2)

| Status | Task | Owner | Notes |
|---|---|---|---|
| [ ] | Statistics: bootstrap CIs, Wilcoxon signed-rank tests | | |
| [ ] | Fig/Table 1: model schematic | | Exists in the proposal |
| [ ] | Fig/Table 2: replication table | | |
| [ ] | Fig/Table 3: error vs N_u and noise (both regimes) | | |
| [ ] | Fig/Table 4: error over time beyond t = 10 s | | |
| [ ] | Fig/Table 5: identification + λ/N_f ablations + cost | | |

## Phase 6: Presentation and repo (Dec 2 – Dec 8)

| Status | Task | Owner | Notes |
|---|---|---|---|
| [~] | Overview animation, scene 1 (rotor swing) | Nischal | `scripts/animate_overview.py` |
| [ ] | Animation scenes 2–4 (sparse data, training split screen, extrapolation) | | Needs training snapshots |
| [ ] | Slides | | |
| [ ] | README final: results summary, how to reproduce, limitations | | |
| [ ] | `scripts/reproduce.sh` end to end | | |

---

## Decisions log

| Date | Decision |
|---|---|
| Oct 8 | Use PyTorch float64 on CPU, L-BFGS, the authors' settings recovered from their code |
| Oct 18 | Noise levels 0, 5, 15, 30% (of target std); ~820 runs total, ~450 PINN |
| Oct 18 | Extrapolation trains on t ∈ [0, 10] s (one full swing period ≈ 9 s) |

## Open questions

- Sweeps end Nov 18 (Kaiyue's suggestion, 2 weeks for analysis before Dec 2) or Nov 22 (proposal)?
- Presentation time limit (not announced yet) decides whether the 30 s animation fits.
- Lower the L-BFGS cap from 50k? The error plateaus by ~2k iterations in the prototype.
