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

## Phase 2: Replication (done Oct 18)

| Status | Task | Owner | Notes |
|---|---|---|---|
| [x] | Forward replication, faithful settings (full grid, boundary sampling) | Nischal | PINN 1.04e-2, 1.03e-2, 3.39e-2 (3 seeds) vs paper 1.34e-2; data-only 0.10–0.67. ~32k iters, ~23 min per PINN run under load |
| [x] | Generate the 10 (m, d) identification datasets | Nischal | Latin hypercube, seed 0 (`identification.make_pairs`); 40 trajectories per pair |
| [x] | Inverse PINN (trainable m, d) | Nischal | 9 of 10 pairs within 2.6%, median 0.5%; pair 2 stuck with seed 0 (fine with seeds 1, 2). ~3 min per run |
| [x] | Two-stage baseline: data-only fit, then least squares for (m, d) | Nischal | Median 17% error even on clean data; 0 of 10 pairs within 10% for m |
| [x] | Replication table (results figure/table 2) | Nischal | `scripts/summarize_replication.py` → `results/table1_replication.md` (Table 1) |

## Phase 3: Sweep infrastructure (done Oct 18)

| Status | Task | Owner | Notes |
|---|---|---|---|
| [x] | `configs/experiments/*.yaml`, one per sweep | Nischal | S1–S6, 960 runs total |
| [x] | `scripts/run_sweep.py`: expand the grid × seeds, run in parallel (1 thread each), skip finished runs | Nischal | Restart-safe; `--shard i/n` splits a sweep across laptops; `--dry-run` |
| [x] | `scripts/aggregate.py`: raw JSON → CSV with mean, 95% bootstrap CI, failed-run count | Nischal | Failure rule: crash/NaN, error > 1, or physics residual > 100× setting median |

## Phase 4: Sweeps (Nov 2 – Nov 18/22)

| Status | Sweep | Runs | Owner | Notes |
|---|---|---|---|---|
| [x] | S1 Data size × noise, boundary sampling | 320 | Nischal | Done Nov 10, 0 errors. PINN beats data-only in all 160 seed-matched pairs (median 7–46× lower error). PINN never above error 1; data-only above 1 in 78/160 runs because it overfits the noise (train MSE ~10× below noise variance), so more noisy data makes it worse |
| [x] | S2 Data size × noise, interior sampling | 320 | Nischal | Done Nov 14, 0 errors. PINN better in 154/160 pairs, median ratio 1.7–8× (smaller gap than boundary sampling: data-only interpolates well when points cover the domain). Data-only error > 1 in 23/160, PINN 0/160 |
| [x] | S3 Physics weight λ ∈ {0.01, 0.1, 10} | 30 | Nischal | 0 failed. Error falls as λ grows: 7.6e-2 (0.01), 3.2e-2 (0.1), 1.4e-2 (10) |
| [x] | S4 Collocation N_f ∈ {500, 2000} | 20 | Nischal | 0 failed. 500 and 2000 points give the same error (~2.1e-2); 500 is 2× faster |
| [x] | S5 Extrapolation (train on t ≤ 10 s) | 30 | Nischal | Physics on t ≤ 20 s: 1.8e-2, error stays flat past 10 s. Physics only on t ≤ 10 s: 0.26 (fine inside the window, diverges after). Data-only: 0.51 (1 failed). Figure 4 |
| [x] | S6 Identification × noise | 240 | Nischal | Done Nov 7, 0 errors. Clean: PINN 0.5% vs two-stage 13–15%; 5% noise: PINN ~15–17% vs two-stage 43–92%; 15–30% noise: both fail (>80%). Re-run Nov 7 on stable pairs only (first run's pairs 0, 1, 4, 9 lost synchronism; archived in `results/archive_unstable_pairs/`) |

## Phase 5: Analysis (Nov 19 – Dec 2)

| Status | Task | Owner | Notes |
|---|---|---|---|
| [x] | Statistics: bootstrap CIs, Wilcoxon signed-rank tests | Nischal | Final run Nov 14: `results/stats_paired_tests.{csv,md}` |
| [x] | Table 1: replication, ours vs. the paper | Nischal | `results/table1_replication.md` (`summarize_replication.py`) |
| [x] | Fig. 1: error vs N_u and noise (both layouts) | Nischal | `figures/fig1_error_vs_data_noise.png` |
| [x] | Fig. 2: error over time beyond t = 10 s | Nischal | `figures/fig2_extrapolation.png`; curves in `results/S5_extrapolation_error_by_time.csv` |
| [x] | Table 2: identification, PINN vs. two-stage across noise | Nischal | `results/table2_identification.md` (median over all runs, matches the slides) |
| [x] | Table 3: design choices (λ, N_f), error, residual, time | Nischal | `results/table3_design_choices.md` |

## Phase 6: Presentation and repo (Dec 2 – Dec 8)

| Status | Task | Owner | Notes |
|---|---|---|---|
| [~] | Overview animation, scene 1 (rotor swing) | Nischal | `scripts/animate_overview.py` |
| [x] | Animation scenes 2 (with vs. without physics, extrapolation) and 3 (identifying m, d) | Nischal | Full recordings Nov 14; scene 2: data-only 0.79 vs PINN 0.019 after 10 s; scene 3: PINN 0.6%/1.2% vs two-stage 13%/16% |
| [ ] | Animation scene 4 (results summary) | | Needs S1/S2 |
| [~] | Slides | Nischal | PowerPoint v1 built Nov 10: 12 slides + backup, speaker notes, videos embedded (local presentation/build_deck.py, rebuilds from results). Rebuilt Nov 14: S2, final animations, the five numbered results as Fig. 1–2 and native Tables 1–3, AI disclosure line on the takeaways slide; QA and validate.py passed. Next: rehearse with a timer |
| [x] | README final: problem, methods, results summary, data, how to reproduce, limitations | Nischal | Nov 14 |
| [x] | `scripts/reproduce.sh` (fast path from CSVs tested; `--full` re-runs everything) | Nischal | Nov 14 |

---

## Decisions log

| Date | Decision |
|---|---|
| Oct 8 | Use PyTorch float64 on CPU, L-BFGS, the authors' settings recovered from their code |
| Oct 18 | Noise levels 0, 5, 15, 30% (of target std); ~820 runs total, ~450 PINN |
| Oct 18 | Extrapolation trains on t ∈ [0, 10] s (one full swing period ≈ 9 s) |
| Oct 18 | Keep the 50k L-BFGS cap (runs stop on their own at ~32k iterations) |
| Oct 18 | Failure rule adds "stuck": physics-trained run whose final residual is > 100× its setting's median (catches cases like pair 2) |
| Oct 18 | Identification sweep uses 3 seeds per pair: 240 runs; total 960 runs |
| Oct 18 | Identification: 10 (m, d) pairs by Latin hypercube (seed 0); authors' init m = 0, d = −6; residual at the 100 data points |
| Nov 7 | Identification pairs must be stable: all 40 trajectories stay below δ = π. Pairs that lose synchronism are a different regime and inflate the noise scale. Effective d range becomes 0.085–0.148 (report as a limitation) |
| Nov 7 | Sweeps run one at a time (`--workers 1 --min-free-mem 25`) to keep the laptop's RAM safe; queue S6 → S1 → S2, estimated to finish ~Nov 11–13 |
| Nov 10 | Presentation: 18-min slot = 15 min talk + 3 min Q&A, strictly enforced; no poster. 12 slides planned at 13:40 with ~3.5 min per speaker; the ~30 s of animation fits |
| Nov 14 | Course README (github.com/ci4ds/piml): exactly five figures and tables combined, in both the slides and the repo; repo due Dec 8 11:59 PM; AI use must be disclosed in submitted work. Chosen five: Table 1 replication, Fig. 1 data size × noise, Fig. 2 extrapolation, Table 2 identification, Table 3 design choices. Animations are explanatory, not counted |
| Nov 10 | Figure 3 shows median and interquartile range over all 10 seeds, failed runs included (mean over non-failed runs hid the data-only overfitting) |
| Nov 7 | Identification replication row = S6 clean-data runs (10 pairs × 3 seeds) |

## Open questions

- Sweeps end Nov 18 (Kaiyue's suggestion, 2 weeks for analysis before Dec 2) or Nov 22 (proposal)?

