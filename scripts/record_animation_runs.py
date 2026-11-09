"""Record what the networks predict during training, for animation scenes 2 and 3.

Scene 2 (forward, extrapolation setup of sweep S5): data-only vs. PINN, measurements only on t <= 10 s,
  physics on t <= 20 s. Predictions on three test trajectories at log-spaced training steps.
Scene 3 (identification, pair 1, clean data): the PINN's m and d at every loss evaluation, plus the
  two-stage baseline's final estimate on the same measurements.

Runs one training at a time. Full budget: ~20 min (PINN) + ~1 min + ~4 min + ~1 min.
  python scripts/record_animation_runs.py                   ->  data/animation/*.npz
  python scripts/record_animation_runs.py --max_iter 50     (quick smoke test)
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pinn_swing.data import make_forward_data
from pinn_swing.models import DTYPE
import run_forward
import run_identification

OUT = ROOT / "data" / "animation"
FORWARD_SEED = 7          # typical S5 seed: data-only 0.58, PINN 0.022 (rel. L2, test set)
ID_PAIR, ID_SEED = 1, 0   # typical S6 case: PINN 0.6% / 1.2%, two-stage 13% / 16% (m / d)
STEPS = set(np.unique(np.geomspace(1, 70000, 180).astype(int)).tolist())


def record_forward(lam, max_iter):
    args = run_forward.parse_args(["--seed", str(FORWARD_SEED), "--lam", str(lam), "--t_train_max", "10",
                                   "--colloc_t_max", "20"] + (["--max_iter", str(max_iter)] if max_iter else []))
    frames, steps, grid = [], [], {}

    def snap(model, n_eval):
        if n_eval in STEPS:
            with torch.no_grad():
                frames.append(model(grid["P"], grid["t"]).numpy().reshape(grid["shape"]))
            steps.append(n_eval)

    # the test trajectories are known before training (split_seed=0), so build the grid up front
    data = make_forward_data(yaml.safe_load(open(args.config)), split_seed=0)
    idx = data.test_idx[np.linspace(0, len(data.test_idx) - 1, 3).round().astype(int)]
    PP, TT = np.meshgrid(data.P[idx], data.t, indexing="ij")
    grid.update(P=torch.as_tensor(PP.reshape(-1, 1), dtype=DTYPE), t=torch.as_tensor(TT.reshape(-1, 1), dtype=DTYPE),
                shape=PP.shape)
    r = run_forward.run(args, snapshot_fn=snap)
    with torch.no_grad():  # always keep the final model
        frames.append(r["model"](grid["P"], grid["t"]).numpy().reshape(grid["shape"]))
    steps.append(r["train"]["n_eval"])
    path = OUT / f"scene2_lam{lam:g}.npz"
    np.savez_compressed(path, steps=np.array(steps), pred=np.array(frames), t=data.t, P=data.P[idx],
                        delta=data.delta[idx], X_u=r["X_u"], rel_l2_test=r["test"]["rel_l2_delta"],
                        rel_l2_extrap=r["test"]["rel_l2_delta_extrap"])
    print(f"wrote {path}: {len(steps)} frames, test rel. L2 {r['test']['rel_l2_delta']:.3g}")


def record_identification(method, max_iter):
    args = run_identification.parse_args(["--pair", str(ID_PAIR), "--seed", str(ID_SEED), "--method", method]
                                         + (["--max_iter", str(max_iter)] if max_iter else []))
    trace = []

    def snap(model, n_eval):
        if model.inverse:
            trace.append((n_eval, model.m.item(), model.d.item()))

    r = run_identification.run(args, snapshot_fn=snap)
    path = OUT / f"scene3_{method}.npz"
    np.savez_compressed(path, trace=np.array(trace).reshape(-1, 3), m_true=r["config"]["m_true"],
                        d_true=r["config"]["d_true"], m_hat=r["m_hat"], d_hat=r["d_hat"],
                        err_m_pct=r["err_m_pct"], err_d_pct=r["err_d_pct"], n_eval=r["train"]["n_eval"])
    print(f"wrote {path}: m {r['err_m_pct']:.2f}% d {r['err_d_pct']:.2f}% error")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max_iter", type=int, help="cap for a quick test (default: the configs' 50000)")
    a = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for lam in (0.0, 1.0):
        record_forward(lam, a.max_iter)
    for method in ("pinn", "two_stage"):
        record_identification(method, a.max_iter)


if __name__ == "__main__":
    main()
