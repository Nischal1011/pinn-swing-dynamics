"""Identify m and d for one (m, d) pair with the PINN or the two-stage baseline, and write JSON.

Examples (from the repo root):
  python scripts/run_identification.py --pair 0 --method pinn
  python scripts/run_identification.py --pair 0 --method two_stage --noise 0.15
  python scripts/run_identification.py --released            # the authors' released pair (m=0.25, d=0.15)
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pinn_swing.eval import rel_l2
from pinn_swing.identification import (fit_pinn, fit_two_stage, make_pair_data, make_pairs, predict_delta,
                                       sample_grid)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "configs" / "identification.yaml"))
    p.add_argument("--pair", type=int, default=0, help="index into the 10 generated pairs")
    p.add_argument("--released", action="store_true", help="use m=0.25, d=0.15 (the authors' released pair)")
    p.add_argument("--method", choices=["pinn", "two_stage"], default="pinn")
    p.add_argument("--noise", type=float, default=0.0, help="fraction of target std, e.g. 0.05")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N_u", type=int)
    p.add_argument("--max_iter", type=int)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out")
    return p.parse_args(argv)


def run(args):
    cfg = yaml.safe_load(open(args.config))
    m_true, d_true = (0.25, 0.15) if args.released else make_pairs(cfg)[args.pair]
    N_u = args.N_u or cfg["data"]["N_u"]
    max_iter = args.max_iter or cfg["train"]["max_iter"]
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    P, t, delta = make_pair_data(cfg, m_true, d_true)
    X_u, u = sample_grid(P, t, delta, N_u, rng)
    if args.noise > 0:
        u = u + args.noise * delta.std() * rng.standard_normal(u.shape)

    fit = fit_pinn if args.method == "pinn" else fit_two_stage
    model, m_hat, d_hat, info = fit(cfg, X_u, u, max_iter)
    info.pop("history", None)
    return {
        "config": {"pair": None if args.released else args.pair, "m_true": float(m_true), "d_true": float(d_true),
                   "method": args.method, "noise": args.noise, "seed": args.seed, "N_u": N_u, "max_iter": max_iter},
        "m_hat": m_hat, "d_hat": d_hat,
        "err_m_pct": abs(m_hat - m_true) / m_true * 100, "err_d_pct": abs(d_hat - d_true) / d_true * 100,
        "rel_l2_delta": rel_l2(predict_delta(model, P, t), delta),
        "train": info,
    }


def main(argv=None):
    args = parse_args(argv)
    r = run(args)
    print(json.dumps({k: r[k] for k in ("config", "m_hat", "d_hat", "err_m_pct", "err_d_pct", "rel_l2_delta")}
                     | {"train_time_s": round(r["train"]["train_time_s"], 1), "n_iter": r["train"]["n_iter"],
                        "failed": r["train"]["failed"]}, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(r, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
