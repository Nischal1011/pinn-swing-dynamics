"""Train one forward-problem model (PINN or data-only) and write its metrics to JSON.

Examples (from the repo root):
  python scripts/run_forward.py --lam 1 --seed 0                     # PINN, faithful settings
  python scripts/run_forward.py --lam 0 --seed 0                     # data-only baseline
  python scripts/run_forward.py --lam 1 --N_u 10 --noise 0.15 --regime interior
  python scripts/run_forward.py --lam 1 --t_train_max 10 --colloc_t_max 20   # extrapolation
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
from pinn_swing.data import add_noise, collocation_points, make_forward_data, sample_measurements
from pinn_swing.eval import evaluate_forward
from pinn_swing.models import SwingNet
from pinn_swing.train import train


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "configs" / "forward.yaml"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lam", type=float, help="physics weight (0 = data-only); default from config")
    p.add_argument("--N_u", type=int)
    p.add_argument("--N_f", type=int)
    p.add_argument("--noise", type=float, help="noise level as a fraction of target std, e.g. 0.05")
    p.add_argument("--regime", choices=["boundary", "interior"])
    p.add_argument("--max_iter", type=int)
    p.add_argument("--t_train_max", type=float, help="only measure t <= this (extrapolation sweep)")
    p.add_argument("--colloc_t_max", type=float, help="collocation covers t <= this")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", help="output JSON path (default: print only)")
    return p.parse_args(argv)


def run(args):
    cfg = yaml.safe_load(open(args.config))
    s, dom = cfg["system"], cfg["domain"]
    run_cfg = {
        "seed": args.seed,
        "lam": cfg["train"]["physics_weight"] if args.lam is None else args.lam,
        "N_u": args.N_u or cfg["data"]["N_u"],
        "N_f": args.N_f or cfg["collocation"]["N_f"],
        "noise": cfg["data"]["noise"] if args.noise is None else args.noise,
        "regime": args.regime or cfg["data"]["sampling"],
        "max_iter": args.max_iter or cfg["train"]["max_iter"],
        "t_train_max": args.t_train_max,
        "colloc_t_max": args.colloc_t_max,
        "layers": cfg["model"]["layers"],
    }
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = make_forward_data(cfg, split_seed=0)  # same test trajectories for every run
    X_u, u = sample_measurements(data, run_cfg["N_u"], run_cfg["regime"], rng, t_max=run_cfg["t_train_max"])
    ref_std = data.delta[data.train_idx].std()
    u_noisy = add_noise(u, run_cfg["noise"], ref_std, rng) if run_cfg["noise"] > 0 else u

    lb = np.array([dom["P_range"][0], dom["t_range"][0]])
    ub = np.array([dom["P_range"][1], dom["t_range"][1]])
    ub_f = ub.copy()
    if run_cfg["colloc_t_max"] is not None:
        ub_f[1] = run_cfg["colloc_t_max"]
    X_f = collocation_points(run_cfg["N_f"], lb, ub_f, seed=args.seed,
                             extra=X_u if cfg["collocation"].get("include_training_points") else None)

    phys = {k: s[k] for k in ("m", "d", "B", "V1", "V2")}
    model = SwingNet(run_cfg["layers"], lb, ub)
    info = train(model, X_u, u_noisy, X_f, run_cfg["lam"], phys, max_iter=run_cfg["max_iter"])

    t_split = run_cfg["t_train_max"]
    result = {"config": run_cfg, "train": info,
              "test": evaluate_forward(model, data, phys, idx=data.test_idx, t_split=t_split),
              "full_grid": evaluate_forward(model, data, phys, t_split=t_split)}
    return result


def main(argv=None):
    args = parse_args(argv)
    result = run(args)
    summary = {k: result[k] for k in ("test", "full_grid")}
    print(json.dumps({"config": result["config"], "n_iter": result["train"]["n_iter"],
                      "train_time_s": round(result["train"]["train_time_s"], 1),
                      "failed": result["train"]["failed"], **summary}, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
