"""Build the replication table (ours vs. the paper): forward runs in results/replication/, identification
from the clean-data runs of the S6 sweep (results/S6_identification_runs.csv).

Table 1 of the project. Writes results/table1_replication.md and results/table1_replication.csv.
"""
import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "replication"


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(str(RES / pattern)))]


def mean_sd(xs):
    xs = np.asarray(xs, dtype=float)
    return f"{xs.mean():.2e} ± {xs.std(ddof=1):.1e}" if len(xs) > 1 else f"{xs.mean():.2e}"


def pct(xs):
    xs = np.asarray(xs, dtype=float)
    miss = int((xs > 10).sum())
    return f"mean {xs.mean():.2f}%, median {np.median(xs):.2f}%; {miss} of {len(xs)} runs above 10%"


rows = []
for lam, name in ((1, "PINN"), (0, "Data-only")):
    rs = [r for r in load(f"forward_lam{lam}_seed*.json") if not r["train"]["failed"]]
    if rs:
        rows.append({
            "experiment": f"Forward, {name}",
            "metric": "rel. L2 of δ, full grid",
            "ours": mean_sd([r["full_grid"]["rel_l2_delta"] for r in rs]) + f" (n={len(rs)} seeds)",
            "paper": "1.34e-2" if lam == 1 else "not reported",
            "train_time": f"{np.mean([r['train']['train_time_s'] for r in rs]):.0f} s",
        })
S6 = ROOT / "results" / "S6_identification_runs.csv"
if S6.exists():  # identification replication = clean-data runs of the S6 sweep (10 stable pairs x 3 seeds)
    import pandas as pd
    s6 = pd.read_csv(S6)
    s6 = s6[s6["noise"] == 0]
    for meth, name in (("pinn", "PINN"), ("two_stage", "Two-stage")):
        g = s6[s6["method"] == meth]
        if g.empty:
            continue
        for p, key in (("m", "err_m_pct"), ("d", "err_d_pct")):
            rows.append({
                "experiment": f"Identification, {name}",
                "metric": f"error in {p}, {g['pair'].nunique()} pairs x {g['seed'].nunique()} seeds, clean data",
                "ours": pct(g[key]) + f"; {int(g['failed'].sum())} flagged failed",
                "paper": {"m": "0.74%", "d": "1.28%"}[p] if meth == "pinn" else "not reported",
                "train_time": f"{g['train_time_s'].mean():.0f} s",
            })

out_md, out_csv = ROOT / "results" / "table1_replication.md", ROOT / "results" / "table1_replication.csv"
cols = ["experiment", "metric", "ours", "paper", "train_time"]
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, cols)
    w.writeheader()
    w.writerows(rows)
lines = ["## Table 1. Replication of Misyris et al. (2019): ours vs. the paper", "",
         "Forward: 40 boundary measurements, 8000 collocation points, L-BFGS; mean ± sd over seeds. "
         "Identification: clean-data runs of sweep S6.", "",
         "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
out_md.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
