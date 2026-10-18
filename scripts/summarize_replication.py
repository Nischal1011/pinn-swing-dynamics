"""Collect results/replication/*.json into the replication table (ours vs. the paper).

Writes results/replication_table.md and results/replication_table.csv.
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
    return f"mean {xs.mean():.2f}%, median {np.median(xs):.2f}%; {miss} of {len(xs)} pairs above 10%"


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
for meth, name in (("pinn", "PINN"), ("two_stage", "Two-stage")):
    rs = [r for r in load(f"ident_pair*_{meth}.json") if not r["train"]["failed"]]
    if rs:
        for p, key in (("m", "err_m_pct"), ("d", "err_d_pct")):
            rows.append({
                "experiment": f"Identification, {name}",
                "metric": f"error in {p} over {len(rs)} pairs (seed 0)",
                "ours": pct([r[key] for r in rs]),
                "paper": {"m": "0.74%", "d": "1.28%"}[p] if meth == "pinn" else "not reported",
                "train_time": f"{np.mean([r['train']['train_time_s'] for r in rs]):.0f} s",
            })

out_md, out_csv = ROOT / "results" / "replication_table.md", ROOT / "results" / "replication_table.csv"
cols = ["experiment", "metric", "ours", "paper", "train_time"]
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, cols)
    w.writeheader()
    w.writerows(rows)
lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
out_md.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
