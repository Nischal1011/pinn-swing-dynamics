"""Turn results/raw/<name>/*.json into one row per run and one row per setting (mean, 95% bootstrap CI).

  python scripts/aggregate.py S1_size_noise_boundary          ->  results/S1_size_noise_boundary_runs.csv
                                                                  results/S1_size_noise_boundary_summary.csv
A run counts as failed if it crashed or produced NaN, if its relative L2 error of delta is above 1, or
(for runs trained with the physics loss) if its final physics residual MSE_f is more than RESIDUAL_FACTOR
times the median of its setting: the optimizer got stuck even though the error may look moderate.
Failed runs are kept in the run table and counted in the summary; statistics use the successful runs.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESIDUAL_FACTOR = 100
FORWARD_KEYS = ["lam", "N_u", "N_f", "noise", "regime", "t_train_max", "colloc_t_max"]
IDENT_KEYS = ["pair", "method", "noise", "m_true", "d_true"]
FORWARD_METRICS = ["rel_l2_delta", "rel_l2_omega_autodiff", "rel_l2_omega_fd", "residual_rms",
                   "rel_l2_delta_extrap", "rel_l2_delta_interp"]
IDENT_METRICS = ["err_m_pct", "err_d_pct", "rel_l2_delta"]


def flatten(r):
    c, t = r["config"], r["train"]
    row = {"seed": c["seed"], "crashed": bool(t["failed"]), "mse_u": t["mse_u"], "mse_f": t.get("mse_f"),
           "n_iter": t["n_iter"], "train_time_s": t["train_time_s"]}
    if "test" in r:  # forward
        row.update({k: c.get(k) for k in FORWARD_KEYS})
        row.update({k: r["test"].get(k) for k in FORWARD_METRICS})
        row["physics_trained"] = (c.get("lam") or 0) > 0
    else:            # identification
        row.update({k: c.get(k) for k in IDENT_KEYS})
        row.update({k: r.get(k) for k in IDENT_METRICS})
        row["physics_trained"] = c["method"] == "pinn"
    return row


def flag_failures(df, keys):
    df = df.copy()
    bad = df["crashed"] | ~np.isfinite(df["rel_l2_delta"]) | (df["rel_l2_delta"] > 1)
    med = df.groupby(keys, dropna=False)["mse_f"].transform("median")
    stuck = df["physics_trained"] & (df["mse_f"] > RESIDUAL_FACTOR * med)
    df["failed"] = bad | stuck
    df["fail_reason"] = np.select([df["crashed"], df["rel_l2_delta"] > 1, stuck], ["crashed", "error>1", "stuck"], "")
    return df


def bootstrap_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return x[0], x[0]
    means = np.random.default_rng(seed).choice(x, (n_boot, len(x))).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def summarize(df, keys, metrics):
    rows = []
    for group, g in df.groupby(keys, dropna=False):
        ok = g[~g["failed"]]
        row = dict(zip(keys, group if isinstance(group, tuple) else (group,)))
        row.update({"n": len(g), "n_failed": int(g["failed"].sum()),
                    "train_time_s_mean": g["train_time_s"].mean()})
        for m in metrics:
            if g[m].notna().any():
                lo, hi = bootstrap_ci(ok[m])
                row.update({f"{m}_mean": ok[m].mean(), f"{m}_ci_lo": lo, f"{m}_ci_hi": hi,
                            f"{m}_median_all": g[m].median()})
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="experiment name, e.g. S1_size_noise_boundary")
    p.add_argument("--raw-root", default=str(ROOT / "results" / "raw"))
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    a = p.parse_args(argv)

    files = sorted(glob.glob(str(Path(a.raw_root) / a.name / "*.json")))
    if not files:
        raise SystemExit(f"no results in {Path(a.raw_root) / a.name}")
    df = pd.DataFrame([flatten(json.load(open(f))) for f in files])
    ident = "method" in df.columns
    keys = [k for k in (IDENT_KEYS if ident else FORWARD_KEYS) if k in df.columns]
    metrics = [m for m in (IDENT_METRICS if ident else FORWARD_METRICS) if m in df.columns]
    df = flag_failures(df, keys)
    summary = summarize(df, keys, metrics)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.sort_values(keys + ["seed"]).to_csv(out / f"{a.name}_runs.csv", index=False)
    summary.to_csv(out / f"{a.name}_summary.csv", index=False)
    print(f"{a.name}: {len(df)} runs, {int(df['failed'].sum())} failed "
          f"({df['fail_reason'].value_counts().to_dict()}), {len(summary)} settings")
    print(summary.drop(columns=[c for c in summary.columns if c.endswith(("_ci_lo", "_ci_hi"))]).to_string(index=False))


if __name__ == "__main__":
    main()
