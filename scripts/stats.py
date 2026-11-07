"""Paired significance tests: PINN vs. data-only (S1, S2) and PINN vs. two-stage (S6).

Runs that share a seed (and, for S6, a pair) see the same measurements and noise, so they are compared
pair by pair with a two-sided Wilcoxon signed-rank test on the error. Failed runs are kept: the comparison
is about what a practitioner would get from one training run.

  python scripts/stats.py    ->  results/stats_paired_tests.csv and .md
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def paired(df, setting_keys, pair_keys, arm_col, arm_a, arm_b, metric):
    rows = []
    for setting, g in df.groupby(setting_keys, dropna=False):
        a = g[g[arm_col] == arm_a].set_index(pair_keys)[metric]
        b = g[g[arm_col] == arm_b].set_index(pair_keys)[metric]
        common = a.index.intersection(b.index)
        if len(common) < 5:
            continue
        a, b = a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)
        diff = a - b
        p = wilcoxon(a, b).pvalue if np.any(diff != 0) else 1.0
        row = dict(zip(setting_keys, setting if isinstance(setting, tuple) else (setting,)))
        row.update({"metric": metric, "n_pairs": len(common), f"median_{arm_a}": np.median(a),
                    f"median_{arm_b}": np.median(b), f"frac_{arm_a}_better": float(np.mean(a < b)),
                    "median_ratio": float(np.median(b / a)), "p_wilcoxon": p})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    out = []
    for name in ("S1_size_noise_boundary", "S2_size_noise_interior"):
        f = RES / f"{name}_runs.csv"
        if not f.exists():
            print(f"{name}: not aggregated yet, skipped")
            continue
        df = pd.read_csv(f)
        df["arm"] = np.where(df["lam"] > 0, "pinn", "data_only")
        t = paired(df, ["N_u", "noise"], ["seed"], "arm", "pinn", "data_only", "rel_l2_delta")
        t.insert(0, "sweep", name)
        out.append(t)
    f = RES / "S6_identification_runs.csv"
    if f.exists():
        df = pd.read_csv(f)
        for metric in ("err_m_pct", "err_d_pct"):
            t = paired(df, ["noise"], ["pair", "seed"], "method", "pinn", "two_stage", metric)
            t.insert(0, "sweep", "S6_identification")
            out.append(t)
    if not out:
        return print("nothing to test yet")
    res = pd.concat(out, ignore_index=True)
    res.to_csv(RES / "stats_paired_tests.csv", index=False)
    fmt = res.copy()
    for c in fmt.columns:
        if fmt[c].dtype == float:
            fmt[c] = fmt[c].map(lambda v: f"{v:.3g}")
    lines = ["| " + " | ".join(fmt.columns) + " |", "|" + "---|" * len(fmt.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in fmt.itertuples(index=False)]
    (RES / "stats_paired_tests.md").write_text(
        "Paired two-sided Wilcoxon signed-rank tests (same seed = same data). `median_ratio` = median of "
        "(baseline error / PINN error); above 1 means the PINN is better.\n\n" + "\n".join(lines) + "\n")
    print(fmt.to_string(index=False))


if __name__ == "__main__":
    main()
