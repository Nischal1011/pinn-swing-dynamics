"""Build the result figures and tables from results/*_summary.csv and results/raw/.

  python scripts/make_figures.py      ->  figures/fig3_error_vs_data_noise.png
                                          figures/fig4_extrapolation.png
                                          results/table5_identification_and_ablations.md
Figures whose sweeps have not run yet are skipped with a message.
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
# Reference categorical palette, slots 1-3 in fixed order; every series also has its own marker/linestyle
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
SERIES = {1.0: ("PINN (λ = 1)", BLUE, "o", "-"), 0.0: ("Data-only (λ = 0)", ORANGE, "s", "--")}


def read(name):
    f = RES / f"{name}_summary.csv"
    return pd.read_csv(f) if f.exists() else None


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig3():
    """Median and interquartile range over ALL seeds, failed runs included: dropping runs with error > 1 would
    hide exactly the cases where data-only training overfits the noise."""
    rows = []
    for regime, name in (("boundary", "S1_size_noise_boundary"), ("interior", "S2_size_noise_interior")):
        f = RES / f"{name}_runs.csv"
        if f.exists():
            rows.append((regime, pd.read_csv(f)))
    if not rows:
        return print("fig3: no S1/S2 results yet, skipped")
    noises = [0.0, 0.05, 0.15, 0.30]
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.4 * len(rows) + 0.6), sharex=True, sharey=True,
                             squeeze=False)
    for i, (regime, df) in enumerate(rows):
        for j, nz in enumerate(noises):
            ax = axes[i, j]
            style(ax)
            ax.axhline(1, color=MUTED, lw=0.8, ls=":")
            for lam, (label, col, mk, ls) in SERIES.items():
                sel = df[(df["lam"] == lam) & np.isclose(df["noise"], nz)]
                if sel.empty:
                    continue
                g = sel.groupby("N_u")["rel_l2_delta"].agg(
                    med="median", lo=lambda x: x.quantile(0.25), hi=lambda x: x.quantile(0.75),
                    n="size", n_bad=lambda x: int((x > 1).sum())).reset_index()
                ax.fill_between(g["N_u"], g["lo"], g["hi"], color=col, alpha=0.15, lw=0)
                ax.plot(g["N_u"], g["med"], color=col, marker=mk, ms=7, ls=ls, lw=2, label=label,
                        markeredgecolor="white", markeredgewidth=1.2)
                for r in g[g["n_bad"] > 0].itertuples():
                    ax.annotate(f"{int(r.n_bad)}/{int(r.n)}", (r.N_u, r.hi), fontsize=8, color=col,
                                ha="center", xytext=(0, 5), textcoords="offset points")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([10, 20, 40, 100], ["10", "20", "40", "100"])
            ax.minorticks_off()
            if i == 0:
                ax.set_title(f"noise {int(nz * 100)}%", color=INK, fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{regime} sampling\nrel. L2 error of δ (test)")
            if i == len(rows) - 1:
                ax.set_xlabel("measurements N_u")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Physics loss vs. data alone as data shrinks and noise grows",
                 x=0.01, y=0.995, ha="left", fontsize=13, color=INK, weight="bold")
    fig.text(0.01, 0.92 if len(rows) == 1 else 0.955,
             "Median and interquartile range over 10 seeds, all runs included. "
             "k/10 = runs with relative error above 1 (dotted line).", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if len(rows) == 1 else 0.94))
    out = FIG / "fig3_error_vs_data_noise.png"
    fig.savefig(out)
    print("wrote", out)


def fig4():
    files = sorted(glob.glob(str(RES / "raw" / "S5_extrapolation" / "*.json")))
    runs = [json.load(open(f)) for f in files]
    runs = [r for r in runs if "rms_err_delta_by_time" in r["test"] and not r["train"]["failed"]]
    if not runs:
        return print("fig4: no S5 results with per-time errors yet, skipped")
    t = np.linspace(0, 20, len(runs[0]["test"]["rms_err_delta_by_time"]))
    variants = [("Data-only", lambda c: c["lam"] == 0, ORANGE, "--"),
                ("PINN, physics only on t ≤ 10 s", lambda c: c["lam"] > 0 and c["colloc_t_max"] == 10, AQUA, "-."),
                ("PINN, physics on t ≤ 20 s", lambda c: c["lam"] > 0 and c["colloc_t_max"] == 20, BLUE, "-")]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    style(ax)
    ax.axvspan(10, 20, color=GRID, alpha=0.5, lw=0)
    ax.axvline(10, color=MUTED, lw=1.2, ls=":")
    ax.text(10.2, 0.97, "no measurements beyond here", transform=ax.get_xaxis_transform(), color=MUTED,
            fontsize=10, va="top")
    for label, sel, col, ls in variants:
        E = np.array([r["test"]["rms_err_delta_by_time"] for r in runs if sel(r["config"])])
        if len(E) == 0:
            continue
        mean = E.mean(axis=0)
        ax.fill_between(t, np.percentile(E, 10, axis=0), np.percentile(E, 90, axis=0), color=col, alpha=0.15, lw=0)
        ax.plot(t, mean, color=col, lw=2, ls=ls, label=f"{label} (n = {len(E)})")
    ax.set_yscale("log")
    ax.set_xlim(0, 20)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("RMS error of δ over test trajectories (rad)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Extrapolation beyond the training window (mean, 10–90% band over seeds)", loc="left",
                 color=INK, fontsize=13, weight="bold")
    fig.tight_layout()
    out = FIG / "fig4_extrapolation.png"
    fig.savefig(out)
    print("wrote", out)


def table5():
    parts = []
    s6 = read("S6_identification")
    if s6 is not None:
        g = s6.groupby(["method", "noise"]).agg(
            err_m_median=("err_m_pct_median_all", "median"), err_d_median=("err_d_pct_median_all", "median"),
            failed=("n_failed", "sum"), runs=("n", "sum"), time_s=("train_time_s_mean", "mean")).reset_index()
        lines = ["### Identification of m and d (median over 10 pairs of each pair's median over 3 seeds)", "",
                 "| method | noise | error m (%) | error d (%) | failed / runs | time per run (s) |", "|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| {r['method']} | {r['noise'] * 100:.0f}% | {r['err_m_median']:.2f} | {r['err_d_median']:.2f} "
                         f"| {int(r['failed'])} / {int(r['runs'])} | {r['time_s']:.0f} |")
        parts.append("\n".join(lines))
    s1, s3, s4 = read("S1_size_noise_boundary"), read("S3_physics_weight"), read("S4_collocation")
    base = None if s1 is None else s1[(s1["N_u"] == 40) & (s1["noise"] == 0)]
    lam_rows = pd.concat([d for d in (base, s3) if d is not None]) if (base is not None or s3 is not None) else None
    if lam_rows is not None and len(lam_rows):
        lines = ["### Physics weight λ (N_u = 40, no noise, boundary sampling)", "",
                 "| λ | rel. L2 error of δ (mean [95% CI]) | residual RMS | failed / runs | time per run (s) |",
                 "|---|---|---|---|---|"]
        for _, r in lam_rows.sort_values("lam").iterrows():
            lines.append(f"| {r['lam']:g} | {r['rel_l2_delta_mean']:.2e} [{r['rel_l2_delta_ci_lo']:.1e}, "
                         f"{r['rel_l2_delta_ci_hi']:.1e}] | {r['residual_rms_mean']:.1e} | {int(r['n_failed'])} / {int(r['n'])} "
                         f"| {r['train_time_s_mean']:.0f} |")
        parts.append("\n".join(lines))
    nf_rows = []
    if s4 is not None:
        nf_rows.append(s4)
    if base is not None:
        nf_rows.append(base[base["lam"] == 1])
    if nf_rows:
        d = pd.concat(nf_rows).sort_values("N_f")
        lines = ["### Collocation points N_f (PINN, N_u = 40, no noise)", "",
                 "| N_f | rel. L2 error of δ (mean [95% CI]) | failed / runs | time per run (s) |", "|---|---|---|---|"]
        for _, r in d.iterrows():
            lines.append(f"| {int(r['N_f'])} | {r['rel_l2_delta_mean']:.2e} [{r['rel_l2_delta_ci_lo']:.1e}, "
                         f"{r['rel_l2_delta_ci_hi']:.1e}] | {int(r['n_failed'])} / {int(r['n'])} | {r['train_time_s_mean']:.0f} |")
        parts.append("\n".join(lines))
    if not parts:
        return print("table5: no S3/S4/S6 results yet, skipped")
    out = RES / "table5_identification_and_ablations.md"
    out.write_text("\n\n".join(parts) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig3()
    fig4()
    table5()
