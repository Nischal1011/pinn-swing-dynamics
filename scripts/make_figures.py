"""Build the result figures and tables from the committed CSVs in results/.

The project reports five results (Table 1, the replication, comes from scripts/summarize_replication.py):
  python scripts/make_figures.py      ->  figures/fig1_error_vs_data_noise.png   (sweeps S1, S2)
                                          figures/fig2_extrapolation.png         (S5)
                                          results/table2_identification.md       (S6)
                                          results/table3_design_choices.md       (S3, S4, and S1 for the defaults)
Fig. 2 needs per-time errors, which only the raw run files hold; when results/raw/S5_extrapolation/ exists they
are condensed into the committed results/S5_extrapolation_error_by_time.csv, which the figure then reads.
Results whose sweeps have not run yet are skipped with a message.
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


def fig1():
    """Median and interquartile range over ALL seeds, failed runs included: dropping runs with error > 1 would
    hide exactly the cases where data-only training overfits the noise."""
    rows = []
    for regime, name in (("boundary", "S1_size_noise_boundary"), ("interior", "S2_size_noise_interior")):
        f = RES / f"{name}_runs.csv"
        if f.exists():
            rows.append((regime, pd.read_csv(f)))
    if not rows:
        return print("fig1: no S1/S2 results yet, skipped")
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
    fig.suptitle("Fig. 1. Physics loss vs. data alone as data shrinks and noise grows",
                 x=0.01, y=0.995, ha="left", fontsize=13, color=INK, weight="bold")
    fig.text(0.01, 0.92 if len(rows) == 1 else 0.955,
             "Median and interquartile range over 10 seeds, all runs included. "
             "k/10 = runs with relative error above 1 (dotted line).", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if len(rows) == 1 else 0.94))
    out = FIG / "fig1_error_vs_data_noise.png"
    fig.savefig(out)
    print("wrote", out)


EXTRAP_VARIANTS = [("Data-only", lambda c: c["lam"] == 0, ORANGE, "--"),
                   ("PINN, physics only on t ≤ 10 s", lambda c: c["lam"] > 0 and c["colloc_t_max"] == 10, AQUA, "-."),
                   ("PINN, physics on t ≤ 20 s", lambda c: c["lam"] > 0 and c["colloc_t_max"] == 20, BLUE, "-")]


def extrapolation_curves():
    """Mean and 10-90% band of the RMS error of delta at each time, per variant (non-failed runs)."""
    out = RES / "S5_extrapolation_error_by_time.csv"
    files = sorted(glob.glob(str(RES / "raw" / "S5_extrapolation" / "*.json")))
    runs = [json.load(open(f)) for f in files]
    runs = [r for r in runs if "rms_err_delta_by_time" in r["test"] and not r["train"]["failed"]]
    if runs:
        t = np.linspace(0, 20, len(runs[0]["test"]["rms_err_delta_by_time"]))
        parts = []
        for label, sel, _, _ in EXTRAP_VARIANTS:
            E = np.array([r["test"]["rms_err_delta_by_time"] for r in runs if sel(r["config"])])
            if len(E):
                parts.append(pd.DataFrame({"variant": label, "t": t, "mean": E.mean(axis=0), "n": len(E),
                                           "p10": np.percentile(E, 10, axis=0), "p90": np.percentile(E, 90, axis=0)}))
        pd.concat(parts).to_csv(out, index=False, float_format="%.6g")
    return pd.read_csv(out) if out.exists() else None


def fig2():
    curves = extrapolation_curves()
    if curves is None:
        return print("fig2: no S5 results with per-time errors yet, skipped")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    style(ax)
    ax.axvspan(10, 20, color=GRID, alpha=0.5, lw=0)
    ax.axvline(10, color=MUTED, lw=1.2, ls=":")
    ax.text(10.2, 0.97, "no measurements beyond here", transform=ax.get_xaxis_transform(), color=MUTED,
            fontsize=10, va="top")
    for label, _, col, ls in EXTRAP_VARIANTS:
        c = curves[curves["variant"] == label]
        if c.empty:
            continue
        ax.fill_between(c["t"], c["p10"], c["p90"], color=col, alpha=0.15, lw=0)
        ax.plot(c["t"], c["mean"], color=col, lw=2, ls=ls, label=f"{label} (n = {int(c['n'].iloc[0])})")
    ax.set_yscale("log")
    ax.set_xlim(0, 20)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("RMS error of δ over test trajectories (rad)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Fig. 2. Extrapolation beyond the training window (mean, 10–90% band over seeds)", loc="left",
                 color=INK, fontsize=13, weight="bold")
    fig.tight_layout()
    out = FIG / "fig2_extrapolation.png"
    fig.savefig(out)
    print("wrote", out)


def table2():
    """Identification error per method and noise: median and interquartile range over all 30 runs
    (10 pairs x 3 seeds), failed runs included, so it matches results/stats_paired_tests.md."""
    f = RES / "S6_identification_runs.csv"
    if not f.exists():
        return print("table2: no S6 results yet, skipped")
    df = pd.read_csv(f)
    q = lambda x: f"{x.median():.1f} [{x.quantile(0.25):.1f}, {x.quantile(0.75):.1f}]"
    lines = ["## Table 2. Identifying inertia m and damping d: PINN vs. two-stage baseline", "",
             "Relative error of the estimates, median [interquartile range] over 10 (m, d) pairs x 3 seeds, "
             "100 measurements each. Two-stage: fit a data-only network, differentiate it, solve least squares.", "",
             "| noise | method | error m (%) | error d (%) | failed / runs | time per run (s) |",
             "|---|---|---|---|---|---|"]
    for nz, g in df.groupby("noise"):
        for meth, name in (("pinn", "PINN"), ("two_stage", "Two-stage")):
            r = g[g["method"] == meth]
            if len(r):
                lines.append(f"| {nz * 100:.0f}% | {name} | {q(r['err_m_pct'])} | {q(r['err_d_pct'])} "
                             f"| {int(r['failed'].sum())} / {len(r)} | {r['train_time_s'].mean():.0f} |")
    out = RES / "table2_identification.md"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


def table3():
    """Design choices around the default PINN (lambda = 1, N_f = 8000, N_u = 40, no noise, boundary sampling)."""
    s1, s3, s4 = read("S1_size_noise_boundary"), read("S3_physics_weight"), read("S4_collocation")
    base = None if s1 is None else s1[(s1["N_u"] == 40) & (s1["noise"] == 0)]
    ci = lambda r: f"{r['rel_l2_delta_mean']:.2e} [{r['rel_l2_delta_ci_lo']:.1e}, {r['rel_l2_delta_ci_hi']:.1e}]"
    lines = ["## Table 3. Design choices: physics weight λ and collocation points N_f", "",
             "N_u = 40 clean measurements, boundary sampling, 10 seeds per row. Error: test relative L2 of δ, "
             "mean [95% bootstrap CI]. Defaults (λ = 1, N_f = 8000) come from sweep S1.", "",
             "| setting | value | rel. L2 error of δ | residual RMS | failed / runs | time per run (s) |",
             "|---|---|---|---|---|---|"]
    if base is not None or s3 is not None:
        rows = pd.concat([d for d in (base, s3) if d is not None]).sort_values("lam")
        for _, r in rows.iterrows():
            lines.append(f"| λ | {r['lam']:g} | {ci(r)} | {r['residual_rms_mean']:.1e} | {int(r['n_failed'])} / "
                         f"{int(r['n'])} | {r['train_time_s_mean']:.0f} |")
    nf = [d for d in (s4, None if base is None else base[base["lam"] == 1]) if d is not None]
    if nf:
        for _, r in pd.concat(nf).sort_values("N_f").iterrows():
            lines.append(f"| N_f (λ = 1) | {int(r['N_f'])} | {ci(r)} | {r['residual_rms_mean']:.1e} | "
                         f"{int(r['n_failed'])} / {int(r['n'])} | {r['train_time_s_mean']:.0f} |")
    if len(lines) == 6:
        return print("table3: no S1/S3/S4 results yet, skipped")
    out = RES / "table3_design_choices.md"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig1()
    fig2()
    table2()
    table3()
