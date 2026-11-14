"""Animation scenes 2 and 3 for the presentation, from the recordings of scripts/record_animation_runs.py.

Scene 2: data-only vs. PINN learning the swing from 40 measurements on t <= 10 s, shown on unseen test
         trajectories over the full 20 s.
Scene 3: the PINN identifies m and d during training; the two-stage baseline's answer for comparison.

  python scripts/animate_training.py   ->  figures/overview_scene2.mp4, figures/overview_scene3.mp4
"""
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "data" / "animation"
FIG = ROOT / "figures"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 14,
                     "animation.ffmpeg_path": imageio_ffmpeg.get_ffmpeg_exe()})
FPS = 30
INK, QUIET, GRID, SHADE = "#1a1a1a", "#6b6b6b", "#e3e3e3", "#f0efed"
PINN, DATA = "#2a78d6", "#eb6834"      # same colours as the result figures


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def header(fig, title, subtitle):
    fig.text(0.05, 0.92, title, fontsize=22, weight="bold", color=INK)
    fig.text(0.05, 0.875, subtitle, fontsize=13, color=QUIET)


def save(fig, draw, n_frames, out):
    writer = FFMpegWriter(fps=FPS, bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(out), dpi=100):
        for f in range(n_frames):
            draw(f)
            writer.grab_frame()
    plt.close(fig)
    print("wrote", out)


def latest(steps, s):
    """Index of the last snapshot taken at or before training step s."""
    return max(int(np.searchsorted(steps, s, side="right")) - 1, 0)


def scene2(seconds=8, hold=2.5):
    runs = {"data": np.load(REC / "scene2_lam0.npz"), "pinn": np.load(REC / "scene2_lam1.npz")}
    t, delta = runs["pinn"]["t"], runs["pinn"]["delta"]
    last = max(int(r["steps"][-1]) for r in runs.values())
    n_anim, n_hold = int(seconds * FPS), int(hold * FPS)
    step_at = np.geomspace(1, last, n_anim)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
    header(fig, "Same network, same 40 measurements: with and without physics",
           "Measurements only on t ≤ 10 s. Curves: three test trajectories the networks never saw.")
    step_txt = fig.text(0.95, 0.875, "", fontsize=13, color=QUIET, ha="right")
    caption = fig.text(0.05, 0.05, "", fontsize=16, color=INK, alpha=0)
    lo, hi = delta.min() - 0.15, delta.max() + 0.15
    panels = {}
    for i, (key, label, col) in enumerate([("data", "Data only (λ = 0)", DATA),
                                           ("pinn", "PINN (λ = 1, physics on 0–20 s)", PINN)]):
        ax = fig.add_axes([0.06 + 0.47 * i, 0.17, 0.42, 0.6])
        ax.set_xlim(0, 20); ax.set_ylim(lo, hi)
        ax.axvspan(10, 20, color=SHADE, zorder=0)
        ax.axvline(10, color=QUIET, ls=":", lw=1.2)
        ax.text(10.3, lo + 0.04, "no measurements", fontsize=11, color=QUIET, va="bottom")
        ax.set_xlabel("time (s)")
        if i == 0:
            ax.set_ylabel("rotor angle δ (rad)")
        style(ax)
        for y in delta:
            ax.plot(t, y, color=INK, lw=1.2, alpha=0.35)
        lines = [ax.plot([], [], color=col, lw=2.6)[0] for _ in delta]
        ax.set_title(label, color=col, weight="bold", fontsize=15, loc="left")
        panels[key] = lines
    fig.legend([plt.Line2D([], [], color=INK, alpha=0.35, lw=1.2)], ["true trajectory (simulated)"],
               loc="upper right", bbox_to_anchor=(0.95, 0.86), frameon=False, fontsize=12)
    e = {k: float(r["rel_l2_extrap"]) for k, r in runs.items()}
    caption.set_text(f"Error after 10 s (relative L2): data only {e['data']:.2f}, PINN {e['pinn']:.3f}. "
                     "The physics term carries the solution forward.")

    def draw(f):
        s = step_at[min(f, n_anim - 1)]
        for key, lines in panels.items():
            r = runs[key]
            pred = r["pred"][latest(r["steps"], s)]
            for line, y in zip(lines, pred):
                line.set_data(t, y)
        step_txt.set_text(f"training step {int(s):,}")
        caption.set_alpha(np.clip((f - n_anim) / (0.6 * FPS), 0, 1))

    save(fig, draw, n_anim + n_hold, FIG / "overview_scene2.mp4")


def rolling_median(x, w):
    pad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, w), axis=1)


def scene3(seconds=7, hold=3):
    p, b = np.load(REC / "scene3_pinn.npz"), np.load(REC / "scene3_two_stage.npz")
    steps, m_tr, d_tr = p["trace"].T
    m_true, d_true = float(p["m_true"]), float(p["d_true"])
    err_m = np.abs(m_tr - m_true) / m_true * 100
    err_d = np.abs(d_tr - d_true) / d_true * 100
    # the trace has one point per loss evaluation, including L-BFGS line-search trials; a rolling median
    # over 25 evaluations shows the trend instead of the trial-to-trial jitter
    err_m, err_d, m_tr, d_tr = (rolling_median(x, 25) for x in (err_m, err_d, m_tr, d_tr))
    n_anim, n_hold = int(seconds * FPS), int(hold * FPS)
    step_at = np.geomspace(1, steps[-1], n_anim)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
    header(fig, "Learning the machine's inertia m and damping d from 100 measurements",
           f"m and d are trainable PINN parameters, started at m = 0, d = −6. True m = {m_true:.3f}, d = {d_true:.3f}.")
    ax = fig.add_axes([0.08, 0.17, 0.56, 0.6])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, steps[-1] * 1.5); ax.set_ylim(0.01, 1e4)
    ax.set_xlabel("training step"); ax.set_ylabel("error of the estimate (%)")
    style(ax)
    lm, = ax.plot([], [], color=PINN, lw=2.6, label="PINN: m")
    ld, = ax.plot([], [], color=PINN, lw=2.6, ls="--", label="PINN: d")
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    base = [ax.axhline(v, color=DATA, lw=1.6, ls=ls, alpha=0) for v, ls in
            ((float(b["err_m_pct"]), "-"), (float(b["err_d_pct"]), "--"))]
    base_txt = ax.text(steps[-1] * 1.3, max(float(b["err_m_pct"]), float(b["err_d_pct"])) * 1.3,
                       "two-stage baseline\n(fit, differentiate, least squares)",
                       color=DATA, fontsize=12, alpha=0, ha="right", va="bottom")

    readout = fig.add_axes([0.69, 0.17, 0.27, 0.6]); readout.axis("off")
    rows = [readout.text(0, y, "", fontsize=17, color=c, va="top", family="Helvetica Neue")
            for y, c in ((0.95, INK), (0.8, PINN), (0.68, PINN), (0.45, DATA), (0.33, DATA))]
    rows[0].set_text("Estimates")
    caption = fig.text(0.05, 0.05, f"PINN: {float(p['err_m_pct']):.1f}% / {float(p['err_d_pct']):.1f}% error. "
                       f"Two-stage: {float(b['err_m_pct']):.0f}% / {float(b['err_d_pct']):.0f}%. "
                       "Fitting data and physics together beats differentiating a fit.",
                       fontsize=16, color=INK, alpha=0)

    def draw(f):
        k = latest(steps, step_at[min(f, n_anim - 1)])
        lm.set_data(steps[:k + 1], np.maximum(err_m[:k + 1], 1e-3))
        ld.set_data(steps[:k + 1], np.maximum(err_d[:k + 1], 1e-3))
        rows[1].set_text(f"PINN  m = {m_tr[k]:.3f}")
        rows[2].set_text(f"PINN  d = {d_tr[k]:.3f}")
        a = np.clip((f - n_anim) / (0.6 * FPS), 0, 1)
        for h in base:
            h.set_alpha(a)
        base_txt.set_alpha(a)
        if a > 0:
            rows[3].set_text(f"two-stage  m = {float(b['m_hat']):.3f}")
            rows[4].set_text(f"two-stage  d = {float(b['d_hat']):.3f}")
        caption.set_alpha(np.clip((f - n_anim - FPS) / (0.6 * FPS), 0, 1))

    save(fig, draw, n_anim + n_hold, FIG / "overview_scene3.mp4")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    scene2()
    scene3()
