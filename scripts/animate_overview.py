"""Overview animation for the presentation.

Scene 1: after a disturbance, the rotor angle swings before settling, and lower inertia swings faster.
Run from the repo root:  python scripts/animate_overview.py   ->  figures/overview_scene1.mp4
"""
from pathlib import Path
import sys

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pinn_swing.sim import simulate

plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 14,
                     "animation.ffmpeg_path": imageio_ffmpeg.get_ffmpeg_exe()})

OUT = Path(__file__).resolve().parents[1] / "figures" / "overview_scene1.mp4"
FPS, SECONDS, HOLD = 30, 7, 1.5
INK, QUIET, GRID = "#1a1a1a", "#6b6b6b", "#e3e3e3"
CASES = [  # (label, m, colour)
    ("Normal inertia (m = 0.4)", 0.4, "#1f4e79"),
    ("Low inertia (m = 0.15)", 0.15, "#d55e00"),
]
P, D, B = 0.15, 0.15, 0.2
DELTA_EQ = np.arcsin(P / B)

t = np.linspace(0, 20, FPS * SECONDS + 1)
traj = [simulate(m, D, P, t)[0] for _, m, _ in CASES]

fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
fig.text(0.05, 0.92, "After a disturbance, the generator rotor swings before settling",
         fontsize=22, weight="bold", color=INK)
fig.text(0.05, 0.875, "Single machine connected to an infinite bus, swing equation simulated with SciPy",
         fontsize=13, color=QUIET)
caption = fig.text(0.05, 0.05, "Lower inertia means faster swings, and grids with more renewables have less inertia.",
                   fontsize=16, color=INK, alpha=0)

# Left: two rotor dials, one per inertia case
dials = []
for i, (label, m, col) in enumerate(CASES):
    ax = fig.add_axes([0.04, 0.47 - 0.37 * i, 0.26, 0.34])
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.2, 1.35); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, lw=2, color=GRID))
    ax.plot([0, 1.08], [0, 0], ls="--", lw=1.2, color=QUIET)                      # infinite-bus reference
    ax.plot([0, 1.08 * np.cos(DELTA_EQ)], [0, 1.08 * np.sin(DELTA_EQ)], ls=":", lw=1.5, color=QUIET)
    ax.text(1.12, -0.08, "bus", fontsize=10, color=QUIET, va="top")
    ax.text(1.12 * np.cos(DELTA_EQ), 1.12 * np.sin(DELTA_EQ) + 0.05, "new equilibrium", fontsize=10, color=QUIET)
    arrow, = ax.plot([], [], lw=5, color=col, solid_capstyle="round")
    ax.plot(0, 0, "o", ms=10, color=col)
    ax.text(0, 1.22, label, ha="center", fontsize=13, color=col, weight="bold")
    angle_txt = ax.text(0, -0.25, "", ha="center", va="top", fontsize=12, color=INK)
    dials.append((arrow, angle_txt))

# Right: rotor angle over time
axr = fig.add_axes([0.40, 0.17, 0.56, 0.62])
axr.set_xlim(0, 20); axr.set_ylim(0, 1.25)
axr.set_xlabel("time (s)"); axr.set_ylabel("rotor angle δ (rad)")
axr.axhline(DELTA_EQ, ls=":", color=QUIET, lw=1.2)
axr.text(0.3, DELTA_EQ + 0.03, "equilibrium", ha="left", fontsize=11, color=QUIET)
for s in ("top", "right"):
    axr.spines[s].set_visible(False)
axr.grid(color=GRID, lw=0.8); axr.set_axisbelow(True)
lines = [axr.plot([], [], lw=2.8, color=col, label=label)[0] for label, _, col in CASES]
dots = [axr.plot([], [], "o", ms=8, color=col)[0] for _, _, col in CASES]
axr.legend(loc="lower right", frameon=False, fontsize=12)

n_anim, n_hold = len(t), int(HOLD * FPS)


def draw(frame):
    k = min(frame, n_anim - 1)
    for (arrow, angle_txt), y, line, dot in zip(dials, traj, lines, dots):
        arrow.set_data([0, np.cos(y[k])], [0, np.sin(y[k])])
        angle_txt.set_text(f"δ = {np.degrees(y[k]):5.1f}°")
        line.set_data(t[:k + 1], y[:k + 1])
        dot.set_data([t[k]], [y[k]])
    caption.set_alpha(np.clip((frame - 0.55 * n_anim) / (0.15 * n_anim), 0, 1))


OUT.parent.mkdir(exist_ok=True)
writer = FFMpegWriter(fps=FPS, bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
with writer.saving(fig, str(OUT), dpi=100):
    for f in range(n_anim + n_hold):
        draw(f)
        writer.grab_frame()
print("wrote", OUT)
