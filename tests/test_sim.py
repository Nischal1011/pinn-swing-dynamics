"""Simulator sanity checks, plus agreement with the paper's released data."""
from pathlib import Path

import numpy as np
import scipy.io

from pinn_swing.sim import energy, simulate, simulate_grid

ROOT = Path(__file__).resolve().parents[1]
T = np.linspace(0, 20, 201)


def test_equilibrium_stays_put():
    P = 0.15
    delta_eq = np.arcsin(P / 0.2)
    delta, omega = simulate(0.4, 0.15, P, T, delta0=delta_eq, omega0=0.0)
    assert np.abs(delta - delta_eq).max() < 1e-9
    assert np.abs(omega).max() < 1e-9


def test_loose_and_tight_tolerances_agree():
    tight, _ = simulate(0.4, 0.15, 0.15, T)
    loose, _ = simulate(0.4, 0.15, 0.15, T, method="RK45", rtol=1e-6, atol=1e-9)
    assert np.abs(tight - loose).max() < 1e-5


def test_damped_system_loses_energy():
    delta, omega = simulate(0.4, 0.15, 0.15, T)
    E = energy(delta, omega, 0.4, 0.15)
    assert np.all(np.diff(E) <= 1e-10)
    assert E[-1] < E[0]


def test_matches_paper_released_data():
    for name, m, d in [("inference", 0.4, 0.15), ("identification", 0.25, 0.15)]:
        D = scipy.io.loadmat(ROOT / f"reference/misyris2019_original/Data/swingEquation_{name}.mat")
        delta, _ = simulate_grid(m, d, D["x"].ravel(), D["t"].ravel())
        rel = np.linalg.norm(delta - D["usol"]) / np.linalg.norm(D["usol"])
        assert rel < 1e-3, (name, rel)
