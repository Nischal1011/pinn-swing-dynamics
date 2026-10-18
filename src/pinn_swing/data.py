"""Datasets for the forward problem: simulated grid, trajectory split, measurement sampling, noise, collocation.

Arrays are indexed [trajectory (one per P), time]. Training inputs are (P, t) pairs, as in the paper.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from .sim import simulate_grid


@dataclass
class ForwardData:
    P: np.ndarray          # (n_P,)
    t: np.ndarray          # (n_t,)
    delta: np.ndarray      # (n_P, n_t)
    omega: np.ndarray      # (n_P, n_t)
    train_idx: np.ndarray  # trajectory indices
    val_idx: np.ndarray
    test_idx: np.ndarray


def make_forward_data(cfg, split_seed=0):
    """Simulate the forward grid from a config dict (configs/forward.yaml) and split trajectories."""
    s, dom = cfg["system"], cfg["domain"]
    P = np.linspace(*dom["P_range"], dom["n_P"])
    t0, t1 = dom["t_range"]
    t = np.linspace(t0, t1, int(round((t1 - t0) / dom["dt"])) + 1)
    delta, omega = simulate_grid(s["m"], s["d"], P, t, delta0=s["delta0"], omega0=s["omega0"],
                                 B=s["B"], V1=s["V1"], V2=s["V2"], **cfg.get("solver", {}))
    tr, va, te = split_trajectories(len(P), seed=split_seed)
    return ForwardData(P, t, delta, omega, tr, va, te)


def split_trajectories(n, fracs=(0.6, 0.2, 0.2), seed=0, always_train=None):
    """Split whole trajectories into train/val/test. The two edge trajectories always go to train,
    because boundary sampling observes them."""
    always_train = np.array([0, n - 1] if always_train is None else always_train)
    rest = np.setdiff1d(np.arange(n), always_train)
    rest = np.random.default_rng(seed).permutation(rest)
    n_val, n_test = round(fracs[1] * n), round(fracs[2] * n)
    val, test = rest[:n_val], rest[n_val:n_val + n_test]
    train = np.concatenate([always_train, rest[n_val + n_test:]])
    return np.sort(train), np.sort(val), np.sort(test)


def candidate_points(data, regime, t_max=None):
    """(i, j) index pairs that may be measured.

    boundary: the initial line t = 0 for every P (the initial condition is known by construction)
              plus the two edge trajectories, as in the authors' code.
    interior: every time point of the training trajectories.
    t_max restricts measurements to t <= t_max (extrapolation sweep)."""
    n_P, n_t = data.delta.shape
    j_ok = np.arange(n_t) if t_max is None else np.flatnonzero(data.t <= t_max + 1e-9)
    if regime == "boundary":
        ic = [(i, 0) for i in range(n_P)]
        edges = [(i, j) for i in (0, n_P - 1) for j in j_ok if j > 0]
        pairs = ic + edges
    elif regime == "interior":
        pairs = [(i, j) for i in data.train_idx for j in j_ok]
    else:
        raise ValueError(f"unknown sampling regime {regime!r}")
    return np.array(pairs)


def sample_measurements(data, N_u, regime, rng, t_max=None):
    """Pick N_u measurement points. Returns X (N_u, 2) with columns (P, t) and clean targets u (N_u, 1)."""
    pairs = candidate_points(data, regime, t_max)
    pick = pairs[rng.choice(len(pairs), N_u, replace=False)]
    X = np.c_[data.P[pick[:, 0]], data.t[pick[:, 1]]]
    u = data.delta[pick[:, 0], pick[:, 1]][:, None]
    return X, u


def add_noise(u, level, ref_std, rng):
    """Zero-mean Gaussian noise with std = level * ref_std (level 0.05 means 5%)."""
    return u + level * ref_std * rng.standard_normal(u.shape)


def collocation_points(N_f, lb, ub, seed, extra=None):
    """Latin hypercube over the box [lb, ub] in (P, t). `extra` points (e.g. the measurement inputs)
    are appended, as in the authors' code."""
    X = np.asarray(lb) + (np.asarray(ub) - np.asarray(lb)) * qmc.LatinHypercube(d=2, seed=seed).random(N_f)
    return X if extra is None else np.vstack([X, extra])
