"""Inverse problem: recover m and d from measured trajectories.

Two methods on the same data and network size:
  pinn       m and d are trainable and fitted jointly with the network (loss = MSE_u + MSE_f at the data points)
  two_stage  fit a data-only network first, differentiate it, then solve the swing equation for (m, d)
             by linear least squares at the same points
"""
import numpy as np
import torch
from scipy.stats import qmc

from .models import DTYPE, SwingNet
from .physics import derivatives
from .sim import simulate_grid
from .train import train


def make_pairs(cfg):
    """The (m, d) pairs, drawn by Latin hypercube with a fixed seed. Shape (n_pairs, 2)."""
    pc = cfg["pairs"]
    u = qmc.LatinHypercube(d=2, seed=pc["seed"]).random(pc["n_pairs"])
    lo = np.array([pc["m_range"][0], pc["d_range"][0]])
    hi = np.array([pc["m_range"][1], pc["d_range"][1]])
    return lo + (hi - lo) * u


def make_pair_data(cfg, m, d):
    """Simulate one pair's grid. Returns P (n_P,), t (n_t,), delta (n_P, n_t)."""
    s, dom = cfg["system"], cfg["domain"]
    P = np.linspace(*dom["P_range"], cfg["pairs"]["trajectories_per_pair"])
    t0, t1 = dom["t_range"]
    t = np.linspace(t0, t1, int(round((t1 - t0) / dom["dt"])) + 1)
    delta, _ = simulate_grid(m, d, P, t, delta0=s["delta0"], omega0=s["omega0"],
                             B=s["B"], V1=s["V1"], V2=s["V2"], **cfg.get("solver", {}))
    return P, t, delta


def sample_grid(P, t, delta, N_u, rng):
    """N_u random points from the full grid, as in the authors' identification code."""
    flat = rng.choice(delta.size, N_u, replace=False)
    i, j = np.unravel_index(flat, delta.shape)
    return np.c_[P[i], t[j]], delta[i, j][:, None]


def bounds(cfg):
    dom = cfg["domain"]
    lb = np.array([dom.get("scaling_lb_P", dom["P_range"][0]), dom["t_range"][0]])
    ub = np.array([dom["P_range"][1], dom["t_range"][1]])
    return lb, ub


def fit_pinn(cfg, X_u, u, max_iter):
    s = cfg["system"]
    lb, ub = bounds(cfg)
    init = cfg["params_init"]
    model = SwingNet(cfg["model"]["layers"], lb, ub, m_init=init["m"], d_init=init["d"])
    info = train(model, X_u, u, X_u, 1.0, {k: s[k] for k in ("B", "V1", "V2")}, max_iter=max_iter)
    return model, info["m_hat"], info["d_hat"], info


def fit_two_stage(cfg, X_u, u, max_iter):
    s = cfg["system"]
    lb, ub = bounds(cfg)
    model = SwingNet(cfg["model"]["layers"], lb, ub)
    info = train(model, X_u, u, X_u, 0.0, {k: s[k] for k in ("B", "V1", "V2")}, max_iter=max_iter)
    P = torch.as_tensor(X_u[:, :1], dtype=DTYPE)
    t = torch.as_tensor(X_u[:, 1:], dtype=DTYPE)
    delta, delta_t, delta_tt = (x.detach().numpy().ravel() for x in derivatives(model, P, t, order=2))
    # m * delta'' + d * delta' = P - B V1 V2 sin(delta)
    A = np.c_[delta_tt, delta_t]
    b = X_u[:, 0] - s["B"] * s["V1"] * s["V2"] * np.sin(delta)
    (m_hat, d_hat), *_ = np.linalg.lstsq(A, b, rcond=None)
    return model, float(m_hat), float(d_hat), info


def predict_delta(model, P, t):
    PP, TT = np.meshgrid(P, t, indexing="ij")
    with torch.no_grad():
        out = model(torch.as_tensor(PP.reshape(-1, 1), dtype=DTYPE), torch.as_tensor(TT.reshape(-1, 1), dtype=DTYPE))
    return out.numpy().reshape(PP.shape)
