"""Evaluation on whole trajectories: relative L2 errors for delta and omega, extrapolation error, residual."""
import numpy as np
import torch

from .models import DTYPE
from .physics import derivatives


def rel_l2(pred, true):
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def predict_grid(model, P, t, phys):
    """Predict delta, omega (autodiff) and the swing residual on the full P x t grid."""
    PP, TT = np.meshgrid(P, t, indexing="ij")
    P_ = torch.as_tensor(PP.reshape(-1, 1), dtype=DTYPE)
    t_ = torch.as_tensor(TT.reshape(-1, 1), dtype=DTYPE)
    delta, delta_t, delta_tt = derivatives(model, P_, t_, order=2)
    m, d = (model.m, model.d) if model.inverse else (phys["m"], phys["d"])
    f = m * delta_tt + d * delta_t + phys["B"] * phys["V1"] * phys["V2"] * torch.sin(delta) - P_
    shape = PP.shape
    return (delta.detach().numpy().reshape(shape), delta_t.detach().numpy().reshape(shape),
            f.detach().numpy().reshape(shape))


def evaluate_forward(model, data, phys, idx=None, t_split=None):
    """Metrics on the trajectories `idx` (all trajectories if None).

    omega is scored two ways: autodiff of the network, and central finite differences of the predicted
    delta on the 0.1 s grid (what the authors' code does). With t_split, also report the error on t > t_split."""
    idx = np.arange(len(data.P)) if idx is None else np.asarray(idx)
    delta_hat, omega_ad, f = predict_grid(model, data.P[idx], data.t, phys)
    delta, omega = data.delta[idx], data.omega[idx]
    omega_fd = np.gradient(delta_hat, data.t, axis=1)
    out = {"rel_l2_delta": rel_l2(delta_hat, delta),
           "rel_l2_omega_autodiff": rel_l2(omega_ad, omega),
           "rel_l2_omega_fd": rel_l2(omega_fd, omega),
           "residual_rms": float(np.sqrt(np.mean(f ** 2)))}
    if t_split is not None:
        late = data.t > t_split + 1e-9
        out["rel_l2_delta_extrap"] = rel_l2(delta_hat[:, late], delta[:, late])
        out["rel_l2_delta_interp"] = rel_l2(delta_hat[:, ~late], delta[:, ~late])
    return out
