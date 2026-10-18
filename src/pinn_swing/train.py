"""L-BFGS training with loss = MSE_u + lambda * MSE_f.

The PINN (lambda > 0) and the data-only baseline (lambda = 0) go through exactly the same code path,
optimizer settings and iteration cap. Only the physics term is switched off.
"""
import time

import numpy as np
import torch

from .models import DTYPE
from .physics import residual


def _t(a):
    return torch.as_tensor(np.asarray(a), dtype=DTYPE)


def train(model, X_u, u, X_f, lam, phys, max_iter=50000, history_every=50, snapshot_fn=None, snapshot_every=None):
    """Train `model` in place.

    X_u (N_u, 2), u (N_u, 1): measurements. X_f (N_f, 2): collocation points (unused when lam == 0).
    phys: dict with B, V1, V2 and, for the forward problem, the known m and d.
    snapshot_fn(model, n_eval): optional callback, e.g. to store predictions for the animation.
    Returns a dict with final losses, iteration counts, wall time, history and a `failed` flag."""
    P_u, t_u, u = _t(X_u[:, :1]), _t(X_u[:, 1:]), _t(u)
    P_f, t_f = _t(X_f[:, :1]), _t(X_f[:, 1:])
    B, V1, V2 = phys["B"], phys["V1"], phys["V2"]

    def m_d():
        return (model.m, model.d) if model.inverse else (phys["m"], phys["d"])

    def losses(with_physics):
        mse_u = ((model(P_u, t_u) - u) ** 2).mean()
        mse_f = ((residual(model, P_f, t_f, *m_d(), B, V1, V2)) ** 2).mean() if with_physics else None
        return mse_u, mse_f

    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=max_iter, max_eval=int(1.25 * max_iter),
                            history_size=50, tolerance_grad=1e-9, tolerance_change=1e-15,
                            line_search_fn="strong_wolfe")
    history, n_eval = [], [0]

    def closure():
        opt.zero_grad()
        mse_u, mse_f = losses(lam > 0)
        loss = mse_u + lam * mse_f if lam > 0 else mse_u
        loss.backward()
        n_eval[0] += 1
        if n_eval[0] % history_every == 1:
            history.append((n_eval[0], loss.item(), mse_u.item(), None if mse_f is None else mse_f.item()))
        if snapshot_fn is not None and snapshot_every and n_eval[0] % snapshot_every == 1:
            snapshot_fn(model, n_eval[0])
        return loss

    start = time.perf_counter()
    failed = False
    try:
        opt.step(closure)
    except RuntimeError:  # e.g. NaN in the line search
        failed = True
    elapsed = time.perf_counter() - start

    mse_u, mse_f = losses(True)  # report the physics residual for both models
    mse_u, mse_f = mse_u.item(), mse_f.item()
    failed = failed or not np.isfinite(mse_u) or not np.isfinite(mse_f)
    state = opt.state[next(iter(opt.state))] if opt.state else {}
    out = {"mse_u": mse_u, "mse_f": mse_f, "n_iter": int(state.get("n_iter", 0)), "n_eval": n_eval[0],
           "train_time_s": elapsed, "history": history, "failed": failed}
    if model.inverse:
        out["m_hat"], out["d_hat"] = model.m.item(), model.d.item()
    return out
