"""SMIB swing-equation simulator (reference solver / ground truth).

    m * delta'' + d * delta' + B * V1 * V2 * sin(delta) - P = 0      (Misyris et al., eq. 2)

Matches the paper's released MATLAB ode45 data to ~2e-4 relative L2
(the residual is ode45's default tolerance, not ours).
"""
import numpy as np
from scipy.integrate import solve_ivp


def swing_rhs(t, y, m, d, P, B=0.2, V1=1.0, V2=1.0):
    delta, omega = y
    return [omega, (P - d * omega - B * V1 * V2 * np.sin(delta)) / m]


def simulate(m, d, P, t_eval, delta0=0.1, omega0=0.1, B=0.2, V1=1.0, V2=1.0,
             method="DOP853", rtol=1e-9, atol=1e-12):
    """Return (delta, omega) sampled at t_eval for one trajectory."""
    sol = solve_ivp(swing_rhs, (t_eval[0], t_eval[-1]), [delta0, omega0],
                    t_eval=t_eval, method=method, rtol=rtol, atol=atol,
                    args=(m, d, P, B, V1, V2))
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[0], sol.y[1]


def simulate_grid(m, d, P_values, t_eval, **kw):
    """Simulate one trajectory per P. Returns arrays of shape (len(P), len(t))."""
    out = [simulate(m, d, P, t_eval, **kw) for P in P_values]
    delta = np.stack([o[0] for o in out])
    omega = np.stack([o[1] for o in out])
    return delta, omega


def energy(delta, omega, m, P, B=0.2, V1=1.0, V2=1.0):
    """Lyapunov-style energy; non-increasing in time when d > 0."""
    return 0.5 * m * omega**2 - P * delta - B * V1 * V2 * np.cos(delta)
