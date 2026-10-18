"""Tests for data sampling, the physics residual and the training loop."""
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from pinn_swing.data import (add_noise, candidate_points, collocation_points, make_forward_data,
                             sample_measurements, split_trajectories)
from pinn_swing.eval import evaluate_forward
from pinn_swing.models import DTYPE, SwingNet
from pinn_swing.physics import residual
from pinn_swing.train import train

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs" / "forward.yaml"))
PHYS = {k: CFG["system"][k] for k in ("m", "d", "B", "V1", "V2")}
LB, UB = np.array([0.08, 0.0]), np.array([0.18, 20.0])


@pytest.fixture(scope="module")
def data():
    return make_forward_data(CFG)


def test_split_is_a_partition_with_edges_in_train():
    tr, va, te = split_trajectories(100, seed=3)
    assert len(np.intersect1d(tr, va)) == len(np.intersect1d(tr, te)) == len(np.intersect1d(va, te)) == 0
    assert len(tr) + len(va) + len(te) == 100
    assert {0, 99} <= set(tr) and (len(va), len(te)) == (20, 20)


def test_boundary_points_lie_on_t0_or_edge_trajectories(data):
    X, u = sample_measurements(data, 40, "boundary", np.random.default_rng(0))
    on_ic = X[:, 1] == 0
    on_edge = np.isclose(X[:, 0], data.P[0]) | np.isclose(X[:, 0], data.P[-1])
    assert X.shape == (40, 2) and u.shape == (40, 1)
    assert np.all(on_ic | on_edge)


def test_interior_points_come_from_training_trajectories_only(data):
    pairs = candidate_points(data, "interior")
    assert set(np.unique(pairs[:, 0])) == set(data.train_idx)


def test_t_max_restricts_measurement_times(data):
    X, _ = sample_measurements(data, 100, "interior", np.random.default_rng(0), t_max=10)
    assert X[:, 1].max() <= 10 + 1e-9


def test_noise_has_requested_scale():
    u = np.zeros((200000, 1))
    noisy = add_noise(u, 0.15, 2.0, np.random.default_rng(0))
    assert noisy.std() == pytest.approx(0.30, rel=0.01)


def test_collocation_inside_box_and_counts():
    X = collocation_points(500, LB, [0.18, 10.0], seed=0, extra=np.zeros((7, 2)))
    assert X.shape == (507, 2)
    assert X[:500, 0].min() >= 0.08 and X[:500, 0].max() <= 0.18 and X[:500, 1].max() <= 10


class _Analytic(torch.nn.Module):
    """delta(P, t) = sin(2 t) + P, so delta' = 2 cos(2 t) and delta'' = -4 sin(2 t)."""
    inverse = False

    def forward(self, P, t):
        return torch.sin(2 * t) + P


def test_residual_matches_hand_derivation():
    P = torch.tensor([[0.1], [0.15]], dtype=DTYPE)
    t = torch.tensor([[0.3], [1.7]], dtype=DTYPE)
    m, d, B = 0.4, 0.15, 0.2
    f = residual(_Analytic(), P, t, m, d, B)
    delta = torch.sin(2 * t) + P
    expected = m * (-4 * torch.sin(2 * t)) + d * (2 * torch.cos(2 * t)) + B * torch.sin(delta) - P
    assert torch.allclose(f, expected, atol=1e-12)


def test_residual_is_small_on_simulated_solution(data):
    """Finite-difference check that the simulated data satisfy the swing equation (sign convention)."""
    i, dt = 50, data.t[1] - data.t[0]
    delta = data.delta[i]
    d1 = np.gradient(delta, dt)
    d2 = np.gradient(d1, dt)
    f = PHYS["m"] * d2 + PHYS["d"] * d1 + PHYS["B"] * np.sin(delta) - data.P[i]
    assert np.abs(f[5:-5]).max() < 5e-3


def test_training_reduces_loss_for_both_models(data):
    rng = np.random.default_rng(0)
    X_u, u = sample_measurements(data, 40, "boundary", rng)
    X_f = collocation_points(300, LB, UB, seed=0)
    for lam in (0.0, 1.0):
        torch.manual_seed(0)
        model = SwingNet([2, 8, 8, 1], LB, UB)
        before = ((model(torch.as_tensor(X_u[:, :1]), torch.as_tensor(X_u[:, 1:])) - torch.as_tensor(u)) ** 2).mean()
        info = train(model, X_u, u, X_f, lam, PHYS, max_iter=60)
        assert not info["failed"] and info["mse_u"] < before.item()
        metrics = evaluate_forward(model, data, PHYS, idx=data.test_idx, t_split=10)
        assert {"rel_l2_delta", "rel_l2_delta_extrap", "residual_rms"} <= set(metrics)


def test_inverse_model_trains_m_and_d(data):
    rng = np.random.default_rng(1)
    X_u, u = sample_measurements(data, 100, "interior", rng)
    torch.manual_seed(0)
    model = SwingNet([2, 8, 8, 1], LB, UB, m_init=0.2, d_init=0.1)
    info = train(model, X_u, u, X_u, 1.0, PHYS, max_iter=30)
    assert {"m_hat", "d_hat"} <= set(info)
    assert (info["m_hat"], info["d_hat"]) != (0.2, 0.1)
