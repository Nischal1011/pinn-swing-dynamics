"""Swing-equation residual and time derivatives of the network output, by automatic differentiation."""
import torch


def derivatives(model, P, t, order=2):
    """Return delta and its time derivatives up to `order` at points (P, t). Builds a graph for training."""
    t = t.detach().clone().requires_grad_(True)
    delta = model(P, t)
    out = [delta]
    for _ in range(order):
        out.append(torch.autograd.grad(out[-1].sum(), t, create_graph=True)[0])
    return out


def residual(model, P, t, m, d, B=0.2, V1=1.0, V2=1.0):
    """f = m * delta'' + d * delta' + B V1 V2 sin(delta) - P   (zero for an exact solution)."""
    delta, delta_t, delta_tt = derivatives(model, P, t, order=2)
    return m * delta_tt + d * delta_t + B * V1 * V2 * torch.sin(delta) - P
