"""Fully connected tanh network (P, t) -> delta, with optional trainable swing parameters m and d."""
import torch
from torch import nn

DTYPE = torch.float64


class SwingNet(nn.Module):
    def __init__(self, layers, lb, ub, m_init=None, d_init=None):
        """layers: e.g. [2, 10, 10, 10, 10, 10, 1]. lb, ub: (P, t) bounds used to scale inputs to [-1, 1].
        Passing m_init and d_init makes m and d trainable (inverse problem)."""
        super().__init__()
        self.linears = nn.ModuleList(nn.Linear(a, b, dtype=DTYPE) for a, b in zip(layers[:-1], layers[1:]))
        for lin in self.linears:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.register_buffer("lb", torch.as_tensor(lb, dtype=DTYPE))
        self.register_buffer("ub", torch.as_tensor(ub, dtype=DTYPE))
        self.inverse = m_init is not None
        if self.inverse:
            self.m = nn.Parameter(torch.tensor(float(m_init), dtype=DTYPE))
            self.d = nn.Parameter(torch.tensor(float(d_init), dtype=DTYPE))

    def forward(self, P, t):
        h = 2 * (torch.cat([P, t], dim=1) - self.lb) / (self.ub - self.lb) - 1
        for lin in self.linears[:-1]:
            h = torch.tanh(lin(h))
        return self.linears[-1](h)
