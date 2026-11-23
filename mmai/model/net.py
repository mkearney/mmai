"""Antisymmetric matchup network.

margin(A, B, ctx) = g(A, B, ctx) - g(B, A, swap(ctx)) + w . (xA - xB)
sigma(A, B, ctx)  = symmetric function (same for both orderings)
P(A wins)         = Phi(margin / sigma)

The pair head sees [eA, eB, eA*eB, ctx], so style interactions (e.g. A's offensive
rebounding vs B's defensive rebounding) can move the prediction away from what the two
teams' one-dimensional ratings imply.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# context features; index 0 must be `loc` (antisymmetric: flipped when teams swap)
CTX = ["loc", "neutral", "is_ncaa", "is_conftourney", "is_post", "same_conf", "day"]


class MatchupNet(nn.Module):
    def __init__(self, d_team: int, d_ctx: int = len(CTX), h: int = 64, e: int = 24, dropout: float = 0.10):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_team, h), nn.GELU(approximate="tanh"), nn.Dropout(dropout),
                                 nn.Linear(h, e), nn.GELU(approximate="tanh"))
        self.pair = nn.Sequential(nn.Linear(3 * e + d_ctx, h), nn.GELU(approximate="tanh"), nn.Dropout(dropout),
                                  nn.Linear(h, 1))
        self.skip = nn.Linear(d_team, 1, bias=False)
        self.hca = nn.Parameter(torch.tensor(3.0))
        self.sig = nn.Sequential(nn.Linear(2 * e + d_ctx - 1, 16), nn.GELU(approximate="tanh"), nn.Linear(16, 1))

    @staticmethod
    def swap_ctx(ctx):
        c = ctx.clone()
        c[:, 0] = -c[:, 0]
        return c

    def forward(self, xa, xb, ctx):
        ea, eb = self.enc(xa), self.enc(xb)
        ctx_s = self.swap_ctx(ctx)
        g_ab = self.pair(torch.cat([ea, eb, ea * eb, ctx], 1)).squeeze(1)
        g_ba = self.pair(torch.cat([eb, ea, ea * eb, ctx_s], 1)).squeeze(1)
        mu = g_ab - g_ba + self.skip(xa - xb).squeeze(1) + self.hca * ctx[:, 0]
        sym = torch.cat([ea + eb, ea * eb, ctx[:, 1:]], 1)
        sigma = 6.0 + F.softplus(self.sig(sym).squeeze(1) + 4.0)
        return mu, sigma


def gaussian_nll(mu, sigma, y, w=None):
    nll = torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2
    if w is not None:
        return (nll * w).sum() / w.sum()
    return nll.mean()
