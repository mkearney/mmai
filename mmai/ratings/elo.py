"""Margin-of-victory Elo across all seasons (point-in-time pre-game values)."""
from __future__ import annotations

import numpy as np
import pandas as pd

K = 28.0
HCA = 75.0           # Elo points for a true home game
CARRY = 0.72         # season-to-season retention (rest regresses to conference mean)
NEW_TEAM = 1300.0


def run(games: pd.DataFrame, k: float = K, hca: float = HCA, carry: float = CARRY):
    """Return (pregame elo per game [elo_a, elo_b], final ratings dict, history DataFrame)."""
    g = games[(games["type"] != "nond1")].sort_values(["date", "gid"]).reset_index()
    R: dict[str, float] = {}
    conf_of: dict[str, str] = {}
    last_season = None
    ea = np.full(len(g), np.nan)
    eb = np.full(len(g), np.nan)
    hist = []
    for i, r in enumerate(g.itertuples()):
        if r.season != last_season:
            if last_season is not None:
                # regress toward the mean of the team's peers (all teams) at season turn
                mean = np.mean(list(R.values())) if R else 1500.0
                R = {t: carry * v + (1 - carry) * mean for t, v in R.items()}
                hist += [(last_season, t, v) for t, v in R.items()]
            last_season = r.season
        a, b = r.team_a, r.team_b
        ra = R.get(a, NEW_TEAM)
        rb = R.get(b, NEW_TEAM)
        ea[i], eb[i] = ra, rb
        if not r.played or pd.isna(r.pts_a):
            continue
        diff = ra - rb + hca * r.loc
        p = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        m = r.pts_a - r.pts_b
        win = 1.0 if m > 0 else 0.0
        wdiff = diff if m > 0 else -diff
        mult = ((abs(m) + 3.0) ** 0.8) / (7.5 + 0.006 * wdiff)
        delta = k * mult * (win - p)
        R[a] = ra + delta
        R[b] = rb - delta
    out = pd.DataFrame({"gid": g["gid"], "season": g["season"], "elo_a": ea, "elo_b": eb})
    return out, R


def elo_prob(diff):
    return 1.0 / (1.0 + 10 ** (-np.asarray(diff) / 400.0))
