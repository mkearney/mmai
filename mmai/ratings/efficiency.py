"""Opponent-adjusted efficiency + style profiles (KenPom-style), point-in-time.

For every team-game "offense" observation:

    stat = mu + Off[team] + Def[opp] + hca * loc        (loc = +1 home, 0 neutral, -1 away)

solved by recency-weighted ridge regression where each team's Off/Def is pulled toward a
preseason prior (pseudo-observations worth ``prior_games`` games). The same design matrix
is reused for many stats (multi right-hand side), so we get adjusted offense/defense for
efficiency, tempo, eFG%, TO%, OR%, FT rate, 3PA rate, 3P%, 2P%, block%, steal%, assist%.
Snapshots are produced *before* each game date -> leak-free features for modelling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve

STATS = ["eff", "pace", "efg", "tov", "orb", "ftr", "tpar", "tp", "twop", "blk", "stl", "ast"]
GROUPS = [["eff", "pace"], ["efg", "tov", "orb", "ftr", "tpar", "tp", "twop", "blk", "stl", "ast"]]
HALF_LIFE_DAYS = 45.0
WEIGHT_FLOOR = 0.40


def team_game_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Two offense rows per game with per-game rate stats."""
    g = games
    rows = []
    for side, opp, s in (("a", "b", 1), ("b", "a", -1)):
        mins = g["minutes"].fillna(200.0).clip(lower=200.0)
        poss = g["poss"]
        fga = g[f"{side}_fga"]
        two_a = fga - g[f"{side}_tpa"]
        r = pd.DataFrame({
            "season": g["season"], "date": g["date"], "gid": g["gid"],
            "team": g[f"team_{side}"], "opp": g[f"team_{opp}"], "loc": g["loc"] * s,
            "pts": g[f"pts_{side}"], "opp_pts": g[f"pts_{opp}"], "poss": poss,
            "eff": 100 * g[f"pts_{side}"] / poss,
            "pace": poss * 200.0 / mins,
            "efg": 100 * (g[f"{side}_fgm"] + 0.5 * g[f"{side}_tpm"]) / fga,
            "tov": 100 * g[f"{side}_tov"] / poss,
            "orb": 100 * g[f"{side}_orb"] / (g[f"{side}_orb"] + g[f"{opp}_drb"]),
            "ftr": 100 * g[f"{side}_fta"] / fga,
            "tpar": 100 * g[f"{side}_tpa"] / fga,
            "tp": 100 * g[f"{side}_tpm"] / g[f"{side}_tpa"].where(g[f"{side}_tpa"] > 0),
            "twop": 100 * (g[f"{side}_fgm"] - g[f"{side}_tpm"]) / two_a.where(two_a > 0),
            "blk": 100 * g[f"{opp}_blk"] / two_a.where(two_a > 0),
            "stl": 100 * g[f"{opp}_stl"] / poss,
            "ast": 100 * g[f"{side}_ast"] / g[f"{side}_fgm"].where(g[f"{side}_fgm"] > 0),
        })
        rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    return out.replace([np.inf, -np.inf], np.nan)


class SeasonSolver:
    """Holds one season's observations and solves snapshots."""

    def __init__(self, rows: pd.DataFrame, teams: list[str], prior: pd.DataFrame | None,
                 prior_games: float = 4.0, ridge: float = 0.5):
        self.rows = rows.sort_values("date").reset_index(drop=True)
        self.teams = teams
        self.tix = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        self.n = n
        self.k = 2 + 2 * n
        r = self.rows
        ti = r["team"].map(self.tix).to_numpy()
        oi = r["opp"].map(self.tix).to_numpy()
        m = len(r)
        data = np.concatenate([np.ones(m), r["loc"].to_numpy(float), np.ones(m), np.ones(m)])
        cols = np.concatenate([np.zeros(m), np.ones(m), 2 + ti, 2 + n + oi])
        rr = np.concatenate([np.arange(m)] * 4)
        self.X = sp.csr_matrix((data, (rr, cols)), shape=(m, self.k))
        self.Y = r[STATS].to_numpy(float)
        self.mask = ~np.isnan(self.Y)
        self.Y = np.nan_to_num(self.Y)
        self.days = (r["date"] - r["date"].min()).dt.days.to_numpy()
        self.dates = r["date"].to_numpy()
        # priors (deviation from league mean) for Off/Def of each stat
        self.prior = np.zeros((self.k, len(STATS)))
        if prior is not None and len(prior):
            p = prior.set_index("team")
            for j, s in enumerate(STATS):
                for side, off in (("o", 2), ("d", 2 + n)):
                    col = f"{s}_{side}"
                    if col in p.columns:
                        vals = p[col].reindex(teams).fillna(0.0).to_numpy()
                        self.prior[off: off + n, j] = vals
        self.lam = np.zeros(self.k)
        self.lam[2:] = prior_games  # each team-game contributes one Off row and one Def row
        self.ridge = ridge

    def solve(self, asof) -> pd.DataFrame:
        """Ratings using games strictly before ``asof`` (numpy datetime64)."""
        use = self.dates < np.datetime64(asof)
        out = pd.DataFrame({"team": self.teams})
        nobs = np.zeros(self.n)
        if use.sum() > 0:
            ti = self.rows.loc[use, "team"].map(self.tix).to_numpy()
            nobs = np.bincount(ti, minlength=self.n)
        out["games"] = nobs
        age = (np.datetime64(asof) - self.dates[use]).astype("timedelta64[D]").astype(float)
        w = np.maximum(WEIGHT_FLOOR, 0.5 ** (age / HALF_LIFE_DAYS))
        X = self.X[use]
        Yu = self.Y[use]
        Mu = self.mask[use]
        lam = self.lam.copy()
        lam[0] = 1e-6
        lam[1] = 2.0
        reg = lam + self.ridge * (np.arange(self.k) >= 2)
        for grp in GROUPS:
            js = [STATS.index(x) for x in grp]
            mk = Mu[:, js].all(1)
            if mk.sum() < 10:
                for x in grp:
                    out[f"{x}_o"] = np.nan
                    out[f"{x}_d"] = np.nan
                if grp is GROUPS[0]:
                    out["hca"] = np.nan
                    out["mu"] = np.nan
                continue
            Xj = X[mk]
            wj = w[mk]
            Y = Yu[mk][:, js]
            mu0 = (wj[:, None] * Y).sum(0) / wj.sum()
            A = ((Xj.T.multiply(wj)) @ Xj).toarray()
            A[np.diag_indices_from(A)] += reg
            Bm = Xj.T @ (wj[:, None] * (Y - mu0)) + lam[:, None] * self.prior[:, js]
            theta = cho_solve(cho_factor(A), Bm)
            for c, x in enumerate(grp):
                mu = mu0[c] + theta[0, c]
                out[f"{x}_o"] = mu + theta[2: 2 + self.n, c]
                out[f"{x}_d"] = mu + theta[2 + self.n:, c]
                if x == "eff":
                    out["hca"] = theta[1, c]
                    out["mu"] = mu
        out["adjem"] = out["eff_o"] - out["eff_d"]
        out["adjt"] = (out["pace_o"] + out["pace_d"]) / 2  # pace when on O / pace allowed on D
        return out


def season_snapshots(rows: pd.DataFrame, teams: list[str], prior: pd.DataFrame | None,
                     snap_dates, prior_games: float = 4.0) -> pd.DataFrame:
    solver = SeasonSolver(rows, teams, prior, prior_games=prior_games)
    frames = []
    for d in snap_dates:
        f = solver.solve(np.datetime64(d))
        f["snap"] = pd.Timestamp(d)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)
