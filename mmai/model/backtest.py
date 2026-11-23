"""Rolling-origin backtest on NCAA tournament games.

For each tournament year, every model is fit only on games played before that
tournament began. Tournament games are then scored with *pre-game knowable*
availability (recent missed games only), not the realised lineups.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import ARTIFACTS
from . import train as TR


def _metrics(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return {"n": int(len(y)), "log_loss": TR.log_loss(y, p), "brier": float(np.mean((p - y) ** 2)),
            "accuracy": float(np.mean((p > 0.5) == (y > 0.5)))}


def _fit_linear(tr):
    tempo = (tr["a_adjt"] + tr["b_adjt"]) / 200.0
    X = np.column_stack([(tr["a_adjem"] - tr["b_adjem"]) * tempo, tr["loc"],
                         (tr["a_avail_delta"] - tr["b_avail_delta"]) * tempo])
    ok = ~np.isnan(X).any(1) & tr["margin"].notna().to_numpy()
    beta = np.linalg.lstsq(X[ok], tr["margin"].to_numpy()[ok], rcond=None)[0]
    sigma = float(np.std(tr["margin"].to_numpy()[ok] - X[ok] @ beta))
    return beta, sigma


def _pred_linear(beta, sigma, te):
    tempo = (te["a_adjt"] + te["b_adjt"]) / 200.0
    X = np.column_stack([(te["a_adjem"] - te["b_adjem"]) * tempo, te["loc"],
                         (te["a_avail_delta"] - te["b_avail_delta"]) * tempo])
    return norm.cdf(np.nan_to_num(X @ beta) / sigma)


def _fit_logit(x, y):
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=10.0)
    m.fit(x, y)
    return m


def run(feat: pd.DataFrame, seasons=None, n_models=3, epochs=TR.EPOCHS, verbose=True):
    feat = feat[feat["season"] >= 2009]
    smeans = TR.season_means(feat)
    seasons = seasons or sorted(s for s in feat.loc[feat["is_ncaa"] & feat["played"], "season"].unique() if s >= 2012)
    rows, per_season, games_out = [], [], []
    for s in seasons:
        te = feat[(feat["season"] == s) & feat["is_ncaa"] & feat["played"]].copy()
        if te.empty:
            continue
        cutoff = te["date"].min()
        tr = feat[(feat["date"] < cutoff) & feat["played"] & feat["margin"].notna()]
        # honest availability at forecast time
        te["a_avail_delta"] = te["a_avail_delta_pre"]
        te["b_avail_delta"] = te["b_avail_delta_pre"]
        y = (te["margin"] > 0).to_numpy()

        models, nrm = TR.train_models(tr, smeans, n_models=n_models, epochs=epochs, seed=s)
        mu, sg, p_nn = TR.predict_df(models, nrm, smeans, te)

        beta, sigma = _fit_linear(tr)
        p_lin = _pred_linear(beta, sigma, te)

        ed = (tr["a_elo"] - tr["b_elo"]).to_numpy()[:, None]
        elo_m = _fit_logit(np.column_stack([ed, tr["loc"]]), (tr["margin"] > 0).to_numpy())
        p_elo = elo_m.predict_proba(np.column_stack([(te["a_elo"] - te["b_elo"]).to_numpy(), te["loc"]]))[:, 1]

        prev_nc = tr[tr["is_ncaa"] & tr["seed_a"].notna() & tr["seed_b"].notna()]
        seed_m = _fit_logit((prev_nc["seed_b"] - prev_nc["seed_a"]).to_numpy()[:, None], (prev_nc["margin"] > 0).to_numpy())
        sd = (te["seed_b"] - te["seed_a"]).fillna(0).to_numpy()[:, None]
        p_seed = seed_m.predict_proba(sd)[:, 1]

        p_tv = te["torvik_p_a"].to_numpy()
        res = {"season": int(s)}
        for name, p in (("MMAI net", p_nn), ("Efficiency only", p_lin), ("Elo", p_elo), ("Seed", p_seed)):
            res[name] = _metrics(y, p)
        ok = ~np.isnan(p_tv)
        res["T-Rank (pre-game)"] = _metrics(y[ok], p_tv[ok]) if ok.any() else None
        res["MMAI net (T-Rank games)"] = _metrics(y[ok], p_nn[ok]) if ok.any() else None
        per_season.append(res)
        games_out.append(pd.DataFrame({
            "season": s, "date": te["date"].dt.strftime("%Y-%m-%d"), "round": te["ncaa_round"], "team_a": te["team_a"],
            "team_b": te["team_b"], "seed_a": te["seed_a"], "seed_b": te["seed_b"], "pts_a": te["pts_a"],
            "pts_b": te["pts_b"], "p_nn": p_nn, "mu": mu, "p_lin": p_lin, "p_elo": p_elo, "p_seed": p_seed, "p_tv": p_tv}))
        if verbose:
            print(s, {k: round(v["log_loss"], 4) for k, v in res.items() if isinstance(v, dict)}, flush=True)
    g = pd.concat(games_out, ignore_index=True)
    y = (g["pts_a"] > g["pts_b"]).to_numpy()
    overall = {name: _metrics(y, g[col]) for name, col in
               (("MMAI net", "p_nn"), ("Efficiency only", "p_lin"), ("Elo", "p_elo"), ("Seed", "p_seed"))}
    ok = g["p_tv"].notna().to_numpy()
    overall["T-Rank (pre-game)"] = _metrics(y[ok], g.loc[ok, "p_tv"])
    overall["MMAI net (T-Rank games)"] = _metrics(y[ok], g.loc[ok, "p_nn"])
    # upset detection: games where the worse seed won
    up = (g["seed_a"] != g["seed_b"]).to_numpy()
    out = {"overall": overall, "per_season": per_season}
    with open(ARTIFACTS / "backtest.json", "w") as f:
        json.dump(out, f, indent=1)
    g.to_csv(ARTIFACTS / "backtest_games.csv", index=False)
    return out
