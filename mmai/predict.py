"""Live team state, arbitrary matchup predictions, power ratings, upcoming games."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import geo
from .config import current_season, today
from .features import TEAM_FEATS
from .model import train as TR

STATE_FEATS = [c for c in TEAM_FEATS if c not in ("rest", "dist")]


def team_state(snaps: pd.DataFrame, extras: dict, conf: pd.DataFrame, meta: pd.DataFrame,
               season: int | None = None) -> pd.DataFrame:
    season = season or current_season()
    s = snaps[snaps["season"] == season]
    last = s[s["snap"] == s["snap"].max()].set_index("team")
    pf = extras["player_feats"]
    live = pf[(pf["season"] == season) & (pf["gid"] == "__live__")].set_index("team")
    st = last.join(live.drop(columns=["season", "gid"]), how="left")
    st["elo"] = pd.Series(extras["elo_final"]).reindex(st.index)
    form = extras["team_form"]
    st["form"] = form.loc[season].reindex(st.index).fillna(0.0) if season in form.index.get_level_values(0) else 0.0
    st = st[st["games"] > 0].copy()  # D-I teams that have played this season
    # impute missing player-layer features (rare data gaps) from efficiency so the net stays in-distribution
    miss = st["roster"].isna()
    if miss.any():
        st.loc[miss, "roster"] = st.loc[miss, "adjem"]
        st.loc[miss, "roster_o"] = st.loc[miss, "eff_o"] - st["mu"].mean()
        st.loc[miss, "roster_d"] = st["mu"].mean() - st.loc[miss, "eff_d"]
        st.loc[miss, "avail_delta"] = 0.0
        for c in ("height", "experience", "star", "bench", "continuity"):
            st.loc[miss, c] = st[c].median() if st[c].notna().any() else np.nan
    c = conf[conf["season"] == season].drop_duplicates("team").set_index("team")["conf"]
    st["conf"] = c.reindex(st.index)
    m = meta.drop_duplicates("team").set_index("team")
    for col in ("espn_id", "location", "name", "abbr", "color", "alt_color", "logo", "lat", "lon", "city", "state"):
        if col in m.columns:
            st[col] = m[col].reindex(st.index)
    st["season"] = season
    return st


def pair_frame(state: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """pairs: team_a, team_b, loc (+ optional v_lat, v_lon, is_ncaa, is_conftourney, is_post, day, rest_a, rest_b)."""
    p = pairs.copy().reset_index(drop=True)
    season = int(state["season"].iloc[0])
    p["season"] = season
    for side, col in (("a", "team_a"), ("b", "team_b")):
        sub = state.reindex(p[col])
        for f in STATE_FEATS:
            p[f"{side}_{f}"] = sub[f].to_numpy()
        p[f"{side}_rest"] = p.get(f"rest_{side}", pd.Series(4.0, index=p.index)).fillna(4.0).to_numpy()
        if "v_lat" in p:
            d = geo.haversine_km(sub["lat"].to_numpy(), sub["lon"].to_numpy(), p["v_lat"].to_numpy(),
                                 p["v_lon"].to_numpy())
            d = np.where(np.isnan(d), 800.0, d)
        else:
            d = np.full(len(p), 800.0)
        p[f"{side}_dist"] = np.log1p(d)
        p[f"conf_{side}"] = sub["conf"].to_numpy()
    for c, dflt in (("is_ncaa", False), ("is_conftourney", 0.0), ("is_post", 0.0)):
        if c not in p:
            p[c] = dflt
    if "day" not in p:
        p["day"] = (pd.Timestamp(today()) - pd.Timestamp(f"{season - 1}-11-01")).days
    p["same_conf"] = (p["conf_a"] == p["conf_b"]).astype(float)
    p["played"] = False
    return p


def predict_pairs(bundle, state, pairs):
    models, norm, smeans = bundle
    df = pair_frame(state, pairs)
    mu, sg, pr = TR.predict_df(models, norm, smeans, df)
    df["margin"] = mu
    df["sigma"] = sg
    df["p_a"] = pr
    tempo = (df["a_adjt"] + df["b_adjt"]) / 2
    mu_eff = (df["a_eff_o"] + df["b_eff_d"]) / 2
    total = 2 * mu_eff.fillna(105) * tempo.fillna(68) / 100
    df["pts_a"] = total / 2 + mu / 2
    df["pts_b"] = total / 2 - mu / 2
    return df


def power_ratings(bundle, state: pd.DataFrame) -> pd.DataFrame:
    """Round-robin through the network on a neutral floor -> each team's average margin."""
    teams = state.index.tolist()
    n = len(teams)
    ia, ib = np.triu_indices(n, 1)
    pairs = pd.DataFrame({"team_a": np.array(teams)[ia], "team_b": np.array(teams)[ib], "loc": 0})
    df = predict_pairs(bundle, state, pairs)
    M = np.zeros((n, n))
    P = np.full((n, n), 0.5)
    M[ia, ib] = df["margin"].to_numpy()
    M[ib, ia] = -df["margin"].to_numpy()
    P[ia, ib] = df["p_a"].to_numpy()
    P[ib, ia] = 1 - df["p_a"].to_numpy()
    out = pd.DataFrame(index=teams)
    out["power"] = M.sum(1) / (n - 1)
    out["win_vs_avg"] = P.sum(1) / (n - 1)
    # win probability vs a typical top-25 team (bubble -> elite context)
    top = np.argsort(-out["power"].to_numpy())[:25]
    out["win_vs_top25"] = P[:, top].mean(1)
    out = out.sort_values("power", ascending=False)
    out["rank"] = np.arange(1, n + 1)
    return out, (teams, M, P)


def upcoming(bundle, state, games: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    t0 = pd.Timestamp(today())
    g = games[(~games["played"]) & (games["date"] >= t0) & (games["date"] <= t0 + pd.Timedelta(days=days))
              & (games["type"] != "nond1")]
    g = g[g["team_a"].isin(state.index) & g["team_b"].isin(state.index)]
    if g.empty:
        return pd.DataFrame()
    pairs = g[["team_a", "team_b", "loc", "v_lat", "v_lon", "date", "type", "gid", "city", "state"]].copy()
    pairs["is_conftourney"] = (pairs["type"] == "conftourney").astype(float)
    pairs["is_post"] = (pairs["type"] == "post").astype(float)
    pairs["day"] = (pairs["date"] - pd.Timestamp(f"{int(state['season'].iloc[0]) - 1}-11-01")).dt.days
    last = state.get("last_date")
    df = predict_pairs(bundle, state, pairs)
    df["date"] = pairs["date"].to_numpy()
    df["gid"] = pairs["gid"].to_numpy()
    df["city"] = pairs["city"].to_numpy()
    return df
