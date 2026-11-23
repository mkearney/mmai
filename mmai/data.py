"""Assemble canonical tables: games, team crosswalk / metadata, NCAA tournament info."""
from __future__ import annotations

import datetime as dt
import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from . import geo
from .config import ARTIFACTS, FIRST_SEASON, PROC, RAW, current_season
from .fetch import espn, torvik


# --------------------------------------------------------------------------- games
def torvik_games(seasons, refresh_current=True) -> pd.DataFrame:
    frames = []
    for s in seasons:
        p = PROC / f"tv_games_{s}.parquet"
        if p.exists() and not (refresh_current and s >= current_season()):
            frames.append(pd.read_parquet(p))
            continue
        g = torvik.super_sked(s)
        g.to_parquet(p)
        frames.append(g)
    g = pd.concat(frames, ignore_index=True)
    return g.drop_duplicates(["season", "gid"], keep="last")


def espn_games(seasons, refresh_current=True) -> pd.DataFrame:
    frames = []
    for s in seasons:
        p = PROC / f"espn_games_{s}.parquet"
        if p.exists() and not (refresh_current and s >= current_season()):
            frames.append(pd.read_parquet(p))
            continue
        a = espn.fetch_season(s, 50)
        b = espn.tournament_games(s)
        e = pd.concat([a, b], ignore_index=True)
        if e.empty:
            continue
        e = e.drop_duplicates("event_id", keep="last")
        e["season"] = s
        for c in ("home_score", "away_score", "home_rank", "away_rank"):
            e[c] = pd.to_numeric(e[c], errors="coerce")
        e = e.astype({c: "string" for c in e.columns if e[c].dtype == object})
        e.to_parquet(p)
        frames.append(e)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------- NCAA parsing
NCAA_RE = re.compile(r"^(NCAA )?MEN'S BASKETBALL CHAMP", re.I)
ROUND_NAMES = ["First Four", "Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]


def ncaa_round(notes: str, season: int) -> int | None:
    n = (notes if isinstance(notes, str) else "").upper()
    if not NCAA_RE.search(n):
        return None
    if "OPENING ROUND" in n or "FIRST FOUR" in n:
        return 0
    if "NATIONAL CHAMPIONSHIP" in n or "CHAMPIONSHIP GAME" in n or n.endswith("- FINAL") or "NATIONAL FINAL" in n:
        return 6
    if "FINAL FOUR" in n or "NATIONAL SEMI" in n:
        return 5
    if "ELITE" in n or "REGIONAL FINAL" in n:
        return 4
    if "SWEET" in n or "REGIONAL SEMI" in n:
        return 3
    shift = 1 if 2011 <= season <= 2015 else 0  # 2011-15: First Four was called "1st round"
    if re.search(r"1ST R|FIRST ROUND", n):
        return 1 - shift
    if re.search(r"2ND R|SECOND ROUND", n):
        return 2 - shift
    if re.search(r"3RD R|THIRD ROUND", n):
        return 3 - shift
    return None


def ncaa_region(notes: str) -> str | None:
    m = re.search(r"- ([A-Z][A-Za-z.' ]+?) REGION", (notes if isinstance(notes, str) else "").upper())
    return m.group(1).title() if m else None


# --------------------------------------------------------------------- crosswalk
def build_crosswalk(tg: pd.DataFrame, eg: pd.DataFrame) -> pd.DataFrame:
    """Map Torvik team names -> ESPN ids by matching games on (date, score pair)."""
    eg = eg[eg["completed"].astype(str) == "True"].copy()
    eg["date"] = pd.to_datetime(eg["date"])
    idx = defaultdict(list)
    for r in eg.itertuples():
        if pd.isna(r.home_score) or pd.isna(r.away_score):
            continue
        for off in (-1, 0, 1):
            d = r.date + pd.Timedelta(days=off)
            idx[(d, int(r.home_score), int(r.away_score))].append((r.home_id, r.away_id, r.home_location, r.away_location))
    votes = defaultdict(Counter)
    tp = tg[tg["played"] & tg["pts_a"].notna()]
    for r in tp.itertuples():
        a, b = int(r.pts_a), int(r.pts_b)
        m1 = idx.get((r.date, a, b), [])  # a = home side
        m2 = idx.get((r.date, b, a), [])  # a = away side
        if len(m1) + len(m2) != 1:
            continue
        if m1:
            ha, hb = m1[0][0], m1[0][1]
        else:
            ha, hb = m2[0][1], m2[0][0]
        votes[r.team_a][ha] += 1
        votes[r.team_b][hb] += 1
    rows = []
    for name, c in votes.items():
        eid, n = c.most_common(1)[0]
        tot = sum(c.values())
        rows.append({"team": name, "espn_id": str(eid), "votes": n, "share": n / tot})
    cw = pd.DataFrame(rows)
    # an ESPN id should map to one Torvik name; keep the best supported
    cw = cw.sort_values("votes", ascending=False).drop_duplicates("espn_id")
    return cw[(cw["votes"] >= 3) & (cw["share"] >= 0.6)].reset_index(drop=True)


def team_meta(cw: pd.DataFrame, eg: pd.DataFrame) -> pd.DataFrame:
    """Logos/colors + home city (mode of non-neutral home venues) + lat/lon."""
    try:
        et = espn.teams()
    except Exception as e:  # pragma: no cover
        print("espn teams failed:", e)
        et = pd.DataFrame(columns=["espn_id"])
    meta = cw.merge(et, on="espn_id", how="left")
    # fallback names from scoreboards
    names = (eg[["home_id", "home_location", "home_name", "home_abbr"]]
             .rename(columns=lambda c: c.replace("home_", "")).dropna(subset=["id"])
             .drop_duplicates("id", keep="last").rename(columns={"id": "espn_id"}))
    meta = meta.merge(names, on="espn_id", how="left", suffixes=("", "_sb"))
    for c in ("location", "name", "abbr"):
        meta[c] = meta[c].fillna(meta[f"{c}_sb"])
    meta = meta.drop(columns=[c for c in meta.columns if c.endswith("_sb")])
    meta["logo"] = meta["logo"].fillna(
        "https://a.espncdn.com/i/teamlogos/ncaa/500/" + meta["espn_id"].astype(str) + ".png")

    recent = eg[(eg["neutral"].astype(str) == "False")]
    recent = recent.sort_values("season")
    home = {}
    for eid, grp in recent.groupby("home_id"):
        g2 = grp[grp["season"] >= grp["season"].max() - 2]
        cs = Counter(zip(g2["city"].fillna(""), g2["state"].fillna("")))
        home[eid] = cs.most_common(1)[0][0] if cs else ("", "")
    meta["city"] = meta["espn_id"].map(lambda i: home.get(i, ("", ""))[0])
    meta["state"] = meta["espn_id"].map(lambda i: home.get(i, ("", ""))[1])
    ll = [geo.geocode(c, s) for c, s in zip(meta["city"], meta["state"])]
    meta["lat"] = [x[0] if x else np.nan for x in ll]
    meta["lon"] = [x[1] if x else np.nan for x in ll]
    return meta


# ------------------------------------------------------------------ enrichment
def attach_espn(tg: pd.DataFrame, eg: pd.DataFrame, cw: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    name2id = dict(zip(cw["team"], cw["espn_id"]))
    g = tg.copy()
    g["a_id"] = g["team_a"].map(name2id)
    g["b_id"] = g["team_b"].map(name2id)

    e = eg.copy()
    e["date"] = pd.to_datetime(e["date"])
    e["ncaa_round"] = [ncaa_round(n, s) for n, s in zip(e["notes"].fillna("").astype(str), e["season"])]
    e["region"] = e["notes"].fillna("").astype(str).map(ncaa_region)
    look = {}
    for r in e.itertuples():
        info = (r.event_id, str(r.neutral) == "True", r.city, r.state, r.ncaa_round, r.region,
                r.home_rank, r.away_rank, r.notes)
        for off in (0, -1, 1):
            d = r.date + pd.Timedelta(days=off)
            look.setdefault((d, r.home_id, r.away_id), (info, False))
            look.setdefault((d, r.away_id, r.home_id), (info, True))
    cols = defaultdict(list)
    for r in g.itertuples():
        hit = look.get((r.date, r.a_id, r.b_id))
        if hit is None:
            for k in ("event_id", "e_neutral", "city", "state", "ncaa_round", "region", "seed_a", "seed_b", "notes"):
                cols[k].append(None)
            continue
        (eid, neu, city, state, rnd, reg, hr, ar, notes), a_is_away = hit
        cols["event_id"].append(eid)
        cols["e_neutral"].append(neu)
        cols["city"].append(city)
        cols["state"].append(state)
        cols["ncaa_round"].append(rnd)
        cols["region"].append(reg)
        cols["seed_a"].append(ar if a_is_away else hr)
        cols["seed_b"].append(hr if a_is_away else ar)
        cols["notes"].append(notes)
    for k, v in cols.items():
        g[k] = v
    # ESPN knows home/neutral reliably; prefer it where the game was matched
    a_away = pd.Series([h[1] if (h := look.get((d, a, b))) else None
                        for d, a, b in zip(g["date"], g["a_id"], g["b_id"])], index=g.index)
    matched = g["event_id"].notna()
    espn_loc = np.where(g["e_neutral"] == True, 0, np.where(a_away == True, -1, 1))  # noqa: E712
    g["loc"] = np.where(matched, espn_loc, g["loc"]).astype(int)
    g["ncaa_round"] = pd.to_numeric(g["ncaa_round"], errors="coerce")
    g["is_ncaa"] = g["ncaa_round"].notna()
    for c in ("seed_a", "seed_b"):
        g[c] = pd.to_numeric(g[c], errors="coerce").where(g["is_ncaa"])

    # venue coordinates: home team's city for true home games, ESPN venue otherwise
    m = meta.set_index("espn_id")
    lat_a, lon_a = g["a_id"].map(m["lat"]), g["a_id"].map(m["lon"])
    lat_b, lon_b = g["b_id"].map(m["lat"]), g["b_id"].map(m["lon"])
    vll = [geo.geocode(c, s) if isinstance(c, str) and c else None for c, s in zip(g["city"], g["state"])]
    vlat = pd.Series([x[0] if x else np.nan for x in vll], index=g.index)
    vlon = pd.Series([x[1] if x else np.nan for x in vll], index=g.index)
    vlat = np.where(g["loc"] == 1, lat_a, np.where(g["loc"] == -1, lat_b, vlat))
    vlon = np.where(g["loc"] == 1, lon_a, np.where(g["loc"] == -1, lon_b, vlon))
    g["v_lat"], g["v_lon"] = vlat, vlon
    g["dist_a"] = geo.haversine_km(lat_a, lon_a, g["v_lat"], g["v_lon"])
    g["dist_b"] = geo.haversine_km(lat_b, lon_b, g["v_lat"], g["v_lon"])
    return g


def build(seasons=None, refresh_current=True):
    seasons = seasons or list(range(FIRST_SEASON, current_season() + 1))
    tg = torvik_games(seasons, refresh_current)
    eg = espn_games(seasons, refresh_current)
    cw_path = ARTIFACTS / "crosswalk.csv"
    cw = build_crosswalk(tg, eg)
    if cw_path.exists():  # keep previously resolved names (e.g. defunct programs)
        old = pd.read_csv(cw_path, dtype={"espn_id": str})
        cw = pd.concat([cw, old[~old["team"].isin(cw["team"])]], ignore_index=True)
    cw.to_csv(cw_path, index=False)
    meta = team_meta(cw, eg)
    meta.to_csv(ARTIFACTS / "team_meta.csv", index=False)
    games = attach_espn(tg, eg, cw, meta)
    games.to_parquet(PROC / "games.parquet")
    return games, meta
