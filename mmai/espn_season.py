"""ESPN fallback for the *current* season when barttorvik.com is unreachable.

Builds the same tables the Torvik loaders produce (games with box scores, player-game
rows with box-score plus/minus, conferences) from ESPN's public scoreboard, game summary
and standings endpoints. Player plus/minus uses a box-score formula calibrated on
Torvik's own game values (R² ≈ 0.87; see ``BOX_COEF``). Player ids are matched to Torvik
ids by name so last season's values carry over.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import history
from .config import PROC, RAW
from .fetch.espn import BASE
from .http import get
from .injuries import norm_name

OUT = RAW / "espn" / "summary"
MARKER = PROC / "current_source.txt"

# per-40-minute box rates -> game plus/minus (fit on Torvik 2023-26 player-game logs)
_FEATS = ["twopm", "twopa", "tpm", "tpa", "ftm", "fta", "orb", "drb", "ast", "tov", "stl", "blk", "pf"]
BOX_COEF = {  # last term: opponent's adjusted efficiency margin (opponent-strength adjustment)
    "obpm": [1.6599, -0.5849, 3.1294, -0.6163, 0.8757, -0.2866, 0.6774, -0.0199, 0.5446, -1.0001, 0.3023,
             0.2844, -0.6045, -4.9146, -3.5877, 0.083],
    "dbpm": [0.6071, -0.3845, 0.7885, -0.3594, 0.3048, -0.1851, -0.1518, 0.3299, 0.2912, -0.3399, 1.3645,
             0.8526, 0.0839, 0.7666, -2.951, 0.0559],
}
STAT_KEYS = {"fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
             "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("tpm", "tpa"),
             "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta"), "offensiveRebounds": "orb",
             "defensiveRebounds": "drb", "totalRebounds": "trb", "assists": "ast", "steals": "stl",
             "blocks": "blk", "totalTurnovers": "tov", "fouls": "pf"}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _split(v):
    a, _, b = str(v).partition("-")
    return _num(a), _num(b)


def summary(season: int, event_id: str) -> dict | None:
    p = OUT / str(season) / f"{event_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    try:
        d = get(f"{BASE}/summary?event={event_id}", timeout=40).json()
    except Exception as e:
        print(f"  espn summary {event_id}: {e}")
        return None
    box = d.get("boxscore") or {}
    teams = {}
    for t in box.get("teams", []):
        rec = {}
        for s in t.get("statistics", []):
            k = STAT_KEYS.get(s.get("name"))
            if isinstance(k, tuple):
                rec[k[0]], rec[k[1]] = _split(s.get("displayValue"))
            elif k:
                rec[k] = _num(s.get("displayValue"))
        teams[str(t["team"]["id"])] = rec
    players = []
    for grp in box.get("players", []):
        tid = str(grp["team"]["id"])
        for st in grp.get("statistics", [])[:1]:
            keys = st.get("keys") or []
            for a in st.get("athletes", []):
                if a.get("didNotPlay") or not a.get("stats"):
                    continue
                row = dict(zip(keys, a["stats"]))
                fgm, fga = _split(row.get("fieldGoalsMade-fieldGoalsAttempted"))
                tpm, tpa = _split(row.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
                ftm, fta = _split(row.get("freeThrowsMade-freeThrowsAttempted"))
                players.append({
                    "team_id": tid, "athlete_id": str(a["athlete"].get("id")), "name": a["athlete"].get("displayName"),
                    "min": _num(row.get("minutes")), "pts": _num(row.get("points")), "fgm": fgm, "fga": fga,
                    "tpm": tpm, "tpa": tpa, "ftm": ftm, "fta": fta, "orb": _num(row.get("offensiveRebounds")),
                    "drb": _num(row.get("defensiveRebounds")), "ast": _num(row.get("assists")),
                    "tov": _num(row.get("turnovers")), "stl": _num(row.get("steals")), "blk": _num(row.get("blocks")),
                    "pf": _num(row.get("fouls"))})
    out = {"teams": teams, "players": players}
    if teams and players:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out))
    return out


def _game_type(r) -> str:
    notes = str(r.notes or "")
    if str(r.season_type) == "3" or re.search(r"MEN'S BASKETBALL CHAMP|NIT|CBI|CROWN", notes.upper()):
        return "post"
    if re.search(r"Championship|Tournament", notes, re.I) and "Men's" in notes:
        return "conftourney"
    return "conf" if str(r.conf_game) == "True" else "nonconf"


def _pid_index(season: int):
    """(name, team) -> pid and unique name -> pid from the last two Torvik seasons."""
    frames = []
    snap = history.load_snapshot(season)
    if snap is not None and len(snap[1]):
        cur = snap[1].drop_duplicates("pid", keep="last")[["player", "team", "pid", "cls"]]
        frames.append(cur.assign(season=season, cls=cur["cls"].map({"So": "Fr", "Jr": "So", "Sr": "Jr"})))
    for s in (season - 1, season - 2):
        if history.path("ps", s).exists():
            frames.append(history.load_ps(s)[["player", "team", "pid", "cls"]].assign(season=s))
    if not frames:
        return {}, {}, {}
    ps = pd.concat(frames).sort_values("season", ascending=False)
    ps["nk"] = ps["player"].map(norm_name)
    by_team = {(t, n): int(p) for t, n, p in zip(ps["team"], ps["nk"], ps["pid"])}
    counts = ps.drop_duplicates("pid")["nk"].value_counts()
    uniq = ps[ps["nk"].map(counts) == 1].drop_duplicates("nk")
    by_name = dict(zip(uniq["nk"], uniq["pid"].astype(int)))
    nxt = {"Fr": "So", "So": "Jr", "Jr": "Sr", "Sr": "Sr"}
    last = ps.drop_duplicates("pid")
    cls = {int(p): nxt.get(str(c), None) for p, c in zip(last["pid"], last["cls"])}
    if snap is not None and len(snap[1]):  # this season's own classes win
        cur_cls = snap[1].dropna(subset=["cls"]).drop_duplicates("pid", keep="last")
        cls.update({int(p): str(c) for p, c in zip(cur_cls["pid"], cur_cls["cls"])})
    return by_team, by_name, cls


def box_pm(df: pd.DataFrame, opp_em: dict | None = None) -> pd.DataFrame:
    df["opp_em"] = df["opponent"].map(opp_em or {})
    m = df["min"].clip(lower=1)
    f = 40.0 / m
    X = np.column_stack([
        (df["fgm"] - df["tpm"]) * f, (df["fga"] - df["tpa"]) * f, df["tpm"] * f, df["tpa"] * f, df["ftm"] * f,
        df["fta"] * f, df["orb"] * f, df["drb"] * f, df["ast"] * f, df["tov"] * f, df["stl"] * f, df["blk"] * f,
        df["pf"] * f, 1.0 / m, np.ones(len(df)), df["opp_em"].fillna(0.0)])
    X = np.nan_to_num(X)
    df["obpm"] = X @ np.array(BOX_COEF["obpm"])
    df["dbpm"] = X @ np.array(BOX_COEF["dbpm"])
    # calibration offset vs Torvik game values for players covered by both sources (Nov 2026: +1.2)
    df["obpm"] -= 0.7
    df["dbpm"] -= 0.5
    df["bpm"] = df["obpm"] + df["dbpm"]
    return df


def build_current(season: int, eg: pd.DataFrame, cw: pd.DataFrame, workers: int = 6):
    """Write Torvik-format current-season tables built from ESPN; returns the games frame."""
    id2team = dict(zip(cw["espn_id"].astype(str), cw["team"]))
    e = eg[(eg["season"] == season)].copy()
    e = e[e["home_id"].astype(str).isin(id2team) & e["away_id"].astype(str).isin(id2team)]
    done = e[e["completed"].astype(str) == "True"]
    with ThreadPoolExecutor(workers) as ex:
        sums = dict(zip(done["event_id"], ex.map(lambda i: summary(season, i), done["event_id"])))
    print(f"  ESPN fallback: {len(e)} D-I games, {sum(v is not None for v in sums.values())} box scores")
    by_team, by_name, cls_map = _pid_index(season)
    conf = conferences(season, cw)
    d1 = set(conf["team"])

    g_rows, p_rows = [], []
    for r in e.itertuples():
        a_id, b_id = str(r.away_id), str(r.home_id)
        ta, tb = id2team[a_id], id2team[b_id]
        gid = f"E{r.event_id}"
        neutral = str(r.neutral) == "True"
        gtype = _game_type(r) if (ta in d1 and tb in d1) else "nond1"
        row = {"gid": gid, "date": pd.Timestamp(r.date), "type": gtype, "neutral": neutral,
               "team_a": ta, "team_b": tb, "played": str(r.completed) == "True",
               "pts_a": _num(r.away_score), "pts_b": _num(r.home_score), "season": season,
               "loc": 0 if neutral else -1, "torvik_p_a": np.nan, "ot": 0, "minutes": 200.0, "poss": np.nan}
        s = sums.get(r.event_id)
        if row["played"] and s:
            A, B = s["teams"].get(a_id, {}), s["teams"].get(b_id, {})
            for k in ("fgm", "fga", "tpm", "tpa", "ftm", "fta", "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf"):
                row[f"a_{k}"], row[f"b_{k}"] = A.get(k, np.nan), B.get(k, np.nan)
            row["a_pts"], row["b_pts"] = row["pts_a"], row["pts_b"]
            pa = A.get("fga", np.nan) - A.get("orb", np.nan) + A.get("tov", np.nan) + 0.475 * A.get("fta", np.nan)
            pb = B.get("fga", np.nan) - B.get("orb", np.nan) + B.get("tov", np.nan) + 0.475 * B.get("fta", np.nan)
            row["poss"] = (pa + pb) / 2
            team_min = sum(p["min"] for p in s["players"] if p["team_id"] == a_id and not np.isnan(p["min"]))
            if team_min > 0:
                row["minutes"] = max(200.0, round(team_min / 25.0) * 25.0)
                row["ot"] = int(max(0, (row["minutes"] - 200) / 25))
            for p in s["players"]:
                if np.isnan(p["min"]) or p["min"] <= 0:
                    continue
                team = id2team.get(p["team_id"])
                nk = norm_name(p["name"])
                pid = by_team.get((team, nk)) or by_name.get(nk) or 90_000_000 + int(p["athlete_id"] or 0)
                p_rows.append({**p, "season": season, "date": row["date"], "muid": gid, "team": team,
                               "opponent": tb if team == ta else ta,
                               "loc": "N" if neutral else ("A" if team == ta else "H"),
                               "player": p["name"], "pid": float(pid), "cls": cls_map.get(int(pid)),
                               "inches": np.nan, "ortg": np.nan, "usage": np.nan, "poss": row["poss"]})
        row["a_won"] = bool(row["pts_a"] > row["pts_b"]) if row["played"] else False
        g_rows.append(row)
    games = pd.DataFrame(g_rows)
    snap = history.load_snapshot(season)
    snap_pg = snap_ps = None
    if snap is not None:
        sg, snap_pg, snap_ps = snap
        cutoff = sg.loc[sg["played"], "date"].max()
        games = pd.concat([sg[sg["played"] & (sg["date"] <= cutoff)], games[games["date"] > cutoff]],
                          ignore_index=True)
        p_rows = [r for r in p_rows if r["date"] > cutoff]
        print(f"  using Torvik snapshot through {cutoff:%Y-%m-%d}; ESPN for later games")
    for side in ("a", "b"):
        for k in ("fgm", "fga", "tpm", "tpa", "ftm", "fta", "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf", "pts"):
            if f"{side}_{k}" not in games:
                games[f"{side}_{k}"] = np.nan
    pg = box_pm(pd.DataFrame(p_rows), _current_em(games)) if p_rows else pd.DataFrame(columns=["season"])
    if len(pg):
        prev = history.load_pg(season - 1) if history.path("pg", season - 1).exists() else None
        if prev is not None:  # carry heights over from last season
            h = prev.dropna(subset=["inches"]).drop_duplicates("pid", keep="last").set_index("pid")["inches"]
            pg["inches"] = pg["pid"].map(h)
        keep = ["season", "date", "muid", "team", "opponent", "loc", "player", "pid", "cls", "inches", "min",
                "ortg", "usage", "bpm", "obpm", "dbpm", "pts", "poss"]
        pg = pg[keep].astype({"cls": "string", "loc": "string"})
    if snap_pg is not None:
        pg = pd.concat([snap_pg, pg], ignore_index=True) if len(pg) else snap_pg
        pg = pg.astype({"cls": "string", "loc": "string"})
    games.to_parquet(PROC / f"tv_games_{season}.parquet")
    pg.to_parquet(PROC / f"pg_{season}.parquet")
    (snap_ps if snap_ps is not None and len(snap_ps) else pd.DataFrame(columns=history.PS_COLS)).to_parquet(
        PROC / f"ps_{season}.parquet")
    conf.to_parquet(PROC / f"conf_{season}.parquet")
    cut = ""
    if snap is not None:
        cut = f" {sg.loc[sg['played'], 'date'].max():%Y-%m-%d}"
    MARKER.write_text(f"espn {season}{cut}")
    return games


def _current_em(games: pd.DataFrame) -> dict:
    """Quick season-to-date adjusted efficiency margins (for opponent-adjusting player values)."""
    from .ratings import efficiency as EF

    pl = games[games["played"] & games["poss"].notna() & (games["type"] != "nond1")]
    if len(pl) < 50:
        return {}
    rows = EF.team_game_rows(pl)
    teams = sorted(set(rows["team"]) | set(rows["opp"]))
    f = EF.SeasonSolver(rows, teams, None, prior_games=2.0).solve(np.datetime64(pl["date"].max() + pd.Timedelta(days=1)))
    return dict(zip(f["team"], f["adjem"]))


def conferences(season: int, cw: pd.DataFrame) -> pd.DataFrame:
    """ESPN standings -> conference membership, labelled with Torvik's abbreviations."""
    id2team = dict(zip(cw["espn_id"].astype(str), cw["team"]))
    prev = pd.read_parquet(history.path("conf"))
    prev = prev[prev["season"] == prev["season"].max()].set_index("team")["conf"]
    rows = []
    try:
        d = get("https://site.web.api.espn.com/apis/v2/sports/basketball/mens-college-basketball/standings"
                "?region=us&lang=en&contentorigin=espn&group=50", timeout=60).json()
        confs = []
        for c in d.get("children", []):
            members = []
            for ent in (c.get("standings") or {}).get("entries", []):
                t = id2team.get(str(ent["team"]["id"]))
                if t:
                    members.append(t)
            vote = Counter(prev.get(t) for t in members if prev.get(t))
            top, n = vote.most_common(1)[0] if vote else (None, 0)
            confs.append((n / max(len(members), 1), top, c.get("abbreviation") or c.get("name"), members))
        used = set()
        # most clear-cut matches claim Torvik labels first; realigned/new leagues keep ESPN's name
        for share, top, espn_label, members in sorted(confs, key=lambda x: -x[0]):
            label = top if (top and share >= 0.6 and top not in used) else espn_label
            used.add(label)
            rows += [{"team": t, "conf": label} for t in members]
    except Exception as e:
        print("  ESPN standings unavailable, using last season's conferences:", e)
    df = pd.DataFrame(rows, columns=["team", "conf"])
    missing = prev[~prev.index.isin(df["team"])]
    df = pd.concat([df, missing.reset_index()], ignore_index=True)
    df["season"] = season
    for c in ("t_adjoe", "t_adjde", "t_barthag"):
        df[c] = np.nan
    return df[["season", "team", "conf", "t_adjoe", "t_adjde", "t_barthag"]]


def active(season: int) -> bool:
    return MARKER.exists() and MARKER.read_text().split()[:2] == ["espn", str(season)]


def snapshot_date(season: int) -> str | None:
    if not active(season):
        return None
    parts = MARKER.read_text().split()
    return parts[2] if len(parts) > 2 else None
