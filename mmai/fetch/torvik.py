"""barttorvik.com public files (no login, no key).

* ``{year}_super_sked.csv``       every game (played + scheduled) with full box score
* ``{year}_all_advgames.json.gz`` player-game logs (minutes, usage, ORtg, BPM ...)
* ``getadvstats.php?year=..``     player-season table (height, class, recruit rank ...)
* ``{year}_team_results.csv``     T-Rank ratings (used only as an external benchmark)
"""
from __future__ import annotations

import ast
import gzip
import io
import json
import re

import numpy as np
import pandas as pd

from ..config import RAW, current_season
from ..http import cached_bytes

BASE = "https://barttorvik.com"
OUT = RAW / "torvik"

GAME_TYPE = {0: "nonconf", 1: "conf", 2: "conftourney", 3: "post", 99: "nond1"}
BOX = ["fgm", "fga", "tpm", "tpa", "ftm", "fta", "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf", "pts"]

PLAYER_GAME_COLS = [
    "numdate", "datetext", "opstyle", "quality", "win1", "opponent", "muid", "win2", "min", "ortg",
    "usage", "efg", "ts", "orb_pct", "drb_pct", "ast_pct", "to_pct", "dunks_m", "dunks_a", "rim_m",
    "rim_a", "mid_m", "mid_a", "twopm", "twopa", "tpm", "tpa", "ftm", "fta", "bpm_rd", "obpm",
    "dbpm", "bpm_net", "pts", "orb", "drb", "ast", "tov", "stl", "blk", "stl_pct", "blk_pct",
    "pf", "poss", "bpm", "sbpm", "loc", "team", "player", "inches", "cls", "pid", "year",
]

ADV_COLS = {
    0: "player", 1: "team", 2: "conf", 3: "gp", 4: "min_pct", 5: "ortg", 6: "usg", 7: "efg",
    8: "ts", 9: "orb_pct", 10: "drb_pct", 11: "ast_pct", 12: "to_pct", 19: "tpm", 20: "tpa",
    21: "tp_pct", 22: "blk_pct", 23: "stl_pct", 24: "ftr", 25: "cls", 26: "ht", 28: "porpag",
    29: "adjoe", 31: "year", 32: "pid", 33: "hometown", 34: "rec_rank", 46: "drtg", 47: "adrtg",
    48: "dporpag", 50: "bpm", 51: "obpm", 52: "dbpm", 53: "gbpm", 54: "mpg", 64: "role",
}


def _refresh(season: int) -> bool:
    return season >= current_season()


def _parse_pred(s: str) -> tuple[str | None, float | None]:
    """'Connecticut -5.9, 73-67 (72%)' -> ('Connecticut', 0.72)."""
    m = re.match(r"^(.*?) -?[\d.]+, \d+-\d+ \((\d+)%\)", str(s))
    if not m:
        return None, None
    return m.group(1), int(m.group(2)) / 100.0


def super_sked(season: int, raw: bytes | None = None) -> pd.DataFrame:
    if raw is None:
        raw = cached_bytes(f"{BASE}/{season}_super_sked.csv", OUT / f"{season}_super_sked.csv",
                           refresh=_refresh(season))
    s = pd.read_csv(io.BytesIO(raw), header=None, low_memory=False)
    out = pd.DataFrame({
        "gid": s[0].astype(str),
        "date": pd.to_datetime(s[1], format="%m/%d/%y"),
        "type": s[6].map(GAME_TYPE).fillna("other"),
        "neutral": s[7].astype(int) == 1,
        "team_a": s[8],
        "team_b": s[14],
        "a_won": s[12].astype(int) == 1,
        "played": s[23].fillna(0).astype(int) == 1,
        "pts_a": pd.to_numeric(s[27], errors="coerce"),
        "pts_b": pd.to_numeric(s[28], errors="coerce"),
        "poss": pd.to_numeric(s[26], errors="coerce"),
        "ot": pd.to_numeric(s[51], errors="coerce").fillna(0).astype(int),
    })
    out["season"] = season
    # location from team_a perspective: +1 home, 0 neutral, -1 away.
    # Torvik writes "away at home" / "x vs. y" (neutral), team_a listed first.
    at_game = s[3].astype(str).str.contains(" at ", regex=False)
    out["loc"] = np.where(out["neutral"] | ~at_game, 0, -1)

    # full box score (list literal)
    boxes = []
    for v in s[50].fillna(""):
        try:
            b = ast.literal_eval(v.replace("u'", "'")) if v else None
        except Exception:
            b = None
        boxes.append(b)
    box_a = np.full((len(s), len(BOX)), np.nan)
    box_b = np.full((len(s), len(BOX)), np.nan)
    minutes = np.full(len(s), np.nan)
    for i, b in enumerate(boxes):
        if not b or len(b) < 35:
            continue
        a_first = b[2] == out.at[i, "team_a"]
        xa, xb = b[4:19], b[19:34]
        if not a_first:
            xa, xb = xb, xa
        box_a[i] = pd.to_numeric(pd.Series(xa), errors="coerce").to_numpy()
        box_b[i] = pd.to_numeric(pd.Series(xb), errors="coerce").to_numpy()
        minutes[i] = pd.to_numeric(b[1], errors="coerce")
    for j, c in enumerate(BOX):
        out[f"a_{c}"] = box_a[:, j]
        out[f"b_{c}"] = box_b[:, j]
    out["minutes"] = minutes

    # T-Rank's own pre-game prediction (benchmark only)
    fav, prob = [], []
    for v in s[54].fillna(""):
        try:
            lst = ast.literal_eval(v) if v else None
            f, p = _parse_pred(lst[2]) if lst else (None, None)
        except Exception:
            f, p = None, None
        fav.append(f)
        prob.append(p)
    fav = pd.Series(fav)
    prob = pd.Series(prob, dtype=float)
    out["torvik_p_a"] = np.where(fav == out["team_a"], prob, np.where(fav == out["team_b"], 1 - prob, np.nan))
    return out


def probe(season: int) -> bytes | None:
    """Fresh download of the current super_sked, or None if Torvik refuses (no stale fallback)."""
    import os

    from ..http import get

    try:
        if os.environ.get("MMAI_FORCE_ESPN"):
            raise RuntimeError("MMAI_FORCE_ESPN set")
        raw = get(f"{BASE}/{season}_super_sked.csv", retries=2).content
    except Exception as e:
        print(f"  Torvik unavailable ({e}); switching current season to ESPN")
        return None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{season}_super_sked.csv").write_bytes(raw)
    return raw


def player_games(season: int) -> pd.DataFrame:
    raw = cached_bytes(f"{BASE}/{season}_all_advgames.json.gz", OUT / f"{season}_all_advgames.json.gz",
                       refresh=_refresh(season))
    try:
        data = json.loads(gzip.decompress(raw))
    except OSError:  # server already decompressed
        data = json.loads(raw)
    df = pd.DataFrame(data)
    df = df.iloc[:, : len(PLAYER_GAME_COLS)]
    df.columns = PLAYER_GAME_COLS[: df.shape[1]]
    df["date"] = pd.to_datetime(df["numdate"].astype(str), format="%Y%m%d", errors="coerce")
    num = ["min", "ortg", "usage", "bpm", "obpm", "dbpm", "pts", "poss", "inches", "pid"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["season"] = season
    keep = ["season", "date", "muid", "team", "opponent", "loc", "player", "pid", "cls", "inches",
            "min", "ortg", "usage", "bpm", "obpm", "dbpm", "pts", "poss"]
    return df[keep]


def player_seasons(season: int) -> pd.DataFrame:
    raw = cached_bytes(f"{BASE}/getadvstats.php?year={season}&csv=1", OUT / f"{season}_advstats.csv",
                       refresh=_refresh(season))
    a = pd.read_csv(io.BytesIO(raw), header=None, low_memory=False)
    a = a[[c for c in ADV_COLS if c in a.columns]].rename(columns=ADV_COLS)
    a["season"] = season
    return a


def team_results(season: int) -> pd.DataFrame:
    raw = cached_bytes(f"{BASE}/{season}_team_results.csv", OUT / f"{season}_team_results.csv",
                       refresh=_refresh(season))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = pd.read_csv(io.BytesIO(raw), index_col=False)
    t = t.rename(columns={"team": "team", "adjoe": "t_adjoe", "adjde": "t_adjde", "barthag": "t_barthag"})
    t["season"] = season
    return t[["season", "team", "conf", "t_adjoe", "t_adjde", "t_barthag"]]
