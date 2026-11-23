"""Compact, committed copy of completed seasons (``history/``).

Daily CI runs only download the current season; everything older is read from
these files, so the job never needs to re-pull ~400 MB from the data sources.
Refresh after a season ends with ``python -m mmai history``.
"""
from __future__ import annotations

import pandas as pd

from .config import ROOT, current_season

HIST = ROOT / "history"

PG_COLS = ["season", "muid", "team", "pid", "cls", "inches", "min", "bpm", "obpm", "dbpm"]
PS_COLS = ["season", "player", "team", "pid", "cls", "gp", "mpg", "bpm", "obpm", "dbpm", "rec_rank", "conf"]


def path(kind: str, season: int | None = None):
    return HIST / (f"{kind}_{season}.parquet" if season is not None else f"{kind}.parquet")


def has(kind: str, season: int) -> bool:
    return season < current_season() and path(kind, season).exists()


def _w(df: pd.DataFrame, p):
    HIST.mkdir(exist_ok=True)
    df.to_parquet(p, compression="zstd", compression_level=19, index=False)


def save_games(df: pd.DataFrame, season: int):
    _w(df, path("tv_games", season))


def save_espn(df: pd.DataFrame, season: int):
    keep = [c for c in df.columns if c not in ("start", "venue", "status")]
    _w(df[keep], path("espn_games", season))


def save_pg(df: pd.DataFrame, season: int):
    k = df[PG_COLS].copy()
    for c in ("muid", "team", "cls"):
        k[c] = k[c].astype("category")
    k["inches"] = k["inches"].fillna(0).round().astype("int8")
    # minutes kept to 0.1; box-score plus/minus to 0.1
    k["min"] = (k["min"] * 10).round().clip(0, 32000).astype("int16")
    for c in ("bpm", "obpm", "dbpm"):
        k[c] = (k[c].fillna(0) * 10).round().clip(-32000, 32000).astype("int16")
    k["pid"] = k["pid"].fillna(-1).astype("int32")
    k["season"] = k["season"].astype("int16")
    _w(k.sort_values(["team", "muid", "pid"]), path("pg", season))


def load_pg(season: int) -> pd.DataFrame:
    k = pd.read_parquet(path("pg", season))
    k["season"] = k["season"].astype(int)
    for c in ("muid", "team"):
        k[c] = k[c].astype(str)
    k["cls"] = k["cls"].astype("string")
    k["inches"] = k["inches"].astype(float).replace(0, float("nan"))
    k["min"] = k["min"].astype(float) / 10
    for c in ("bpm", "obpm", "dbpm"):
        k[c] = k[c].astype(float) / 10
    k["pid"] = k["pid"].astype(float)
    return k


def save_ps(df: pd.DataFrame, season: int):
    k = df[[c for c in PS_COLS if c in df.columns]].copy()
    for c in ("gp", "mpg", "bpm", "obpm", "dbpm", "rec_rank"):
        if c in k:
            k[c] = pd.to_numeric(k[c], errors="coerce").astype("float32")
    _w(k, path("ps", season))


def load_ps(season: int) -> pd.DataFrame:
    k = pd.read_parquet(path("ps", season))
    for c in k.columns:
        if str(k[c].dtype) == "float32":
            k[c] = k[c].astype(float)
    return k


def save_conf(conf: pd.DataFrame):
    _w(conf, path("conf"))


def export(seasons=None):
    """Write the compact history for all completed seasons from the local caches."""
    from . import players as PL
    from .config import FIRST_SEASON
    from .data import espn_games, torvik_games
    from .features import conferences

    seasons = seasons or list(range(FIRST_SEASON, current_season()))
    for s in seasons:
        save_games(torvik_games([s], refresh_current=False), s)
        save_espn(espn_games([s], refresh_current=False), s)
        save_pg(PL.load_player_games([s], refresh_current=False), s)
        save_ps(PL.load_player_seasons([s], refresh_current=False), s)
        print("history", s, flush=True)
    save_conf(conferences(seasons))
