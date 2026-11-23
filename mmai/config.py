"""Project-wide paths and constants."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
ARTIFACTS = ROOT / "artifacts"
OVERRIDES = ROOT / "overrides"
SITE = ROOT / "docs"
SITE_DATA = SITE / "data"

for _p in (RAW, PROC, ARTIFACTS, SITE_DATA):
    _p.mkdir(parents=True, exist_ok=True)

FIRST_SEASON = 2008  # earliest season Torvik publishes game + player data for

USER_AGENT = os.environ.get(
    "MMAI_UA",
    "mmai-research/0.2 (non-commercial academic research)",
)
# minimum seconds between requests to a host (barttorvik.com asks for a 10s crawl delay)
HOST_MIN_INTERVAL = {"barttorvik.com": 10.0}


def today() -> dt.date:
    override = os.environ.get("MMAI_TODAY")
    if override:
        return dt.date.fromisoformat(override)
    return dt.date.today()


def season_for(d: dt.date) -> int:
    """Season label = calendar year in which the season ends (2026-27 -> 2027)."""
    return d.year + 1 if d.month >= 8 else d.year


def current_season() -> int:
    return season_for(today())


def season_dates(season: int) -> tuple[dt.date, dt.date]:
    return dt.date(season - 1, 11, 1), dt.date(season, 4, 12)
