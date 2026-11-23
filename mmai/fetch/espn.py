"""ESPN public (unauthenticated) JSON endpoints.

Used for: NCAA tournament games/seeds/regions, game venues (neutral-site
proximity), team metadata (logos, colors, home city), and the 2027 bracket.
"""
from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ..config import RAW, season_dates, today
from ..http import get

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
OUT = RAW / "espn"


def _compact(ev: dict, qdate: dt.date) -> dict | None:
    try:
        c = ev["competitions"][0]
    except (KeyError, IndexError):
        return None
    comps = {t["homeAway"]: t for t in c.get("competitors", [])}
    if "home" not in comps or "away" not in comps:
        return None
    venue = c.get("venue") or {}
    addr = venue.get("address") or {}
    notes = c.get("notes") or []
    status = (c.get("status") or {}).get("type") or {}
    rec = {
        "event_id": str(ev.get("id")),
        "date": qdate.isoformat(),
        "start": ev.get("date"),
        "season": (ev.get("season") or {}).get("year"),
        "season_type": (ev.get("season") or {}).get("type"),
        "neutral": bool(c.get("neutralSite")),
        "conf_game": bool(c.get("conferenceCompetition")),
        "venue": venue.get("fullName"),
        "city": addr.get("city"),
        "state": addr.get("state"),
        "notes": notes[0].get("headline") if notes else None,
        "tournament_id": c.get("tournamentId"),
        "completed": bool(status.get("completed")),
        "status": status.get("name"),
    }
    for side, t in comps.items():
        team = t.get("team") or {}
        rank = (t.get("curatedRank") or {}).get("current")
        rec[f"{side}_id"] = str(team.get("id"))
        rec[f"{side}_location"] = team.get("location")
        rec[f"{side}_name"] = team.get("displayName")
        rec[f"{side}_abbr"] = team.get("abbreviation")
        rec[f"{side}_score"] = pd.to_numeric(t.get("score"), errors="coerce")
        rec[f"{side}_rank"] = rank if rank and rank < 99 else None
        rec[f"{side}_winner"] = t.get("winner")
    return rec


def scoreboard(d: dt.date, groups: int = 50) -> list[dict]:
    url = f"{BASE}/scoreboard?dates={d:%Y%m%d}&groups={groups}&limit=500"
    data = get(url).json()
    return [r for ev in data.get("events", []) if (r := _compact(ev, d))]


def _date_path(season: int, d: dt.date, groups: int) -> Path:
    return OUT / f"sb{groups}" / str(season) / f"{d:%Y%m%d}.json"


def fetch_season(season: int, groups: int = 50, *, refresh_recent_days: int = 3,
                 lookahead_days: int = 7, workers: int = 6) -> pd.DataFrame:
    """Fetch every scoreboard date of a season (cached per date)."""
    start, end = season_dates(season)
    t = today()
    end = min(end, t + dt.timedelta(days=lookahead_days))
    dates = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]

    def work(d: dt.date):
        p = _date_path(season, d, groups)
        stale = d >= t - dt.timedelta(days=refresh_recent_days)
        if p.exists() and not stale:
            return json.loads(p.read_text())
        try:
            recs = scoreboard(d, groups)
        except Exception as e:  # keep going; a missing day is not fatal
            print(f"  espn {d}: {e}")
            return json.loads(p.read_text()) if p.exists() else []
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(recs, default=str))
        return recs

    with ThreadPoolExecutor(workers) as ex:
        rows = [r for recs in ex.map(work, dates) for r in recs]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates("event_id", keep="last")
    return df


def teams() -> pd.DataFrame:
    data = get(f"{BASE}/teams?limit=1000").json()
    rows = []
    for t in data["sports"][0]["leagues"][0]["teams"]:
        t = t["team"]
        logos = t.get("logos") or []
        rows.append({
            "espn_id": str(t["id"]),
            "location": t.get("location"),
            "name": t.get("displayName"),
            "short": t.get("shortDisplayName"),
            "abbr": t.get("abbreviation"),
            "nickname": t.get("nickname"),
            "color": t.get("color"),
            "alt_color": t.get("alternateColor"),
            "logo": logos[0]["href"] if logos else None,
        })
    return pd.DataFrame(rows)


def tournament_games(season: int) -> pd.DataFrame:
    """NCAA tournament games (ESPN groups=100) for a season, incl. scheduled ones."""
    start = dt.date(season, 3, 10)
    end = dt.date(season, 4, 12)
    df = fetch_season_range(season, start, end, groups=100)
    if df.empty:
        return df
    df = df[df["notes"].fillna("").str.contains("Championship", case=False)]
    df = df[~df["notes"].fillna("").str.contains("Women", case=False)]
    return df.reset_index(drop=True)


def fetch_season_range(season: int, start: dt.date, end: dt.date, groups: int) -> pd.DataFrame:
    t = today()
    end = min(end, t + dt.timedelta(days=30))
    dates = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    rows = []
    for d in dates:
        p = _date_path(season, d, groups)
        if p.exists() and d < t - dt.timedelta(days=3):
            rows += json.loads(p.read_text())
            continue
        try:
            recs = scoreboard(d, groups)
        except Exception as e:
            print(f"  espn {d}: {e}")
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(recs, default=str))
        rows += recs
    df = pd.DataFrame(rows)
    return df.drop_duplicates("event_id", keep="last") if not df.empty else df
