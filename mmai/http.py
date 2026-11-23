"""Tiny polite HTTP layer with retries and on-disk caching."""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import requests

from .config import USER_AGENT

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})


def get(url: str, *, timeout: int = 60, retries: int = 4, backoff: float = 2.0) -> requests.Response:
    last = None
    for i in range(retries):
        try:
            r = _session.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (404, 403):
                r.raise_for_status()
            last = RuntimeError(f"HTTP {r.status_code} for {url}")
        except requests.HTTPError:
            raise
        except Exception as e:  # network hiccup
            last = e
        time.sleep(backoff * (2**i))
    raise RuntimeError(f"failed GET {url}: {last}")


def cached_bytes(url: str, path: Path, *, refresh: bool = False) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh:
        return path.read_bytes()
    try:
        content = get(url).content
    except Exception as e:
        if path.exists():  # source unavailable: fall back to the last good copy
            print(f"  WARNING: refresh failed for {url} ({e}); using cached copy")
            return path.read_bytes()
        raise
    path.write_bytes(content)
    return content


def cached_json(url: str, path: Path, *, refresh: bool = False):
    """JSON cached gzip-compressed on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh:
        with gzip.open(path, "rt", encoding="utf8") as f:
            return json.load(f)
    data = get(url).json()
    with gzip.open(path, "wt", encoding="utf8") as f:
        json.dump(data, f)
    return data
