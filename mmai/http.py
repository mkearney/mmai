"""Tiny polite HTTP layer with retries and on-disk caching."""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import requests

import threading
from urllib.parse import urlparse

from .config import HOST_MIN_INTERVAL, USER_AGENT

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
_last: dict[str, float] = {}
_lock = threading.Lock()


def _throttle(url: str):
    """Honor per-host crawl delays (requests to such hosts are serialized)."""
    host = urlparse(url).hostname or ""
    gap = next((v for h, v in HOST_MIN_INTERVAL.items() if host == h or host.endswith("." + h)), 0.0)
    if not gap:
        return
    with _lock:
        wait = _last.get(host, 0.0) + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[host] = time.time()


def get(url: str, *, timeout: int = 60, retries: int = 4, backoff: float = 2.0) -> requests.Response:
    last = None
    for i in range(retries):
        try:
            _throttle(url)
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
