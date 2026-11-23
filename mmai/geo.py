"""City geocoding (offline table) + great-circle distance."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import RAW
from .http import cached_bytes

CITIES_URL = "https://raw.githubusercontent.com/kelvins/US-Cities-Database/main/csv/us_cities.csv"

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

# venues outside the US-cities table (MTEs abroad, territories, odd spellings)
MANUAL = {
    ("nassau", ""): (25.06, -77.35), ("paradise island", ""): (25.08, -77.32),
    ("nassau", "bahamas"): (25.06, -77.35), ("paradise island", "bahamas"): (25.08, -77.32),
    ("freeport", "bahamas"): (26.53, -78.70), ("cancun", ""): (21.16, -86.85),
    ("cancun", "mexico"): (21.16, -86.85), ("cancún", "mexico"): (21.16, -86.85),
    ("montego bay", "jamaica"): (18.47, -77.92), ("montego bay", ""): (18.47, -77.92),
    ("san juan", "pr"): (18.47, -66.11), ("san juan", "puerto rico"): (18.47, -66.11),
    ("carolina", "pr"): (18.38, -65.96), ("bayamon", "pr"): (18.40, -66.16),
    ("st. thomas", "vi"): (18.34, -64.93), ("saint thomas", "vi"): (18.34, -64.93),
    ("st. thomas", "virgin islands"): (18.34, -64.93), ("st. thomas", "usvi"): (18.34, -64.93),
    ("charlotte amalie", "vi"): (18.34, -64.93),
    ("vancouver", "bc"): (49.28, -123.12), ("victoria", "bc"): (48.43, -123.37),
    ("toronto", "on"): (43.65, -79.38), ("montreal", "qc"): (45.50, -73.57),
    ("niagara falls", "on"): (43.09, -79.08),
    ("dublin", "ireland"): (53.35, -6.26), ("belfast", "northern ireland"): (54.60, -5.93),
    ("paris", "france"): (48.86, 2.35), ("berlin", "germany"): (52.52, 13.40),
    ("melbourne", "australia"): (-37.81, 144.96), ("sydney", "australia"): (-33.87, 151.21),
    ("tokyo", "japan"): (35.68, 139.69), ("abu dhabi", "uae"): (24.45, 54.38),
    ("honolulu", "hi"): (21.31, -157.86), ("lahaina", "hi"): (20.88, -156.68),
    ("kahului", "hi"): (20.89, -156.47), ("laie", "hi"): (21.65, -157.93),
    ("anchorage", "ak"): (61.22, -149.90), ("fairbanks", "ak"): (64.84, -147.72),
    ("uncasville", "ct"): (41.43, -72.11), ("mashantucket", "ct"): (41.47, -71.97),
    ("estero", "fl"): (26.44, -81.81), ("fort myers", "fl"): (26.64, -81.87),
    ("lake buena vista", "fl"): (28.39, -81.53), ("kissimmee", "fl"): (28.29, -81.41),
    ("brooklyn", "ny"): (40.68, -73.94), ("bronx", "ny"): (40.84, -73.86),
    ("queens", "ny"): (40.73, -73.79), ("jamaica", "ny"): (40.70, -73.80),
    ("staten island", "ny"): (40.58, -74.15), ("new york city", "ny"): (40.71, -74.01),
    ("west point", "ny"): (41.39, -73.96), ("annapolis", "md"): (38.98, -76.49),
    ("usaf academy", "co"): (38.99, -104.86), ("air force academy", "co"): (38.99, -104.86),
    ("colorado springs", "co"): (38.83, -104.82), ("college park", "md"): (38.99, -76.94),
    ("st. louis", "mo"): (38.63, -90.20), ("saint louis", "mo"): (38.63, -90.20),
    ("st. paul", "mn"): (44.95, -93.09), ("st. petersburg", "fl"): (27.77, -82.64),
    ("st. bonaventure", "ny"): (42.08, -78.47), ("st. charles", "mo"): (38.79, -90.48),
    ("st. george", "ut"): (37.10, -113.58), ("st. augustine", "fl"): (29.90, -81.31),
    ("st. cloud", "mn"): (45.56, -94.16), ("st. joseph", "mo"): (39.77, -94.85),
    ("ft. worth", "tx"): (32.76, -97.33), ("ft. myers", "fl"): (26.64, -81.87),
    ("ft. wayne", "in"): (41.08, -85.14), ("ft. lauderdale", "fl"): (26.12, -80.14),
    ("mt. pleasant", "mi"): (43.60, -84.77), ("mt. pleasant", "sc"): (32.79, -79.86),
    ("winston salem", "nc"): (36.10, -80.24), ("winston-salem", "nc"): (36.10, -80.24),
    ("macomb", "il"): (40.46, -90.67), ("riverside", "ca"): (33.95, -117.40),
    ("emmitsburg", "md"): (39.70, -77.33), ("loretto", "pa"): (40.50, -78.63),
    ("moon township", "pa"): (40.52, -80.22), ("chestnut hill", "ma"): (42.33, -71.17),
    ("villanova", "pa"): (40.04, -75.34), ("storrs", "ct"): (41.81, -72.25),
    ("university park", "pa"): (40.80, -77.86), ("state college", "pa"): (40.79, -77.86),
    ("notre dame", "in"): (41.70, -86.24), ("south bend", "in"): (41.68, -86.25),
    ("boiling springs", "nc"): (35.25, -81.67), ("buies creek", "nc"): (35.41, -78.74),
    ("elon", "nc"): (36.10, -79.51), ("cullowhee", "nc"): (35.31, -83.18),
    ("hempstead", "ny"): (40.71, -73.62), ("brookville", "ny"): (40.81, -73.57),
    ("kings point", "ny"): (40.82, -73.74), ("smithfield", "ri"): (41.92, -71.55),
    ("kingston", "ri"): (41.48, -71.52), ("durham", "nh"): (43.13, -70.93),
    ("orono", "me"): (44.88, -68.67), ("lowell", "ma"): (42.63, -71.32),
    ("amherst", "ma"): (42.37, -72.52), ("worcester", "ma"): (42.26, -71.80),
    ("fairfield", "ct"): (41.14, -73.26), ("hamden", "ct"): (41.38, -72.90),
    ("west long branch", "nj"): (40.29, -74.02), ("teaneck", "nj"): (40.89, -74.01),
    ("piscataway", "nj"): (40.55, -74.46), ("south orange", "nj"): (40.75, -74.26),
    ("lawrenceville", "nj"): (40.30, -74.73), ("ewing", "nj"): (40.27, -74.80),
    ("princeton", "nj"): (40.35, -74.66), ("jersey city", "nj"): (40.73, -74.08),
    ("newark", "nj"): (40.74, -74.17), ("east rutherford", "nj"): (40.81, -74.07),
    ("atlantic city", "nj"): (39.36, -74.42), ("glassboro", "nj"): (39.70, -75.11),
    ("saint thomas", "virgin islands"): (18.34, -64.93), ("charlotte amalie", "virgin islands"): (18.34, -64.93),
    ("tortola", "virgin islands"): (18.43, -64.62), ("maui", "hi"): (20.88, -156.68),
    ("george town", "cayman islands"): (19.29, -81.37), ("sunrise", "fl"): (26.13, -80.11),
    ("hot springs", "ar"): (34.50, -93.06), ("college park", "ga"): (33.65, -84.45),
    ("laval", "pq"): (45.57, -73.69), ("laval", "qc"): (45.57, -73.69), ("bimini", "bahamas"): (25.73, -79.28),
    ("bayamon", "puerto rico"): (18.40, -66.16), ("aguadilla", "puerto rico"): (18.43, -67.15),
    ("puerto vallarta", "mexico"): (20.65, -105.23), ("homewood", "al"): (33.47, -86.80),
    ("moncton", "nb"): (46.09, -64.78), ("london", "england"): (51.51, -0.13),
    ("lethbridge", "ab"): (49.69, -112.84), ("shanghai", "china"): (31.23, 121.47),
    ("ashwaubenon", "wi"): (44.48, -88.07), ("highland heights", "ky"): (39.03, -84.45),
    ("coral gables", "fl"): (25.72, -80.27), ("ramstein", "germany"): (49.44, 7.60),
    ("camp humphreys", "south korea"): (36.96, 127.03), ("rome", "italy"): (41.90, 12.50),
    ("wilkes barre township", "pa"): (41.22, -75.88), ("riverdale", "ny"): (40.89, -73.91),
    ("ave maria", "fl"): (26.33, -81.44), ("rosemont", "pa"): (40.03, -75.33),
}


def _norm(s) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


@lru_cache(maxsize=1)
def _cities() -> dict:
    raw = cached_bytes(CITIES_URL, RAW / "geo" / "us_cities.csv")
    import io

    df = pd.read_csv(io.BytesIO(raw))
    df["c"] = df["CITY"].map(_norm)
    df["s"] = df["STATE_CODE"].str.lower()
    g = df.groupby(["c", "s"])[["LATITUDE", "LONGITUDE"]].mean()
    return {k: (float(v.LATITUDE), float(v.LONGITUDE)) for k, v in g.iterrows()}


def geocode(city, state) -> tuple[float, float] | None:
    c, s = _norm(city), _norm(state)
    if s in (x.lower() for x in STATE_ABBR):
        s = STATE_ABBR[next(k for k in STATE_ABBR if k.lower() == s)].lower()
    if (c, s) in MANUAL:
        return MANUAL[(c, s)]
    table = _cities()
    for cand in (c, c.replace("st. ", "saint "), c.replace("saint ", "st. "), c.replace("ft. ", "fort "),
                 c.replace("mt. ", "mount "), c.replace("-", " "), c.split(",")[0]):
        if (cand, s) in table:
            return table[(cand, s)]
    for (mc, ms), v in MANUAL.items():
        if mc == c and (not s or ms == s):
            return v
    return None


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))
