"""NCAA bracket: build (actual or projected), exact advancement probabilities, picks.

Works for any field size: each of the 64 main-bracket slots holds a team or an
opening-round game (68-team era: 4 such games; 2027's 76-team field: 12).
Advancement probabilities are computed exactly by propagating win distributions up
the tree (no Monte Carlo noise); each game uses venue-specific travel distances.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from . import geo
from .config import OVERRIDES, current_season
from .data import ROUND_NAMES
from .predict import predict_pairs

SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
POD_OF_SEED = {1: 0, 16: 0, 8: 0, 9: 0, 5: 1, 12: 1, 4: 1, 13: 1, 6: 2, 11: 2, 3: 2, 14: 2, 7: 3, 10: 3, 2: 3, 15: 3}
POD_TOP = {0: 1, 1: 4, 2: 3, 3: 2}


def load_config(season: int | None = None) -> dict:
    season = season or current_season()
    p = OVERRIDES / f"bracket_{season}.yaml"
    if p.exists():
        return yaml.safe_load(p.read_text()) or {}
    return {}


def _site(s):
    if s is None:
        return None
    if isinstance(s, dict) and "lat" in s:
        return s
    city, state = s.split(",") if isinstance(s, str) else s
    ll = geo.geocode(city.strip(), state.strip())
    return {"city": f"{city.strip()}, {state.strip()}", "lat": ll[0] if ll else np.nan, "lon": ll[1] if ll else np.nan}


@dataclass
class Node:
    round: int
    site: dict | None
    left: "Node | None" = None
    right: "Node | None" = None
    team: str | None = None
    region: str | None = None
    seed: int | None = None
    slot: str = ""
    dist: dict = field(default_factory=dict)
    pick: str | None = None
    pick_prob: float | None = None
    game_probs: dict = field(default_factory=dict)


# --------------------------------------------------------------------- projection
def project_field(state: pd.DataFrame, power: pd.DataFrame, games: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Auto bids (conf. tournament champion if decided, else best power) + best at-larges."""
    n_field = int(cfg.get("field_size", 76))
    n_auto_open = int(cfg.get("opening_auto", 12 if n_field == 76 else 2))
    n_large_open = int(cfg.get("opening_at_large", 12 if n_field == 76 else 6))
    st = state.copy() if "power" in state.columns else state.join(power[["power"]], how="inner")
    st = st[st["conf"].notna() & (st["conf"] != "ind")]
    champs = conf_champions(games, st)
    auto = {}
    for c, grp in st.groupby("conf"):
        if c in champs:
            auto[c] = champs[c]
        else:
            auto[c] = grp["power"].idxmax()
    auto_teams = set(auto.values())
    n_large = n_field - len(auto_teams)
    pool = st[~st.index.isin(auto_teams)].sort_values("power", ascending=False)
    at_large = list(pool.index[:n_large])
    fld = st.loc[list(auto_teams) + at_large].copy()
    fld["bid"] = ["auto" if t in auto_teams else "at-large" for t in fld.index]
    fld = fld.sort_values("power", ascending=False)
    fld["overall"] = np.arange(1, len(fld) + 1)
    # opening round: lowest-ranked autos and at-larges
    autos = fld[fld["bid"] == "auto"].sort_values("overall")
    larges = fld[fld["bid"] == "at-large"].sort_values("overall")
    fld["opening"] = False
    fld.loc[autos.index[-n_auto_open:] if n_auto_open else [], "opening"] = True
    fld.loc[larges.index[-n_large_open:] if n_large_open else [], "opening"] = True
    return fld


def conf_champions(games: pd.DataFrame, st: pd.DataFrame) -> dict:
    season = int(st["season"].iloc[0])
    g = games[(games["season"] == season) & (games["type"] == "conftourney")]
    if g.empty:
        return {}
    out = {}
    cmap = st["conf"].to_dict()
    for c in set(cmap.values()):
        gc = g[g["team_a"].map(cmap).eq(c) & g["team_b"].map(cmap).eq(c)]
        if gc.empty or (~gc["played"]).any():
            continue
        last = gc.sort_values("date").iloc[-1]
        if last["date"] < gc["date"].max():
            continue
        out[c] = last["team_a"] if last["pts_a"] > last["pts_b"] else last["team_b"]
    return out


def projected_bracket(fld: pd.DataFrame, cfg: dict) -> dict:
    """S-curve seeding into 4 regions; opening games placed on the lowest lines they belong to."""
    regions = cfg.get("regions", DEFAULT_SITES["regions"])
    names = [r["name"] for r in regions]
    rsites = {r["name"]: _site(r["site"]) for r in regions}
    direct = fld[~fld["opening"]].sort_values("overall")
    entries = [(row.overall, [t]) for t, row in direct.iterrows()]
    for bid in ("at-large", "auto"):
        o = fld[fld["opening"] & (fld["bid"] == bid)].sort_values("overall")
        teams = list(o.index)
        for i in range(0, len(teams) - 1, 2):
            pair = teams[i:i + 2]
            entries.append((o.loc[pair, "overall"].mean(), pair))
    entries.sort(key=lambda x: x[0])
    entries = entries[:64]
    # 1-seeds choose the closest regional site in S-curve order
    one_seeds = [e[1][0] for e in entries[:4]]
    order = []
    free = list(names)
    for t in one_seeds:
        lat, lon = fld.at[t, "lat"], fld.at[t, "lon"]
        best = min(free, key=lambda r: geo.haversine_km(lat, lon, rsites[r]["lat"], rsites[r]["lon"])
                   if not np.isnan(lat) else 0)
        free.remove(best)
        order.append(best)
    slots = {r: {} for r in names}
    for i, (_, teams) in enumerate(entries):
        line = i // 4 + 1
        k = i % 4
        reg = order[k] if line % 2 == 1 else order[3 - k]
        slots[reg][line] = teams
    return {"regions": [{"name": r, "site": rsites[r], "slots": slots[r]} for r in names],
            "final_four": cfg.get("final_four_pairs", [[names[0], names[1]], [names[2], names[3]]]),
            "projected": True}


def assign_pods(br: dict, state: pd.DataFrame, cfg: dict):
    """First/second-round sites: protected seeds (1-4) take the nearest site with capacity."""
    sites = [_site(s) for s in cfg.get("first_round_sites", DEFAULT_SITES["first_round_sites"])]
    cap = {i: 2 for i in range(len(sites))}
    pods = []
    for reg in br["regions"]:
        for pod, top in POD_TOP.items():
            team = reg["slots"].get(top, [None])[0]
            pods.append((top, reg["name"], pod, team))
    pods.sort(key=lambda x: x[0])
    out = {}
    for top, rname, pod, team in pods:
        lat = state.at[team, "lat"] if team in state.index else np.nan
        lon = state.at[team, "lon"] if team in state.index else np.nan
        cands = [i for i in cap if cap[i] > 0]
        best = min(cands, key=lambda i: geo.haversine_km(lat, lon, sites[i]["lat"], sites[i]["lon"])
                   if not np.isnan(lat) else i)
        cap[best] -= 1
        out[(rname, pod)] = sites[best]
    return out


# ---------------------------------------------------------------------- ESPN actual
def actual_bracket(tourney: pd.DataFrame, name2team: dict, cfg: dict) -> dict | None:
    """Build from ESPN tournament games once the bracket is out (opening + first-round games)."""
    if tourney is None or tourney.empty:
        return None
    t = tourney.copy()
    from .data import ncaa_region, ncaa_round
    t["round"] = [ncaa_round(n, int(s)) for n, s in zip(t["notes"], t["season"])]
    t["region"] = t["notes"].map(ncaa_region)
    r1 = t[t["round"] == 1]
    if len(r1) < 32:
        return None
    regions = {}
    pod_sites = {}
    for r in t[t["round"].isin([0, 1])].itertuples():
        reg = regions.setdefault(r.region, {})
        for side in ("home", "away"):
            team = name2team.get(getattr(r, f"{side}_id"))
            seed = getattr(r, f"{side}_rank")
            if team is None or pd.isna(seed):
                continue
            seed = int(seed)
            lst = reg.setdefault(seed, [])
            if team not in lst:
                lst.append(team)
        if r.round == 1 and r.city:
            for side in ("home", "away"):
                seed = getattr(r, f"{side}_rank")
                if not pd.isna(seed):
                    pod_sites[(r.region, POD_OF_SEED[int(seed)])] = _site(f"{r.city}, {r.state}")
    # a slot whose opening game is finished keeps only the winner
    for r in t[(t["round"] == 0) & (t["completed"].astype(str) == "True")].itertuples():
        winner = r.home_id if str(r.home_winner) == "True" else r.away_id
        loser = r.away_id if winner == r.home_id else r.home_id
        lt = name2team.get(loser)
        for seed, lst in regions.get(r.region, {}).items():
            if lt in lst and len(lst) > 1:
                lst.remove(lt)
    cfg_regions = {x["name"]: x for x in cfg.get("regions", DEFAULT_SITES["regions"])}
    br = {"regions": [], "projected": False, "pod_sites": pod_sites}
    for name, slots in regions.items():
        site = cfg_regions.get(name, {}).get("site")
        br["regions"].append({"name": name, "site": _site(site) if site else None, "slots": slots})
    names = [r["name"] for r in br["regions"]]
    br["final_four"] = cfg.get("final_four_pairs") or [names[:2], names[2:4]]
    return br


# ------------------------------------------------------------------- probabilities
ROUND_DATES_2027 = {0: "2027-03-16", 1: "2027-03-19", 2: "2027-03-21", 3: "2027-03-26", 4: "2027-03-28",
                    5: "2027-04-03", 6: "2027-04-05"}
REST = {0: 4, 1: 4, 2: 2, 3: 5, 4: 2, 5: 6, 6: 2}


def build_tree(br: dict, pod_sites: dict, cfg: dict) -> Node:
    ff_site = _site(cfg.get("final_four_site", DEFAULT_SITES["final_four_site"]))
    open_sites = [_site(s) for s in cfg.get("opening_sites", DEFAULT_SITES["opening_sites"])]
    reg_nodes = {}
    for reg in br["regions"]:
        name = reg["name"]
        leaves = []
        for seed in SEED_ORDER:
            teams = reg["slots"].get(seed) or reg["slots"].get(str(seed)) or []
            if len(teams) == 1:
                leaves.append(Node(0, None, team=teams[0], region=name, seed=seed, slot=f"{name}-{seed}"))
            elif len(teams) >= 2:
                a = Node(0, None, team=teams[0], region=name, seed=seed)
                b = Node(0, None, team=teams[1], region=name, seed=seed)
                leaves.append(Node(0, {"multi": open_sites}, a, b, region=name, seed=seed, slot=f"{name}-{seed}-OR"))
            else:
                leaves.append(Node(0, None, team=None, region=name, seed=seed, slot=f"{name}-{seed}"))
        level = leaves
        rnd = 1
        while len(level) > 1:
            nxt = []
            for i in range(0, len(level), 2):
                if rnd <= 2:
                    pod = POD_OF_SEED[SEED_ORDER[(i * 2 ** (rnd - 1)) % 16]]
                    site = pod_sites.get((name, pod)) or reg["site"]
                else:
                    site = reg["site"]
                nxt.append(Node(rnd, site, level[i], level[i + 1], region=name, slot=f"{name}-R{rnd}-{i // 2}"))
            level = nxt
            rnd += 1
        reg_nodes[name] = level[0]
    (a, b), (c, d) = br["final_four"]
    semi1 = Node(5, ff_site, reg_nodes[a], reg_nodes[b], slot="F4-1")
    semi2 = Node(5, ff_site, reg_nodes[c], reg_nodes[d], slot="F4-2")
    return Node(6, ff_site, semi1, semi2, slot="Final")


def solve(tree: Node, bundle, state: pd.DataFrame, season: int):
    """Fill node.dist = {team: P(win this node)} bottom-up with batched network calls."""
    def rec(node: Node):
        if node.left is None:
            node.dist = {node.team: 1.0} if node.team else {}
            return
        rec(node.left)
        rec(node.right)
        L, R = node.left.dist, node.right.dist
        pairs = [(a, b) for a in L for b in R]
        if not pairs:
            node.dist = dict(L or R)
            return
        site = node.site
        if site and "multi" in site:  # opening round: nearest of the opening sites to both teams
            def tot(s):
                return sum(geo.haversine_km(state.at[t, "lat"], state.at[t, "lon"], s["lat"], s["lon"])
                           for t in (pairs[0][0], pairs[0][1]) if t in state.index)
            site = min(site["multi"], key=tot)
            node.site = site
        df = pd.DataFrame(pairs, columns=["team_a", "team_b"])
        df["loc"] = 0
        df["v_lat"] = site["lat"] if site else np.nan
        df["v_lon"] = site["lon"] if site else np.nan
        df["is_ncaa"] = True
        df["is_post"] = 1.0
        df["is_conftourney"] = 0.0
        date = pd.Timestamp(ROUND_DATES_2027[node.round]).replace(year=season)
        df["day"] = (date - pd.Timestamp(f"{season - 1}-11-01")).days
        df["rest_a"] = df["rest_b"] = REST[node.round]
        pr = predict_pairs(bundle, state, df)
        dist = {}
        for (a, b), p in zip(pairs, pr["p_a"].to_numpy()):
            w = L[a] * R[b]
            dist[a] = dist.get(a, 0.0) + w * p
            dist[b] = dist.get(b, 0.0) + w * (1 - p)
            node.game_probs[(a, b)] = float(p)
        node.dist = dist

    rec(tree)
    return tree


def picks(tree: Node, mode: str = "likely"):
    """'likely': winner of each game = more likely of the two predicted entrants (bottom-up).
    'pool': champion = max title probability, then each sub-bracket's winner = max P(reaching)."""
    if mode == "likely":
        def rec(n: Node):
            if n.left is None:
                n.pick = n.team
                return n.team
            a, b = rec(n.left), rec(n.right)
            if a is None or b is None:
                n.pick = a or b
                n.pick_prob = 1.0
                return n.pick
            p = n.game_probs.get((a, b))
            if p is None:
                p = 1 - n.game_probs.get((b, a), 0.5)
            n.pick = a if p >= 0.5 else b
            n.pick_prob = p if p >= 0.5 else 1 - p
            return n.pick
        rec(tree)
    else:
        def rec(n: Node, forced=None):
            if n.left is None:
                n.pick = n.team
                return
            winner = forced or max(n.dist, key=n.dist.get) if n.dist else None
            n.pick = winner
            n.pick_prob = n.dist.get(winner)
            in_left = winner in n.left.dist
            rec(n.left, winner if in_left else None)
            rec(n.right, winner if not in_left else None)
        rec(tree)
    return tree


def advancement(tree: Node) -> pd.DataFrame:
    """P(team wins a game in round r) for every round, from node distributions."""
    rows = {}
    seeds, regions = {}, {}

    def rec(n: Node):
        if n.left is None:
            if n.team:
                seeds[n.team] = n.seed
                regions[n.team] = n.region
            return
        rec(n.left)
        rec(n.right)
        for t, p in n.dist.items():
            rows.setdefault(t, {})
            rows[t][n.round] = rows[t].get(n.round, 0) + p

    rec(tree)
    df = pd.DataFrame(rows).T.fillna(0.0)
    for r in range(0, 7):
        if r not in df.columns:
            df[r] = 0.0
    df = df[sorted(df.columns)]
    # P(in main bracket): 1 for direct entrants, P(win opening game) otherwise
    df["main"] = [df.at[t, 0] if df.at[t, 0] > 0 else 1.0 for t in df.index]
    df["seed"] = pd.Series(seeds)
    df["region"] = pd.Series(regions)
    df = df.rename(columns={1: "r32", 2: "s16", 3: "e8", 4: "f4", 5: "final", 6: "champ"}).drop(columns=[0])
    return df.sort_values("champ", ascending=False)


def tree_json(tree: Node) -> dict:
    def rec(n: Node):
        d = {"round": n.round, "slot": n.slot, "pick": n.pick,
             "pick_prob": None if n.pick_prob is None else round(float(n.pick_prob), 4)}
        if n.site and "city" in n.site:
            d["site"] = n.site["city"]
        if n.left is None:
            d.update({"team": n.team, "seed": n.seed, "region": n.region})
            return d
        d["region"] = n.region
        top = sorted(n.dist.items(), key=lambda x: -x[1])[:6]
        d["top"] = [[t, round(p, 4)] for t, p in top]
        d["children"] = [rec(n.left), rec(n.right)]
        return d
    return rec(tree)


DEFAULT_SITES = {
    # 2027 sites (per Wikipedia, Oct 2026). Edit overrides/bracket_2027.yaml if they change.
    "regions": [
        {"name": "South", "site": "San Antonio, TX"},
        {"name": "West", "site": "Los Angeles, CA"},
        {"name": "Midwest", "site": "Kansas City, MO"},
        {"name": "East", "site": "New York, NY"},
    ],
    "first_round_sites": ["Pittsburgh, PA", "Omaha, NE", "Fort Worth, TX", "Spokane, WA",
                          "Charlotte, NC", "Minneapolis, MN", "Louisville, KY", "Sacramento, CA"],
    "opening_sites": ["Dayton, OH", "Wichita, KS"],
    "final_four_site": "Detroit, MI",
}
