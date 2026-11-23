"""End-to-end pipeline: data -> features -> (train) -> predictions -> site JSON."""
from __future__ import annotations

import datetime as dt
import json
import math
import pickle
import re
import time

import numpy as np
import pandas as pd

from . import bracket as BR
from . import data as D
from . import features as F
from . import injuries as INJ
from . import players as PL
from . import predict as PR
from .config import ARTIFACTS, FIRST_SEASON, PROC, SITE_DATA, current_season, today
from .fetch import espn
from .model import backtest as BT
from .model import train as TR

MODEL_PATH = ARTIFACTS / "model.pt"


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def _clean(v):
    if v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (np.floating, float)):
        return None if (math.isnan(v) or math.isinf(v)) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (pd.Timestamp, dt.date)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict(orient="records")]


def write_json(name, obj):
    p = SITE_DATA / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, separators=(",", ":"), default=_clean)


# ------------------------------------------------------------------------ stages
def stage_data(refresh=True):
    seasons = list(range(FIRST_SEASON, current_season() + 1))
    log("fetching games (Torvik + ESPN)")
    games, meta = D.build(seasons, refresh_current=refresh)
    from . import espn_season, history
    cur = current_season()
    log("player data (history + Torvik snapshot + ESPN box scores)")
    cw = pd.read_csv(ARTIFACTS / "crosswalk.csv", dtype={"espn_id": str})
    espn_season.current_player_games(cur, games, cw)
    pg = PL.load_player_games(seasons, refresh_current=refresh)
    ps = PL.load_player_seasons(seasons, refresh_current=refresh)
    conf = F.conferences(seasons)
    if not espn_season.active(cur) and (PROC / f"tv_games_{cur}.parquet").exists():
        # games + player-season table only; the player-game snapshot stays frozen (no json fetches)
        history.save_snapshot(cur, pd.read_parquet(PROC / f"tv_games_{cur}.parquet"), None,
                              ps[ps["season"] == cur])
        log("updated Torvik snapshot (games, player seasons) in history/")
    return games, meta, pg, ps, conf


def stage_features(games, meta, pg, ps, conf):
    season = current_season()
    cur = pg[pg["season"] == season][["team", "pid", "player"]].drop_duplicates("pid", keep="last")
    log("injury report")
    avail, inj_report = INJ.availability(meta, cur)
    log(f"  {len(inj_report)} injury rows, {len(avail)} matched to players")
    log("building point-in-time features")
    feat, snaps, live, extras = F.build(games, pg, ps, conf, availability=avail)
    feat.to_parquet(PROC / "features.parquet")
    snaps.to_parquet(PROC / "snapshots.parquet")
    with open(PROC / "live_extras.pkl", "wb") as f:
        pickle.dump((live, extras, inj_report), f)
    return feat, snaps, live, extras, inj_report


def stage_train(feat, n_models=TR.N_MODELS):
    # completed seasons only -> current-season tracking on the site is genuinely out-of-sample
    f = feat[(feat["season"] >= 2009) & (feat["season"] < current_season()) & feat["played"] & feat["margin"].notna()]
    smeans = TR.season_means(feat[feat["season"] >= 2009])
    log(f"training {n_models} networks on {len(f):,} games")
    models, norm = TR.train_models(f, smeans, n_models=n_models)
    TR.save(models, norm, smeans, MODEL_PATH)
    return models, norm, smeans


def stage_backtest(feat):
    log("rolling backtest on NCAA tournaments")
    return BT.run(feat)


# ------------------------------------------------------------------------ publish
def stage_publish(games, meta, pg, conf, feat, snaps, live, extras, inj_report):
    season = current_season()
    models, norm, _ = TR.load(MODEL_PATH)
    smeans = TR.season_means(feat[feat["season"] >= 2009])
    bundle = (models, norm, smeans)
    state = PR.team_state(snaps, extras, conf, meta, season)
    log(f"power ratings for {len(state)} teams")
    power, (teams, M, P) = PR.power_ratings(bundle, state)
    st = state.join(power)
    st["slug"] = [slug(t) for t in st.index]

    # ranks for headline metrics
    for c, asc in (("adjem", False), ("eff_o", False), ("eff_d", True), ("adjt", False), ("elo", False),
                   ("roster", False)):
        st[f"{c}_rank"] = st[c].rank(ascending=asc, method="min")

    # records / SOS from played games
    gcur = games[(games["season"] == season) & (games["type"] != "nond1")]
    played = gcur[gcur["played"]]
    rec = {}
    for r in played.itertuples():
        for t, won in ((r.team_a, r.pts_a > r.pts_b), (r.team_b, r.pts_b > r.pts_a)):
            w, l = rec.get(t, (0, 0))
            rec[t] = (w + won, l + (not won))
    st["wins"] = [rec.get(t, (0, 0))[0] for t in st.index]
    st["losses"] = [rec.get(t, (0, 0))[1] for t in st.index]
    opp_power = st["power"].to_dict()
    sos = {}
    for t in st.index:
        opps = list(played.loc[played["team_a"] == t, "team_b"]) + list(played.loc[played["team_b"] == t, "team_a"])
        vals = [opp_power[o] for o in opps if o in opp_power]
        sos[t] = float(np.mean(vals)) if vals else np.nan
    st["sos"] = pd.Series(sos)
    st["sos_rank"] = st["sos"].rank(ascending=False, method="min")

    # trend: rank change vs a week ago (AdjEM ranks; power history is NN-heavy)
    s = snaps[snaps["season"] == season]
    dates = sorted(s["snap"].unique())
    if len(dates) > 7:
        wk = s[s["snap"] == dates[-8]].set_index("team")["adjem"].rank(ascending=False)
        st["adjem_rank_week_ago"] = wk.reindex(st.index)

    # live injury impact in points (per game at team tempo)
    st["injury_pts"] = st["avail_delta"].fillna(0) * st["adjt"] / 100

    cols = ["slug", "rank", "power", "power_net", "win_vs_avg", "win_vs_top25", "conf", "wins", "losses", "eff_o", "eff_d",
            "adjem", "adjt", "elo", "roster", "avail_delta", "injury_pts", "sos", "games", "form",
            "eff_o_rank", "eff_d_rank", "adjem_rank", "adjt_rank", "elo_rank", "roster_rank", "sos_rank",
            "adjem_rank_week_ago", "espn_id", "location", "abbr", "color", "alt_color", "logo", "height", "experience",
            "continuity", "star", "bench"] + [f"{x}_{y}" for x in F.STYLE for y in ("o", "d")]
    cols = [c for c in cols if c in st.columns]
    out = st[cols].copy()
    out.insert(0, "team", out.index)
    write_json("ratings.json", {"season": season, "updated": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                                "teams": records(out.sort_values("rank"))})

    # ---- web model (normalised team vectors + weights)
    TR.export_web(models, norm, SITE_DATA / "model.json")
    vec_df = PR.pair_frame(state, pd.DataFrame({"team_a": state.index, "team_b": state.index, "loc": 0}))
    xa = TR.team_matrix(vec_df, "a", smeans)
    z = norm(xa)
    feats = F.TEAM_FEATS
    vectors = {t: [round(float(v), 4) for v in z[i]] for i, t in enumerate(state.index)}
    venues = venue_list(st)
    write_json("vectors.json", {"features": feats, "teams": vectors,
                                "loc": {t: [_clean(st.at[t, "lat"]), _clean(st.at[t, "lon"])] for t in st.index},
                                "conf": {t: st.at[t, "conf"] for t in st.index},
                                "day": int((pd.Timestamp(today()) - pd.Timestamp(f"{season - 1}-11-01")).days),
                                "venues": venues})

    # ---- games: upcoming predictions + recent results with pre-game predictions
    log("game predictions")
    up = PR.upcoming(bundle, state, gcur, days=7)
    rec_feat = feat[(feat["season"] == season) & feat["played"]].copy()
    rec_feat["a_avail_delta"] = rec_feat["a_avail_delta_pre"]  # what was knowable pre-game
    rec_feat["b_avail_delta"] = rec_feat["b_avail_delta_pre"]
    if len(rec_feat):
        mu, sg, pr = TR.predict_df(models, norm, smeans, rec_feat)
        rec_feat["pred_margin"], rec_feat["p_a"] = mu, pr
    gcols = ["date", "team_a", "team_b", "loc", "city", "p_a", "margin", "pts_a", "pts_b"]
    upcoming = records(up[gcols].sort_values("date")) if len(up) else []
    recent = []
    if len(rec_feat):
        rf = rec_feat.sort_values("date")
        rf = rf.rename(columns={"margin": "actual_margin"})
        recent = records(rf[["date", "team_a", "team_b", "loc", "city", "pts_a", "pts_b", "p_a", "pred_margin"]])
    acc = None
    if len(rec_feat):
        y = rec_feat["margin"] > 0
        acc = {"n": int(len(y)), "accuracy": float(np.mean((rec_feat["p_a"] > 0.5) == y)),
               "log_loss": TR.log_loss(y, rec_feat["p_a"]), "mae": float(np.mean(np.abs(rec_feat["margin"] - rec_feat["pred_margin"])))}
    write_json("games.json", {"upcoming": upcoming, "recent": recent[-400:], "season_accuracy": acc})

    # ---- per-team pages
    log("team pages")
    write_team_pages(st, snaps, season, live, pg, inj_report, gcur, rec_feat, up)

    # ---- bracket
    log("bracket")
    br_out = make_bracket(bundle, st, power, games, meta, season)
    write_json("bracket.json", br_out)

    # ---- backtest + meta
    bt = ARTIFACTS / "backtest.json"
    if bt.exists():
        write_json("backtest.json", json.loads(bt.read_text()))
    from . import espn_season
    source = "espn" if espn_season.active(season) else "torvik"
    write_json("meta.json", {
        "updated": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "season": season, "source": source,
        "torvik_through": espn_season.snapshot_date(season),
        "n_teams": int(len(st)), "n_games_train": int(feat["played"].sum()),
        "seasons": [int(feat["season"].min()), int(feat["season"].max())],
        "injuries": records(inj_report.drop(columns=["pid"], errors="ignore")) if len(inj_report) else [],
        "season_accuracy": acc,
    })
    log("publish done")


def venue_list(st):
    v = []
    for name, lst in (("Opening round", BR.DEFAULT_SITES["opening_sites"]),
                      ("First/second round", BR.DEFAULT_SITES["first_round_sites"]),
                      ("Regional", [r["site"] for r in BR.DEFAULT_SITES["regions"]]),
                      ("Final Four", [BR.DEFAULT_SITES["final_four_site"]])):
        for s in lst:
            site = BR._site(s)
            v.append({"group": name, "city": site["city"], "lat": _clean(site["lat"]), "lon": _clean(site["lon"])})
    return v


def write_team_pages(st, snaps, season, live, pg, inj_report, gcur, rec_feat, up):
    s = snaps[snaps["season"] == season].copy()
    s["rk"] = s.groupby("snap")["adjem"].rank(ascending=False, method="min")
    hist = {t: g for t, g in s.groupby("team")}
    names = pg[pg["season"] == season].drop_duplicates("pid", keep="last").set_index("pid")
    inj = inj_report.set_index("player") if len(inj_report) else None
    pred_lookup = {}
    if len(rec_feat):
        for r in rec_feat.itertuples():
            pred_lookup[r.gid] = (r.p_a, r.pred_margin)
    if len(up):
        for r in up.itertuples():
            pred_lookup[r.gid] = (r.p_a, r.margin)
    for t, row in st.iterrows():
        h = hist.get(t)
        history = []
        if h is not None:
            h = h.sort_values("snap")
            history = [{"date": d.strftime("%Y-%m-%d"), "adjem": _clean(a), "o": _clean(o), "d": _clean(dd),
                        "rank": _clean(r)} for d, a, o, dd, r in zip(h["snap"], h["adjem"], h["eff_o"], h["eff_d"], h["rk"])]
        roster = []
        lp = live.get(t)
        if lp is not None:
            for p in lp.sort_values("share", ascending=False).itertuples():
                if p.minutes <= 0 and p.avail <= 0:
                    continue
                nm = names["player"].get(p.pid, str(p.pid))
                status = None
                if inj is not None and nm in inj.index:
                    r_ = inj.loc[nm]
                    r_ = r_.iloc[0] if isinstance(r_, pd.DataFrame) else r_
                    if r_["team"] == t:
                        status = f"{r_['status']} ({r_['injury']})" if r_["injury"] else r_["status"]
                roster.append({"player": nm, "cls": _clean(names["cls"].get(p.pid)), "inches": _clean(names["inches"].get(p.pid)),
                               "value": _clean(p.value), "o": _clean(p.o_value), "d": _clean(p.d_value),
                               "mpg": _clean(p.minutes / max(p.games, 1)), "games": int(p.games), "share": _clean(p.share),
                               "avail": _clean(p.avail), "status": status})
        sched = []
        tg = gcur[(gcur["team_a"] == t) | (gcur["team_b"] == t)].sort_values("date")
        for r in tg.itertuples():
            is_a = r.team_a == t
            opp = r.team_b if is_a else r.team_a
            loc = r.loc if is_a else -r.loc
            p, m = pred_lookup.get(r.gid, (None, None))
            if p is not None and not is_a:
                p, m = 1 - p, -m
            sched.append({"date": r.date.strftime("%Y-%m-%d"), "opp": opp, "loc": int(loc), "played": bool(r.played),
                          "pts": _clean(r.pts_a if is_a else r.pts_b), "opp_pts": _clean(r.pts_b if is_a else r.pts_a),
                          "p": _clean(p), "margin": _clean(m), "type": r.type})
        write_json(f"teams/{row['slug']}.json", {"team": t, "history": history, "roster": roster, "schedule": sched})


def make_bracket(bundle, st, power, games, meta, season):
    cfg = BR.load_config(season)
    name2team = dict(zip(meta["espn_id"].astype(str), meta["team"]))
    actual = None
    try:
        tg = espn.tournament_games(season)
        actual = BR.actual_bracket(tg, name2team, cfg)
    except Exception as e:
        print("espn tournament lookup failed:", e)
    if cfg.get("bracket"):  # manual bracket in YAML wins
        br = {"regions": cfg["bracket"]["regions"], "final_four": cfg["final_four_pairs"], "projected": False}
        for r in br["regions"]:
            r["site"] = BR._site(r.get("site"))
        pods = BR.assign_pods(br, st, cfg)
        fld = None
    elif actual:
        br = actual
        pods = actual.get("pod_sites") or BR.assign_pods(br, st, cfg)
        fld = None
    else:
        fld = BR.project_field(st, power, games, cfg)
        br = BR.projected_bracket(fld, cfg)
        pods = BR.assign_pods(br, st, cfg)
    tree = BR.build_tree(br, pods, cfg)
    BR.solve(tree, bundle, st, season)
    adv = BR.advancement(tree)
    likely = BR.tree_json(BR.picks(tree, "likely"))
    pool = BR.tree_json(BR.picks(tree, "pool"))
    adv = adv.join(st[["slug", "power", "conf", "logo", "rank"]], how="left")
    adv.insert(0, "team", adv.index)
    out = {"projected": bool(br.get("projected")), "season": season,
           "regions": [r["name"] for r in br["regions"]], "final_four": br["final_four"],
           "advancement": records(adv), "likely": likely, "pool": pool,
           "field_size": int(sum(len(v) for r in br["regions"] for v in r["slots"].values())),
           "note": ("Projected field: auto-bids go to each conference's tournament champion (or highest-rated team "
                    "until decided); at-large bids and seeds follow MMAI power. 76-team format with a 12-game "
                    "Opening Round.") if br.get("projected") else "Official bracket."}
    if fld is not None:
        out["field"] = records(fld.reset_index().rename(columns={"index": "team"})[["team", "conf", "bid", "overall", "opening", "power"]])
    return out


# ------------------------------------------------------------------------ entry
def run(mode="daily"):
    t0 = time.time()
    if mode == "publish":  # re-publish from cached features (fast iteration)
        games = pd.read_parquet(PROC / "games.parquet")
        meta = pd.read_csv(ARTIFACTS / "team_meta.csv", dtype={"espn_id": str})
        seasons = list(range(FIRST_SEASON, current_season() + 1))
        pg = PL.load_player_games(seasons, refresh_current=False)
        conf = F.conferences(seasons)
        feat = pd.read_parquet(PROC / "features.parquet")
        snaps = pd.read_parquet(PROC / "snapshots.parquet")
        with open(PROC / "live_extras.pkl", "rb") as f:
            live, extras, inj = pickle.load(f)
        stage_publish(games, meta, pg, conf, feat, snaps, live, extras, inj)
        log(f"finished in {time.time() - t0:.0f}s")
        return
    games, meta, pg, ps, conf = stage_data(refresh=True)
    feat, snaps, live, extras, inj = stage_features(games, meta, pg, ps, conf)
    if mode in ("train", "all") or not MODEL_PATH.exists():
        stage_train(feat)
    if mode in ("backtest", "all"):
        stage_backtest(feat)
    if not (feat["season"].eq(current_season()) & feat["played"]).any():
        log("no games played yet this season - keeping last published site")
        return
    stage_publish(games, meta, pg, conf, feat, snaps, live, extras, inj)
    log(f"finished in {time.time() - t0:.0f}s")
