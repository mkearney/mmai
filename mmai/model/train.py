"""Training, inference and backtesting for the matchup network."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import torch

from ..features import STYLE, TEAM_FEATS
from .net import CTX, MatchupNet, gaussian_nll

torch.set_num_threads(4)

CENTER_SEASON = ["eff_o", "eff_d"] + [f"{s}_{side}" for s in STYLE for side in ("o", "d")] + ["adjt"]
EPOCHS = 14
ROSTER_COLS = ["roster", "roster_o", "roster_d", "avail_delta", "height", "experience", "star", "bench", "continuity"]
ROSTER_IDX = torch.tensor([TEAM_FEATS.index(c) for c in ROSTER_COLS])
ROSTER_DROPOUT = 0.0  # tested 0.05-0.15: hurts accuracy; missing roster is imputed instead
N_MODELS = 5


def season_means(feat: pd.DataFrame) -> pd.DataFrame:
    """Per-season league means of the centred columns (from both sides of every game)."""
    cols = {}
    for c in CENTER_SEASON:
        both = pd.concat([feat[["season", f"a_{c}"]].rename(columns={f"a_{c}": c}),
                          feat[["season", f"b_{c}"]].rename(columns={f"b_{c}": c})])
        cols[c] = both.groupby("season")[c].mean()
    return pd.DataFrame(cols)


def team_matrix(df: pd.DataFrame, side: str, smeans: pd.DataFrame) -> np.ndarray:
    out = []
    sm = smeans.reindex(df["season"].to_numpy())
    for c in TEAM_FEATS:
        v = df[f"{side}_{c}"].to_numpy(float).copy()
        if c in CENTER_SEASON:
            v = v - sm[c].to_numpy()
        elif c == "elo":
            v = v - 1500.0
        elif c == "games":
            v = np.log1p(np.nan_to_num(v))
        elif c == "rest":
            v = np.clip(v, 0, 10)
        out.append(v)
    return np.column_stack(out)


def ctx_matrix(df: pd.DataFrame) -> np.ndarray:
    d = df.copy()
    d["neutral"] = (d["loc"] == 0).astype(float)
    d["is_ncaa"] = d["is_ncaa"].astype(float)
    d["day"] = d["day"] / 100.0
    return d[CTX].to_numpy(float)


class Normalizer:
    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, xa, xb):
        x = np.vstack([xa, xb])
        self.mean = np.nanmean(x, 0)
        self.std = np.nanstd(x, 0) + 1e-6
        return self

    def __call__(self, x):
        z = (x - self.mean) / self.std
        return np.nan_to_num(np.clip(z, -5, 5))

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def prepare(df, smeans, norm=None):
    xa = team_matrix(df, "a", smeans)
    xb = team_matrix(df, "b", smeans)
    if norm is None:
        norm = Normalizer().fit(xa, xb)
    return norm(xa), norm(xb), ctx_matrix(df), norm


def train_models(df: pd.DataFrame, smeans, n_models=N_MODELS, epochs=EPOCHS, seed=0, verbose=False,
                 val: pd.DataFrame | None = None):
    df = df[df["played"] & df["margin"].notna()]
    xa, xb, cx, norm = prepare(df, smeans)
    y = df["margin"].to_numpy(float)
    w = np.where(df["is_post"] > 0, 1.5, 1.0) * np.where((df["a_games"] < 3) | (df["b_games"] < 3), 0.6, 1.0)
    T = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    xa_t, xb_t, cx_t, y_t, w_t = map(T, (xa, xb, cx, y, w))
    if val is not None:
        vxa, vxb, vcx, _ = prepare(val, smeans, norm)
        vy = val["margin"].to_numpy(float)
    models = []
    n = len(y)
    for m in range(n_models):
        torch.manual_seed(seed * 100 + m)
        rng = np.random.default_rng(seed * 100 + m)
        net = MatchupNet(xa.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * math.ceil(n / 1024))
        for ep in range(epochs):
            net.train()
            perm = rng.permutation(n)
            # random side swap = data augmentation (model is antisymmetric anyway)
            for i in range(0, n, 1024):
                idx = perm[i:i + 1024]
                ba, bb = xa_t[idx].clone(), xb_t[idx].clone()
                # feature-group dropout: teach the net to cope with missing player data
                for bx in (ba, bb) if ROSTER_DROPOUT > 0 else ():
                    drop = torch.rand(len(idx)) < ROSTER_DROPOUT
                    bx[drop.nonzero(as_tuple=True)[0][:, None], ROSTER_IDX] = 0.0
                mu, sg = net(ba, bb, cx_t[idx])
                loss = gaussian_nll(mu, sg, y_t[idx], w_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            if verbose and val is not None:
                mu, sg = predict_raw([net], vxa, vxb, vcx)
                p = norm_cdf(mu / sg)
                ll = log_loss(vy > 0, p)
                print(f"  model {m} epoch {ep}: val rmse {np.sqrt(np.mean((vy - mu) ** 2)):.3f} ll {ll:.4f}")
        net.eval()
        models.append(net)
    return models, norm


def predict_raw(models, xa, xb, cx):
    T = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    mus, sgs = [], []
    with torch.no_grad():
        for net in models:
            net.eval()
            mu, sg = net(T(xa), T(xb), T(cx))
            mus.append(mu.numpy())
            sgs.append(sg.numpy())
    mu = np.mean(mus, 0)
    # mixture variance: within + between model spread
    var = np.mean(np.square(sgs), 0) + np.var(mus, 0)
    return mu, np.sqrt(var)


def predict_df(models, norm, smeans, df):
    xa, xb, cx, _ = prepare(df, smeans, norm)
    mu, sg = predict_raw(models, xa, xb, cx)
    return mu, sg, norm_cdf(mu / sg)


def norm_cdf(z):
    from scipy.stats import norm

    return norm.cdf(z)


def log_loss(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------------------------------------------------------- persistence / web
def save(models, norm, smeans, path):
    torch.save({"state": [m.state_dict() for m in models], "d_team": len(TEAM_FEATS),
                "norm": norm.to_dict(), "smeans": smeans.reset_index().to_dict(orient="list")}, path)


def load(path):
    blob = torch.load(path, weights_only=False)
    models = []
    for st in blob["state"]:
        m = MatchupNet(blob["d_team"])
        m.load_state_dict(st)
        m.eval()
        models.append(m)
    norm = Normalizer(np.array(blob["norm"]["mean"]), np.array(blob["norm"]["std"]))
    smeans = pd.DataFrame(blob["smeans"]).set_index("season")
    return models, norm, smeans


def export_web(models, norm, path):
    """Weights as JSON for the in-browser matchup calculator."""
    def r(t):
        return np.round(t.detach().numpy().astype(float), 5).tolist()

    nets = []
    for m in models:
        nets.append({
            "enc0": [r(m.enc[0].weight), r(m.enc[0].bias)], "enc1": [r(m.enc[3].weight), r(m.enc[3].bias)],
            "pair0": [r(m.pair[0].weight), r(m.pair[0].bias)], "pair1": [r(m.pair[3].weight), r(m.pair[3].bias)],
            "skip": r(m.skip.weight)[0], "hca": float(m.hca.detach()),
            "sig0": [r(m.sig[0].weight), r(m.sig[0].bias)], "sig1": [r(m.sig[2].weight), r(m.sig[2].bias)],
        })
    blob = {"features": TEAM_FEATS, "ctx": CTX, "norm": norm.to_dict(), "nets": nets}
    with open(path, "w") as f:
        json.dump(blob, f, separators=(",", ":"))
