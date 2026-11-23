# MMAI — March Madness AI

Daily men's college basketball **power ratings** and **NCAA tournament predictions**. Everything comes from public data, with no keys or subscriptions. A matchup-aware neural network sits on top of KenPom-style adjusted efficiency, a player-quality layer, and Elo.

**Site:** https://mkearney.github.io/mmai/ (GitHub Pages, served from `docs/`)

| | |
|---|---|
| Ratings | Power (network round-robin), AdjO/AdjD/AdjEM/Tempo, Elo, roster strength, injury adjustment, SOS |
| Bracket | 2027 76-team format; projected until Selection Sunday, then read from ESPN automatically |
| Matchup Lab | any two teams, any venue; the network runs in the browser |
| Backtest | 2012–2026 tournaments: MMAI net log loss **0.526** vs efficiency-only 0.544, Elo 0.565, seeds 0.577; **0.519 vs 0.542** for T-Rank pre-game lines on the same games |

## How it works

See [`PLAN.md`](PLAN.md) and the site's *Method* page. In short:

1. **Data.** barttorvik.com: games with box scores and player-game logs, 2008 onward. ESPN API: venues, seeds, logos. Covers/Rotowire: injuries.
2. **Point-in-time ratings.** Recency-weighted ridge regression for offense, defense and 10 style stats (four factors and friends), opponent- and venue-adjusted. Ratings are re-solved before every game day and anchored by a learned preseason prior: last season plus the values of the incoming roster.
3. **Players and injuries.** Player values come from box-score impact plus a prior. Team roster strength is built from those values. Availability adjustments use injury reports, recent DNPs and manual overrides.
4. **Matchup network.** An antisymmetric PyTorch model with team embeddings and a style-interaction head. It also uses travel distance to the venue, home/neutral context and game-type context, and outputs a margin and an uncertainty σ. It's a 5-seed ensemble.
5. **Bracket.** Exact advancement probabilities, with travel computed for each venue.

## Usage

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m mmai daily     # refresh current season -> ratings, predictions, bracket, site JSON
python -m mmai train     # same + retrain the network ensemble
python -m mmai all       # same + rerun the rolling NCAA backtest
python -m mmai publish   # re-export the site from cached features (fast)
python -m mmai history   # after a season ends: refresh the committed history/ bundle
```

If barttorvik.com refuses a request (it has blocked GitHub's runners), the current season is rebuilt from ESPN instead: games and box scores from the game-summary endpoint, player values from a box-score formula calibrated on Torvik's game values, and conferences from ESPN standings. This path agrees with the Torvik path at about r = 0.99 on efficiency and 0.97 on power ratings. The site header notes when it was used.

Completed seasons ship in `history/` (about 23 MB of compact parquet), so a run only downloads the current season. After a season ends, run `python -m mmai history` and commit `history/`. To preview locally: `cd docs && python -m http.server`.

### Manual controls

* `overrides/injuries.yaml`: force a player's availability (e.g. a star ruled out).
* `overrides/bracket_2027.yaml`: tournament sites, Final Four pairings, or a hand-entered bracket.

### Automation

* `.github/workflows/daily.yml`: every morning, November through April.
* `.github/workflows/retrain.yml`: monthly retrain and backtest, or on demand.

Enable Pages once: **Settings → Pages → Deploy from a branch → `main` / `/docs`**.

## Layout

```
mmai/            python package
  fetch/         torvik.py, espn.py (data sources)
  data.py        games table, team crosswalk, venues, NCAA rounds/seeds
  ratings/       efficiency.py (adjusted ratings), elo.py
  players.py     player values, roster strength, availability
  injuries.py    injury feeds + overrides
  features.py    leak-free per-game modelling table
  model/         net.py (network), train.py, backtest.py
  predict.py     live team state, power ratings, upcoming games
  bracket.py     projected/actual bracket, exact probabilities, picks
  pipeline.py    end-to-end + JSON export
history/         compact committed copy of completed seasons (games, players, venues)
docs/            static site (index.html, assets/, data/)
artifacts/       trained model, crosswalk, team metadata, backtest results
```
