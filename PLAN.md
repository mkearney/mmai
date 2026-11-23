# MMAI — design plan

Goal: a refreshable, zero-cost pipeline that (1) predicts every game of the 2027
NCAA men's tournament bracket and (2) exposes daily power ratings (KenPom-style)
derived from the same model, published on a GitHub Pages site.

## 1. Data inventory (all free, no keys)

| Source | What | Span | Used for |
|---|---|---|---|
| barttorvik.com `{yr}_super_sked.csv` | every D-I game (played + scheduled), venue flag, game type, **full box score**, T-Rank's pre-game line | 2008– | game results, possessions, four factors, benchmark |
| barttorvik.com `{yr}_all_advgames.json.gz` | **player-game logs** (~110k rows/season): minutes, usage, ORtg, game BPM | 2008– | player values, roster strength, availability/injury detection |
| barttorvik.com `getadvstats.php` | player-season table: height, class, recruit rank, BPM | 2008– | freshman priors, roster profile |
| ESPN site API scoreboard | every game with venue city/state, neutral flag, tournament notes, **seeds/regions** | 2008– | tournament games + seeds, neutral-site locations, team home cities |
| ESPN teams API | logos, colors, names | current | website |
| Rotowire injury JSON feed | current injuries (player, team, status) | live | injury adjustments |
| kelvins/US-Cities-Database (GitHub raw) | city → lat/lon | static | travel distance / tournament proximity |
| `overrides/*.yaml` (hand-edited) | injury overrides, 2027 bracket & sites if needed | — | manual control |

Blocked / not used: kenpom (paywall), Massey (blocked), Kaggle (login), GitHub release
downloads (blocked from the build sandbox).

## 2. Modeling stack

All features are computed **point-in-time** (only games before the game date), so the
historical training set mimics what is known on game day.

1. **Adjusted efficiency (the KenPom-like core).** Per season, a recency-weighted ridge
   regression of per-possession offense on `mu + Off[team] + Def[opp] + hca*loc`, with
   a learned preseason prior as pseudo-observations. The same design matrix is solved
   for ~12 other per-game stats at once (eFG%, TO%, OR%, FT rate, 3PA rate, 3P%, 2P%,
   block, steal, assist rates, tempo), giving **opponent-adjusted style profiles** for
   offense and defense. Snapshots are taken before every game day of every season.
2. **Elo.** Margin-of-victory Elo across seasons with home-court and season reversion.
3. **Player layer.** Each player gets a value (points/100 poss. above average): a
   Bayesian blend of last season's value (+ class progression; recruit-rank prior for
   freshmen) and this season's minute-weighted game BPM. Team **roster strength** =
   Σ minutes-share × value. Height, experience, depth, star power, continuity are
   derived too. The preseason prior regresses last season's rating toward the roster
   strength of the players actually on the team (so transfers count).
4. **Availability / injuries.** The rating reflects the roster as it actually played.
   Adjustment = Σ (availability now − historical participation rate) × share × (value −
   replacement). Historical training uses who actually played; live predictions use
   Rotowire statuses, "missed the last N games" detection and manual overrides.
5. **Matchup neural net (PyTorch).** A team encoder turns each side's ~40 features
   (efficiency, style O/D, roster, Elo, form, availability) into an embedding; a pair
   head scores `[eA, eB, eA*eB, context]`. Margin = g(A,B) − g(B,A) (exactly
   antisymmetric); a symmetric head predicts the uncertainty σ, so
   P(win) = Φ(margin/σ). Context = home/neutral/away, **travel distance of each team to
   the venue** (proximity in neutral/tournament games), rest, game type (NCAA,
   conference tourney), same conference. A 5-seed ensemble. Style interactions
   (e.g. elite offensive rebounding vs poor defensive rebounding) let a weaker team
   that "matches up well" get a higher probability; the site reports this as the
   **matchup edge** = NN margin − ratings-only margin.
6. **Ratings from the model.** The daily power rating = each team's average predicted
   neutral-site margin against every other D-I team (round-robin through the NN),
   alongside the interpretable AdjO/AdjD/AdjT/AdjEM, Elo and roster numbers.

## 3. Validation

Rolling-origin backtest over the NCAA tournaments 2014–2026 (no 2020): for each year
train only on games before that tournament, then score tournament games. Compare log
loss / Brier / accuracy against: seed-only, Elo, efficiency-only linear model, and
T-Rank's own pre-game lines. Results published on the Method page.

## 4. Bracket

* **Actual bracket** (auto-detected from ESPN once announced, or `overrides/bracket_2027.yaml`).
* **Projected bracket** before Selection Sunday: auto-bids to each conference's best
  team (or tournament champion once known), best remaining at-large, S-curve seeding.
* 20k Monte Carlo sims with venue-specific distances → round-by-round probabilities,
  a "most likely" bracket and a pool-oriented bracket (highest advancement probability).

## 5. Refresh + site

* `python -m mmai daily` — refresh current-season data, ratings, predictions, bracket,
  export JSON into `docs/data/`. `python -m mmai train` — rebuild features, retrain the
  NN ensemble, run backtests. `python -m mmai all` does both.
* GitHub Actions: daily cron (data + ratings + predictions), weekly retrain.
* `docs/` = static site (vanilla JS, no build): Ratings, Bracket, Matchup tool (the NN
  runs in the browser from exported weights), Team pages, Games, Method.
