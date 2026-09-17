# Is Fantasy Premier League luck or skill?

> I measured it on 24,041 real managers, from the world champion down to rank 6 million. The answer: **85% skill**. Then I built an engine that plays the game like a top-2,500 manager to prove it.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-projections-green)
![MILP](https://img.shields.io/badge/PuLP-squad%20optimizer-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Third project in a series about football and prediction. [football-prediction-ml](https://github.com/zakariae-boui/football-prediction-ml) showed that ML cannot beat an efficient betting market; [value-betting-scanner](https://github.com/zakariae-boui/value-betting-scanner) showed the edge lives in prices instead. Fantasy Premier League is the opposite environment: 11 million players, no bookmaker, no closing line to price your decisions. If skill exists anywhere in football games, it should be measurable here. It is.

## The dataset

In July 2026, days before the FPL API wiped the season, I archived **24,041 complete manager seasons**, stratified by final rank: the entire top 100 (including the world champion), ~900 of the top 1k, ~9,000 of the top 10k, and 1,000 managers per band at every level down to rank 6M. For each manager: all 38 gameweek squads with captains, every transfer with its timestamp, chips, points, and ranks.

An autopsy engine scores **21 decision metrics** per manager: net points gained from transfers, points paid for hits, captaincy vs the elite consensus pick, chip timing loss, unused chips, squad value built, hours before deadline, dead starters, and more. The derived tables (per-manager metrics plus the full 24k x 38 gameweek score matrix) ship with this repo, so everything below is reproducible without the 2.5 GB raw archive.

## Part 1: the answer

**Decomposing the variance** of season scores in a random-manager sample: within-season noise (the luck a fixed manager experiences) accounts for **14.7%** of the spread between managers; persistent differences between managers account for **85.3%**. And it is not hidden skill: a regression on just 8 measured decision habits explains **R² = 87%** of season points.

But skill controls your *neighborhood*, not your exact address. Bootstrap-replaying each manager's own season 2,000 times shows how wide the luck bands are:

<p align="center">
  <img src="assets/luck_bands.png" width="850" alt="90% rank bands when the same manager replays the season">
</p>

The sharpest version of this result is at the very top: **the actual world champion wins the replayed season only 3.6% of the time** (median replay position: 25th of the top 100, and 102 different managers win at least one replay). Reaching the top 100 is skill. Finishing exactly first is luck.

**The decision space is visibly ordered by rank.** Principal component analysis on 16 decision metrics for 21,043 managers, with no rank or points information given to the PCA, arranges managers from elite (left) to rank 6M (right) on the first component alone:

<p align="center">
  <img src="assets/pca_decision_space.png" width="850" alt="PCA of the decision space, colored by final rank tier">
</p>

The loadings say what PC1 *is*: captaincy aligned with the elite consensus, armband skill above random, and transfer gains pull left; unused chips, dead starters, and captaincy losses pull right. Behavior is rank.

**What the elite actually do differently**, tier by tier:

<p align="center">
  <img src="assets/elite_fingerprints.png" width="900" alt="Decision habits by rank tier">
</p>

| Habit | Top 100 | Rank 6M |
|---|---|---|
| Captain matches elite consensus | 76% of GWs | 45% |
| Median captaincy loss vs the elite pick | **-11 pts (they beat it)** | +72 pts |
| Chips never played | 0 of 8 | 5 of 8 |
| Net transfer gain | +325 pts | +136 pts |
| Median points paid for hits | 4 | (noisy, up to 80 in mid bands) |

The world champion is the cleanest case study: a first-season player who made 36 transfers with **zero hits**, captained Haaland 22 of 38 weeks, used all 8 chips, and rose from rank 8.9M (GW1) to 1.

## Part 2: the proof

If FPL is 85% skill expressed through measurable decisions, then a machine making good decisions should rank highly. Testing that claim end to end:

**Projection models.** Expected points are decomposed as P(minutes bucket) x E[points | bucket]: a 3-class LightGBM minutes model plus two conditional points regressors, and q10/q90 quantile models for ceiling/floor. Trained on **253,509 player-match rows across 10 seasons** with 53 leakage-safe features (shift-then-roll form windows, previous season, fixture-derived team form, context). Validation is walk-forward only, each season predicted using strictly earlier seasons:

<p align="center">
  <img src="assets/model_validation.png" width="850" alt="Walk-forward MAE vs baselines">
</p>

The model beats the rolling-form and position-mean baselines on every test season, on both MAE and the quality of its top-10 picks per gameweek (4.50 vs 3.77 actual points per pick in 2025/26).

**Squad optimizer.** Squad selection is a mixed-integer linear program (PuLP): maximize expected points over a horizon subject to the £100m budget, position quotas, max 3 per club, valid formations, free-transfer banking, and hit penalties.

**The backtest.** Replay the full 2025/26 season with pure model decisions, using models trained only on 2016-2024 (strictly out-of-sample) and only pre-deadline information, with real price movement and FPL's selling rule:

<p align="center">
  <img src="assets/backtest_ladder.png" width="900" alt="Backtest results by decision style">
</p>

The best policy scores **2,431 points, an estimated rank of 2,324 out of 13.1 million** (top 0.02%), above the top-10k cutoff and 16 points off the top 1k, with zero hits taken all season. Two findings generalize beyond the bot:

1. **Hits are poison.** Allowing paid transfers loses 90 to 195 points across every configuration tested, even with a strict 2-hit budget. Root cause: the model's largest EV edges sit exactly where calibration shows overconfidence, so a hit that "pays on paper" usually does not.
2. **Use every chip.** A use-it-or-lose-it rule (fire before each half ends, even imperfectly) beats waiting for the perfect week by 39 points. This mirrors the human data: unused chips are one of the strongest rank predictors in the 24k sample.

**Price model.** A separate LightGBM classifier predicts overnight price rises and falls from transfer flow and ownership: **54% precision@20 against an 8.8% base rate (6x lift)** on the held-out 2025/26 season.

## Negative results (kept on purpose)

- **Wildcard/Free-Hit automation fails.** All five automated chip-scheduling variants lose 11 to 154 points versus simply holding a good squad; the model cannot identify rebuild weeks sharply enough, so the CLI keeps WC/FH advisory-only.
- **"Finishing luck" features hurt.** Adding explicit goals-minus-xG regression features made walk-forward MAE slightly worse on every season; the existing xG form windows already carry the signal. Reverted, kept as a human-readable scout report instead.
- **Deadline-day information is the real moat of commercial tools.** FPL's own xP (computed at the deadline with team news) beats this model's historical-features-only MAE on 2022-2025. Closing that gap needs live availability data, not better ML.

## The toolkit

A CLI (`python engine/fpl.py <command>`) exposes everything. The most demo-able commands:

| Command | What it does |
|---|---|
| `luck --entry ID` | bootstrap "skill band" for any manager: was your rank luck or skill? |
| `coach --entry ID` | your habits vs the top 1k, each priced in season points |
| `sim --entry ID --gw N` | P(top 1k / 10k / 100k) for the rest of the season |
| `rate --entry ID --gw N` | squad rating /100 vs the best possible squad at your budget |
| `draft` / `plan` / `captain` | optimizer: GW1 draft, weekly transfers, captaincy |
| `chips --gw N` | Wildcard / Free-Hit / Bench-Boost / Triple-Captain advisor |
| `xg --gw N` | over/under-performers vs xG (regression candidates, buy-lows) |
| `price --gw N` | tonight's likely price risers and fallers |
| `h2h --entry ID --rival ID` | mini-league war room: differentials, model verdict |
| `ticker --gw N` | fixture-difficulty grid + chip calendar |

Commands that only need the shipped derived data (like `luck` and `coach` benchmarks) run from this repo; commands that need per-manager raw JSONs or trained model files need the local archive (see below).

## Repository structure

```
├── engine/
│   ├── fpl_engine/          the library: autopsy, luck, coach, project, optimizer,
│   │                        strategy, chips, price, scout, ticker, features, data
│   ├── fpl.py               the CLI
│   ├── monte_carlo.py       the luck-vs-skill experiments (reproducible from data/)
│   ├── backtest.py          full-season replay harness
│   ├── build_training.py    10-season training set builder
│   ├── train_models.py      LightGBM training + walk-forward validation
│   └── tests/
├── scripts/
│   ├── extract_summary.py   manager JSONs -> summary tables
│   ├── build_player_gw.py   player-GW table with integrity checks
│   └── make_figures.py      regenerates every figure in assets/
├── data/                    derived tables (shipped): autopsy_all.csv, gw_points_matrix.csv,
│                            train_gw.parquet, model_eval.json, monte_carlo_results.json, ...
├── docs/                    DATA / ENGINE / MODELS / OPTIMIZER / BACKTEST / COACH
└── assets/                  figures
```

## Reproducing the results

```bash
pip install -r requirements.txt

python engine/monte_carlo.py       # luck vs skill: 85/15, champion replay, R2
python scripts/make_figures.py     # regenerate every figure in this README
python engine/train_models.py      # retrain + walk-forward eval (needs ~5 min)
```

The raw archive (24k manager JSONs, 841 player histories, 38 live-event files, 10 seasons of history) is 2.5 GB and stays local; `docs/DATA.md` documents every source, quirk, and lineage step, and the collection scripts are included. Historical seasons come from the excellent [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) archive; everything else is from the public FPL API.

## The three-project story

| Project | Question | Answer |
|---|---|---|
| [football-prediction-ml](https://github.com/zakariae-boui/football-prediction-ml) | Can ML out-predict the betting market? | No. Closing odds already price public data |
| [value-betting-scanner](https://github.com/zakariae-boui/value-betting-scanner) | Where is the edge then? | In prices: +4% CLV buying soft-book mistakes |
| **fpl-luck-or-skill** | No market, no odds. Does skill exist? | **Yes: 85% skill, and a machine can learn it** |

## Author

**Zakariae Boui** ([GitHub](https://github.com/zakariae-boui))

## License

MIT, see [LICENSE](LICENSE).
