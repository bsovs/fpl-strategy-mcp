# From Player Points to FPL Actions: A Rules-Aware, Risk-Aware MCP Decision System

**Working paper v0.1 — 15 September 2026 — not peer reviewed**

## Abstract

Fantasy Premier League (FPL) decisions are sequential portfolio decisions, not
independent weekly player-points picks. A transfer spends a scarce resource,
changes future price exposure, consumes budget, and may alter future transfer
flexibility. This paper describes FPL Strategy MCP, a local Model Context
Protocol server that combines player forecasts, expected availability, price
signals, news/context fields, league position, legal squad constraints, and an
action-value policy. The policy ranks hold, transfer, hit, and chip actions for
the exact squad at a particular deadline.

The system uses temporal validation and a deliberately conservative champion
selection rule. In the current small action-policy experiment, a learned
neural/action cocktail did not clear a paired 95% lower-confidence guardrail
against the free-transfer anchor on development folds. It therefore serves the
anchor by default. This is a useful negative result: a more complex model can
look better on mean simulated points while still failing a conservative test of
reliable improvement. The implementation is a research system, not evidence
of a guaranteed edge over expert managers or overall FPL winners.

## 1. Research question

Given a current 15-player squad, selling values, bank, free transfers, unused
chips, buyable players, current signals, and optional mini-league standings,
which legal action maximizes the chosen season objective under uncertainty?

The objective is intentionally broader than next-gameweek points:

\[
U(a_t) = P_t(a_t) + L_t(a_t) + V_t(a_t) + G_t(a_t) - C_t(a_t) - R_t(a_t),
\]

where `P` is short-horizon expected points, `L` is longer-horizon expected
points, `V` is price and budget value, `G` is rank/ownership leverage, `C` is
transfer and chip opportunity cost, and `R` covers minutes, injury, rotation,
news, uncertainty, and model-risk penalties. The deployed policy does not
pretend these terms are all measured in literal FPL points; their scales and
weights are explicit configuration.

## 2. System design

### 2.1 Forecast layer

The first layer produces point-in-time player signals rather than directly
choosing a squad. It supports short and long expected points, expected minutes,
fixture deltas, form, value, role security, injury/rotation risk, price-change
risk, uncertainty, captain upside, ownership leverage, and structured news or
social fields.

The literature-driven starting benchmark (before the expanded archive run)
used public historical FPL data with
leakage-safe lagged and rolling features. It compared a last-five mean,
EWMA(5), regularized ridge, and position-specific random forests. On a final
2024–25 holdout season, the position-specific forest had the lowest MAE by a
small margin while ridge had the lowest RMSE by a similarly small margin:

| Model | RMSE | MAE |
|---|---:|---:|
| Last-five mean | 2.068 | 1.056 |
| EWMA(5) | 2.039 | 1.035 |
| Ridge | 1.929 | 1.058 |
| Position-specific random forest | 1.928 | 1.034 |

These are player-fixture forecast errors, not proof of transfer profitability.

The expanded local trainer now uses the complete public Vaastav archive from
2016–17 through 2025–26. The temporal protocol is development on 2016–17
through 2023–24, model selection on 2024–25, and an untouched final test on
2025–26. It produces 247,574 player-fixture feature rows from 247,896 raw
rows and uses 286 model inputs: lagged player form and volatility, minutes and
starts, ownership and transfer movement, price momentum, team/opponent and
prior-matchup form, fixture-shape counts, leakage-safe cross-season player
history, breakout-vs-career signals, and optional timestamped context. The
oldest GW files are enriched from season-level `players_raw.csv` metadata;
those rows carry explicit imputation flags because the roster snapshot is not
a point-in-time transfer history. All player and team lags are guarded by
season/gameweek sequence, which blocks double-gameweek result leakage. The
action bridge additionally trains direct 3- and 8-gameweek point targets and
a next-gameweek price-change target on distinct player/gameweek rows; a
double gameweek contributes both fixtures to the point targets but only one
price label.

On the 2024–25 validation season, the career-feature neural candidate had
MAE 1.026 while the stronger ridge had RMSE 1.930; the run selected the neural
model on its primary MAE/ranking rule. On the untouched 2025–26 test, the
selected neural forecast had RMSE 1.954, MAE 0.973, Spearman ranking 0.697,
and a top-20-per-gameweek realized mean of 4.247 points, compared with 1.039
MAE and 3.695 points for EWMA(5). A separate next-price-change ridge had MAE
0.118 in the source price-tenth unit. The career-history ablation improved
MAE and top-20 selection value but slightly reduced RMSE and Spearman, so it
is retained as a research feature set rather than treated as a universal
forecast winner.

### 2.2 Legal decision layer

The decision layer receives the exact squad and buyable pool. It enforces:

- 15-player squad and same-position transfers;
- bank and selling-value affordability;
- three-player-per-team limits;
- free-transfer counts and paid-hit costs;
- chip availability and gameweek gates;
- hold as a first-class action;
- optional defend, neutral, or chase rank modes.

The output is an inspectable recommendation containing the action, legal
transfer set, score components, timing, reasons, risk flags, and model
uncertainty. A player with the highest projected points can therefore lose to
holding when the transfer cost, future flexibility, price economics, or
availability risk is unfavorable.

### 2.3 Action-value policy

The policy model is trained on legal counterfactual actions generated by the
season simulator. Its action features include the proposed short/long point
delta, price delta, ownership leverage, hit cost, chip type, gameweek, bank,
free transfers, squad value, and league deficit. An ensemble supplies a mean
and uncertainty estimate.

The candidate cocktail combines the model with an anchor policy. It can be
configured for cautious, balanced, or win-seeking behavior, but every learned
override must clear an anchor tolerance and a risk/opportunity-cost gate.

### 2.4 MCP interface

The server exposes a prediction tool plus inspectable supporting tools:

1. `fpl_recommend_moves` accepts the current state and returns legal actions.
2. `fpl_lineup_plan` handles formation, XI, bench order, and captaincy.
3. `fpl_search_players` and `fpl_forecast_signals` inspect the full player pool.
4. `fpl_score_moves` applies explicit per-request weight overrides.
5. `fpl_strategy_info` and `fpl_strategy_catalog` report the served model and
   tuning contract.
6. `fpl_backtest_strategy` evaluates supplied scenarios or Vaastav-format
   historical replays.

The default transport is local stdio. An optional Streamable HTTP transport is
included for a deliberately configured remote endpoint. The HTTP listener is
bound to localhost by default and supports an optional bearer token; production
deployment still requires HTTPS and an authentication/reverse-proxy policy.

## 3. Backtest protocol

The current action-policy experiment uses an expanding temporal protocol:

- development seasons: 2016–17 through 2023–24;
- model-selection season: 2024–25;
- untouched evaluation season: 2025–26;
- four legal opening families: `points`, `value`, `template`, and
  `randomized_points`;
- four sampled states per season and opening family for development labels;
- six candidate actions per decision policy family and a three-gameweek label
  horizon;
- wildcard/free-hit training search depth 5 and final-test depth 15;
- the final test season is not used for fitting or model selection.

The simulator now also searches sequentially legal multi-transfer bundles for
wildcard and free-hit actions up to the 15-player squad depth. Each step
recomputes bank, selling price, team caps, position legality, and the already
selected players. This prevents the action learner from treating every move
as an isolated one-for-one transfer.

The forecast bridge also aggregates fixture-level predictions to one
player/gameweek signal by summing double-gameweek fixtures. This is required
for correct Bench Boost, captain, Free Hit, and transfer comparisons; keeping
only one fixture would understate the value of a double gameweek.

The bridge carries three forecast families into the action state: the
one-fixture point model, direct short/long horizon models, and a price-change
ridge. These are deliberately evaluated as ablations rather than blended by
assumption. Direct horizon values are averaged across duplicated
double-gameweek rows because they are repeated labels, while fixture-level
point predictions are summed.

This is still a small benchmark. It is designed to prevent premature claims,
not to establish a final public leaderboard.

## 4. Current results

The corrected career-feature action run generated 1,441 development and 177
validation counterfactual examples. The smaller action ensemble was selected
on validation (RMSE 7.983 versus 8.695 for the default ensemble; state-level
argmax accuracy 18.75% versus 25%). On 2024–25, the neural policy averaged
2,129.0 points across the four opening families versus 2,060.25 for the
points-only free-transfer anchor. This advantage was not uniform: the value
and randomized openings favored the anchor, so the result is not a deployment
guarantee.

On the untouched 2025–26 replay, the corrected neural policy averaged 2,021.0
points versus 1,915.25 for that run's anchor, with paired gains of 78, 65,
170, and 110 points across the four opening families. The best neural opening
scored 2,079 points. This remains 334 points below the 2,413 research target
and is based on one held-out season with synthetic opening squads. The
artifact is therefore not promoted to the public MCP champion yet.

The anchored cocktail was also replayed with the same action model and legal
search. It averaged 2,095.75 on validation and 1,975.75 on 2025–26, versus
the neural policy's 2,021.0 on that test. The cocktail was better than the
anchor on average but worse than raw neural, with negative gains in two of the
four test opening families; the transparent anchor remains the safer deployed
fallback while gate tuning is treated as a separate validation-only
experiment.

Two additional walk-forward ablations make the current boundary clearer. A
direct-horizon neural bridge plus price model scored 1,914.25 mean points on
the untouched test, despite a stronger validation action argmax score. A
ridge point/direct-horizon/price bridge scored 2,113.0 mean and 2,189 in its
best opening family. The ridge result is the strongest current ablation, but
it remains 224 points below 2,413 and is based on one held-out season, so it
is not promoted as the public champion. A separate price-aware neural run
without direct horizons scored 1,994.25 mean and 2,035 best; price forecasts
are therefore retained as an available feature, not assumed to improve
decisions.

As a starting-squad stress test, an optional forecast-optimized legal opening
family was added to the same ridge bridge. It scored 2,162 on 2025–26 versus
1,929 for its free-transfer anchor, but its validation action error was worse
than the four-family run and it did not beat the 2,189 best. This supports
reporting opening-squad sensitivity explicitly rather than selecting the
most favorable starting family after seeing the test season.

The result also illustrates why “the neural network scored more on average” is
not enough. The action space is path-dependent, chip timing has opportunity
cost, and model errors compound over a season. A strategy can win a few
simulations while having an unacceptable downside or unstable behavior across
starting squads.

The current expanded forecast and action layers have not yet produced a
validated 2,413-point strategy. The 2,413 figure is therefore treated as a
strategy benchmark/aspiration, not as a supervised player-point label.
Clearing it requires more historical action states, stronger expected-minutes
and multi-horizon fixture forecasts, a real league/rival model, and held-out
families of starting squads; a player forecast that ranks well is not
sufficient.

## 5. News, social context, and price information

The input contract supports point-in-time `news_sentiment`, `news_risk`,
`social_sentiment`, `context_reliability`, event counts, and triggers. The
production server does not silently scrape Twitter/X or news. A caller must
provide timestamped, leakage-safe context or explicitly request the transparent
official-bootstrap fallback. The fallback uses official availability, form,
price, ownership, and transfer fields; it is not a replacement for a trained
expected-minutes or news model.

Price is treated as an input and a future-state concern. The current
walk-forward price-change model is wired into the action bridge and evaluated
with the same cutoff as the points model. Its distribution is intended to feed
budget flexibility and transfer timing rather than merely add a static value
score, but the current held-out price-aware ablation did not improve strategy
replay.

## 6. Limitations and threats to validity

1. The action-policy sample is small and the simulator uses synthetic starting
   squads and rival behavior. It has not been proven against a representative
   set of real FPL managers or global winners.
2. A simulator can encode its own assumptions. Counterfactual action labels are
   not causal evidence that a real manager could have captured the same result.
3. Forecast quality is still constrained by public FPL data, missing or
   imperfect expected-minutes information, and changing game rules.
4. The live lineup path uses the legal formation optimizer over the supplied
   forecast signals; final autosubs remain dependent on confirmed minutes and
   late team news.
5. News and social fields are supported by the schema but are not a validated,
   continuously ingested production feature set in this release.
6. “Going for the win” needs a real league-state distribution and rival-action
   model. Ownership leverage is a useful signal, not a complete game-theoretic
   equilibrium.

## 7. Reproducibility and release artifacts

The public distribution bundles the served policy model, the bootstrap
snapshot, the champion manifest, and machine-readable evaluation metrics under
`assets/`. The local research workspace contains the larger historical data,
simulators, notebooks, and reports used to produce the release artifacts.

Install and run the server using the commands in the root README. Run the
package tests with:

```bash
python -m unittest discover -s tests -q
```

Future releases should publish the exact data manifest, commit identifier,
environment lock, and all walk-forward predictions alongside the binary.

## 8. Research roadmap

- Expand seasons and legal starting-squad scenarios before tuning again.
- Add a calibrated expected-minutes model and component xPoints features.
- Add source-timestamped news and social ingestion with ablation tests.
- Separate transfer, chip, and captain policy heads and calibrate uncertainty.
- Fit a real-league/rival model and evaluate rank win probability, not only
  net points.
- Keep a permanently untouched season and starting-squad families for each
  release.

## References

- [Official FPL bootstrap API](https://fantasy.premierleague.com/api/bootstrap-static/)
- [OpenFPL paper](https://arxiv.org/html/2508.09992v1) and [OpenFPL code](https://github.com/daniegr/OpenFPL)
- [fpl-forecast](https://github.com/daniel-mehta/fpl-forecast)
- [Ramezani and Dinh, 2025](https://arxiv.org/html/2505.02170v3)
- [Open FPL solver](https://github.com/solioanalytics/open-fpl-solver)
- [MCP Python SDK: connecting a server to a host](https://py.sdk.modelcontextprotocol.io/get-started/real-host/)
- [OpenAI Responses API remote MCP tools](https://developers.openai.com/api/reference/cli/resources/beta/subresources/responses)
