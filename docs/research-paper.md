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
2025–26. It produces 163,082 player-fixture feature rows from 247,896 raw
rows and uses 227 model inputs: lagged player form and volatility, minutes and
starts, ownership and transfer movement, price momentum, team/opponent and
prior-matchup form, fixture-shape counts, and optional timestamped context.
All player and team lags are guarded by season/gameweek sequence, which blocks
double-gameweek result leakage.

On the untouched 2025–26 test, the selected stronger-ridge forecast had RMSE
1.947 and MAE 1.005, compared with MAE 1.038 for EWMA(5). Its top-20-per-
gameweek selection had a realized mean of 4.274 points, compared with 3.695
for EWMA(5). A separate next-price-change ridge had MAE 0.119 in the source
price-tenth unit. The neural forecast candidate was trained and evaluated in
the validation protocol, but was not selected because the regularized model
performed better on 2024–25. This is an intended anti-overfitting result, not
a reason to force a neural model into production.

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

The server exposes two tools:

1. `fpl_recommend_moves` accepts the current state and returns legal actions.
2. `fpl_strategy_info` reports the served champion, evaluation summary, sources,
   and limitations.

The default transport is local stdio. An optional Streamable HTTP transport is
included for a deliberately configured remote endpoint. The HTTP listener is
bound to localhost by default and supports an optional bearer token; production
deployment still requires HTTPS and an authentication/reverse-proxy policy.

## 3. Backtest protocol

The action-policy experiment used an expanding temporal protocol:

- development seasons: 2021–22, 2022–23, and 2023–24;
- temporal validation folds: train on earlier seasons, validate on the next;
- development starting squads: `points_optimal`, `value_balanced`, and
  `template_proxy`;
- untouched holdout season: 2024–25;
- holdout starting squads: two random but legal squads;
- learned-candidate selection: maximize the paired 95% lower confidence bound
  against the free-transfer anchor;
- holdout was not used for selection.

The simulator now also searches sequentially legal multi-transfer bundles for
wildcard and free-hit actions up to the 15-player squad depth. Each step
recomputes bank, selling price, team caps, position legality, and the already
selected players. This prevents the action learner from treating every move
as an isolated one-for-one transfer.

This is still a small benchmark. It is designed to prevent premature claims,
not to establish a final public leaderboard.

## 4. Current results

The development tournament contained six runs per candidate. The free-transfer
anchor averaged 2,037.2 net points. The raw neural policy averaged 2,074.0;
the selective-chip cocktail averaged 2,099.3. However, neither had a positive
paired 95% lower confidence bound against the anchor. The selective-chip
candidate's lower bound was −19.4 points.

On the two-run untouched holdout, the anchor averaged 2,001.0 points and the
raw neural policy averaged 1,960.5, a −40.5 mean difference. The sample is far
too small for a definitive statistical conclusion, but it supports the current
deployment choice: use the anchor unless a future, larger walk-forward study
demonstrates a robust improvement.

The result also illustrates why “the neural network scored more on average” is
not enough. The action space is path-dependent, chip timing has opportunity
cost, and model errors compound over a season. A strategy can win a few
simulations while having an unacceptable downside or unstable behavior across
starting squads.

The current expanded forecast layer has not yet produced a validated
2,413-point strategy. The 2,413 figure is therefore treated as a strategy
benchmark/aspiration, not as a supervised player-point label. Clearing it
requires a full-season legal replay with chips, formation, captaincy,
multi-transfer bundles, price economics, and a held-out family of starting
squads; a player forecast that ranks well is not sufficient. In a first
2025–26 bridge smoke test, the expanded forecasts produced 1,942–1,992 net
points across three legal starting-squad modes under the same free-transfer
policy, versus 1,789–1,807 for the legacy signals. The shipped action cocktail
with chips scored 1,982 on the template start. These are useful integration
checks, not tuned final results, and none clears 2,413.

## 5. News, social context, and price information

The input contract supports point-in-time `news_sentiment`, `news_risk`,
`social_sentiment`, `context_reliability`, event counts, and triggers. The
production server does not silently scrape Twitter/X or news. A caller must
provide timestamped, leakage-safe context or explicitly request the transparent
official-bootstrap fallback. The fallback uses official availability, form,
price, ownership, and transfer fields; it is not a replacement for a trained
expected-minutes or news model.

Price is treated as an input and a future-state concern. The intended next
research step is a separate price-change model, evaluated with the same cutoff
as the points model, whose distribution feeds budget flexibility and transfer
timing rather than merely adding a static value score.

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
