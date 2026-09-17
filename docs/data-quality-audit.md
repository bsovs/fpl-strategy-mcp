# FPL historical data-quality audit

Updated 16 September 2026 after adding the observed elite-manager benchmark,
the hit-aware/multi-transfer action-label path, and the external-style
forecast benchmark.

## Scope

The historical source is the Vaastav Fantasy Premier League archive, covering
2016/17 through 2025/26. The local archive contains 247,896 raw player-fixture
rows across ten seasons. The final evaluation season is 2025/26; it is not used
for fitting or model selection.

The source grain is normally one player-fixture row. That is not the correct
grain for every feature:

- point outcomes are fixture-level and are summed for a player/gameweek;
- lags, rolling form, ownership, transfers, price, and direct horizon labels
  are player/gameweek-level;
- schedule features are team/season/calendar-gameweek-level.

Keeping those grains separate is necessary for double gameweeks and blank
gameweeks. A row count alone is therefore not a sufficient completeness check.

## Checks and findings

| Check | Evidence | Severity | Disposition |
|---|---:|---|---|
| Raw historical coverage | 247,896 rows, 10 seasons | Low | Adequate for a first walk-forward benchmark |
| Duplicate player/gameweek keys | 1,586 in 2024/25–2025/26 | Medium | Expected double-gameweek fixtures; aggregate before lagging |
| Inactive/blank observations | 853 of 1,645 recent player-season entities have fewer than 38 observed gameweeks | Medium | Treat missing rows as structural absence, not zero points without a calendar rule |
| Pre-fix 3-GW target alignment | 293 distinct 2025/26 player-gameweek rows disagreed with the calendar-window target | High | Fixed: horizons now use calendar gameweeks and missing weeks contribute zero |
| Pre-fix double-GW lag consistency | 416 player/gameweek groups had inconsistent lag fields; 189 had inconsistent five-gameweek rolling points | High | Fixed: aggregate to one player/gameweek row before lags/rollups, then broadcast |
| Old metadata | Oldest files lack team/position in the gameweek rows | Medium | Filled from season roster snapshots and retain imputation flags |
| Historical context | No complete timestamped expected-minutes, news/social, or pre-2025/26 rival-manager action archive | High | 1,722 archived official bootstrap snapshots through 2025/26 now produce 9,366 leakage-safe news/role events; 24,041 aggregate manager-season rows plus 100 detailed 2025/26 manager histories are local, but earlier-season weekly behavior remains missing |

The two high-severity temporal issues were capable of making backtests look
better or worse for the wrong reason. Results from before those fixes—such as
the earlier 2,189-point ridge replay—are retained as ablations but should not
be compared directly with the clean run.

## Current clean replay

The corrected run uses 1,476 development counterfactual examples and 186
validation examples. The selected small action ensemble had validation RMSE
9.016, compared with 9.787 for the default candidate.

On the untouched 2025/26 replay across four legal opening squads:

| Policy | Mean net points | Best opening | Gap to 2,413 |
|---|---:|---:|---:|
| Learned neural action policy | 2,076.25 | 2,156 | 257 |
| Free-transfer anchor | 2,008.50 | 2,073 | 340 |
| Anchored cocktail | 2,020.75 | 2,092 | 321 |

The new expected-minutes ablation scored 2,064.75 mean and 2,155 best on the
same untouched test. It still beat its paired anchor (2,031.50 mean), but it
was below the prior clean point/horizon run (2,076.25 mean and 2,156 best), so
it is retained as an inspectable feature family rather than promoted as a
strategy improvement.

## Official context replay

The official-context run uses 9,366 point-in-time events from 1,722 archived
bootstrap snapshots, including injury/status/news events and intervalized
set-piece role changes. It keeps the same development/validation/test season
split and four opening-squad families:

| Metric | Clean baseline | Official context | Change |
|---|---:|---:|---:|
| Validation action RMSE | 9.016 | 8.670 | -0.346 |
| Validation state-argmax accuracy | 6.25% | 18.75% | +12.50 pp |
| 2025/26 neural mean | 2,076.25 | 2,007.50 | -68.75 |
| 2025/26 neural best opening | 2,156 | 2,085 | -71 |
| 2025/26 free-transfer anchor mean | 2,008.50 | 1,984.50 | -24.00 |
| 2025/26 cocktail mean | 2,020.75 | 2,001.00 | -19.75 |

This is an important negative result: better validation action fit did not
translate into better held-out points. Official context is therefore wired as
an inspectable research input and live signal source, but it is not promoted
into the shipped champion. The next experiment should calibrate/gate news and
set-piece features at the action layer rather than allowing every event to
change the transfer policy directly.

This is evidence of a useful improvement over the anchor in this replay, not
evidence of a winning FPL strategy. It is one held-out season with synthetic
opening squads and synthetic continuation/rival behavior.

## Form, fixture, value, and loss-aware action replay

The next ablation promoted the decision-layer signals into explicit action
features: short/long fixture-window deltas, recent form, relative value, role
security, price-change risk, and the unrealized loss on the player being sold.
Missing forecast values are normalized before entering the policy vector.

| Metric | Result |
|---|---:|
| Development counterfactual examples | 1,500 |
| Validation examples | 184 |
| Validation action RMSE | 11.184 |
| Validation action MAE | 6.020 |
| Validation state-argmax accuracy | 43.75% |
| 2025/26 neural mean | 1,990.50 |
| 2025/26 neural best opening | 2,114 |
| 2025/26 anchor mean | 1,983.75 |
| 2025/26 cocktail mean | 1,991.75 |

The neural policy beats its paired anchor by 6.75 points on the held-out mean;
the cocktail beats it by 8.00. The best opening remains 299 points below the
2,413 target, so this is an incremental action-selection improvement rather
than a winning-strategy claim. The artifact is retained at
`runs/action-policy-official-snapshot-v2/` and is not copied into the shipped
MCP assets.

## Observed manager behavior benchmark

The local `data/elite_managers/` input contains 24,041 aggregate manager-season
rows from a public 2025/26 archive. It is a behavioral audit and prior source,
not a point-in-time supervised action table. In the observed rank bands:

| Band | Median points | Median transfer gain | Median transfers | Median hits | Median unused chips | Median captain agreement |
|---|---:|---:|---:|---:|---:|---:|
| Top 100 | 2,498 | 325 | 62 | 4 | 0 | 71.1% |
| Top 1k | 2,460 | 316 | 63 | 4 | 0 | 79.0% |
| Top 10k | 2,412 | 302 | 63 | 4 | 0 | 79.0% |
| 100k band | 2,327 | 260 | 64 | 20 | 0 | 71.1% |

The individual action hypotheses are therefore measurable: transfer-value
capture, hit discipline, chip completion/timing, captaincy alignment, and
deadline timing should be modeled separately and then composed. The current
aggregate import cannot tell us which players were alternatives at each
deadline, so it is not yet used as a direct action label. Exact 38-gameweek
squads, transfers, and chip events are the next data-quality target.

The detailed weekly archive now closes that gap for the final benchmark
season: 100 managers, 3,800 manager-gameweek rows, and 3,700 observed
post-opening transitions. It contains 1,083 single-transfer weeks, 1,048
multi-transfer weeks, 177 paid-hit weeks, and 797 chip weeks. The local
validator confirms 38 gameweeks and 15 picks for every manager. Because all of
these actions are from 2025/26, they are not used to fit the 2025/26 policy;
they are reserved for an honest behavior benchmark and leakage audit.

The decision audit at `runs/elite-manager-benchmark/decision-audit.json` joins
those observed actions to player state through the prior gameweek. In the
top-100 sample, incoming players averaged 13.15 points over the prior three
gameweeks versus 12.61 for outgoing players, with 216.9 versus 223.7 prior
minutes and stronger transfer momentum for incoming players. The next-three-
gameweek points comparison is stored separately as an outcome audit; it is
not used for fitting.

The separate action-head audit at
`runs/elite-manager-benchmark/action-heads.json` turns those same pre-deadline
states into behavior-specific diagnostics. Grouped by manager, its transfer,
bundle, paid-hit, chip, and incoming/outgoing heads achieved AUCs of 0.670,
0.736, 0.623, 0.687, and 0.590 respectively. The bundle head being materially
stronger than the incoming/outgoing head is important: the observed behavior
contains a real action-selection problem, not just a player-ranking problem.
These coefficients are descriptive because 2025/26 is the final test season;
they are not used by the shipped trainer. The archived picks also lack exact
purchase/selling prices, preventing a fully legal historical choice-set model.

## Hit-aware and multi-transfer action replay

The follow-up label run uses `--label-policy points_only`, which permits paid
transfers in the behavior trajectory and exposes legal two-transfer bundles
when the state has enough free transfers. It produced 1,644 development and
203 validation examples, including 165 paid-hit action candidates and 453
multi-transfer candidates. This fixes the previous zero-hit label defect, but
it did not improve the untouched test:

| Policy | 2025/26 mean | Best opening | Paid-hit behavior |
|---|---:|---:|---|
| Hit-aware neural | 1,986.00 | 2,114 | 7–11 paid transfers in the four openings |
| No-hit anchor | 1,983.75 | 2,129 | 0 |
| Hit-aware cocktail | 1,995.25 | 2,060 | 0 after gating |

The run is retained at `runs/action-policy-official-snapshot-v3-hit-aware/`
and is not promoted. The result says the simulator can now represent the
actions; it does not yet know when a hit or bundle is valuable. The next step
is direct point-in-time manager-action training and separate hit/value/chip
behavior heads.

### Corrected season-chip replay

The v3 result above was not a valid final 2025/26 rules benchmark: the
simulator had one token per chip type, while the 2025/26 season uses
season-specific chip copies with first/second-half availability gates. The
implementation now represents those concrete chip tokens and maps them back
to canonical chip kinds for reporting. The corrected replay is saved at
`runs/action-policy-official-snapshot-v4-season-chips/`.

| Policy | 2024/25 validation mean | 2025/26 test mean | 2025/26 best opening |
|---|---:|---:|---:|
| Corrected neural action policy | 2,109.00 | 1,969.50 | 2,053 |
| No-hit free-transfer anchor | 1,973.50 | 1,983.75 | 2,129 |
| Corrected cocktail | 2,058.75 | 1,997.50 | 2,069 |

The v4 test is the authoritative corrected-rules baseline, not a promoted champion. It
shows that correct chip accounting removes a simulator error but does not
close the action-selection gap to 2,413. The policy still needs real,
point-in-time manager-action states and better calibration of hit, bundle,
chip, captain, and minutes decisions.

### External patient-chip challenger

The public [fpl-luck-or-skill repository](https://github.com/zakariae-boui/fpl-luck-or-skill)
reports 2,431 points for its `patient_chips` replay: 36 transfers, zero hits,
and all four tokenized TC/BB uses. Its rule is a useful independently authored
challenger—multi-gameweek EV, no paid hits, legal FT banking, and use-it-or-
lose-it TC/BB timing—but it is not a verified local result. The public clone
does not contain its raw prediction artifact, and the LightGBM/OpenMP runtime
did not run cleanly on this Mac.

The local simulator now includes the same high-level challenger as
`patient_chips`. With the local neural/context forecast cache it scored:

| Season | Mean across four openings | Best opening | Hits |
|---|---:|---:|---:|
| 2024/25 validation | 1,964.0 | 2,109 | 0 |
| 2025/26 test | 1,928.25 | 1,955 | 0 |

The result is a negative replication, not a reason to discard the external
work. It isolates the remaining gap as forecast calibration and opening-squad
quality in addition to transfer/chip discipline. The local artifact is at
`runs/action-policy-official-snapshot-v4-season-chips/patient-chips-evaluation.json`.

### External-style forecast and forecast-optimized opening

The next benchmark uses a Mac-compatible histogram-gradient-boosting fallback
for the public challenger's minutes-plus-conditional-points architecture. It
uses 53 leakage-safe inputs, including player form and minutes/start security,
xG/xA, price, ownership, transfer momentum, true fixture team, opponent/team
form, and previous-season production. The 2024/25 validation season selected
the small action ensemble (RMSE 6.212 versus 7.171 for the default ensemble),
and 2025/26 remained completely out of fitting and model selection.

| Policy | 2024/25 validation | 2025/26 test | Test transfers | Test paid hits |
|---|---:|---:|---:|---:|
| Neural action policy | 2,270 | 2,160 | 62 | 0 |
| Free-transfer anchor | 2,249 | 2,338 | 37 | 0 |
| Cocktail | 2,341 | 2,442 | 53 | 0 |
| **Patient chips candidate** | **2,357** | **2,486** | **37** | **0** |

This is the strongest current research candidate and it clears the 2,413
benchmark in one untouched season replay. It is not yet a production claim:
the forecast architecture is an independent local fallback rather than a
byte-for-byte reproduction of the public LightGBM artifact, the candidate
uses one forecast-optimized opening family, and the local elite-manager
archive is still incomplete for pre-2025/26 weekly alternatives and actions.
The candidate is now exposed through `fpl_backtest_strategy` with
`forecast_model="external_hgb"`, `policy="patient_chips"`, and
`initial_squad_modes=["forecast"]`; it remains a research candidate rather
than the live default until multi-start validation is complete.
The artifact is `runs/action-policy-external-hgb-forecast-v1/metrics.json`.

## Missing data that matters most

The main limitation is not the number of player rows. The highest-value missing
inputs are:

1. point-in-time expected minutes and lineup probability, including manager
   rotation and late fitness news;
2. timestamped historical official news, press conferences, and social signals
   that can be replayed exactly as known before each deadline;
3. richer historical fixture-strength and tactical matchup features;
4. exact point-in-time elite/rival-manager states, available alternatives, and
   actions for the win-seeking objective; and
5. complete pre-2025 weekly elite/rival-manager states, alternatives, and
   actions for direct imitation or inverse-decision modeling. Season-versioned
   chip inventory and half-season gates are now implemented for 2025/26, but
   older seasons still need an explicit rules audit.

Current FPL snapshots do contain form, minutes, ownership, transfers, prices,
team results, and fixture information. Those should be expanded before adding
more model complexity; otherwise the neural policy will mostly learn from
better bookkeeping of the same limited context.

## Remediation order

1. Preserve the current grain and leakage tests as regression tests.
2. Calibrate and selectively gate official availability/news/set-piece signals
   at the action layer; the first full replay improved validation fit but
   degraded held-out points.
3. Add separate timestamped feeds for predicted lineups, press conferences,
   and social signals, plus richer fixture-strength/matchup features using
   only pre-deadline information.
4. Add real rival-state replay where available, otherwise keep the synthetic
   league explicitly labeled as such.
5. Re-run temporal folds and the untouched 2025/26 test only after the new
   inputs are available. Do not tune against the final test while developing.

The corrected baseline neural/cocktail replay does not reach the 2,413-point
target. The separate external-style forecast candidate reaches 2,486 points
on the untouched 2025/26 replay, which is evidence that the expanded forecast
and patient-chip combination is promising—not evidence of a guaranteed or
generalized winning strategy.
