# FPL historical data-quality audit

Updated 16 September 2026 after the official-context action-policy replay.

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
| Historical context | No complete timestamped expected-minutes, news/social, or rival-manager archive | High | 1,722 archived official bootstrap snapshots through 2025/26 now produce 9,366 leakage-safe news/role events; press-conference/social/rival history remains incomplete |

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

## Missing data that matters most

The main limitation is not the number of player rows. The highest-value missing
inputs are:

1. point-in-time expected minutes and lineup probability, including manager
   rotation and late fitness news;
2. timestamped historical official news, press conferences, and social signals
   that can be replayed exactly as known before each deadline;
3. richer historical fixture-strength and tactical matchup features;
4. real mini-league/rival-manager states and actions for the win-seeking
   objective; and
5. season-versioned chip and rule definitions, including newer half-season
   chip slots.

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

The audit supports continuing the project, but it does not support claiming
that the current model has reached the 2,413-point target.
