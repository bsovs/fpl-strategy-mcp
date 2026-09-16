# FPL historical data-quality audit

Updated 16 September 2026 after the clean action-policy replay.

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
| Historical context | No complete timestamped expected-minutes, news/social, or rival-manager archive | High | A leakage-safe expected-minutes ablation now exists; an importer for the public official bootstrap snapshot archive is validated, but the full context replay is not yet promoted |

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
2. Ingest the public timestamped official bootstrap snapshots for availability
   and official-news replay, then add separate timestamped feeds for predicted
   lineups, press conferences, and set-piece roles; the schema, importer, and
   deadline cutoff are implemented in `docs/context-data-contract.md`.
3. Calibrate the expected-minutes model against that context archive and add
   fixture-strength/matchup features using only pre-deadline information.
4. Add real rival-state replay where available, otherwise keep the synthetic
   league explicitly labeled as such.
5. Re-run temporal folds and the untouched 2025/26 test only after the new
   inputs are available. Do not tune against the final test while developing.

The audit supports continuing the project, but it does not support claiming
that the current model has reached the 2,413-point target.
