# Observed elite-manager benchmark

These files are downloaded derived tables from the public
[`zakariae-boui/fpl-luck-or-skill`](https://github.com/zakariae-boui/fpl-luck-or-skill)
repository, retrieved on 2026-09-16:

- `autopsy_all.csv` — one aggregate row per manager season, including transfer
  gains, transfer count, hits, captain agreement, chip usage, and timing.
- `manager_index.csv` — manager entry IDs, observed rank bands, and final
  points.
- `SOURCE.md` — the upstream project README and methodology notes.

The `season_winners_2025-26/` directory is a second, more detailed archive
from the public [`bentindal/FPL-Auto`](https://github.com/bentindal/FPL-Auto)
project. The downloaded archive contains 100 manager directories and 3,800
weekly JSON records. Each record includes the full 15-player squad, starting
XI, captain, vice-captain, active chip, bank/value, points, transfer count,
transfer cost, and rank snapshot. The rank-band crosswalk is taken from
`manager_index.csv`; in this local sample it maps to 14 top-100 managers and
86 top-1k managers.

Validate and summarize the weekly transitions with:

```sh
PYTHONPATH=src python scripts/analyze_observed_manager_archive.py \
  --archive-root data/elite_managers/season_winners_2025-26 \
  --rank-index data/elite_managers/manager_index.csv \
  --output runs/elite-manager-benchmark/weekly-summary.json
```

The loader derives squad overlap and incoming/outgoing IDs between adjacent
gameweeks. Final rank, final points, gameweek points, and cumulative points
remain audit-only outcome fields and must be excluded from pre-deadline model
features. Because this is the 2025/26 holdout, the weekly archive is used for
benchmarking and data validation—not for fitting the 2025/26 action model.

This is a benchmark and behavioral-prior source, not a point-in-time training
label set. Final rank and final points are outcome fields. The local loader
keeps them available for reporting but does not return them as behavior
features. The aggregate table also cannot reconstruct each weekly squad or
the exact transfer alternatives available at each deadline. The detailed
2025/26 weekly archive is now present locally, but pre-2025/26 weekly manager
archives are still needed for leakage-safe direct imitation or
inverse-decision modeling.
