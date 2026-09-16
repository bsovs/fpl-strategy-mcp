#!/usr/bin/env python3
"""Train leakage-safe FPL player and price forecasters on local history.

The script intentionally leaves one complete season untouched.  It writes
metrics and predictions for inspection and serializes the selected forecast
and price models for downstream strategy experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_lab.player_models import load_vaastav_gameweeks, run_extended_player_benchmark


def season_order(season: str) -> int:
    return int(season.split("-")[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", required=True, help="directory containing Vaastav season/gws/gw*.csv files")
    parser.add_argument("--evaluation-season", default="2025-26", help="untouched final test season")
    parser.add_argument("--validation-season", default="2024-25", help="development-only model selection season")
    parser.add_argument("--output-dir", default="runs/player-models", help="directory for metrics, predictions and models")
    args = parser.parse_args()

    history_root = Path(args.history_root)
    seasons = sorted(
        [path.name for path in history_root.iterdir() if path.is_dir() and (path / "gws").exists()],
        key=season_order,
    )
    if args.evaluation_season not in seasons:
        raise SystemExit(f"evaluation season {args.evaluation_season!r} is not present under {history_root}")
    development = tuple(
        season for season in seasons if season_order(season) < season_order(args.validation_season)
    )
    if args.validation_season not in seasons:
        raise SystemExit(f"validation season {args.validation_season!r} is not present under {history_root}")
    if args.evaluation_season in development or args.validation_season in development:
        raise SystemExit("the evaluation and validation seasons must be excluded from development")

    print(f"Loading {len(seasons)} seasons: {', '.join(seasons)}", flush=True)
    raw = load_vaastav_gameweeks(history_root, seasons)
    print(f"Loaded {len(raw):,} raw player-fixture rows", flush=True)
    result = run_extended_player_benchmark(
        raw,
        development_seasons=development,
        validation_season=args.validation_season,
        evaluation_season=args.evaluation_season,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(output_dir / "evaluation-predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps({"metrics": result.metrics, "metadata": result.metadata}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if result.forecast_model is not None:
        joblib.dump(result.forecast_model, output_dir / "player-forecast-model.joblib")
    if result.price_model is not None:
        joblib.dump(result.price_model, output_dir / "price-change-model.joblib")

    # A compact audit artifact makes feature lineage and split boundaries
    # inspectable without serializing the full 200+ column feature matrix.
    audit = {
        "raw_rows": int(len(raw)),
        "feature_rows": result.metadata["feature_rows"],
        "feature_columns": result.metadata["feature_columns"],
        "rows_by_season": {
            season: int((raw["season"].astype(str) == season).sum()) for season in seasons
        },
        "raw_column_coverage": {
            column: round(float(raw[column].notna().mean()), 6)
            for column in (
                "expected_goals",
                "expected_assists",
                "expected_goal_involvements",
                "xP",
                "starts",
                "selected",
                "transfers_in",
                "transfers_out",
                "team_h_score",
                "team_a_score",
            )
            if column in raw
        },
        "development_seasons": list(development),
        "validation_season": args.validation_season,
        "evaluation_season": args.evaluation_season,
        "feature_count_used_by_model": result.metadata["feature_count"],
    }
    (output_dir / "data-audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "selected_model": result.metadata["selected_model"], "metrics": {key: result.metrics[key] for key in ("last5", "ewma5", "extended_selected", "future_price_change_ridge")}}, indent=2))


if __name__ == "__main__":
    main()
