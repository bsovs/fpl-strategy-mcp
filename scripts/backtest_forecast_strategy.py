#!/usr/bin/env python3
"""Compare the extended forecast layer with the legacy simulator signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_lab.player_models import load_vaastav_gameweeks
from fpl_lab.simulator import (
    build_model_signal_cache,
    build_season_data,
    build_signal_cache,
    simulate_season,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", required=True)
    parser.add_argument("--forecast-csv", required=True, help="evaluation-predictions.csv from train_player_models.py")
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--previous-season", default="2024-25")
    parser.add_argument("--output", default="runs/player-models/strategy-backtest.json")
    parser.add_argument("--max-transfer-depth", type=int, default=3)
    args = parser.parse_args()

    raw = load_vaastav_gameweeks(args.history_root, [args.previous_season, args.season])
    current = build_season_data(raw, args.season)
    previous = build_season_data(raw, args.previous_season)
    forecast_rows = pd.read_csv(args.forecast_csv)
    extended_cache = build_model_signal_cache(current, forecast_rows)
    legacy_cache = build_signal_cache(current, previous)
    rows = []
    for source, cache in (("legacy", legacy_cache), ("extended", extended_cache)):
        for mode in ("points", "value", "template"):
            result = simulate_season(
                current,
                previous,
                policy="points_only_free_only",
                initial_squad_mode=mode,
                signal_cache=cache,
                max_transfer_depth=args.max_transfer_depth,
            )
            rows.append(
                {
                    "signal_source": source,
                    "initial_squad_mode": mode,
                    "season": result.season,
                    "gross_points": result.gross_points,
                    "hit_points": result.hit_points,
                    "net_points": result.net_points,
                    "transfers": result.transfers,
                    "paid_transfers": result.paid_transfers,
                    "chip_uses": result.chip_uses,
                    "final_bank": result.final_bank,
                    "final_squad_value": result.final_squad_value,
                }
            )
    rows.sort(key=lambda row: row["net_points"], reverse=True)
    result = {
        "season": args.season,
        "previous_season": args.previous_season,
        "target_points": 2413,
        "max_transfer_depth": args.max_transfer_depth,
        "results": rows,
        "warning": "This is a strategy smoke benchmark, not a tuned final cocktail. It does not claim that the 2,413 target is achieved.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
