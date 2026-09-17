#!/usr/bin/env python3
"""Validate and summarize the point-in-time elite-manager weekly archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_lab.elite_managers import (  # noqa: E402
    load_observed_manager_transitions,
    summarize_observed_manager_transitions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        default="data/elite_managers/season_winners_2025-26",
    )
    parser.add_argument(
        "--rank-index",
        default="data/elite_managers/manager_index.csv",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    frame = load_observed_manager_transitions(args.archive_root, args.rank_index)
    result = summarize_observed_manager_transitions(frame)
    result["leakage_note"] = (
        "season_final_rank, season_final_points, event_points, and total_points "
        "are audit outcomes and must be excluded from pre-deadline features"
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
