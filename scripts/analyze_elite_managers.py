#!/usr/bin/env python3
"""Summarize the observed elite-manager benchmark without training leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_lab.elite_managers import (  # noqa: E402
    elite_behavior_priors,
    load_elite_manager_autopsy,
    summarize_manager_behavior,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--autopsy",
        default="data/elite_managers/autopsy_all.csv",
        help="downloaded derived manager-season table",
    )
    parser.add_argument("--output", default=None, help="optional JSON output path")
    args = parser.parse_args()

    frame = load_elite_manager_autopsy(args.autopsy)
    result = {
        "rows": int(len(frame)),
        "groups": sorted(str(group) for group in frame["group"].unique()),
        "behavior_by_group": summarize_manager_behavior(frame),
        "elite_behavior_priors": elite_behavior_priors(frame),
        "leakage_note": (
            "final_rank and total_points are retained for benchmark reporting only; "
            "they are not runtime/player-model features"
        ),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
