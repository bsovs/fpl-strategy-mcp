#!/usr/bin/env python3
"""Audit what high-performing managers bought and sold before each deadline.

This is an observational audit, not a training-label generator. The forward
points section is deliberately marked as an outcome audit and must never be
passed to a pre-deadline policy fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_lab.elite_managers import load_observed_manager_transitions  # noqa: E402


def _load_player_gameweeks(history_root: Path, season: str) -> pd.DataFrame:
    rows = []
    for path in sorted((history_root / season / "gws").glob("gw*.csv")):
        gameweek = int(path.stem.removeprefix("gw"))
        frame = pd.read_csv(path)
        frame["gw"] = gameweek
        rows.append(frame)
    if len(rows) != 38:
        raise ValueError(f"expected 38 gameweek files for {season}, found {len(rows)}")
    raw = pd.concat(rows, ignore_index=True)
    numeric = [
        "element",
        "total_points",
        "minutes",
        "value",
        "transfers_balance",
        "selected",
    ]
    for column in numeric:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    # Double gameweeks have one row per fixture. Points and minutes are additive;
    # price, ownership, and transfer fields are gameweek state values.
    frame = (
        raw.sort_values(["element", "gw"])
        .groupby(["gw", "element"], as_index=False)
        .agg(
            total_points=("total_points", "sum"),
            minutes=("minutes", "sum"),
            value=("value", "last"),
            transfers_balance=("transfers_balance", "last"),
            selected=("selected", "last"),
        )
        .sort_values(["element", "gw"])
    )
    for column in ("total_points", "minutes", "value", "transfers_balance", "selected"):
        frame[f"prev_{column}"] = frame.groupby("element")[column].shift(1)
    for window in (3, 5):
        frame[f"points_{window}"] = frame.groupby("element")["total_points"].transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).sum()
        )
        frame[f"minutes_{window}"] = frame.groupby("element")["minutes"].transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).sum()
        )
    return frame.set_index(["gw", "element"])


def _player_action_rows(observed: pd.DataFrame, player_state: pd.DataFrame) -> pd.DataFrame:
    rows = []
    transfers = observed[
        (observed["gameweek"] > 1)
        & (observed["action_kind"].isin(["transfer", "transfer_bundle"]))
    ]
    for record in transfers.itertuples(index=False):
        decision_gw = int(record.gameweek)
        for direction, player_ids in (("in", record.incoming_ids), ("out", record.outgoing_ids)):
            for player_id in player_ids:
                key = (decision_gw - 1, int(player_id))
                if key not in player_state.index:
                    continue
                state = player_state.loc[key]
                row = {
                    "rank_band": record.rank_band,
                    "manager_id": int(record.manager_id),
                    "gameweek": decision_gw,
                    "direction": direction,
                    "player_id": int(player_id),
                }
                for column in (
                    "points_3",
                    "points_5",
                    "minutes_3",
                    "minutes_5",
                    "prev_value",
                    "prev_selected",
                    "prev_transfers_balance",
                ):
                    value = state[column]
                    # Vaastav stores player prices in tenths of a million.
                    row[column] = (
                        float(value) / 10.0
                        if column == "prev_value" and pd.notna(value)
                        else (float(value) if pd.notna(value) else None)
                    )
                rows.append(row)
    return pd.DataFrame.from_records(rows)


def _forward_audit(observed: pd.DataFrame, player_state: pd.DataFrame) -> pd.DataFrame:
    # These are realized future points and are output only to measure the
    # observed decisions after the fact. They are never features.
    future_points = {}
    for player_id, subset in player_state.reset_index().groupby("element"):
        points = subset.set_index("gw")["total_points"]
        for gameweek in range(2, 39):
            future_points[(gameweek, int(player_id))] = sum(
                float(points.get(next_gameweek, 0.0))
                for next_gameweek in range(gameweek, min(gameweek + 3, 39))
            )
    rows = []
    transfers = observed[
        (observed["gameweek"] > 1)
        & (observed["action_kind"].isin(["transfer", "transfer_bundle"]))
    ]
    for record in transfers.itertuples(index=False):
        gameweek = int(record.gameweek)
        incoming = sum(future_points.get((gameweek, int(pid)), 0.0) for pid in record.incoming_ids)
        outgoing = sum(future_points.get((gameweek, int(pid)), 0.0) for pid in record.outgoing_ids)
        rows.append(
            {
                "rank_band": record.rank_band,
                "manager_id": int(record.manager_id),
                "gameweek": gameweek,
                "forward_3gw_incoming_points": incoming,
                "forward_3gw_outgoing_points": outgoing,
                "forward_3gw_net_points": incoming - outgoing,
                "positive_net": incoming > outgoing,
                "paid_hit": float(record.observed_transfer_cost) > 0,
            }
        )
    return pd.DataFrame.from_records(rows)


def _summary(frame: pd.DataFrame, numeric: tuple[str, ...]) -> dict[str, dict[str, float | int]]:
    result = {}
    for band, subset in frame.groupby("rank_band", dropna=False, sort=True):
        result[str(band)] = {
            "rows": int(len(subset)),
            **{
                f"{column}_mean": float(subset[column].mean())
                for column in numeric
                if column in subset and subset[column].notna().any()
            },
            **{
                f"{column}_median": float(subset[column].median())
                for column in numeric
                if column in subset and subset[column].notna().any()
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", default="data/elite_managers/season_winners_2025-26")
    parser.add_argument("--rank-index", default="data/elite_managers/manager_index.csv")
    parser.add_argument("--history-root", required=True)
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    observed = load_observed_manager_transitions(args.archive_root, args.rank_index)
    state = _load_player_gameweeks(Path(args.history_root), args.season)
    actions = _player_action_rows(observed, state)
    forward = _forward_audit(observed, state)
    result = {
        "season": args.season,
        "coverage": {
            "manager_count": int(observed.manager_id.nunique()),
            "gameweek_rows": int(len(observed)),
            "transfer_transition_rows": int(
                observed.action_kind.isin(["transfer", "transfer_bundle"]).sum()
            ),
            "player_transfer_rows": int(len(actions)),
        },
        "pre_deadline_behavior": {
            "incoming": _summary(
                actions[actions.direction == "in"],
                ("points_3", "points_5", "minutes_3", "minutes_5", "prev_value", "prev_selected", "prev_transfers_balance"),
            ),
            "outgoing": _summary(
                actions[actions.direction == "out"],
                ("points_3", "points_5", "minutes_3", "minutes_5", "prev_value", "prev_selected", "prev_transfers_balance"),
            ),
        },
        "forward_outcome_audit": {
            "warning": "realized future points are audit outcomes, never pre-deadline features",
            "by_rank_band": _summary(
                forward,
                ("forward_3gw_incoming_points", "forward_3gw_outgoing_points", "forward_3gw_net_points"),
            ),
            "paid_hit_net_mean": float(forward.loc[forward.paid_hit, "forward_3gw_net_points"].mean()),
            "free_transfer_net_mean": float(forward.loc[~forward.paid_hit, "forward_3gw_net_points"].mean()),
            "positive_net_rate": float(forward.positive_net.mean()),
        },
        "leakage_note": (
            "Only points/minutes/price/ownership/transfer state through GW-1 are in "
            "pre_deadline_behavior. Forward outcome fields are an audit section and "
            "must be excluded from model fitting."
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
