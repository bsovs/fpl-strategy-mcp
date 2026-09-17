"""Observed-manager behavior benchmarks for FPL policy research.

The public benchmark currently available to this project is an aggregate
manager-season table. It is useful for calibrating action priors and auditing
our replay, but it is not a point-in-time state/action label set: final rank
and final points must never be passed into a live decision model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_AUTOPSY_COLUMNS = frozenset(
    {
        "entry",
        "final_rank",
        "total_points",
        "transfer_gain_net",
        "transfer_count",
        "hits",
        "cap_agreement",
        "n_unused_chips",
        "median_hours_before_deadline",
        "pct_transfers_last_3h",
        "group",
    }
)

BEHAVIOR_COLUMNS = (
    "value_gain",
    "median_hours_before_deadline",
    "pct_transfers_last_3h",
    "xi_ownership",
    "zero_min_starters",
    "auto_sub_rescues",
    "rebuys",
    "armband_skill_vs_random",
    "transfer_gain_net",
    "transfer_count",
    "hits",
    "cap_agreement",
    "armband_points",
    "chip_timing_loss",
    "n_unused_chips",
    "hindsight_xi_loss",
)


def load_elite_manager_autopsy(path: str | Path) -> pd.DataFrame:
    """Load and validate the derived manager-season benchmark table."""

    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_AUTOPSY_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"elite-manager table is missing columns: {missing}")
    for column in BEHAVIOR_COLUMNS + ("final_rank", "total_points", "entry"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["entry"].isna().any() or frame["group"].isna().any():
        raise ValueError("elite-manager table contains invalid entry/group values")
    if frame["entry"].duplicated().any():
        raise ValueError("elite-manager table contains duplicate manager entries")
    return frame


def summarize_manager_behavior(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return auditable behavior medians by observed rank band.

    This intentionally returns both sample counts and medians. The medians are
    descriptive priors only; they do not imply that a behavior caused rank.
    """

    missing = sorted(REQUIRED_AUTOPSY_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"elite-manager frame is missing columns: {missing}")
    metrics = (
        "total_points",
        "transfer_gain_net",
        "transfer_count",
        "hits",
        "cap_agreement",
        "n_unused_chips",
        "median_hours_before_deadline",
        "pct_transfers_last_3h",
        "zero_min_starters",
        "chip_timing_loss",
    )
    result: dict[str, dict[str, Any]] = {}
    for group, subset in frame.groupby("group", sort=True):
        result[str(group)] = {
            "manager_count": int(len(subset)),
            **{
                f"{metric}_median": float(subset[metric].median())
                for metric in metrics
                if metric in subset
            },
            **{
                f"{metric}_mean": float(subset[metric].mean())
                for metric in metrics
                if metric in subset
            },
        }
    return result


def elite_behavior_priors(
    frame: pd.DataFrame,
    groups: tuple[str, ...] = ("top100", "top1k", "top10k"),
) -> dict[str, float | int | tuple[str, ...]]:
    """Build non-outcome action priors from the requested elite bands.

    Rank and points are used only to select the already-defined benchmark
    bands. The returned values are behavior quantities that can be used for
    regularization or reporting, not as player-level predictive features.
    """

    subset = frame[frame["group"].isin(groups)]
    if subset.empty:
        raise ValueError(f"no rows found for elite groups {groups!r}")
    return {
        "source_groups": groups,
        "manager_count": int(len(subset)),
        "median_transfers_per_gameweek": float(subset["transfer_count"].median() / 38.0),
        "median_hits_per_season": float(subset["hits"].median()),
        "median_cap_agreement": float(subset["cap_agreement"].median()),
        "median_unused_chips": float(subset["n_unused_chips"].median()),
        "median_hours_before_deadline": float(subset["median_hours_before_deadline"].median()),
        "median_pct_transfers_last_3h": float(subset["pct_transfers_last_3h"].median()),
        "median_transfer_gain_net": float(subset["transfer_gain_net"].median()),
    }


CHIP_NAME_MAP = {
    "wildcard": "wildcard",
    "freehit": "free_hit",
    "bboost": "bench_boost",
    "3xc": "triple_captain",
    "manager": "assistant_manager",
}


def _load_manager_gameweeks(manager_dir: Path) -> list[dict[str, Any]]:
    complete_path = manager_dir / "_complete_season.json"
    if complete_path.exists():
        payload = json.loads(complete_path.read_text(encoding="utf-8"))
        return [payload[str(gameweek)] for gameweek in range(1, 39)]
    rows = []
    for gameweek in range(1, 39):
        path = manager_dir / f"gw{gameweek:02d}.json"
        if not path.exists():
            raise ValueError(f"missing GW{gameweek} for manager {manager_dir.name}")
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def load_observed_manager_transitions(
    archive_root: str | Path,
    rank_index_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load weekly manager states and derive observed squad transitions.

    The returned frame retains outcome columns for audit, but callers must
    remove them before fitting a point-in-time model. A transition from GW-1 to
    GW is the observed action for GW; GW1 has no previous squad and is marked
    as an opening state rather than an inferred transfer.
    """

    root = Path(archive_root)
    manager_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not manager_dirs:
        raise ValueError(f"no manager directories found under {root}")
    rank_frame = None
    if rank_index_path is not None:
        rank_frame = pd.read_csv(rank_index_path)
        required = {"entry_id", "overall_rank", "group", "total_points"}
        missing = sorted(required.difference(rank_frame.columns))
        if missing:
            raise ValueError(f"manager rank index is missing columns: {missing}")
        rank_frame = rank_frame.set_index("entry_id")

    records: list[dict[str, Any]] = []
    for manager_dir in manager_dirs:
        try:
            manager_id = int(manager_dir.name)
        except ValueError as exc:
            raise ValueError(f"invalid manager directory name: {manager_dir.name!r}") from exc
        gameweeks = _load_manager_gameweeks(manager_dir)
        if len(gameweeks) != 38:
            raise ValueError(f"manager {manager_id} does not have 38 gameweeks")
        previous_squad: set[int] | None = None
        for expected_gameweek, payload in enumerate(gameweeks, start=1):
            history = payload.get("entry_history", {})
            gameweek = int(history.get("event", expected_gameweek))
            if gameweek != expected_gameweek:
                raise ValueError(f"manager {manager_id} has out-of-order gameweek {gameweek}")
            picks = payload.get("picks", [])
            if len(picks) != 15:
                raise ValueError(f"manager {manager_id} GW{gameweek} has {len(picks)} picks")
            squad = tuple(sorted(int(pick["element"]) for pick in picks))
            if len(set(squad)) != 15:
                raise ValueError(f"manager {manager_id} GW{gameweek} has duplicate picks")
            active_chip_raw = payload.get("active_chip")
            active_chip = CHIP_NAME_MAP.get(active_chip_raw, active_chip_raw)
            captain_ids = [int(pick["element"]) for pick in picks if pick.get("is_captain")]
            vice_ids = [int(pick["element"]) for pick in picks if pick.get("is_vice_captain")]
            squad_set = set(squad)
            incoming = tuple(sorted(squad_set - previous_squad)) if previous_squad is not None else ()
            outgoing = tuple(sorted(previous_squad - squad_set)) if previous_squad is not None else ()
            action_kind = active_chip or ("hold" if not incoming and not outgoing else "transfer")
            if len(incoming) > 1 or len(outgoing) > 1:
                action_kind = "transfer_bundle" if not active_chip else action_kind
            record: dict[str, Any] = {
                "manager_id": manager_id,
                "gameweek": gameweek,
                "action_kind": action_kind,
                "active_chip": active_chip,
                "squad_ids": squad,
                "starting_xi_ids": tuple(sorted(int(pick["element"]) for pick in picks if int(pick.get("multiplier", 0)) > 0)),
                "captain_id": captain_ids[0] if captain_ids else None,
                "vice_captain_id": vice_ids[0] if vice_ids else None,
                "incoming_ids": incoming,
                "outgoing_ids": outgoing,
                "observed_transfer_count": int(history.get("event_transfers", 0)),
                "observed_transfer_cost": float(history.get("event_transfers_cost", 0)) / 1.0,
                "squad_overlap_prev": (
                    float(len(squad_set & previous_squad) / 15.0)
                    if previous_squad is not None
                    else None
                ),
                # These are audit outcomes, not point-in-time features.
                "event_points": float(history.get("points", 0)),
                "total_points": float(history.get("total_points", 0)),
                "overall_rank": history.get("overall_rank"),
                "bank_tenths": int(history.get("bank", 0)),
                "squad_value_tenths": int(history.get("value", 0)),
            }
            if rank_frame is not None and manager_id in rank_frame.index:
                rank_row = rank_frame.loc[manager_id]
                record["rank_band"] = str(rank_row["group"])
                record["season_final_rank"] = int(rank_row["overall_rank"])
                record["season_final_points"] = int(rank_row["total_points"])
            else:
                record["rank_band"] = None
                record["season_final_rank"] = None
                record["season_final_points"] = None
            records.append(record)
            previous_squad = squad_set
    return pd.DataFrame.from_records(records)


def summarize_observed_manager_transitions(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize weekly action behavior and data-integrity counts."""

    required = {"manager_id", "gameweek", "action_kind", "squad_ids", "squad_overlap_prev"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"weekly manager frame is missing columns: {missing}")
    action_counts = frame["action_kind"].value_counts(dropna=False).to_dict()
    observed = frame[frame["gameweek"] > 1]
    by_group = {}
    for group, subset in frame.groupby("rank_band", dropna=False, sort=True):
        weekly = subset[subset["gameweek"] > 1]
        by_group[str(group)] = {
            "manager_count": int(subset["manager_id"].nunique()),
            "gameweek_rows": int(len(subset)),
            "transfer_gameweek_rate": float((weekly["action_kind"].isin(["transfer", "transfer_bundle"])).mean()),
            "multi_transfer_gameweeks": int((weekly["action_kind"] == "transfer_bundle").sum()),
            "paid_hit_gameweeks": int((weekly["observed_transfer_cost"] > 0).sum()),
            "wildcard_gameweeks": int((weekly["action_kind"] == "wildcard").sum()),
            "free_hit_gameweeks": int((weekly["action_kind"] == "free_hit").sum()),
            "median_squad_overlap_prev": float(weekly["squad_overlap_prev"].median()),
        }
    return {
        "manager_count": int(frame["manager_id"].nunique()),
        "gameweek_rows": int(len(frame)),
        "action_counts": {str(key): int(value) for key, value in action_counts.items()},
        "opening_rows": int((frame["gameweek"] == 1).sum()),
        "observed_action_rows": int(len(observed)),
        "by_rank_band": by_group,
    }
