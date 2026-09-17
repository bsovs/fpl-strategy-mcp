#!/usr/bin/env python3
"""Fit descriptive action heads to observed FPL manager decisions.

This script is intentionally an audit tool, not a production trainer.  It
turns exact weekly manager squads into separate pre-deadline behavior heads:

* whether to transfer at all;
* whether to make a bundle;
* whether to accept a paid hit;
* whether to use a chip; and
* the observable profile of an incoming player versus an outgoing player.

The current local archive is 2025/26 and is the final test season, so its
fitted coefficients are descriptive only.  The output is useful for checking
whether the simulator's action assumptions resemble high-performing managers
and for defining the training contract for earlier, non-test archives.

No realized future points, final rank, final points, or post-decision fields
are included in model features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_lab.elite_managers import load_observed_manager_transitions  # noqa: E402


PRE_DEADLINE_FEATURES = (
    "gameweek",
    "gameweek_sin",
    "gameweek_cos",
    "previous_bank_tenths",
    "previous_squad_value_tenths",
    "previous_event_points",
    "previous_total_points",
    "previous_overall_rank",
    "previous_event_transfers",
    "previous_event_transfers_cost",
    "squad_mean_points_3",
    "squad_mean_points_5",
    "squad_mean_minutes_3",
    "squad_mean_minutes_5",
    "squad_mean_value",
    "squad_mean_selected",
    "squad_mean_transfer_momentum",
    "squad_low_minutes_count",
    "squad_zero_points_3_count",
)

PLAYER_SIGNAL_FEATURES = (
    "points_3",
    "points_5",
    "minutes_3",
    "minutes_5",
    "prev_value",
    "prev_selected",
    "prev_transfers_balance",
)


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
    # Aggregate fixture rows to one player/gameweek row before calculating
    # lags.  This prevents double gameweeks from duplicating recent form.
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
    frame["transfer_momentum"] = frame["prev_transfers_balance"]
    return frame.set_index(["gw", "element"])


def _safe_mean(values: Iterable[object], default: float = 0.0) -> float:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else default


def _player_state_rows(observed: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """Return observed in/out rows using state known one GW before the move."""

    rows: list[dict[str, object]] = []
    transfers = observed[
        (observed["gameweek"] > 1)
        & (observed["action_kind"].isin(["transfer", "transfer_bundle"]))
    ]
    for record in transfers.itertuples(index=False):
        decision_gw = int(record.gameweek)
        for direction, player_ids in (("in", record.incoming_ids), ("out", record.outgoing_ids)):
            for player_id in player_ids:
                key = (decision_gw - 1, int(player_id))
                if key not in state.index:
                    continue
                values = state.loc[key]
                row: dict[str, object] = {
                    "manager_id": int(record.manager_id),
                    "gameweek": decision_gw,
                    "direction": direction,
                    "direction_label": int(direction == "in"),
                }
                for column in PLAYER_SIGNAL_FEATURES:
                    value = values.get(column)
                    if column == "prev_value" and pd.notna(value):
                        value = float(value) / 10.0
                    row[column] = float(value) if pd.notna(value) else np.nan
                rows.append(row)
    return pd.DataFrame.from_records(rows)


def _squad_features(observed: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """Build one row per decision using the previous week's squad state."""

    records: list[dict[str, object]] = []
    ordered = observed.sort_values(["manager_id", "gameweek"])
    for manager_id, subset in ordered.groupby("manager_id", sort=False):
        previous = None
        previous_record = None
        for record in subset.itertuples(index=False):
            gameweek = int(record.gameweek)
            if previous is None or gameweek == 1:
                previous = tuple(record.squad_ids)
                previous_record = record
                continue
            player_rows = [state.loc[(gameweek - 1, int(player_id))] for player_id in previous if (gameweek - 1, int(player_id)) in state.index]
            numeric = {column: [row.get(column, np.nan) for row in player_rows] for column in PLAYER_SIGNAL_FEATURES}
            previous_event_points = float(getattr(previous_record, "event_points", 0.0) or 0.0)
            previous_total_points = float(getattr(previous_record, "total_points", 0.0) or 0.0)
            previous_rank = pd.to_numeric(getattr(previous_record, "overall_rank", np.nan), errors="coerce")
            row = {
                "manager_id": int(manager_id),
                "gameweek": gameweek,
                "transfer_any": int(record.action_kind in {"transfer", "transfer_bundle"}),
                "multi_transfer": int(record.action_kind == "transfer_bundle"),
                "paid_hit": int(float(record.observed_transfer_cost) > 0.0),
                "any_chip": int(record.active_chip is not None),
                "chip_kind": record.active_chip or "none",
                # These columns describe the state at the previous deadline.
                "previous_bank_tenths": int(getattr(previous_record, "bank_tenths", 0)),
                "previous_squad_value_tenths": int(getattr(previous_record, "squad_value_tenths", 0)),
                "previous_event_points": previous_event_points,
                "previous_total_points": previous_total_points,
                "previous_overall_rank": float(previous_rank) if pd.notna(previous_rank) else np.nan,
                "previous_event_transfers": int(getattr(previous_record, "observed_transfer_count", 0)),
                "previous_event_transfers_cost": float(getattr(previous_record, "observed_transfer_cost", 0.0)),
                "squad_mean_points_3": _safe_mean(numeric["points_3"]),
                "squad_mean_points_5": _safe_mean(numeric["points_5"]),
                "squad_mean_minutes_3": _safe_mean(numeric["minutes_3"]),
                "squad_mean_minutes_5": _safe_mean(numeric["minutes_5"]),
                "squad_mean_value": _safe_mean(numeric["prev_value"]),
                "squad_mean_selected": _safe_mean(numeric["prev_selected"]),
                "squad_mean_transfer_momentum": _safe_mean(numeric["prev_transfers_balance"]),
                "squad_low_minutes_count": int(sum(float(v or 0.0) < 90.0 for v in numeric["minutes_3"] if pd.notna(v))),
                "squad_zero_points_3_count": int(sum(float(v or 0.0) <= 0.0 for v in numeric["points_3"] if pd.notna(v))),
                "gameweek_sin": float(np.sin(2.0 * np.pi * gameweek / 38.0)),
                "gameweek_cos": float(np.cos(2.0 * np.pi * gameweek / 38.0)),
                "rank_band": getattr(record, "rank_band", None),
            }
            records.append(row)
            previous = tuple(record.squad_ids)
            previous_record = record
    return pd.DataFrame.from_records(records)


def _fit_grouped_head(frame: pd.DataFrame, label: str) -> dict[str, object]:
    subset = frame.dropna(subset=[label]).copy()
    y = subset[label].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return {"rows": int(len(subset)), "positive_rate": float(y.mean()) if len(y) else None, "status": "single_class"}
    x = subset[list(PRE_DEADLINE_FEATURES)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    groups = subset["manager_id"].to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    predictions = np.full(len(subset), np.nan)
    for train_indices, test_indices in GroupKFold(n_splits=n_splits).split(x, y, groups):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("logit", LogisticRegression(max_iter=500, class_weight="balanced")),
            ]
        )
        model.fit(x.iloc[train_indices], y[train_indices])
        predictions[test_indices] = model.predict_proba(x.iloc[test_indices])[:, 1]
    metrics: dict[str, object] = {
        "rows": int(len(subset)),
        "manager_count": int(subset["manager_id"].nunique()),
        "positive_rate": float(y.mean()),
        "group_cv_accuracy": float(accuracy_score(y, predictions >= 0.5)),
        "group_cv_brier": float(brier_score_loss(y, predictions)),
        "status": "fit",
    }
    if len(np.unique(y)) == 2:
        metrics["group_cv_auc"] = float(roc_auc_score(y, predictions))
    full_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    full_model.fit(x, y)
    coefficients = full_model.named_steps["logit"].coef_[0]
    metrics["coefficients"] = [
        {"feature": feature, "coefficient": float(coefficient)}
        for feature, coefficient in sorted(
            zip(PRE_DEADLINE_FEATURES, coefficients), key=lambda item: abs(item[1]), reverse=True
        )
    ]
    return metrics


def _fit_direction_head(frame: pd.DataFrame) -> dict[str, object]:
    subset = frame.dropna(subset=list(PLAYER_SIGNAL_FEATURES) + ["direction_label"]).copy()
    if subset.empty or subset["direction_label"].nunique() < 2:
        return {"rows": int(len(subset)), "status": "insufficient_class_coverage"}
    x = subset[list(PLAYER_SIGNAL_FEATURES)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = subset["direction_label"].astype(int).to_numpy()
    groups = subset["manager_id"].to_numpy()
    predictions = np.full(len(subset), np.nan)
    n_splits = min(5, len(np.unique(groups)))
    for train_indices, test_indices in GroupKFold(n_splits=n_splits).split(x, y, groups):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("logit", LogisticRegression(max_iter=500, class_weight="balanced")),
            ]
        )
        model.fit(x.iloc[train_indices], y[train_indices])
        predictions[test_indices] = model.predict_proba(x.iloc[test_indices])[:, 1]
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    model.fit(x, y)
    coefficients = model.named_steps["logit"].coef_[0]
    return {
        "rows": int(len(subset)),
        "manager_count": int(subset["manager_id"].nunique()),
        "incoming_rate": float(y.mean()),
        "group_cv_accuracy": float(accuracy_score(y, predictions >= 0.5)),
        "group_cv_brier": float(brier_score_loss(y, predictions)),
        "group_cv_auc": float(roc_auc_score(y, predictions)),
        "coefficients": [
            {"feature": feature, "coefficient": float(coefficient)}
            for feature, coefficient in sorted(
                zip(PLAYER_SIGNAL_FEATURES, coefficients), key=lambda item: abs(item[1]), reverse=True
            )
        ],
        "warning": (
            "This is an observed incoming-vs-outgoing profile, not a legal choice-set model. "
            "It must not be interpreted as a transfer recommendation without all affordable, "
            "position-compatible alternatives."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", default="data/elite_managers/season_winners_2025-26")
    parser.add_argument("--rank-index", default="data/elite_managers/manager_index.csv")
    parser.add_argument("--history-root", required=True)
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--output", default="runs/elite-manager-benchmark/action-heads.json")
    args = parser.parse_args()

    observed = load_observed_manager_transitions(args.archive_root, args.rank_index)
    state = _load_player_gameweeks(Path(args.history_root), args.season)
    decision_frame = _squad_features(observed, state)
    player_frame = _player_state_rows(observed, state)

    result = {
        "season": args.season,
        "status": "descriptive_holdout_audit",
        "coverage": {
            "manager_count": int(observed.manager_id.nunique()),
            "decision_rows": int(len(decision_frame)),
            "player_action_rows": int(len(player_frame)),
            "transfer_rows": int(decision_frame.transfer_any.sum()),
            "bundle_rows": int(decision_frame.multi_transfer.sum()),
            "paid_hit_rows": int(decision_frame.paid_hit.sum()),
            "chip_rows": int(decision_frame.any_chip.sum()),
        },
        "pre_deadline_features": list(PRE_DEADLINE_FEATURES),
        "action_heads": {
            "transfer_vs_hold": _fit_grouped_head(decision_frame, "transfer_any"),
            "bundle_vs_single_or_hold": _fit_grouped_head(decision_frame, "multi_transfer"),
            "paid_hit_vs_free_or_hold": _fit_grouped_head(decision_frame, "paid_hit"),
            "chip_vs_no_chip": _fit_grouped_head(decision_frame, "any_chip"),
        },
        "incoming_outgoing_head": _fit_direction_head(player_frame),
        "leakage_note": (
            "Features are lagged to the previous gameweek. Current action fields, realized future "
            "points, final rank, and final points are labels or audit metadata only. The archive "
            "is the untouched 2025/26 benchmark, so these heads are not production-fitted."
        ),
        "next_training_use": (
            "Fit these heads on complete pre-2025/26 weekly manager archives, then use the 2025/26 "
            "output only as a final behavior audit. The incoming/outgoing head must be replaced by "
            "a constrained choice-set ranker once affordable alternatives and selling values are available."
        ),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
