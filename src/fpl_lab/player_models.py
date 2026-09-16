"""Literature-driven player-point benchmarks.

This is deliberately an OpenFPL-style first pass, not a byte-for-byte
reimplementation: it uses public FPL history, position-specific regressors,
short/medium rolling windows, and explicit availability proxies from prior
minutes. Understat features and the paper's K-best search remain separate
follow-up work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OPTIONAL_NUMERIC_COLUMNS = (
    "assists",
    "bonus",
    "bps",
    "clean_sheets",
    "creativity",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals",
    "goals_conceded",
    "goals_scored",
    "ict_index",
    "influence",
    "minutes",
    "starts",
    "saves",
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
    "xP",
    "threat",
    "total_points",
    "value",
    "yellow_cards",
    "team_h_score",
    "team_a_score",
    "element",
    "id",
)


def _season_order(season: str) -> int:
    return int(season.split("-")[0])


def load_vaastav_gameweeks(root: str | Path, seasons: Iterable[str]) -> pd.DataFrame:
    """Load raw Vaastav GW files into one player-match table."""

    root = Path(root)
    frames: list[pd.DataFrame] = []
    for season in seasons:
        for path in sorted((root / season / "gws").glob("gw*.csv"), key=lambda p: int(p.stem[2:])):
            gameweek = int(path.stem[2:])
            try:
                frame = pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                # Some early Vaastav exports contain accented names encoded as
                # Latin-1. Keep the fallback explicit so a future encoding
                # change does not silently replace characters.
                frame = pd.read_csv(path, encoding="latin-1")
            # Some older/newer archive files contain columns that are entirely
            # empty for that file. Dropping those per-file avoids concat dtype
            # drift while the final union still preserves columns from files
            # where they exist.
            frame = frame.dropna(axis=1, how="all")
            frame["season"] = season
            frame["season_order"] = _season_order(season)
            frame["gameweek"] = gameweek
            frames.append(frame)
    if not frames:
        raise ValueError(f"no Vaastav gameweek files found under {root}")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return _enrich_historical_metadata(combined, root)


def _enrich_historical_metadata(frame: pd.DataFrame, root: str | Path) -> pd.DataFrame:
    """Recover missing early-season team/position fields with provenance.

    Vaastav's oldest GW exports predate the merged player columns and contain
    ``element`` but no position or team. ``players_raw.csv`` is the available
    season-level roster snapshot, so it is used only to fill missing values.
    The imputation flags are retained for audit and downstream ablation tests.
    """

    root = Path(root)
    frame = frame.copy()
    if "team" not in frame:
        frame["team"] = np.nan
    if "position" not in frame:
        frame["position"] = np.nan
    frame["team"] = frame["team"].astype(object)
    frame["position"] = frame["position"].astype(object)
    frame["metadata_team_imputed"] = 0.0
    frame["metadata_position_imputed"] = 0.0
    position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    for season in frame["season"].astype(str).unique():
        metadata_path = root / season / "players_raw.csv"
        if not metadata_path.exists():
            continue
        try:
            metadata = pd.read_csv(metadata_path, encoding="utf-8")
        except UnicodeDecodeError:
            metadata = pd.read_csv(metadata_path, encoding="latin-1")
        if "id" in metadata and "element" not in metadata:
            metadata["element"] = metadata["id"]
        needed = [column for column in ("element", "team", "element_type") if column in metadata]
        if "element" not in needed:
            continue
        metadata = metadata[needed].copy()
        metadata["element"] = pd.to_numeric(metadata["element"], errors="coerce")
        metadata = metadata.dropna(subset=["element"]).drop_duplicates("element")
        metadata["metadata_position"] = metadata.get("element_type", pd.Series(index=metadata.index)).map(position_map)
        season_mask = frame["season"].astype(str) == season
        lookup = frame.loc[season_mask, ["element", "team", "position"]].copy()
        lookup["_row_index"] = lookup.index
        lookup["element"] = pd.to_numeric(lookup["element"], errors="coerce")
        lookup = lookup.merge(metadata, on="element", how="left", suffixes=("", "_metadata"))
        team_missing = lookup["team"].isna() & lookup["team_metadata"].notna()
        position_missing = lookup["position"].isna() & lookup["metadata_position"].notna()
        if team_missing.any():
            frame.loc[lookup.loc[team_missing, "_row_index"], "team"] = lookup.loc[team_missing, "team_metadata"].to_numpy()
            frame.loc[lookup.loc[team_missing, "_row_index"], "metadata_team_imputed"] = 1.0
        if position_missing.any():
            frame.loc[lookup.loc[position_missing, "_row_index"], "position"] = lookup.loc[position_missing, "metadata_position"].to_numpy()
            frame.loc[lookup.loc[position_missing, "_row_index"], "metadata_position_imputed"] = 1.0
    return frame


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _strict_prior(frame: pd.DataFrame, column: str, group_column: str = "player_key") -> pd.Series:
    """Return a value known before the current gameweek.

    Vaastav's merged files can contain multiple fixtures for one player in a
    double gameweek.  A plain ``shift(1)`` would let the first fixture's
    realized score become a feature for the second fixture, even though both
    were known at the same FPL deadline.  The sequence guard removes that
    within-gameweek leakage while retaining history from earlier gameweeks and
    seasons.
    """

    grouped = frame.groupby(group_column, sort=False)
    prior = grouped[column].shift(1)
    prior_sequence = grouped["sequence"].shift(1)
    return prior.where(prior_sequence < frame["sequence"])


def _rolling_prior(frame: pd.DataFrame, prior: pd.Series, window: int, group_column: str = "player_key") -> pd.Series:
    return prior.groupby(frame[group_column], sort=False).transform(
        lambda series: series.rolling(window, min_periods=1).mean()
    )


def _ewma_prior(frame: pd.DataFrame, prior: pd.Series, span: int, group_column: str = "player_key") -> pd.Series:
    return prior.groupby(frame[group_column], sort=False).transform(
        lambda series: series.ewm(span=span, adjust=False, min_periods=1).mean()
    )


def _std_prior(frame: pd.DataFrame, prior: pd.Series, window: int, group_column: str = "player_key") -> pd.Series:
    return prior.groupby(frame[group_column], sort=False).transform(
        lambda series: series.rolling(window, min_periods=2).std()
    )


def build_player_feature_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe lagged rolling features for next-match prediction."""

    required = {"name", "position", "team", "opponent_team", "kickoff_time", "was_home", "total_points"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"player history missing columns: {sorted(missing)}")

    frame = raw.copy()
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
    normalized_name = frame["name"].astype(str).str.strip().str.lower()
    element = _numeric(frame, "element")
    frame["player_key"] = np.where(
        element > 0,
        "id:" + element.astype(int).astype(str),
        "name:" + normalized_name,
    )
    frame["position"] = frame["position"].astype(str)
    # Vaastav's newer exports include manager rows as ``AM`` and have used
    # ``GKP`` for goalkeepers in some seasons.  Keep the four FPL player
    # positions and normalize the historical goalkeeper label.
    frame["position"] = frame["position"].replace({"GKP": "GK"})
    frame = frame[frame["position"].isin({"GK", "DEF", "MID", "FWD"})].copy()
    frame["team"] = frame["team"].astype(str)
    frame["opponent_team"] = frame["opponent_team"].astype(str)
    frame["was_home"] = frame["was_home"].astype(int)
    for column in OPTIONAL_NUMERIC_COLUMNS:
        frame[f"{column}_observed"] = frame[column].notna().astype(float) if column in frame else 0.0
        frame[column] = _numeric(frame, column)
    frame = frame.sort_values(["season_order", "gameweek", "kickoff_time", "player_key"]).reset_index(drop=True)
    frame["sequence"] = frame["season_order"] * 100 + frame["gameweek"]
    # Historical context must be cut off at the simulated FPL deadline, not
    # at kickoff.  Use the first fixture of the gameweek so a later
    # double-gameweek article cannot leak into the original transfer decision.
    first_kickoff = frame.groupby(["season", "gameweek"], sort=False)["kickoff_time"].transform("min")
    frame["decision_time"] = first_kickoff - pd.Timedelta(minutes=90)

    grouped = frame.groupby("player_key", sort=False)
    source_features = [
        "total_points",
        "minutes",
        "goals_scored",
        "assists",
        "bonus",
        "bps",
        "ict_index",
        "creativity",
        "threat",
        "expected_goals",
        "expected_assists",
        "value",
    ]
    for column in source_features:
        prior = _strict_prior(frame, column)
        frame[f"{column}_last"] = prior
        for window in (3, 5, 10):
            frame[f"{column}_mean_{window}"] = _rolling_prior(frame, prior, window)
    frame["games_before"] = grouped.cumcount()
    frame["minutes_last"] = frame["minutes_last"].fillna(0.0)
    points_prior = _strict_prior(frame, "total_points")
    frame["points_ewma_5"] = _ewma_prior(frame, points_prior, 5)
    frame["points_mean_3"] = frame["total_points_mean_3"]
    frame["points_mean_5"] = frame["total_points_mean_5"]
    frame["points_mean_10"] = frame["total_points_mean_10"]
    frame["minutes_ewma_5"] = _ewma_prior(frame, _strict_prior(frame, "minutes"), 5)
    frame["target_points"] = frame["total_points"]
    frame["availability_proxy"] = (frame["minutes_ewma_5"] / 90.0).clip(0.0, 1.0)
    frame["season_gameweek"] = frame["season"].astype(str) + "_" + frame["gameweek"].astype(str)

    # The current row's realized columns remain available for the target and
    # audit, but the feature list below contains only lagged values plus match
    # context known before kickoff.
    return frame


EXTENDED_SOURCE_FEATURES = (
    "total_points",
    "xP",
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "clean_sheets",
    "saves",
    "goals_conceded",
    "yellow_cards",
    "value",
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
)


def _fixture_key(frame: pd.DataFrame) -> pd.Series:
    """Build a stable row key without trusting the season's fixture id."""

    return (
        frame["season_order"].astype(str)
        + "|"
        + frame["gameweek"].astype(str)
        + "|"
        + frame["kickoff_time"].astype(str)
        + "|"
        + frame["team"].astype(str)
        + "|"
        + frame["opponent_team"].astype(str)
        + "|"
        + frame["was_home"].astype(str)
    )


def _build_team_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create as-of team and matchup features from prior completed fixtures."""

    key_columns = [
        "season_order",
        "gameweek",
        "kickoff_time",
        "team",
        "opponent_team",
        "was_home",
    ]
    team_rows = frame[key_columns + ["team_h_score", "team_a_score"]].drop_duplicates(key_columns).copy()
    team_rows["fixture_key"] = _fixture_key(team_rows)
    team_rows["reverse_fixture_key"] = (
        team_rows["season_order"].astype(str)
        + "|"
        + team_rows["gameweek"].astype(str)
        + "|"
        + team_rows["kickoff_time"].astype(str)
        + "|"
        + team_rows["opponent_team"].astype(str)
        + "|"
        + team_rows["team"].astype(str)
        + "|"
        + (1 - team_rows["was_home"]).astype(str)
    )
    home_score = team_rows["team_h_score"]
    away_score = team_rows["team_a_score"]
    team_rows["goals_for"] = np.where(team_rows["was_home"].astype(bool), home_score, away_score)
    team_rows["goals_against"] = np.where(team_rows["was_home"].astype(bool), away_score, home_score)
    team_rows["match_points"] = np.select(
        [team_rows["goals_for"] > team_rows["goals_against"], team_rows["goals_for"] == team_rows["goals_against"]],
        [3.0, 1.0],
        default=0.0,
    )
    team_rows["clean_sheet"] = (team_rows["goals_against"] == 0).astype(float)
    team_rows = team_rows.sort_values(["team", "season_order", "gameweek", "kickoff_time"])
    team_rows["sequence"] = team_rows["season_order"] * 100 + team_rows["gameweek"]
    group = team_rows.groupby("team", sort=False)
    for column in ("match_points", "goals_for", "goals_against", "clean_sheet"):
        prior = group[column].shift(1)
        prior_sequence = group["sequence"].shift(1)
        prior = prior.where(prior_sequence < team_rows["sequence"])
        team_rows[f"team_{column}_last"] = prior
        for window in (3, 5, 10):
            team_rows[f"team_{column}_mean_{window}"] = _rolling_prior(
                team_rows, prior, window, group_column="team"
            )
    team_rows["matchup_group"] = team_rows["team"].astype(str) + "|" + team_rows["opponent_team"].astype(str)
    matchup = team_rows.sort_values(["matchup_group", "season_order", "gameweek", "kickoff_time"])
    matchup_group = matchup.groupby("matchup_group", sort=False)
    prior_points = matchup_group["match_points"].shift(1)
    prior_sequence = matchup_group["sequence"].shift(1)
    prior_points = prior_points.where(prior_sequence < matchup["sequence"])
    matchup["matchup_points_mean_5"] = _rolling_prior(matchup, prior_points, 5, group_column="matchup_group")
    matchup["matchup_goals_for_mean_5"] = _rolling_prior(
        matchup,
        matchup_group["goals_for"].shift(1).where(prior_sequence < matchup["sequence"]),
        5,
        group_column="matchup_group",
    )
    team_rows = team_rows.merge(
        matchup[["fixture_key", "matchup_points_mean_5", "matchup_goals_for_mean_5"]],
        on="fixture_key",
        how="left",
        validate="one_to_one",
    )
    feature_columns = [
        "fixture_key",
        "reverse_fixture_key",
        "team_match_points_mean_5",
        "team_match_points_mean_10",
        "team_goals_for_mean_5",
        "team_goals_for_mean_10",
        "team_goals_against_mean_5",
        "team_goals_against_mean_10",
        "team_clean_sheet_mean_5",
        "team_clean_sheet_mean_10",
        "matchup_points_mean_5",
        "matchup_goals_for_mean_5",
    ]
    return team_rows[feature_columns].drop_duplicates("fixture_key")


def _add_context_features(frame: pd.DataFrame, context_store: object | None) -> None:
    """Attach point-in-time news/social aggregates when supplied.

    Historical Vaastav archives do not contain an auditable news stream, so
    the default remains zero rather than silently joining modern articles to
    old seasons.  A caller can pass ``ContextStore`` records with publication
    timestamps to turn these columns on for a live or replayed decision.
    """

    context_columns = (
        "news_availability_delta",
        "news_role_security_delta",
        "context_set_piece_delta",
        "context_transfer_role_delta",
        "news_risk",
        "news_sentiment",
        "social_sentiment",
        "news_price_pressure",
        "context_reliability",
        "context_event_count",
        "context_news_count",
        "context_social_count",
    )
    if context_store is None:
        for column in context_columns:
            frame[column] = 0.0
        return
    rows: list[dict[str, float]] = []
    # Fixture-grain history repeats the same player/deadline context on every
    # fixture in a double gameweek.  Cache those immutable aggregates so a
    # large archived role/news stream does not make feature construction
    # quadratic in the number of fixture rows.
    feature_cache: dict[tuple[str, str, str, str], object] = {}
    for _, row in frame.iterrows():
        player_id = str(int(row["element"])) if float(row.get("element", 0.0)) > 0 else ""
        player_name = str(row["name"])
        team = str(row["team"])
        as_of = row.get("decision_time", row["kickoff_time"])
        cache_key = (player_id, player_name, team, str(as_of))
        features = feature_cache.get(cache_key)
        if features is None:
            features = context_store.features_for_player(player_id, player_name, team, as_of)
            feature_cache[cache_key] = features
        rows.append(
            {
                "news_availability_delta": features.availability_delta,
                "news_role_security_delta": features.role_security_delta,
                "context_set_piece_delta": features.set_piece_delta,
                "context_transfer_role_delta": features.transfer_role_delta,
                "news_risk": features.news_risk,
                "news_sentiment": features.news_sentiment,
                "social_sentiment": features.social_sentiment,
                "news_price_pressure": features.price_pressure,
                "context_reliability": features.reliability,
                "context_event_count": float(features.event_count),
                "context_news_count": float(features.news_count),
                "context_social_count": float(features.social_count),
            }
        )
    context_frame = pd.DataFrame(rows, index=frame.index)
    for column in context_columns:
        frame[column] = context_frame[column].astype(float)


def _add_official_snapshot_features(frame: pd.DataFrame, official_snapshot_store: object | None) -> None:
    """Attach latest archived bootstrap estimates available at each cutoff."""

    columns = (
        "official_snapshot_available",
        "official_availability_probability",
        "official_chance_this_round",
        "official_chance_next_round",
        "official_ep_this",
        "official_ep_next",
        "official_form",
        "official_points_per_game",
        "official_selected_by_percent",
        "official_transfers_in_event",
        "official_transfers_out_event",
        "official_value",
        "official_news_present",
        "official_snapshot_age_days",
    )
    if official_snapshot_store is None:
        for column in columns:
            frame[column] = 0.0
        return
    rows: list[dict[str, float]] = []
    feature_cache: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in frame.iterrows():
        player_id = str(int(row["element"])) if float(row.get("element", 0.0)) > 0 else ""
        as_of = row.get("decision_time", row["kickoff_time"])
        cache_key = (player_id, str(as_of))
        features = feature_cache.get(cache_key)
        if features is None:
            features = official_snapshot_store.features_for_player(player_id, as_of)
            feature_cache[cache_key] = features
        rows.append(features)
    snapshot_frame = pd.DataFrame(rows, index=frame.index)
    for column in columns:
        frame[column] = pd.to_numeric(snapshot_frame[column], errors="coerce").fillna(0.0).astype(float)


PLAYER_GAMEWEEK_LAST_COLUMNS = frozenset(
    {"value", "selected", "transfers_in", "transfers_out", "transfers_balance"}
)


def _build_player_gameweek_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse fixture rows to the information known at one player/GW cutoff."""

    keys = ["player_key", "player_season_key", "season", "season_order", "gameweek", "sequence"]
    ordered = frame.sort_values(["player_key", "season_order", "gameweek", "kickoff_time"])
    aggregations = {
        column: ("last" if column in PLAYER_GAMEWEEK_LAST_COLUMNS else "sum")
        for column in EXTENDED_SOURCE_FEATURES
    }
    summary = ordered.groupby(keys, as_index=False, sort=False).agg(aggregations)
    return summary.sort_values(["player_key", "season_order", "gameweek"]).reset_index(drop=True)


def _broadcast_player_gameweek_features(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Broadcast distinct player/GW features to every fixture row in that GW."""

    columns = list(columns)
    if not columns:
        return frame
    lookup = summary.set_index(["player_season_key", "sequence"])[columns]
    keys = pd.MultiIndex.from_frame(frame[["player_season_key", "sequence"]])
    values = lookup.reindex(keys)
    values.index = frame.index
    overlapping = [column for column in columns if column in frame.columns]
    if overlapping:
        frame = frame.drop(columns=overlapping)
    return pd.concat([frame, values], axis=1)


def build_extended_player_feature_table(
    raw: pd.DataFrame,
    context_store: object | None = None,
    official_snapshot_store: object | None = None,
) -> pd.DataFrame:
    """Build the richer point-in-time feature table used for model training.

    The table intentionally includes only information available before the
    target fixture: prior player form and role, ownership/transfer momentum,
    lagged team and matchup form, fixture-shape counts, and optional as-of
    context.  Future rows are used only to create labels or schedule counts;
    their realized scores never enter a predictor.
    """

    frame = build_player_feature_table(raw)
    for column in ("metadata_team_imputed", "metadata_position_imputed"):
        if column not in frame:
            frame[column] = 0.0
    frame["player_season_key"] = frame["player_key"].astype(str) + "|" + frame["season"].astype(str)
    player_gameweeks = _build_player_gameweek_summary(frame)

    # Keep a second history keyed only by player.  The season-local features
    # below are useful for recency, but resetting them at every season means
    # a GW1 decision cannot use the prior season that was genuinely known at
    # that deadline.  These career features are still point-in-time safe:
    # ``_strict_prior`` excludes the current gameweek, including double-GW
    # fixtures, before any rolling statistic is computed.  The calculation is
    # performed on distinct player/gameweek rows so the second fixture in a
    # double GW cannot create a fake missing or repeated lag.
    career_source_features = (
        "total_points",
        "xP",
        "minutes",
        "starts",
        "goals_scored",
        "assists",
        "expected_goals",
        "expected_assists",
        "value",
        "selected",
        "transfers_in",
        "transfers_out",
        "transfers_balance",
    )
    career_features: dict[str, pd.Series] = {}
    for column in career_source_features:
        prior = _strict_prior(player_gameweeks, column)
        career_features[f"career_{column}_last"] = prior
        career_features[f"career_{column}_mean_5"] = _rolling_prior(player_gameweeks, prior, 5)
        career_features[f"career_{column}_mean_10"] = _rolling_prior(player_gameweeks, prior, 10)
        career_features[f"career_{column}_ewma_5"] = _ewma_prior(player_gameweeks, prior, 5)
    player_gameweeks = pd.concat(
        [player_gameweeks, pd.DataFrame(career_features, index=player_gameweeks.index)], axis=1
    )
    player_gameweeks["career_games_before"] = player_gameweeks.groupby("player_key", sort=False).cumcount()

    computed_features: dict[str, pd.Series] = {}
    for column in EXTENDED_SOURCE_FEATURES:
        prior = _strict_prior(player_gameweeks, column, group_column="player_season_key")
        computed_features[f"{column}_last"] = prior
        for window in (3, 5, 10):
            computed_features[f"{column}_mean_{window}"] = _rolling_prior(
                player_gameweeks, prior, window, group_column="player_season_key"
            )
        computed_features[f"{column}_ewma_5"] = _ewma_prior(
            player_gameweeks, prior, 5, group_column="player_season_key"
        )
    player_gameweeks = pd.concat(
        [player_gameweeks, pd.DataFrame(computed_features, index=player_gameweeks.index)], axis=1
    )
    player_gameweeks = pd.concat(
        [
            player_gameweeks,
            pd.DataFrame(
                {
                    "points_ewma_5": player_gameweeks["total_points_ewma_5"],
                    "points_mean_3": player_gameweeks["total_points_mean_3"],
                    "points_mean_5": player_gameweeks["total_points_mean_5"],
                    "points_mean_10": player_gameweeks["total_points_mean_10"],
                },
                index=player_gameweeks.index,
            ),
        ],
        axis=1,
    )

    std_features: dict[str, pd.Series] = {}
    for column in EXTENDED_SOURCE_FEATURES:
        std_features[f"{column}_std_5"] = _std_prior(
            player_gameweeks,
            _strict_prior(player_gameweeks, column, group_column="player_season_key"),
            5,
            group_column="player_season_key",
        )
    player_gameweeks = pd.concat(
        [player_gameweeks, pd.DataFrame(std_features, index=player_gameweeks.index)], axis=1
    )

    points_prior = player_gameweeks["total_points_last"]
    minutes_prior = player_gameweeks["minutes_last"]
    transfers_in_prior = player_gameweeks["transfers_in_last"]
    transfers_out_prior = player_gameweeks["transfers_out_last"]
    value_prior = player_gameweeks["value_last"]
    derived_features: dict[str, pd.Series] = {}
    derived_features["points_ewma_3"] = _ewma_prior(
        player_gameweeks, points_prior, 3, group_column="player_season_key"
    )
    derived_features["points_ewma_10"] = _ewma_prior(
        player_gameweeks, points_prior, 10, group_column="player_season_key"
    )
    derived_features["form_acceleration"] = derived_features["points_ewma_3"] - derived_features["points_ewma_10"]
    derived_features["points_vs_career"] = player_gameweeks["points_mean_5"] - player_gameweeks["career_total_points_mean_5"]
    derived_features["minutes_vs_career"] = player_gameweeks["minutes_mean_5"] - player_gameweeks["career_minutes_mean_5"]
    derived_features["price_vs_career"] = player_gameweeks["value_mean_5"] - player_gameweeks["career_value_mean_5"]
    derived_features["minutes_ewma_10"] = _ewma_prior(
        player_gameweeks, minutes_prior, 10, group_column="player_season_key"
    )
    derived_features["minutes_trend"] = player_gameweeks["minutes_ewma_5"] - derived_features["minutes_ewma_10"]
    derived_features["start_rate_5"] = _rolling_prior(
        player_gameweeks,
        player_gameweeks["starts_last"],
        5,
        group_column="player_season_key",
    )
    derived_features["minutes_share_5"] = player_gameweeks["minutes_mean_5"] / 90.0
    derived_features["breakout_signal"] = (
        derived_features["points_vs_career"].fillna(0.0)
        * derived_features["minutes_share_5"].fillna(0.0)
    )
    derived_features["blank_rate_5"] = _rolling_prior(
        player_gameweeks,
        (points_prior <= 2).astype(float),
        5,
        group_column="player_season_key",
    )
    derived_features["hauler_rate_5"] = _rolling_prior(
        player_gameweeks,
        (points_prior >= 5).astype(float),
        5,
        group_column="player_season_key",
    )
    selected_prior = player_gameweeks["selected_last"]
    derived_features["ownership_change_5"] = _rolling_prior(
        player_gameweeks,
        selected_prior,
        3,
        group_column="player_season_key",
    ) - _rolling_prior(
        player_gameweeks,
        selected_prior,
        10,
        group_column="player_season_key",
    )
    derived_features["transfer_in_out_ratio_5"] = (
        _rolling_prior(player_gameweeks, transfers_in_prior, 5, group_column="player_season_key")
        / (_rolling_prior(player_gameweeks, transfers_out_prior, 5, group_column="player_season_key") + 1.0)
    )
    derived_features["transfer_balance_mean_5"] = _rolling_prior(
        player_gameweeks,
        player_gameweeks["transfers_balance_last"],
        5,
        group_column="player_season_key",
    )
    derived_features["price_momentum"] = value_prior - _rolling_prior(
        player_gameweeks, value_prior, 5, group_column="player_season_key"
    )
    derived_features["value_per_point_5"] = player_gameweeks["value_mean_5"] / (player_gameweeks["points_mean_5"] + 1.0)
    derived_features["points_per_value_5"] = player_gameweeks["points_mean_5"] / (player_gameweeks["value_mean_5"] + 1.0)
    player_gameweeks = pd.concat(
        [player_gameweeks, pd.DataFrame(derived_features, index=player_gameweeks.index)], axis=1
    )

    feature_columns = [
        *career_features,
        "career_games_before",
        *computed_features,
        *std_features,
        *derived_features,
    ]
    # Availability targets are gameweek-level outcomes.  Keeping them under
    # explicit target names prevents the current realized minutes from being
    # accidentally included in the predictor feature list.
    player_gameweeks["target_minutes_gw"] = player_gameweeks["minutes"]
    player_gameweeks["target_starts_gw"] = player_gameweeks["starts"]
    player_gameweeks["target_played_gw"] = (player_gameweeks["minutes"] > 0).astype(float)
    feature_columns.extend(["target_minutes_gw", "target_starts_gw", "target_played_gw"])
    frame = _broadcast_player_gameweek_features(frame, player_gameweeks, feature_columns)
    frame["points_ewma_5"] = frame["total_points_ewma_5"]
    frame["points_mean_3"] = frame["total_points_mean_3"]
    frame["points_mean_5"] = frame["total_points_mean_5"]
    frame["points_mean_10"] = frame["total_points_mean_10"]
    # Price changes are gameweek-level labels.  Using the next fixture row
    # would create a spurious zero/within-double-GW movement when a player has
    # two fixtures in the same gameweek.  Build the label on distinct
    # player/gameweek rows, then broadcast it back to every fixture row.
    price_labels = (
        frame[["player_season_key", "sequence", "value"]]
        .drop_duplicates(["player_season_key", "sequence"], keep="last")
        .sort_values(["player_season_key", "sequence"])
    )
    price_labels["next_gameweek_value"] = price_labels.groupby("player_season_key", sort=False)["value"].shift(-1)
    price_labels["target_price_change"] = price_labels["next_gameweek_value"] - price_labels["value"]
    frame = frame.merge(
        price_labels[["player_season_key", "sequence", "target_price_change"]],
        on=["player_season_key", "sequence"],
        how="left",
        validate="many_to_one",
    )

    # Direct multi-horizon labels are more useful for transfer decisions than
    # multiplying a one-fixture forecast by a fixture count.  Aggregate the
    # realized points to distinct player/gameweek rows first, so a double GW
    # contributes both fixtures exactly once to the 3- and 8-GW targets.
    horizon_labels = (
        frame.groupby(["player_season_key", "season_order", "gameweek", "sequence"], as_index=False, sort=False)
        .agg(
            gameweek_points=("total_points", "sum"),
            gameweek_minutes=("minutes", "sum"),
            gameweek_starts=("starts", "sum"),
        )
        .sort_values(["player_season_key", "gameweek"])
    )
    point_lookup = horizon_labels.set_index(["player_season_key", "gameweek"])["gameweek_points"]
    minutes_lookup = horizon_labels.set_index(["player_season_key", "gameweek"])["gameweek_minutes"]
    starts_lookup = horizon_labels.set_index(["player_season_key", "gameweek"])["gameweek_starts"]
    for horizon in (3, 8):
        total = np.zeros(len(horizon_labels), dtype=float)
        minutes_total = np.zeros(len(horizon_labels), dtype=float)
        starts_total = np.zeros(len(horizon_labels), dtype=float)
        for offset in range(horizon):
            lookup_keys = pd.MultiIndex.from_arrays(
                [horizon_labels["player_season_key"], horizon_labels["gameweek"] + offset]
            )
            total += point_lookup.reindex(lookup_keys).fillna(0.0).to_numpy()
            minutes_total += minutes_lookup.reindex(lookup_keys).fillna(0.0).to_numpy()
            starts_total += starts_lookup.reindex(lookup_keys).fillna(0.0).to_numpy()
        horizon_labels[f"target_horizon_points_{horizon}"] = total
        horizon_labels[f"target_horizon_minutes_{horizon}"] = minutes_total
        horizon_labels[f"target_horizon_starts_{horizon}"] = starts_total
    frame = frame.merge(
        horizon_labels[
            [
                "player_season_key",
                "sequence",
                "target_horizon_points_3",
                "target_horizon_points_8",
                "target_horizon_minutes_3",
                "target_horizon_minutes_8",
                "target_horizon_starts_3",
                "target_horizon_starts_8",
            ]
        ],
        on=["player_season_key", "sequence"],
        how="left",
        validate="many_to_one",
    )

    # Fixture shape is allowed to look forward only at schedule fields, never
    # at future scores or player outcomes.  The final archive schedule can
    # include later rescheduling, so this is recorded as a schedule feature,
    # not treated as a historical news signal.
    schedule_rows = frame[
        ["season_order", "gameweek", "kickoff_time", "team", "opponent_team", "was_home"]
    ].drop_duplicates()
    schedule = (
        schedule_rows.groupby(["season_order", "team", "gameweek"], as_index=False, sort=False)
        .agg(fixture_count=("opponent_team", "size"), home_count=("was_home", "sum"))
        .set_index(["season_order", "team", "gameweek"])
    )
    base_schedule_keys = [frame["season_order"], frame["team"], frame["gameweek"]]
    for horizon in (2, 3, 5, 7, 8):
        fixture_counts = np.zeros(len(frame), dtype=float)
        home_counts = np.zeros(len(frame), dtype=float)
        for offset in range(1, horizon + 1):
            lookup_keys = pd.MultiIndex.from_arrays(
                [base_schedule_keys[0], base_schedule_keys[1], base_schedule_keys[2] + offset]
            )
            future = schedule.reindex(lookup_keys)
            fixture_counts += future["fixture_count"].fillna(0.0).to_numpy()
            home_counts += future["home_count"].fillna(0.0).to_numpy()
        frame[f"fixtures_next_{horizon}"] = fixture_counts
        frame[f"home_share_next_{horizon}"] = np.divide(
            home_counts,
            fixture_counts,
            out=np.zeros(len(frame), dtype=float),
            where=fixture_counts > 0,
        )
    current_schedule_keys = pd.MultiIndex.from_arrays(base_schedule_keys)
    frame["fixtures_current_gw"] = schedule.reindex(current_schedule_keys)["fixture_count"].fillna(0.0).to_numpy()

    team_features = _build_team_history_features(frame)
    frame["fixture_key"] = _fixture_key(frame)
    frame["reverse_fixture_key"] = (
        frame["season_order"].astype(str)
        + "|"
        + frame["gameweek"].astype(str)
        + "|"
        + frame["kickoff_time"].astype(str)
        + "|"
        + frame["opponent_team"].astype(str)
        + "|"
        + frame["team"].astype(str)
        + "|"
        + (1 - frame["was_home"]).astype(str)
    )
    own_columns = [column for column in team_features.columns if column not in {"fixture_key", "reverse_fixture_key"}]
    frame = frame.merge(
        team_features[["fixture_key", *own_columns]],
        on="fixture_key",
        how="left",
        validate="many_to_one",
    )
    opponent_features = team_features[["fixture_key", *own_columns]].rename(
        columns={column: f"opponent_{column}" for column in own_columns}
    )
    frame = frame.merge(
        opponent_features.rename(columns={"fixture_key": "reverse_fixture_key"}),
        on="reverse_fixture_key",
        how="left",
        validate="many_to_one",
    )
    _add_context_features(frame, context_store)
    _add_official_snapshot_features(frame, official_snapshot_store)
    frame["target_price_change"] = pd.to_numeric(frame["target_price_change"], errors="coerce")
    return frame.copy()


NUMERIC_FEATURES = [
    "was_home",
    "games_before",
    "availability_proxy",
    "minutes_last",
    "minutes_mean_3",
    "minutes_mean_5",
    "minutes_mean_10",
    "points_ewma_5",
    "points_mean_3",
    "points_mean_5",
    "points_mean_10",
    "goals_scored_mean_5",
    "assists_mean_5",
    "bonus_mean_5",
    "bps_mean_5",
    "ict_index_mean_5",
    "creativity_mean_5",
    "threat_mean_5",
    "expected_goals_mean_5",
    "expected_assists_mean_5",
    "value_last",
]
CATEGORICAL_FEATURES = ["position", "team", "opponent_team"]


EXTENDED_NUMERIC_FEATURES = list(
    dict.fromkeys(
        NUMERIC_FEATURES
        + [
            "gameweek",
            "season_order",
            "fixtures_current_gw",
            "fixtures_next_2",
            "fixtures_next_3",
            "fixtures_next_5",
            "fixtures_next_7",
            "fixtures_next_8",
            "home_share_next_2",
            "home_share_next_3",
            "home_share_next_5",
            "home_share_next_7",
            "home_share_next_8",
            "form_acceleration",
            "minutes_trend",
            "start_rate_5",
            "minutes_share_5",
            "blank_rate_5",
            "hauler_rate_5",
            "ownership_change_5",
            "transfer_in_out_ratio_5",
            "transfer_balance_mean_5",
            "price_momentum",
            "value_per_point_5",
            "points_per_value_5",
            "career_games_before",
            "points_vs_career",
            "minutes_vs_career",
            "price_vs_career",
            "breakout_signal",
            "metadata_team_imputed",
            "metadata_position_imputed",
            *[f"{column}_observed" for column in EXTENDED_SOURCE_FEATURES],
            "news_availability_delta",
            "news_role_security_delta",
            "context_set_piece_delta",
            "context_transfer_role_delta",
            "news_risk",
            "news_sentiment",
            "social_sentiment",
            "news_price_pressure",
            "context_reliability",
            "context_event_count",
            "context_news_count",
            "context_social_count",
            "official_snapshot_available",
            "official_availability_probability",
            "official_chance_this_round",
            "official_chance_next_round",
            "official_ep_this",
            "official_ep_next",
            "official_form",
            "official_points_per_game",
            "official_selected_by_percent",
            "official_transfers_in_event",
            "official_transfers_out_event",
            "official_value",
            "official_news_present",
            "official_snapshot_age_days",
        ]
        + [
            f"{column}_{suffix}"
            for column in EXTENDED_SOURCE_FEATURES
            for suffix in ("last", "mean_3", "mean_5", "mean_10", "ewma_5", "std_5")
        ]
        + [
            f"career_{column}_{suffix}"
            for column in (
                "total_points",
                "xP",
                "minutes",
                "starts",
                "goals_scored",
                "assists",
                "expected_goals",
                "expected_assists",
                "value",
                "selected",
                "transfers_in",
                "transfers_out",
                "transfers_balance",
            )
            for suffix in ("last", "mean_5", "mean_10", "ewma_5")
        ]
        + [
            "team_match_points_mean_5",
            "team_match_points_mean_10",
            "team_goals_for_mean_5",
            "team_goals_for_mean_10",
            "team_goals_against_mean_5",
            "team_goals_against_mean_10",
            "team_clean_sheet_mean_5",
            "team_clean_sheet_mean_10",
            "matchup_points_mean_5",
            "matchup_goals_for_mean_5",
            "opponent_team_match_points_mean_5",
            "opponent_team_match_points_mean_10",
            "opponent_team_goals_for_mean_5",
            "opponent_team_goals_for_mean_10",
            "opponent_team_goals_against_mean_5",
            "opponent_team_goals_against_mean_10",
            "opponent_team_clean_sheet_mean_5",
            "opponent_team_clean_sheet_mean_10",
            "opponent_matchup_points_mean_5",
            "opponent_matchup_goals_for_mean_5",
        ]
    )
)
EXTENDED_CATEGORICAL_FEATURES = ["position", "team", "opponent_team"]


def _feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].fillna("unknown").astype(str)
    return features


def _extended_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame[EXTENDED_CATEGORICAL_FEATURES + EXTENDED_NUMERIC_FEATURES].copy()
    for column in EXTENDED_NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for column in EXTENDED_CATEGORICAL_FEATURES:
        features[column] = features[column].fillna("unknown").astype(str)
    return features


@dataclass(frozen=True)
class BenchmarkResult:
    metrics: dict[str, dict[str, float]]
    predictions: pd.DataFrame
    metadata: dict
    forecast_model: object | None = None
    price_model: object | None = None


def _metric_row(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(actual)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
    }


def _category(value: float) -> str:
    if value == 0:
        return "zero"
    if value <= 2:
        return "blank"
    if value <= 4:
        return "ticker"
    return "hauler"


def _fit_ridge(train: pd.DataFrame) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    model = Pipeline([("features", preprocessor), ("regressor", Ridge(alpha=10.0))])
    model.fit(_feature_columns(train), train["target_points"])
    return model


def _fit_position_forests(train: pd.DataFrame) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {}
    for position, position_frame in train.groupby("position"):
        preprocessor = ColumnTransformer(
            [
                ("categorical", OneHotEncoder(handle_unknown="ignore"), ["team", "opponent_team"]),
                ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ]
        )
        model = Pipeline(
            [
                ("features", preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=150,
                        max_depth=12,
                        min_samples_leaf=8,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(_feature_columns(position_frame), position_frame["target_points"])
        models[str(position)] = model
    return models


def run_player_benchmark(
    raw: pd.DataFrame,
    development_seasons: tuple[str, ...],
    evaluation_season: str,
) -> BenchmarkResult:
    """Evaluate rolling baselines, ridge, and position-specific RF on one season."""

    frame = build_player_feature_table(raw)
    train = frame[frame["season"].isin(development_seasons)].copy()
    test = frame[frame["season"] == evaluation_season].copy()
    if train.empty or test.empty:
        raise ValueError("development or evaluation split is empty")

    predictions = test[["season", "gameweek", "name", "position", "team", "target_points"]].copy()
    predictions["category"] = predictions["target_points"].map(_category)
    predictions["last5"] = test["points_mean_5"].fillna(train["target_points"].mean())
    predictions["ewma5"] = test["points_ewma_5"].fillna(train["target_points"].mean())
    ridge = _fit_ridge(train)
    predictions["ridge"] = ridge.predict(_feature_columns(test))
    forests = _fit_position_forests(train)
    predictions["position_rf"] = np.nan
    for position, position_frame in test.groupby("position"):
        index = position_frame.index
        predictions.loc[index, "position_rf"] = forests[str(position)].predict(_feature_columns(position_frame))
    predictions = predictions.reset_index(drop=True)

    metrics: dict[str, dict[str, float]] = {}
    actual = predictions["target_points"].to_numpy(float)
    for model_name in ("last5", "ewma5", "ridge", "position_rf"):
        predicted = predictions[model_name].to_numpy(float)
        metrics[model_name] = _metric_row(actual, predicted)
        for category, category_frame in predictions.groupby("category"):
            metrics[f"{model_name}:{category}"] = _metric_row(
                category_frame["target_points"].to_numpy(float), category_frame[model_name].to_numpy(float)
            )

    metadata = {
        "raw_rows": int(len(raw)),
        "feature_rows": int(len(frame)),
        "feature_columns": int(len(frame.columns)),
        "development_seasons": list(development_seasons),
        "evaluation_season": evaluation_season,
        "feature_count": len(CATEGORICAL_FEATURES) + len(NUMERIC_FEATURES),
        "models": ["last5", "ewma5", "ridge", "position_rf"],
        "notes": [
            "Public FPL history only; no Understat features yet.",
            "Features are lagged before the target fixture; current realized statistics are not used as predictors.",
            "The position_rf model is OpenFPL-inspired, not a reproduction of its XGBoost/Random Forest K-best ensemble.",
            "Vaastav rows labelled AM (managers) are excluded; GKP is normalized to GK.",
            "The availability feature is a prior-minutes proxy, not a calibrated expected-minutes model.",
        ],
    }
    return BenchmarkResult(metrics=metrics, predictions=predictions, metadata=metadata)


def _fit_extended_ridge(train: pd.DataFrame, alpha: float = 20.0, target: str = "target_points") -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), EXTENDED_CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), EXTENDED_NUMERIC_FEATURES),
        ]
    )
    model = Pipeline([("features", preprocessor), ("regressor", Ridge(alpha=alpha))])
    model.fit(_extended_feature_columns(train), train[target].to_numpy(float))
    return model


def _fit_extended_neural(train: pd.DataFrame, target: str = "target_points") -> Pipeline:
    """Fit a compact neural forecast model on the expanded feature vector."""

    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), EXTENDED_CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), EXTENDED_NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("features", preprocessor),
            (
                "regressor",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=0.003,
                    batch_size=1024,
                    learning_rate_init=0.001,
                    max_iter=120,
                    early_stopping=True,
                    validation_fraction=0.10,
                    n_iter_no_change=8,
                    random_state=42,
                    verbose=False,
                ),
            ),
        ]
    )
    model.fit(_extended_feature_columns(train), train[target].to_numpy(float))
    return model


def _ranking_metrics(predictions: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    """Report ranking quality, which matters more than RMSE for transfers."""

    rows: list[pd.DataFrame] = []
    for _, group in predictions.groupby(["season", "gameweek"], sort=False):
        if len(group) < 2:
            continue
        rows.append(group.nlargest(min(20, len(group)), prediction_column))
    if not rows:
        return {"spearman_by_gw": 0.0, "top20_actual_mean": 0.0, "top20_actual_sum": 0.0}
    top = pd.concat(rows, ignore_index=True)
    correlations = []
    for _, group in predictions.groupby(["season", "gameweek"], sort=False):
        if len(group) < 2 or group[prediction_column].nunique() < 2 or group["target_points"].nunique() < 2:
            continue
        correlations.append(float(group[prediction_column].rank().corr(group["target_points"].rank())))
    return {
        "spearman_by_gw": float(np.nanmean(correlations)) if correlations else 0.0,
        "top20_actual_mean": float(top["target_points"].mean()),
        "top20_actual_sum": float(top["target_points"].sum()),
    }


def run_extended_player_benchmark(
    raw: pd.DataFrame,
    development_seasons: tuple[str, ...],
    evaluation_season: str,
    validation_season: str | None = None,
    context_store: object | None = None,
    official_snapshot_store: object | None = None,
) -> BenchmarkResult:
    """Tune the expanded player/price forecasts without touching the test season.

    ``validation_season`` is used only to select between the regularized
    linear model and the neural model.  The selected model is then refit on
    development plus validation data before the untouched evaluation season
    is scored.  This is deliberately a forecast benchmark; the transfer
    policy must still be judged inside the legal simulator.
    """

    frame = build_extended_player_feature_table(
        raw,
        context_store=context_store,
        official_snapshot_store=official_snapshot_store,
    )
    development = frame[frame["season"].isin(development_seasons)].copy()
    validation = frame[frame["season"] == validation_season].copy() if validation_season else pd.DataFrame()
    test = frame[frame["season"] == evaluation_season].copy()
    if development.empty or test.empty:
        raise ValueError("development or evaluation split is empty")

    candidate_scores: dict[str, dict[str, float]] = {}
    candidates: dict[str, Pipeline] = {}
    tuning_frame = development if validation.empty else validation
    fit_frame = development
    for name, factory in (
        ("extended_ridge", lambda train: _fit_extended_ridge(train, alpha=20.0)),
        ("extended_ridge_stronger", lambda train: _fit_extended_ridge(train, alpha=60.0)),
        ("extended_neural", _fit_extended_neural),
    ):
        model = factory(fit_frame)
        scored = validation if not validation.empty else tuning_frame
        predicted = model.predict(_extended_feature_columns(scored))
        candidate_scores[name] = _metric_row(scored["target_points"].to_numpy(float), predicted)
        candidate_scores[name].update(_ranking_metrics(scored.assign(candidate_prediction=predicted), "candidate_prediction"))
        candidates[name] = model
    selected_name = min(candidate_scores, key=lambda name: (candidate_scores[name]["mae"], -candidate_scores[name]["spearman_by_gw"]))
    final_train = pd.concat([development, validation], ignore_index=True) if not validation.empty else development
    if selected_name == "extended_neural":
        selected_model = _fit_extended_neural(final_train)
    elif selected_name == "extended_ridge_stronger":
        selected_model = _fit_extended_ridge(final_train, alpha=60.0)
    else:
        selected_model = _fit_extended_ridge(final_train, alpha=20.0)

    predictions = test[
        [
            "season",
            "gameweek",
            "element",
            "player_key",
            "name",
            "position",
            "team",
            "opponent_team",
            "value",
            "selected",
            "transfers_in",
            "transfers_out",
            "target_points",
            "target_price_change",
            "fixtures_next_3",
            "fixtures_next_5",
            "home_share_next_3",
            "home_share_next_5",
            "form_acceleration",
            "minutes_share_5",
            "news_risk",
            "social_sentiment",
        ]
    ].copy()
    predictions["last5"] = test["points_mean_5"].fillna(final_train["target_points"].mean()).to_numpy()
    predictions["ewma5"] = test["points_ewma_5"].fillna(final_train["target_points"].mean()).to_numpy()
    predictions["extended_selected"] = selected_model.predict(_extended_feature_columns(test))
    price_train = final_train.dropna(subset=["target_price_change"]).copy()
    price_test = test.dropna(subset=["target_price_change"]).copy()
    price_model = None
    if price_train.empty or price_test.empty:
        predictions["future_price_change_ridge"] = 0.0
    else:
        price_model = _fit_extended_ridge(price_train, alpha=10.0, target="target_price_change")
        price_predictions = price_model.predict(_extended_feature_columns(price_test))
        predictions["future_price_change_ridge"] = 0.0
        predictions.loc[price_test.index, "future_price_change_ridge"] = price_predictions
    predictions = predictions.reset_index(drop=True)

    metrics: dict[str, dict[str, float]] = {}
    for model_name in ("last5", "ewma5", "extended_selected"):
        actual = predictions["target_points"].to_numpy(float)
        predicted = predictions[model_name].to_numpy(float)
        metrics[model_name] = _metric_row(actual, predicted)
        metrics[model_name].update(_ranking_metrics(predictions, model_name))
        for position, group in predictions.groupby("position"):
            metrics[f"{model_name}:{position}"] = _metric_row(
                group["target_points"].to_numpy(float), group[model_name].to_numpy(float)
            )
    price_rows = predictions.dropna(subset=["target_price_change"])
    metrics["future_price_change_ridge"] = _metric_row(
        price_rows["target_price_change"].to_numpy(float),
        price_rows["future_price_change_ridge"].to_numpy(float),
    ) if not price_rows.empty else {"n": 0, "rmse": 0.0, "mae": 0.0}
    metadata = {
        "raw_rows": int(len(raw)),
        "feature_rows": int(len(frame)),
        "feature_columns": int(len(frame.columns)),
        "development_seasons": list(development_seasons),
        "validation_season": validation_season,
        "evaluation_season": evaluation_season,
        "feature_count": len(EXTENDED_CATEGORICAL_FEATURES) + len(EXTENDED_NUMERIC_FEATURES),
        "selected_model": selected_name,
        "candidate_validation_scores": candidate_scores,
        "models": ["last5", "ewma5", "extended_ridge", "extended_ridge_stronger", "extended_neural", "future_price_change_ridge"],
        "notes": [
            "2016/17 onward public Vaastav FPL history; 2025/26 remains an untouched test season when supplied as evaluation_season.",
            "Player lags are guarded by season/gameweek sequence so double-gameweek outcomes cannot leak across the same FPL deadline.",
            "Team and matchup features use prior completed results only; fixture-shape features use schedule fields without future outcomes.",
            "News and social columns are zero unless an auditable timestamped ContextStore is supplied; modern context is never backfilled into old seasons.",
            "The 2,413-point goal is a strategy-level simulator benchmark, not a player-point regression target.",
        ],
    }
    return BenchmarkResult(
        metrics=metrics,
        predictions=predictions,
        metadata=metadata,
        forecast_model=selected_model,
        price_model=price_model,
    )
