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
    return pd.concat(frames, ignore_index=True, sort=False)


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
    for _, row in frame.iterrows():
        player_id = str(int(row["element"])) if float(row.get("element", 0.0)) > 0 else ""
        features = context_store.features_for_player(
            player_id,
            str(row["name"]),
            str(row["team"]),
            row["kickoff_time"],
        )
        rows.append(
            {
                "news_availability_delta": features.availability_delta,
                "news_role_security_delta": features.role_security_delta,
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


def build_extended_player_feature_table(
    raw: pd.DataFrame,
    context_store: object | None = None,
) -> pd.DataFrame:
    """Build the richer point-in-time feature table used for model training.

    The table intentionally includes only information available before the
    target fixture: prior player form and role, ownership/transfer momentum,
    lagged team and matchup form, fixture-shape counts, and optional as-of
    context.  Future rows are used only to create labels or schedule counts;
    their realized scores never enter a predictor.
    """

    frame = build_player_feature_table(raw)
    frame["player_season_key"] = frame["player_key"].astype(str) + "|" + frame["season"].astype(str)
    player_group = frame.groupby("player_season_key", sort=False)
    computed_features: dict[str, pd.Series] = {}
    for column in EXTENDED_SOURCE_FEATURES:
        prior = _strict_prior(frame, column, group_column="player_season_key")
        computed_features[f"{column}_last"] = prior
        for window in (3, 5, 10):
            computed_features[f"{column}_mean_{window}"] = _rolling_prior(
                frame, prior, window, group_column="player_season_key"
            )
        computed_features[f"{column}_ewma_5"] = _ewma_prior(frame, prior, 5, group_column="player_season_key")
    overlapping = [column for column in computed_features if column in frame.columns]
    if overlapping:
        frame = frame.drop(columns=overlapping)
    frame = pd.concat([frame, pd.DataFrame(computed_features, index=frame.index)], axis=1)
    frame["points_ewma_5"] = frame["total_points_ewma_5"]
    frame["points_mean_3"] = frame["total_points_mean_3"]
    frame["points_mean_5"] = frame["total_points_mean_5"]
    frame["points_mean_10"] = frame["total_points_mean_10"]

    points_prior = _strict_prior(frame, "total_points", group_column="player_season_key")
    minutes_prior = _strict_prior(frame, "minutes", group_column="player_season_key")
    selected_prior = _strict_prior(frame, "selected", group_column="player_season_key")
    transfers_in_prior = _strict_prior(frame, "transfers_in", group_column="player_season_key")
    transfers_out_prior = _strict_prior(frame, "transfers_out", group_column="player_season_key")
    value_prior = _strict_prior(frame, "value", group_column="player_season_key")
    std_features: dict[str, pd.Series] = {}
    for column in EXTENDED_SOURCE_FEATURES:
        std_features[f"{column}_std_5"] = _std_prior(
            frame,
            _strict_prior(frame, column, group_column="player_season_key"),
            5,
            group_column="player_season_key",
        )
    frame = pd.concat([frame, pd.DataFrame(std_features, index=frame.index)], axis=1)

    frame["points_ewma_3"] = _ewma_prior(frame, points_prior, 3, group_column="player_season_key")
    frame["points_ewma_10"] = _ewma_prior(frame, points_prior, 10, group_column="player_season_key")
    frame["form_acceleration"] = frame["points_ewma_3"] - frame["points_ewma_10"]
    frame["minutes_ewma_10"] = _ewma_prior(frame, minutes_prior, 10, group_column="player_season_key")
    frame["minutes_trend"] = frame["minutes_ewma_5"] - frame["minutes_ewma_10"]
    frame["start_rate_5"] = _rolling_prior(
        frame, _strict_prior(frame, "starts", group_column="player_season_key"), 5, group_column="player_season_key"
    ) / 1.0
    frame["minutes_share_5"] = frame["minutes_mean_5"] / 90.0
    frame["blank_rate_5"] = _rolling_prior(
        frame,
        (points_prior <= 2).astype(float),
        5,
        group_column="player_season_key",
    )
    frame["hauler_rate_5"] = _rolling_prior(
        frame,
        (points_prior >= 5).astype(float),
        5,
        group_column="player_season_key",
    )
    frame["ownership_change_5"] = _rolling_prior(
        frame,
        _strict_prior(frame, "selected", group_column="player_season_key"),
        3,
        group_column="player_season_key",
    ) - _rolling_prior(
        frame,
        _strict_prior(frame, "selected", group_column="player_season_key"),
        10,
        group_column="player_season_key",
    )
    frame["transfer_in_out_ratio_5"] = (
        _rolling_prior(frame, transfers_in_prior, 5, group_column="player_season_key")
        / (_rolling_prior(frame, transfers_out_prior, 5, group_column="player_season_key") + 1.0)
    )
    frame["transfer_balance_mean_5"] = _rolling_prior(
        frame,
        _strict_prior(frame, "transfers_balance", group_column="player_season_key"),
        5,
        group_column="player_season_key",
    )
    frame["price_momentum"] = value_prior - _rolling_prior(frame, value_prior, 5, group_column="player_season_key")
    frame["value_per_point_5"] = frame["value_mean_5"] / (frame["points_mean_5"] + 1.0)
    frame["points_per_value_5"] = frame["points_mean_5"] / (frame["value_mean_5"] + 1.0)
    frame["target_price_change"] = player_group["value"].shift(-1) - frame["value"]

    # Fixture shape is allowed to look forward only at schedule fields, never
    # at future scores or player outcomes.  The final archive schedule can
    # include later rescheduling, so this is recorded as a schedule feature,
    # not treated as a historical news signal.
    for horizon in (3, 5):
        future_home = [frame.groupby("player_season_key", sort=False)["was_home"].shift(-offset) for offset in range(1, horizon + 1)]
        future_opponents = [frame.groupby("player_season_key", sort=False)["opponent_team"].shift(-offset) for offset in range(1, horizon + 1)]
        frame[f"fixtures_next_{horizon}"] = pd.concat(future_opponents, axis=1).notna().sum(axis=1).astype(float)
        frame[f"home_share_next_{horizon}"] = pd.concat(future_home, axis=1).mean(axis=1).fillna(0.0)
    frame["fixtures_current_gw"] = frame.groupby(["player_season_key", "sequence"])["sequence"].transform("size").astype(float)

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
            "fixtures_next_3",
            "fixtures_next_5",
            "home_share_next_3",
            "home_share_next_5",
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
            *[f"{column}_observed" for column in EXTENDED_SOURCE_FEATURES],
            "news_availability_delta",
            "news_role_security_delta",
            "news_risk",
            "news_sentiment",
            "social_sentiment",
            "news_price_pressure",
            "context_reliability",
            "context_event_count",
            "context_news_count",
            "context_social_count",
        ]
        + [
            f"{column}_{suffix}"
            for column in EXTENDED_SOURCE_FEATURES
            for suffix in ("last", "mean_3", "mean_5", "mean_10", "ewma_5", "std_5")
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
) -> BenchmarkResult:
    """Tune the expanded player/price forecasts without touching the test season.

    ``validation_season`` is used only to select between the regularized
    linear model and the neural model.  The selected model is then refit on
    development plus validation data before the untouched evaluation season
    is scored.  This is deliberately a forecast benchmark; the transfer
    policy must still be judged inside the legal simulator.
    """

    frame = build_extended_player_feature_table(raw, context_store=context_store)
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
