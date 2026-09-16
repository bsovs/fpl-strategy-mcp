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
    "saves",
    "threat",
    "total_points",
    "value",
    "yellow_cards",
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


def build_player_feature_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe lagged rolling features for next-match prediction."""

    required = {"name", "position", "team", "opponent_team", "kickoff_time", "was_home", "total_points"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"player history missing columns: {sorted(missing)}")

    frame = raw.copy()
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
    frame["player_key"] = frame["name"].astype(str).str.strip().str.lower()
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
        frame[column] = _numeric(frame, column)
    frame = frame.sort_values(["season_order", "gameweek", "kickoff_time", "player_key"]).reset_index(drop=True)

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
        prior = grouped[column].shift(1)
        frame[f"{column}_last"] = prior
        for window in (3, 5, 10):
            frame[f"{column}_mean_{window}"] = prior.groupby(frame["player_key"]).transform(
                lambda series: series.rolling(window, min_periods=1).mean()
            )
    frame["games_before"] = grouped.cumcount()
    frame["minutes_last"] = frame["minutes_last"].fillna(0.0)
    frame["points_ewma_5"] = grouped["total_points"].transform(
        lambda series: series.shift(1).ewm(span=5, adjust=False, min_periods=1).mean()
    )
    frame["points_mean_3"] = frame["total_points_mean_3"]
    frame["points_mean_5"] = frame["total_points_mean_5"]
    frame["points_mean_10"] = frame["total_points_mean_10"]
    frame["minutes_ewma_5"] = grouped["minutes"].transform(
        lambda series: series.shift(1).ewm(span=5, adjust=False, min_periods=1).mean()
    )
    frame["target_points"] = frame["total_points"]
    frame["availability_proxy"] = (frame["minutes_ewma_5"] / 90.0).clip(0.0, 1.0)
    frame["season_gameweek"] = frame["season"].astype(str) + "_" + frame["gameweek"].astype(str)

    # The current row's realized columns remain available for the target and
    # audit, but the feature list below contains only lagged values plus match
    # context known before kickoff.
    return frame


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


def _feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].fillna("unknown").astype(str)
    return features


@dataclass(frozen=True)
class BenchmarkResult:
    metrics: dict[str, dict[str, float]]
    predictions: pd.DataFrame
    metadata: dict


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
