"""Mac-compatible reproduction of the external FPL forecast architecture.

The public ``fpl-luck-or-skill`` work uses a 53-feature, walk-forward feature
table and separates expected minutes from conditional point production. This
module keeps that architecture while using scikit-learn's histogram gradient
boosting, which is available in the local package without a platform-specific
LightGBM/OpenMP runtime.

This is a forecast-family ablation. It does not choose transfers by itself;
the legal simulator and action policy remain downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


BENCHMARK_FEATURES = tuple(
    [f"{column}_r{window}" for column in (
        "total_points",
        "minutes",
        "bps",
        "ict_index",
        "threat",
        "creativity",
        "influence",
        "xg",
        "xa",
        "bonus",
    ) for window in (3, 5, 10)]
    + [
        "played_60_r5",
        "appearances_prior",
        "started_share_r5",
        "days_since_last",
        "last_minutes",
        "last_points",
        "prev_points",
        "prev_minutes",
        "prev_ppg",
        "prev_pp90",
        "prev_matches",
        "team_gf_r5",
        "team_ga_r5",
        "opp_gf_r5",
        "opp_ga_r5",
        "was_home",
        "value",
        "round",
        "log_selected",
        "pos_1",
        "pos_2",
        "pos_3",
        "pos_4",
    ]
)

FORM_COLUMNS = (
    "total_points",
    "minutes",
    "bps",
    "ict_index",
    "threat",
    "creativity",
    "influence",
    "xg",
    "xa",
    "bonus",
)


def minutes_bucket(minutes: pd.Series) -> pd.Series:
    """Classify a fixture as DNP, cameo, or 60-plus minutes."""

    return pd.cut(minutes, [-1, 0, 59, 200], labels=False).astype(int)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _player_code_map(history_root: str | Path, season: str) -> dict[int, str]:
    path = Path(history_root) / season / "players_raw.csv"
    if not path.exists():
        return {}
    try:
        roster = pd.read_csv(path, encoding="utf-8", usecols=lambda column: column in {"id", "element", "code"})
    except UnicodeDecodeError:
        roster = pd.read_csv(path, encoding="latin-1", usecols=lambda column: column in {"id", "element", "code"})
    if "element" not in roster and "id" in roster:
        roster["element"] = roster["id"]
    if "code" not in roster or "element" not in roster:
        return {}
    roster["element"] = pd.to_numeric(roster["element"], errors="coerce")
    roster["code"] = pd.to_numeric(roster["code"], errors="coerce")
    roster = roster.dropna(subset=["element", "code"]).drop_duplicates("element")
    return {int(element): str(int(code)) for element, code in zip(roster["element"], roster["code"])}


def _add_persistent_codes(frame: pd.DataFrame, history_root: str | Path | None) -> pd.DataFrame:
    frame = frame.copy()
    frame["element"] = pd.to_numeric(frame.get("element"), errors="coerce")
    if "code" in frame:
        codes = pd.to_numeric(frame["code"], errors="coerce")
    else:
        codes = pd.Series(np.nan, index=frame.index)
    if history_root is not None:
        for season in frame["season"].astype(str).unique():
            mapping = _player_code_map(history_root, season)
            if not mapping:
                continue
            mask = frame["season"].astype(str).eq(season) & codes.isna()
            codes.loc[mask] = frame.loc[mask, "element"].map(mapping)
    fallback = (
        frame.get("name", pd.Series("unknown", index=frame.index))
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    code_text = pd.to_numeric(codes, errors="coerce").astype("Int64").astype("string")
    frame["code"] = code_text.fillna(fallback)
    return frame


def _rolling_prior(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    prior = frame.groupby(["season", "code"], sort=False)[column].shift(1)
    return prior.groupby([frame["season"], frame["code"]], sort=False).rolling(
        window, min_periods=1
    ).mean().reset_index(level=[0, 1], drop=True)


def build_benchmark_feature_table(
    raw: pd.DataFrame,
    history_root: str | Path | None = None,
) -> pd.DataFrame:
    """Build the external-style leakage-safe player-fixture feature table.

    ``raw`` is the output of :func:`load_vaastav_gameweeks`. Every lagged
    player/team feature is computed before the current fixture, and the
    previous-season aggregates are mapped through a persistent player code.
    """

    required = {"season", "gameweek", "element", "name", "position", "team", "opponent_team", "kickoff_time", "was_home", "total_points"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"raw history is missing benchmark columns: {missing}")
    frame = raw.copy()
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
    frame["round"] = pd.to_numeric(frame["gameweek"], errors="coerce").astype(int)
    if "fixture" not in frame:
        frame["fixture"] = frame.groupby(["season", "round"]).cumcount()
    frame = _add_persistent_codes(frame, history_root)
    position_map = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
    frame["pos"] = frame["position"].astype(str).map(position_map).fillna(0).astype(int)
    season_end_team = pd.to_numeric(frame["team"], errors="coerce")
    frame["opponent_team"] = pd.to_numeric(frame["opponent_team"], errors="coerce").fillna(-1).astype(int)
    frame["was_home"] = pd.to_numeric(frame["was_home"], errors="coerce").fillna(0).astype(int)
    # Vaastav's player roster team is season-end metadata. Reconstruct the
    # actual team at each fixture from the opposing side, which matters for
    # players transferred during a season and for team-form features.
    fixture_keys = pd.MultiIndex.from_frame(frame[["season", "fixture"]])
    home_team = frame.loc[frame["was_home"].eq(0)].groupby(["season", "fixture"])["opponent_team"].first()
    away_team = frame.loc[frame["was_home"].eq(1)].groupby(["season", "fixture"])["opponent_team"].first()
    reconstructed_team = np.where(
        frame["was_home"].eq(1),
        home_team.reindex(fixture_keys).to_numpy(),
        away_team.reindex(fixture_keys).to_numpy(),
    )
    frame["team_id"] = pd.to_numeric(pd.Series(reconstructed_team, index=frame.index), errors="coerce")
    frame["team_id"] = frame["team_id"].fillna(season_end_team).fillna(-1).astype(int)
    # Vaastav's archive uses the expanded expected_* names while the external
    # benchmark's unified table uses xg/xa/xgc.
    for source, target in (
        ("expected_goals", "xg"),
        ("expected_assists", "xa"),
        ("expected_goals_conceded", "xgc"),
    ):
        if target not in frame and source in frame:
            frame[target] = frame[source]
    for column in FORM_COLUMNS + (
        "minutes",
        "starts",
        "selected",
        "transfers_balance",
        "value",
        "xg",
        "xa",
        "bps",
        "ict_index",
        "threat",
        "creativity",
        "influence",
        "bonus",
        "team_h_score",
        "team_a_score",
    ):
        frame[column] = _numeric(frame, column)
    home = frame["was_home"].eq(1)
    frame["team_gf"] = frame["team_h_score"].where(home, frame["team_a_score"])
    frame["team_ga"] = frame["team_a_score"].where(home, frame["team_h_score"])
    frame = frame.sort_values(["season", "code", "kickoff_time", "fixture"]).reset_index(drop=True)
    player_groups = frame.groupby(["season", "code"], sort=False)
    for column in FORM_COLUMNS:
        for window in (3, 5, 10):
            frame[f"{column}_r{window}"] = _rolling_prior(frame, column, window)
    started = np.where(frame["starts"].notna(), frame["starts"], (frame["minutes"] >= 60).astype(float))
    frame["started_share_r5"] = (
        pd.Series(started, index=frame.index).groupby([frame["season"], frame["code"]], sort=False)
        .shift(1).groupby([frame["season"], frame["code"]], sort=False).rolling(5, min_periods=1).mean()
        .reset_index(level=[0, 1], drop=True)
    )
    frame["played_60_r5"] = (
        frame["minutes"].ge(60).astype(float).groupby([frame["season"], frame["code"]], sort=False)
        .shift(1).groupby([frame["season"], frame["code"]], sort=False).rolling(5, min_periods=1).mean()
        .reset_index(level=[0, 1], drop=True)
    )
    frame["appearances_prior"] = player_groups.cumcount()
    frame["days_since_last"] = player_groups["kickoff_time"].diff().dt.total_seconds().div(86400).clip(upper=60)
    frame["last_minutes"] = player_groups["minutes"].shift(1)
    frame["last_points"] = player_groups["total_points"].shift(1)

    season_order = {season: index for index, season in enumerate(sorted(frame["season"].unique()))}
    ordered_seasons = sorted(season_order, key=lambda value: season_order[value])
    previous = (
        frame.groupby(["season", "code"], as_index=False)
        .agg(
            prev_points=("total_points", "sum"),
            prev_minutes=("minutes", "sum"),
            prev_matches=("minutes", lambda values: int((values > 0).sum())),
        )
    )
    previous["season"] = previous["season"].map(
        {season: ordered_seasons[index + 1] for index, season in enumerate(ordered_seasons[:-1])}
    )
    previous["prev_ppg"] = previous["prev_points"] / previous["prev_matches"].clip(lower=1)
    previous["prev_pp90"] = 90.0 * previous["prev_points"] / previous["prev_minutes"].clip(lower=90)
    previous = previous.dropna(subset=["season"])
    frame = frame.merge(previous, on=["season", "code"], how="left")

    team_frame = (
        frame.groupby(["season", "team_id", "fixture"], as_index=False)
        .agg(kickoff_time=("kickoff_time", "first"), gf=("team_gf", "first"), ga=("team_ga", "first"))
        .sort_values(["season", "team_id", "kickoff_time", "fixture"])
    )
    team_groups = team_frame.groupby(["season", "team_id"], sort=False)
    for source, target in (("gf", "team_gf_r5"), ("ga", "team_ga_r5")):
        prior = team_groups[source].shift(1)
        team_frame[target] = (
            prior.groupby([team_frame["season"], team_frame["team_id"]], sort=False)
            .rolling(5, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
        )
    own = team_frame[["season", "team_id", "fixture", "team_gf_r5", "team_ga_r5"]]
    frame = frame.merge(own, on=["season", "team_id", "fixture"], how="left")
    opponent = own.rename(columns={"team_id": "opponent_team", "team_gf_r5": "opp_gf_r5", "team_ga_r5": "opp_ga_r5"})
    frame = frame.merge(opponent, on=["season", "opponent_team", "fixture"], how="left")
    frame["log_selected"] = np.log1p(frame["selected"])
    for position in (1, 2, 3, 4):
        frame[f"pos_{position}"] = frame["pos"].eq(position).astype(int)
    return frame


@dataclass
class BenchmarkForecastModels:
    minutes: HistGradientBoostingClassifier
    points_cameo: HistGradientBoostingRegressor
    points_sixty: HistGradientBoostingRegressor

    def predict_expected_points(self, frame: pd.DataFrame) -> np.ndarray:
        features = frame[list(BENCHMARK_FEATURES)]
        probabilities = self.minutes.predict_proba(features)
        cameo = np.clip(self.points_cameo.predict(features), 0.0, None)
        sixty = np.clip(self.points_sixty.predict(features), 0.0, None)
        return np.clip(probabilities[:, 1] * cameo + probabilities[:, 2] * sixty, 0.0, None)

    def predict_expected_minutes(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = self.minutes.predict_proba(frame[list(BENCHMARK_FEATURES)])
        return np.clip(probabilities[:, 1] * 30.0 + probabilities[:, 2] * 75.0, 0.0, 90.0)


def fit_benchmark_forecast_models(
    train: pd.DataFrame,
    *,
    max_iter: int = 180,
    random_state: int = 42,
) -> BenchmarkForecastModels:
    """Fit the minutes-plus-conditional-points model suite."""

    features = train[list(BENCHMARK_FEATURES)]
    minutes = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=max_iter,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=random_state,
    ).fit(features, minutes_bucket(train["minutes"]))
    cameo_mask = train["minutes"].between(1, 59)
    points_cameo = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=random_state,
    ).fit(features.loc[cameo_mask], train.loc[cameo_mask, "total_points"].clip(lower=0.0))
    sixty_mask = train["minutes"] >= 60
    points_sixty = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=random_state,
    ).fit(features.loc[sixty_mask], train.loc[sixty_mask, "total_points"].clip(lower=0.0))
    return BenchmarkForecastModels(minutes, points_cameo, points_sixty)


def build_gameweek_forecast_rows(
    target: pd.DataFrame,
    models: BenchmarkForecastModels,
    *,
    max_gameweek: int = 38,
) -> pd.DataFrame:
    """Create simulator-compatible per-fixture forecasts for one season."""

    frame = target.copy()
    frame["_point_prediction"] = models.predict_expected_points(frame)
    frame["_minutes_prediction"] = models.predict_expected_minutes(frame)
    by_player = {code: subset for code, subset in frame.groupby("code", sort=False)}
    rows: list[dict[str, float | int]] = []
    for (code, gameweek), current in frame.groupby(["code", "round"], sort=False):
        all_player = by_player[code]
        short = all_player[all_player["round"].between(gameweek, min(max_gameweek, gameweek + 2))]
        long = all_player[all_player["round"].between(gameweek, min(max_gameweek, gameweek + 7))]
        for _, row in current.iterrows():
            rows.append(
                {
                    "gameweek": int(gameweek),
                    "element": int(row["element"]),
                    "extended_selected": float(row["_point_prediction"]),
                    "forecast_short_expected_points": float(short["_point_prediction"].sum()),
                    "forecast_long_expected_points": float(long["_point_prediction"].sum()),
                    "forecast_expected_minutes": float(current["_minutes_prediction"].sum()),
                    "forecast_short_expected_minutes": float(short["_minutes_prediction"].sum()),
                    "forecast_long_expected_minutes": float(long["_minutes_prediction"].sum()),
                    "fixtures_current_gw": float(len(current)),
                    "fixtures_next_2": float(len(short) - len(current)),
                    "fixtures_next_3": float(len(all_player[all_player["round"].between(gameweek, min(max_gameweek, gameweek + 3))])),
                    "fixtures_next_5": float(len(all_player[all_player["round"].between(gameweek, min(max_gameweek, gameweek + 5))])),
                    "fixtures_next_7": float(len(long)),
                    "future_price_change_ridge": 0.0,
                }
            )
    return pd.DataFrame.from_records(rows)
