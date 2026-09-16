"""Temporal validation, baselines, and paired-match bootstrap diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Match, TeamObservation, to_team_observations
from .poisson import RegularizedPoissonRegression


@dataclass(frozen=True)
class BacktestResult:
    train_gameweeks: tuple[int, ...]
    test_gameweeks: tuple[int, ...]
    selected_alpha: float
    metrics: dict[str, float]
    predictions: tuple[dict, ...]
    bootstrap: dict[str, float]


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("actual and predicted must be non-empty arrays with the same shape")
    return float(np.mean(np.abs(actual - predicted)))


def _league_average_predictions(train: list[TeamObservation], test: list[TeamObservation]) -> np.ndarray:
    average = float(np.mean([row.goals for row in train]))
    return np.full(len(test), average, dtype=float)


def _home_away_average_predictions(train: list[TeamObservation], test: list[TeamObservation]) -> np.ndarray:
    home = [row.goals for row in train if row.is_home]
    away = [row.goals for row in train if not row.is_home]
    if not home or not away:
        raise ValueError("home/away baseline requires both home and away training observations")
    averages = {1: float(np.mean(home)), 0: float(np.mean(away))}
    return np.asarray([averages[row.is_home] for row in test], dtype=float)


def _tune_alpha(train: list[TeamObservation], gameweeks: list[int], candidates: tuple[float, ...]) -> float:
    if len(gameweeks) < 2:
        return candidates[0]
    validation_week = gameweeks[-1]
    fit = [row for row in train if row.gameweek < validation_week]
    validation = [row for row in train if row.gameweek == validation_week]
    if not fit or not validation:
        return candidates[0]
    scores = []
    actual = np.asarray([row.goals for row in validation], dtype=float)
    for alpha in candidates:
        model = RegularizedPoissonRegression(alpha=alpha).fit(fit)
        scores.append((mean_absolute_error(actual, model.predict(validation)), alpha))
    scores.sort(key=lambda item: (item[0], item[1]))
    return scores[0][1]


def paired_match_bootstrap(
    matches: list[Match],
    actual: np.ndarray,
    model_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    seed: int = 20260914,
    n_bootstrap: int = 5000,
) -> dict[str, float]:
    """Bootstrap model-vs-baseline MAE improvement by whole match."""

    if len(matches) * 2 != actual.size:
        raise ValueError("one actual/prediction pair per team observation is required")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    model_errors = np.abs(actual - model_predictions).reshape(len(matches), 2).mean(axis=1)
    baseline_errors = np.abs(actual - baseline_predictions).reshape(len(matches), 2).mean(axis=1)
    improvement_by_match = baseline_errors - model_errors
    rng = np.random.default_rng(seed)
    samples = rng.choice(improvement_by_match, size=(n_bootstrap, len(matches)), replace=True).mean(axis=1)
    return {
        "observed_improvement": float(np.mean(improvement_by_match)),
        "ci_2_5": float(np.quantile(samples, 0.025)),
        "ci_97_5": float(np.quantile(samples, 0.975)),
        "seed": int(seed),
        "n_bootstrap": int(n_bootstrap),
        "n_matches": len(matches),
    }


def run_temporal_backtest(
    matches: list[Match],
    test_gameweek: int | None = None,
    alpha_candidates: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 30.0),
    bootstrap_seed: int = 20260914,
    n_bootstrap: int = 5000,
) -> BacktestResult:
    """Fit on prior gameweeks and evaluate exactly one later gameweek."""

    if len(matches) < 2:
        raise ValueError("at least two matches are required")
    weeks = sorted({match.gameweek for match in matches})
    if test_gameweek is None:
        test_gameweek = weeks[-1]
    if test_gameweek not in weeks or test_gameweek == weeks[0]:
        raise ValueError("test_gameweek must exist and have earlier training weeks")
    train_matches = [match for match in matches if match.gameweek < test_gameweek]
    test_matches = [match for match in matches if match.gameweek == test_gameweek]
    train = to_team_observations(train_matches)
    test = to_team_observations(test_matches)
    train_weeks = sorted({row.gameweek for row in train})
    alpha = _tune_alpha(train, train_weeks, alpha_candidates)
    model = RegularizedPoissonRegression(alpha=alpha).fit(train)
    actual = np.asarray([row.goals for row in test], dtype=float)
    model_predictions = model.predict(test)
    league_predictions = _league_average_predictions(train, test)
    home_away_predictions = _home_away_average_predictions(train, test)
    metrics = {
        "poisson_mae": mean_absolute_error(actual, model_predictions),
        "league_average_mae": mean_absolute_error(actual, league_predictions),
        "home_away_average_mae": mean_absolute_error(actual, home_away_predictions),
    }
    predictions = tuple(
        {
            "match_id": row.match_id,
            "gameweek": row.gameweek,
            "team": row.team,
            "opponent": row.opponent,
            "is_home": row.is_home,
            "actual_goals": row.goals,
            "poisson_prediction": float(model_prediction),
            "league_average_prediction": float(league_prediction),
            "home_away_average_prediction": float(home_away_prediction),
        }
        for row, model_prediction, league_prediction, home_away_prediction in zip(
            test, model_predictions, league_predictions, home_away_predictions
        )
    )
    bootstrap = paired_match_bootstrap(
        test_matches,
        actual,
        model_predictions,
        league_predictions,
        seed=bootstrap_seed,
        n_bootstrap=n_bootstrap,
    )
    return BacktestResult(
        train_gameweeks=tuple(train_weeks),
        test_gameweeks=(test_gameweek,),
        selected_alpha=alpha,
        metrics=metrics,
        predictions=predictions,
        bootstrap=bootstrap,
    )
