"""Point-in-time tuning, holdout evaluation, and match-cluster bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

from .data import Match, TeamObservation, to_team_observations
from .model import PoissonTeamGoalsModel


ALPHA_CANDIDATES = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


@dataclass(frozen=True)
class BacktestResult:
    train_gameweeks: tuple[int, ...]
    test_gameweek: int
    selected_alpha: float
    model: dict
    metrics: dict[str, float]
    bootstrap: dict[str, float]
    predictions: tuple[dict, ...]


def _assert_temporal_order(train: list[TeamObservation], validation: list[TeamObservation]) -> None:
    if not train or not validation:
        raise ValueError("time split cannot have an empty side")
    if max(row.date for row in train) >= min(row.date for row in validation):
        raise ValueError("training dates overlap or occur after validation dates")


def _baselines(train: list[TeamObservation], test: list[TeamObservation]) -> tuple[np.ndarray, np.ndarray]:
    league = float(np.mean([row.goals for row in train]))
    home_values = [row.goals for row in train if row.home]
    away_values = [row.goals for row in train if not row.home]
    return (
        np.full(len(test), league, dtype=float),
        np.asarray([np.mean(home_values) if row.home else np.mean(away_values) for row in test], dtype=float),
    )


def _tune_alpha(train_matches: list[Match], candidates: tuple[float, ...]) -> float:
    weeks = sorted({match.gameweek for match in train_matches})
    validation_weeks = weeks[1:]
    if not validation_weeks:
        return candidates[0]
    observed_by_alpha: dict[float, list[float]] = {alpha: [] for alpha in candidates}
    actual_by_alpha: dict[float, list[float]] = {alpha: [] for alpha in candidates}
    for validation_week in validation_weeks:
        fit_matches = [match for match in train_matches if match.gameweek < validation_week]
        validation_matches = [match for match in train_matches if match.gameweek == validation_week]
        fit = to_team_observations(fit_matches)
        validation = to_team_observations(validation_matches)
        _assert_temporal_order(fit, validation)
        actual = np.asarray([row.goals for row in validation], dtype=float)
        for alpha in candidates:
            model = PoissonTeamGoalsModel(alpha=alpha).fit(fit)
            observed_by_alpha[alpha].extend(model.predict(validation).tolist())
            actual_by_alpha[alpha].extend(actual.tolist())
    scores = []
    for alpha in candidates:
        scores.append((mean_poisson_deviance(actual_by_alpha[alpha], observed_by_alpha[alpha]), alpha))
    return min(scores, key=lambda item: (item[0], item[1]))[1]


def paired_match_bootstrap(
    matches: list[Match],
    actual: np.ndarray,
    model_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    seed: int = 20260914,
    n_bootstrap: int = 100_000,
) -> dict[str, float]:
    if actual.size != len(matches) * 2:
        raise ValueError("expected two team observations per match")
    errors_model = np.abs(actual - model_prediction).reshape(len(matches), 2).mean(axis=1)
    errors_baseline = np.abs(actual - baseline_prediction).reshape(len(matches), 2).mean(axis=1)
    improvement = errors_baseline - errors_model
    rng = np.random.default_rng(seed)
    samples = rng.choice(improvement, size=(n_bootstrap, len(matches)), replace=True).mean(axis=1)
    return {
        "observed_improvement": float(np.mean(improvement)),
        "ci_2_5": float(np.quantile(samples, 0.025)),
        "ci_97_5": float(np.quantile(samples, 0.975)),
        "seed": int(seed),
        "n_bootstrap": int(n_bootstrap),
        "n_matches": len(matches),
    }


def run_temporal_backtest(
    matches: list[Match],
    test_gameweek: int | None = None,
    candidates: tuple[float, ...] = ALPHA_CANDIDATES,
    bootstrap_seed: int = 20260914,
    n_bootstrap: int = 100_000,
) -> BacktestResult:
    weeks = sorted({match.gameweek for match in matches})
    if len(weeks) < 2:
        raise ValueError("at least two gameweeks are required")
    test_gameweek = weeks[-1] if test_gameweek is None else test_gameweek
    if test_gameweek not in weeks or test_gameweek == weeks[0]:
        raise ValueError("test_gameweek must be a later observed gameweek")
    train_matches = [match for match in matches if match.gameweek < test_gameweek]
    test_matches = [match for match in matches if match.gameweek == test_gameweek]
    selected_alpha = _tune_alpha(train_matches, candidates)
    train = to_team_observations(train_matches)
    test = to_team_observations(test_matches)
    _assert_temporal_order(train, test)
    model = PoissonTeamGoalsModel(alpha=selected_alpha).fit(train)
    actual = np.asarray([row.goals for row in test], dtype=float)
    model_prediction = model.predict(test)
    league_prediction, home_away_prediction = _baselines(train, test)
    metrics = {
        "model_mae": float(mean_absolute_error(actual, model_prediction)),
        "model_poisson_deviance": float(mean_poisson_deviance(actual, model_prediction)),
        "league_average_mae": float(mean_absolute_error(actual, league_prediction)),
        "league_average_poisson_deviance": float(mean_poisson_deviance(actual, league_prediction)),
        "home_away_mae": float(mean_absolute_error(actual, home_away_prediction)),
        "home_away_poisson_deviance": float(mean_poisson_deviance(actual, home_away_prediction)),
    }
    predictions = tuple(
        {
            "match_id": row.match_id,
            "gameweek": row.gameweek,
            "team": row.team,
            "opponent": row.opponent,
            "home": row.home,
            "actual_goals": row.goals,
            "model_prediction": float(model_value),
            "league_average_prediction": float(league_value),
            "home_away_prediction": float(home_away_value),
        }
        for row, model_value, league_value, home_away_value in zip(
            test, model_prediction, league_prediction, home_away_prediction
        )
    )
    return BacktestResult(
        train_gameweeks=tuple(sorted({match.gameweek for match in train_matches})),
        test_gameweek=test_gameweek,
        selected_alpha=selected_alpha,
        model=model.to_dict(),
        metrics=metrics,
        bootstrap=paired_match_bootstrap(
            test_matches,
            actual,
            model_prediction,
            league_prediction,
            seed=bootstrap_seed,
            n_bootstrap=n_bootstrap,
        ),
        predictions=predictions,
    )
