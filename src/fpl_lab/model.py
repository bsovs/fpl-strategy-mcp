"""Scikit-learn implementation of the saved team-goals model specification."""

from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder

from .data import TeamObservation


class PoissonTeamGoalsModel:
    """Regularised Poisson model: attack, opponent defence, and home advantage."""

    def __init__(self, alpha: float = 1.0, max_iter: int = 3000, tol: float = 1e-9):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.encoder: OneHotEncoder | None = None
        self.regressor: PoissonRegressor | None = None
        self.teams: tuple[str, ...] | None = None

    def _categorical(self, observations: list[TeamObservation]) -> np.ndarray:
        return np.asarray([[row.team, row.opponent] for row in observations], dtype=object)

    def _design(self, observations: list[TeamObservation], fit: bool = False) -> np.ndarray:
        if fit:
            teams = sorted({row.team for row in observations} | {row.opponent for row in observations})
            if len(teams) < 2:
                raise ValueError("at least two teams are required")
            self.teams = tuple(teams)
            self.encoder = OneHotEncoder(
                categories=[np.asarray(self.teams, dtype=object), np.asarray(self.teams, dtype=object)],
                handle_unknown="ignore",
                sparse_output=False,
                dtype=float,
            )
            categories = self._categorical(observations)
            encoded = self.encoder.fit_transform(categories)
        else:
            if self.encoder is None:
                raise RuntimeError("model is not fitted")
            encoded = self.encoder.transform(self._categorical(observations))
        home = np.asarray([[row.home] for row in observations], dtype=float)
        return np.column_stack((encoded, home))

    def fit(self, observations: list[TeamObservation]) -> "PoissonTeamGoalsModel":
        if not observations:
            raise ValueError("at least one observation is required")
        design = self._design(observations, fit=True)
        targets = np.asarray([row.goals for row in observations], dtype=float)
        self.regressor = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)
        self.regressor.fit(design, targets)
        return self

    def predict(self, observations: list[TeamObservation]) -> np.ndarray:
        if self.regressor is None or self.encoder is None or self.teams is None:
            raise RuntimeError("model is not fitted")
        unknown = ({row.team for row in observations} | {row.opponent for row in observations}) - set(self.teams)
        if unknown:
            raise ValueError(f"observations contain unseen teams: {sorted(unknown)}")
        return self.regressor.predict(self._design(observations))

    def to_dict(self) -> dict:
        if self.regressor is None or self.encoder is None or self.teams is None:
            raise RuntimeError("model is not fitted")
        return {
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "teams": list(self.teams),
            "feature_names": [*self.encoder.get_feature_names_out(["attack", "defence"]), "home"],
            "intercept": float(self.regressor.intercept_),
            "coefficients": self.regressor.coef_.tolist(),
            "n_iter": int(self.regressor.n_iter_),
        }
