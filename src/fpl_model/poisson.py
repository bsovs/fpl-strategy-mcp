"""A small NumPy-only regularised Poisson regression implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .data import TeamObservation


@dataclass(frozen=True)
class FeatureSchema:
    teams: tuple[str, ...]
    reference_team: str


class RegularizedPoissonRegression:
    """Poisson GLM with team attack, opponent defence, home advantage, and L2.

    The intercept and home coefficient are left unpenalised. One team is used
    as the reference level for attack and defence to keep the design matrix
    identifiable. The implementation uses damped Newton updates so it runs
    with NumPy alone on the small FPL datasets this project targets.
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 200, tol: float = 1e-8):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coef_: np.ndarray | None = None
        self.schema_: FeatureSchema | None = None
        self.n_iter_: int | None = None
        self.objective_: float | None = None

    @property
    def feature_names_(self) -> tuple[str, ...]:
        self._require_fitted()
        assert self.schema_ is not None
        attack = tuple(f"attack[{team}]" for team in self.schema_.teams if team != self.schema_.reference_team)
        defence = tuple(f"defence[{team}]" for team in self.schema_.teams if team != self.schema_.reference_team)
        return ("intercept", *attack, *defence, "home_advantage")

    def _require_fitted(self) -> None:
        if self.coef_ is None or self.schema_ is None:
            raise RuntimeError("model is not fitted")

    def _design(self, observations: list[TeamObservation], schema: FeatureSchema) -> np.ndarray:
        non_reference = [team for team in schema.teams if team != schema.reference_team]
        attack_index = {team: idx for idx, team in enumerate(non_reference)}
        defence_index = {team: idx for idx, team in enumerate(non_reference)}
        matrix = np.zeros((len(observations), 1 + len(non_reference) * 2 + 1), dtype=float)
        matrix[:, 0] = 1.0
        for row_index, observation in enumerate(observations):
            if observation.team != schema.reference_team:
                matrix[row_index, 1 + attack_index[observation.team]] = 1.0
            if observation.opponent != schema.reference_team:
                matrix[row_index, 1 + len(non_reference) + defence_index[observation.opponent]] = 1.0
            matrix[row_index, -1] = observation.is_home
        return matrix

    def fit(self, observations: list[TeamObservation]) -> "RegularizedPoissonRegression":
        if not observations:
            raise ValueError("at least one observation is required")
        teams = sorted({row.team for row in observations} | {row.opponent for row in observations})
        if len(teams) < 2:
            raise ValueError("at least two teams are required")
        schema = FeatureSchema(teams=tuple(teams), reference_team=teams[-1])
        design = self._design(observations, schema)
        targets = np.asarray([row.goals for row in observations], dtype=float)
        penalty = np.full(design.shape[1], self.alpha, dtype=float)
        penalty[0] = 0.0
        penalty[-1] = 0.0
        coefficients = np.zeros(design.shape[1], dtype=float)
        coefficients[0] = math.log(max(float(targets.mean()), 0.1))

        def objective(beta: np.ndarray) -> float:
            eta = np.clip(design @ beta, -30.0, 30.0)
            return float(np.sum(np.exp(eta) - targets * eta) + 0.5 * np.sum(penalty * beta * beta))

        current_objective = objective(coefficients)
        for iteration in range(1, self.max_iter + 1):
            eta = np.clip(design @ coefficients, -30.0, 30.0)
            means = np.exp(eta)
            gradient = design.T @ (means - targets) + penalty * coefficients
            hessian = (design.T * means) @ design
            hessian.flat[:: hessian.shape[0] + 1] += penalty
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError("Poisson Hessian was singular; check input variation") from exc

            if float(np.linalg.norm(step, ord=np.inf)) < self.tol:
                self.n_iter_ = iteration
                break

            step_scale = 1.0
            accepted = False
            while step_scale >= 1e-8:
                candidate = coefficients - step_scale * step
                candidate_objective = objective(candidate)
                if candidate_objective <= current_objective:
                    coefficients = candidate
                    current_objective = candidate_objective
                    accepted = True
                    break
                step_scale *= 0.5
            if not accepted:
                self.n_iter_ = iteration
                break
        else:
            self.n_iter_ = self.max_iter

        self.schema_ = schema
        self.coef_ = coefficients
        self.objective_ = current_objective
        return self

    def predict(self, observations: list[TeamObservation]) -> np.ndarray:
        self._require_fitted()
        assert self.schema_ is not None and self.coef_ is not None
        unknown = {row.team for row in observations} | {row.opponent for row in observations}
        unseen = unknown - set(self.schema_.teams)
        if unseen:
            raise ValueError(f"observations contain unseen teams: {sorted(unseen)}")
        design = self._design(observations, self.schema_)
        return np.exp(np.clip(design @ self.coef_, -30.0, 30.0))

    def to_dict(self) -> dict:
        self._require_fitted()
        assert self.schema_ is not None and self.coef_ is not None
        return {
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "teams": list(self.schema_.teams),
            "reference_team": self.schema_.reference_team,
            "feature_names": list(self.feature_names_),
            "coefficients": self.coef_.tolist(),
            "n_iter": self.n_iter_,
            "objective": self.objective_,
        }
