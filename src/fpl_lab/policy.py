"""Action-value policy model for the strategic FPL simulator.

This is deliberately an action-value model, not another player-point model.
The training target must come from a historical legal-action simulator: future
squad points, price/equity change, transfer hits, chips, and rank utility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ACTION_KINDS = ("hold", "transfer", "wildcard", "free_hit", "bench_boost", "triple_captain")


@dataclass(frozen=True)
class CocktailConfig:
    """Interpretable gate for the multi-model action policy.

    The free-transfer anchor remains the default action.  The action-value
    ensemble can override it only when its risk-adjusted advantage clears a
    separate transfer/chip threshold.  These parameters are selected on
    temporal development folds and then frozen before the final season test.
    """

    name: str = "anchor_free"
    anchor_policy: str = "points_only_free_only"
    model_risk_aversion: float = 0.30
    model_weight: float = 0.75
    risk_penalty: float = 0.10
    transfer_gate: float = 1.00
    chip_gate: float = 4.00
    hit_gate: float = 7.00
    anchor_tolerance: float = 0.10
    allow_hits: bool = False
    allow_chips: bool = True
    min_chip_gameweek: int = 2
    max_chip_gameweek: int = 36

    def __post_init__(self) -> None:
        if self.anchor_policy not in {"points_only_free_only", "points_only", "price_aware", "chase"}:
            raise ValueError(f"unsupported anchor policy: {self.anchor_policy}")
        if self.model_risk_aversion < 0.0 or self.model_weight < 0.0 or self.model_weight > 1.0:
            raise ValueError("model risk/weight values are outside their valid ranges")
        if self.transfer_gate < 0.0 or self.chip_gate < 0.0 or self.hit_gate < 0.0:
            raise ValueError("action gates cannot be negative")
        if self.min_chip_gameweek < 1 or self.max_chip_gameweek > 38:
            raise ValueError("chip gameweek bounds must lie within GW1-GW38")


@dataclass(frozen=True)
class PolicyState:
    """Information known when the manager must choose an action."""

    gameweek: int
    weeks_remaining: int
    bank: float
    free_transfers: int
    squad_value: float
    rank_percentile: float = 0.5
    target_rank_percentile: float = 0.5
    template_ownership: float = 0.0
    chip_flexibility: float = 1.0
    rank_mode: str = "neutral"
    wildcard_available: bool = True
    free_hit_available: bool = True
    bench_boost_available: bool = True
    triple_captain_available: bool = True
    # Aggregate signal regime features.  These let the action learner learn
    # when a squad is operating in a high-news/high-availability-risk state
    # instead of treating every action delta as context-free.
    squad_minutes_probability: float = 0.0
    squad_news_risk: float = 0.0
    squad_context_reliability: float = 0.0
    squad_context_coverage: float = 0.0


@dataclass(frozen=True)
class PolicyAction:
    """A legal action candidate and forecast deltas supplied by the models."""

    kind: str
    player_out_id: str = ""
    player_in_id: str = ""
    hit_cost: float = 0.0
    short_points_delta: float = 0.0
    long_points_delta: float = 0.0
    short_price_delta: float = 0.0
    long_price_delta: float = 0.0
    short_fixture_delta: float = 0.0
    long_fixture_delta: float = 0.0
    form_delta: float = 0.0
    value_delta: float = 0.0
    role_security_delta: float = 0.0
    price_change_risk_delta: float = 0.0
    sell_loss: float = 0.0
    ownership_leverage_delta: float = 0.0
    short_minutes_delta: float = 0.0
    long_minutes_delta: float = 0.0
    news_risk_delta: float = 0.0
    set_piece_delta: float = 0.0
    transfer_role_delta: float = 0.0
    context_reliability_delta: float = 0.0
    uncertainty_delta: float = 0.0
    legal: bool = True


def encode_state_action(state: PolicyState, action: PolicyAction) -> np.ndarray:
    """Encode a state/action pair for the policy network.

    The encoding keeps state variables and action economics together so the
    network cannot learn a player ranking detached from bank, free transfers,
    season timing or risk posture.
    """

    if action.kind not in ACTION_KINDS:
        raise ValueError(f"unknown action kind: {action.kind}")
    rank_mode = {"neutral": 0.0, "chase": 1.0, "defend": -1.0}.get(state.rank_mode)
    if rank_mode is None:
        raise ValueError(f"unknown rank mode: {state.rank_mode}")
    action_one_hot = [float(action.kind == kind) for kind in ACTION_KINDS]
    return np.nan_to_num(np.asarray(
        [
            state.gameweek / 38.0,
            state.weeks_remaining / 38.0,
            state.bank / 15.0,
            state.free_transfers / 3.0,
            state.squad_value / 100.0,
            state.rank_percentile,
            state.target_rank_percentile,
            state.template_ownership,
            state.chip_flexibility,
            float(state.wildcard_available),
            float(state.free_hit_available),
            float(state.bench_boost_available),
            float(state.triple_captain_available),
            float(np.clip(state.squad_minutes_probability, 0.0, 1.0)),
            float(np.clip(state.squad_news_risk, 0.0, 1.0)),
            float(np.clip(state.squad_context_reliability, 0.0, 1.0)),
            float(np.clip(state.squad_context_coverage, 0.0, 1.0)),
            rank_mode,
            *action_one_hot,
            float(action.legal),
            action.hit_cost / 4.0,
            action.short_points_delta,
            action.long_points_delta,
            action.short_price_delta,
            action.long_price_delta,
            action.short_fixture_delta,
            action.long_fixture_delta,
            action.form_delta,
            action.value_delta,
            action.role_security_delta,
            action.price_change_risk_delta,
            action.sell_loss,
            action.ownership_leverage_delta,
            action.short_minutes_delta,
            action.long_minutes_delta,
            action.news_risk_delta,
            action.set_piece_delta,
            action.transfer_role_delta,
            action.context_reliability_delta,
            action.uncertainty_delta,
        ],
        dtype=float,
    ), nan=0.0, posinf=0.0, neginf=0.0)


def _encode_state_action_context_v1(state: PolicyState, action: PolicyAction) -> np.ndarray:
    """Encode the intermediate 38-feature policy schema.

    This is kept only for loading the research model produced before explicit
    form/fixture/value/loss action features were added.
    """

    if action.kind not in ACTION_KINDS:
        raise ValueError(f"unknown action kind: {action.kind}")
    rank_mode = {"neutral": 0.0, "chase": 1.0, "defend": -1.0}.get(state.rank_mode)
    if rank_mode is None:
        raise ValueError(f"unknown rank mode: {state.rank_mode}")
    action_one_hot = [float(action.kind == kind) for kind in ACTION_KINDS]
    return np.nan_to_num(np.asarray(
        [
            state.gameweek / 38.0,
            state.weeks_remaining / 38.0,
            state.bank / 15.0,
            state.free_transfers / 3.0,
            state.squad_value / 100.0,
            state.rank_percentile,
            state.target_rank_percentile,
            state.template_ownership,
            state.chip_flexibility,
            float(state.wildcard_available),
            float(state.free_hit_available),
            float(state.bench_boost_available),
            float(state.triple_captain_available),
            float(np.clip(state.squad_minutes_probability, 0.0, 1.0)),
            float(np.clip(state.squad_news_risk, 0.0, 1.0)),
            float(np.clip(state.squad_context_reliability, 0.0, 1.0)),
            float(np.clip(state.squad_context_coverage, 0.0, 1.0)),
            rank_mode,
            *action_one_hot,
            float(action.legal),
            action.hit_cost / 4.0,
            action.short_points_delta,
            action.long_points_delta,
            action.short_price_delta,
            action.long_price_delta,
            action.ownership_leverage_delta,
            action.short_minutes_delta,
            action.long_minutes_delta,
            action.news_risk_delta,
            action.set_piece_delta,
            action.transfer_role_delta,
            action.context_reliability_delta,
            action.uncertainty_delta,
        ],
        dtype=float,
    ), nan=0.0, posinf=0.0, neginf=0.0)


def _encode_state_action_legacy(state: PolicyState, action: PolicyAction) -> np.ndarray:
    """Encode the original 27-feature shipped policy schema."""

    if action.kind not in ACTION_KINDS:
        raise ValueError(f"unknown action kind: {action.kind}")
    rank_mode = {"neutral": 0.0, "chase": 1.0, "defend": -1.0}.get(state.rank_mode)
    if rank_mode is None:
        raise ValueError(f"unknown rank mode: {state.rank_mode}")
    action_one_hot = [float(action.kind == kind) for kind in ACTION_KINDS]
    return np.nan_to_num(np.asarray(
        [
            state.gameweek / 38.0,
            state.weeks_remaining / 38.0,
            state.bank / 15.0,
            state.free_transfers / 3.0,
            state.squad_value / 100.0,
            state.rank_percentile,
            state.target_rank_percentile,
            state.template_ownership,
            state.chip_flexibility,
            float(state.wildcard_available),
            float(state.free_hit_available),
            float(state.bench_boost_available),
            float(state.triple_captain_available),
            rank_mode,
            *action_one_hot,
            float(action.legal),
            action.hit_cost / 4.0,
            action.short_points_delta,
            action.long_points_delta,
            action.short_price_delta,
            action.long_price_delta,
            action.ownership_leverage_delta,
        ],
        dtype=float,
    ), nan=0.0, posinf=0.0, neginf=0.0)


class ActionValueMLP:
    """Small neural-network policy baseline for offline action-value learning.

    ``realized_values`` must be generated by the historical simulator. Fitting
    this network directly to observed player points would answer the wrong
    question and is intentionally not supported by this wrapper.
    """

    def __init__(self, hidden_layer_sizes: tuple[int, ...] = (64, 32), random_state: int = 42):
        self.model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    TransformedTargetRegressor(
                        regressor=MLPRegressor(
                            hidden_layer_sizes=hidden_layer_sizes,
                            activation="relu",
                            alpha=0.001,
                            max_iter=1200,
                            tol=1e-3,
                            random_state=random_state,
                        ),
                        transformer=StandardScaler(),
                    ),
                ),
            ]
        )

    def fit(
        self,
        states: Iterable[PolicyState],
        actions: Iterable[PolicyAction],
        realized_values: Iterable[float],
    ) -> "ActionValueMLP":
        features = np.vstack([encode_state_action(state, action) for state, action in zip(states, actions)])
        targets = np.asarray(list(realized_values), dtype=float)
        if len(features) != len(targets):
            raise ValueError("states, actions and realized_values must have equal lengths")
        if len(targets) < 2:
            raise ValueError("at least two historical action outcomes are required")
        self.model.fit(features, targets)
        return self

    def _encode_for_fitted_model(self, state: PolicyState, action: PolicyAction) -> np.ndarray:
        expected = getattr(self.model.named_steps["scale"], "n_features_in_", None)
        if expected == 27:
            return _encode_state_action_legacy(state, action)
        if expected == 38:
            return _encode_state_action_context_v1(state, action)
        return encode_state_action(state, action)

    def predict(self, states: Iterable[PolicyState], actions: Iterable[PolicyAction]) -> np.ndarray:
        features = np.vstack([
            self._encode_for_fitted_model(state, action)
            for state, action in zip(states, actions)
        ])
        return self.model.predict(features)

    def rank_actions(
        self,
        state: PolicyState,
        actions: Iterable[PolicyAction],
        risk_aversion: float = 0.0,
    ) -> list[tuple[PolicyAction, float]]:
        """Rank legal actions by predicted value.

        ``risk_aversion`` is accepted so a single network and an ensemble can
        be used by the same simulator. A single network has no epistemic
        uncertainty estimate, so the argument has no effect here.
        """

        candidates = [action for action in actions if action.legal]
        values = self.predict([state] * len(candidates), candidates)
        return sorted(zip(candidates, values.tolist()), key=lambda pair: pair[1], reverse=True)


class ActionValueEnsemble:
    """Bootstrap ensemble of action-value MLPs for strategic FPL decisions.

    The ensemble is intentionally small and inspectable. Each member sees a
    bootstrap resample of the legal-action rollouts, giving the runtime policy
    both a mean value and a disagreement estimate. The simulator can subtract
    a configurable fraction of disagreement before choosing an action, which
    makes the learned policy less eager to exploit sparse regions of the
    training data.
    """

    def __init__(
        self,
        n_models: int = 5,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        random_state: int = 42,
        bootstrap_fraction: float = 0.90,
    ):
        if n_models < 2:
            raise ValueError("an ensemble requires at least two models")
        if not 0.5 <= bootstrap_fraction <= 1.0:
            raise ValueError("bootstrap_fraction must be between 0.5 and 1.0")
        self.n_models = int(n_models)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.random_state = int(random_state)
        self.bootstrap_fraction = float(bootstrap_fraction)
        self.models: list[ActionValueMLP] = []

    def fit(
        self,
        states: Iterable[PolicyState],
        actions: Iterable[PolicyAction],
        realized_values: Iterable[float],
    ) -> "ActionValueEnsemble":
        state_rows = list(states)
        action_rows = list(actions)
        targets = np.asarray(list(realized_values), dtype=float)
        if not (len(state_rows) == len(action_rows) == len(targets)):
            raise ValueError("states, actions and realized_values must have equal lengths")
        if len(targets) < 2:
            raise ValueError("at least two historical action outcomes are required")

        rng = np.random.default_rng(self.random_state)
        sample_size = max(2, int(round(len(targets) * self.bootstrap_fraction)))
        self.models = []
        for member in range(self.n_models):
            indices = rng.integers(0, len(targets), size=sample_size)
            model = ActionValueMLP(
                hidden_layer_sizes=self.hidden_layer_sizes,
                random_state=self.random_state + member,
            )
            model.fit(
                [state_rows[index] for index in indices],
                [action_rows[index] for index in indices],
                targets[indices],
            )
            self.models.append(model)
        return self

    def _member_predictions(
        self,
        states: Iterable[PolicyState],
        actions: Iterable[PolicyAction],
    ) -> np.ndarray:
        if not self.models:
            raise ValueError("ensemble has not been fitted")
        state_rows = list(states)
        action_rows = list(actions)
        if len(state_rows) != len(action_rows):
            raise ValueError("states and actions must have equal lengths")
        return np.vstack(
            [model.predict(state_rows, action_rows) for model in self.models]
        )

    def predict(self, states: Iterable[PolicyState], actions: Iterable[PolicyAction]) -> np.ndarray:
        """Return the ensemble mean action value."""

        return self._member_predictions(states, actions).mean(axis=0)

    def predict_with_uncertainty(
        self,
        states: Iterable[PolicyState],
        actions: Iterable[PolicyAction],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return mean and member disagreement for each state/action pair."""

        predictions = self._member_predictions(states, actions)
        return predictions.mean(axis=0), predictions.std(axis=0)

    def rank_actions(
        self,
        state: PolicyState,
        actions: Iterable[PolicyAction],
        risk_aversion: float = 0.20,
    ) -> list[tuple[PolicyAction, float]]:
        candidates = [action for action in actions if action.legal]
        means, uncertainty = self.predict_with_uncertainty([state] * len(candidates), candidates)
        risk_adjusted = means - float(risk_aversion) * uncertainty
        return sorted(
            zip(candidates, risk_adjusted.tolist()),
            key=lambda pair: pair[1],
            reverse=True,
        )
