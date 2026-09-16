"""League-style strategic simulation for FPL transfer policies.

The ordinary season simulator asks, "how many points did this policy score?"
This module adds the missing game-theory layer: several managers play the same
historical season, see the same pre-deadline information, and are ranked
against one another after each gameweek.  Hybrid policies can use the learned
action-value model, but retain the robust free-transfer policy as an anchor.

The league is synthetic.  It compares strategy rules against one another; it
does not claim to recreate an actual user's mini-league or real manager
decisions until real manager histories are supplied as a separate dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import exp, tanh
from typing import Iterable

import numpy as np

from .decision import PlayerSignal, TransferRecommendation, recommend_transfers
from .policy import ActionValueEnsemble, ActionValueMLP, PolicyAction
from .simulator import (
    SeasonData,
    _apply_transfer,
    _choose_neural_transfer,
    _choose_transfer_bundle,
    _policy_config,
    _scoring_snapshot,
    _states_for_decision,
    build_neural_action_candidates,
    choose_lineup,
    policy_state_from_runtime,
    score_gameweek,
    season_rules,
    selling_price_tenths,
)


NEURAL_TYPES = (ActionValueMLP, ActionValueEnsemble)
LEAGUE_STRATEGIES = (
    "hold",
    "points_only_free_only",
    "points_only",
    "price_aware",
    "chase",
    "neural",
    "hybrid_safe",
    "hybrid_balanced",
    "hybrid_win",
)


@dataclass(frozen=True)
class LeagueManagerSpec:
    """A named manager rule used in one synthetic league."""

    manager_id: str
    strategy: str


@dataclass
class LeagueManagerResult:
    """One manager's season result and final league placement."""

    manager_id: str
    strategy: str
    season: str
    gross_points: float
    hit_points: float
    net_points: float
    transfers: int
    paid_transfers: int
    final_bank: float
    final_squad_value: float
    gameweeks: int
    rank: int = 0
    rank_percentile: float = 0.0
    win: bool = False
    podium: bool = False
    log: list[dict] = field(default_factory=list)

    def to_dict(self, include_log: bool = True) -> dict:
        payload = asdict(self)
        if not include_log:
            payload.pop("log", None)
        return payload


@dataclass(frozen=True)
class HybridProfile:
    """Fixed policy profile; these are selected on development only."""

    base_policy: str
    model_weight: float
    model_risk_aversion: float
    min_model_advantage: float
    minimum_score: float
    allow_hits: bool
    aggression: float
    hit_model_advantage: float
    risk_penalty: float


# These profiles are intentionally simple and interpretable.  They are not
# fitted on the holdout.  The league runner reports the development ranking so
# a later experiment can replace them with a learned policy only after a new
# temporal split is declared.
HYBRID_PROFILES = {
    "hybrid_safe": HybridProfile(
        base_policy="points_only_free_only",
        model_weight=0.20,
        model_risk_aversion=0.35,
        min_model_advantage=1.50,
        minimum_score=0.08,
        allow_hits=False,
        aggression=0.0,
        hit_model_advantage=99.0,
        risk_penalty=0.20,
    ),
    "hybrid_balanced": HybridProfile(
        base_policy="points_only_free_only",
        model_weight=0.50,
        model_risk_aversion=0.20,
        min_model_advantage=0.25,
        minimum_score=0.05,
        allow_hits=False,
        aggression=0.20,
        hit_model_advantage=99.0,
        risk_penalty=0.10,
    ),
    "hybrid_win": HybridProfile(
        base_policy="chase",
        model_weight=0.35,
        model_risk_aversion=0.05,
        min_model_advantage=-0.50,
        minimum_score=0.02,
        allow_hits=True,
        aggression=0.90,
        hit_model_advantage=5.00,
        risk_penalty=0.02,
    ),
}


@dataclass
class _ManagerRuntime:
    spec: LeagueManagerSpec
    squad_ids: list[str]
    purchase_prices: dict[str, int]
    bank_tenths: int
    free_transfers: int
    gross_points: float = 0.0
    hit_points: float = 0.0
    transfers: int = 0
    paid_transfers: int = 0
    log: list[dict] = field(default_factory=list)

    @property
    def net_points(self) -> float:
        return self.gross_points - self.hit_points


def _action_key(action: PolicyAction) -> tuple[str, str, str]:
    return (action.kind, action.player_out_id, action.player_in_id)


def _recommendation_key(recommendation: TransferRecommendation) -> tuple[str, str, str]:
    return ("transfer", recommendation.player_out_id, recommendation.player_in_id)


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -50.0, 50.0))
    return 1.0 / (1.0 + exp(-value))


def _candidate_recommendations(
    current: SeasonData,
    gameweek: int,
    squad_ids: list[str],
    purchase_prices: dict[str, int],
    bank_tenths: int,
    free_transfers: int,
    signals: list[PlayerSignal],
    policy: str,
    candidate_width: int = 18,
) -> list[TransferRecommendation]:
    """Return scored legal one-transfer candidates for a hybrid policy."""

    current_states, buyable = _states_for_decision(
        current, gameweek, squad_ids, purchase_prices, bank_tenths
    )
    config = _policy_config(policy, 3, 8, free_transfers, bank_tenths)
    return recommend_transfers(current_states, buyable, signals, config)[:candidate_width]


def _model_action_scores(
    current: SeasonData,
    gameweek: int,
    squad_ids: list[str],
    purchase_prices: dict[str, int],
    bank_tenths: int,
    free_transfers: int,
    signals: list[PlayerSignal],
    neural_policy: ActionValueMLP | ActionValueEnsemble,
    risk_aversion: float,
    candidate_width: int = 18,
) -> tuple[dict[tuple[str, str, str], tuple[float, float]], dict[tuple[str, str, str], TransferRecommendation]]:
    """Return risk-adjusted action scores and their recommendation mapping."""

    candidates = build_neural_action_candidates(
        current,
        gameweek,
        squad_ids,
        purchase_prices,
        bank_tenths,
        free_transfers,
        signals,
        candidate_width=candidate_width,
    )
    actions = [candidate.action for candidate in candidates]
    state = policy_state_from_runtime(
        current, gameweek, squad_ids, bank_tenths, free_transfers
    )
    if isinstance(neural_policy, ActionValueEnsemble):
        means, uncertainty = neural_policy.predict_with_uncertainty([state] * len(actions), actions)
    else:
        means = neural_policy.predict([state] * len(actions), actions)
        uncertainty = np.zeros(len(actions), dtype=float)
    values = {}
    recommendations = {}
    for candidate, mean, spread in zip(candidates, means, uncertainty):
        action = candidate.action
        key = _action_key(action)
        values[key] = (float(mean - risk_aversion * spread), float(spread))
        if candidate.transfer_bundle and action.kind == "transfer":
            recommendations[key] = candidate.transfer_bundle[0]
    return values, recommendations


def _hybrid_transfer(
    current: SeasonData,
    gameweek: int,
    squad_ids: list[str],
    purchase_prices: dict[str, int],
    bank_tenths: int,
    free_transfers: int,
    signals: list[PlayerSignal],
    neural_policy: ActionValueMLP | ActionValueEnsemble,
    manager_net_points: float,
    leader_net_points: float,
    profile_name: str,
    league_size: int,
    candidate_width: int = 18,
) -> list[TransferRecommendation]:
    """Choose one transfer using a free-transfer anchor plus model/rank signals."""

    if profile_name not in HYBRID_PROFILES:
        raise ValueError(f"unknown hybrid profile: {profile_name}")
    profile = HYBRID_PROFILES[profile_name]
    model_values, model_recommendations = _model_action_scores(
        current,
        gameweek,
        squad_ids,
        purchase_prices,
        bank_tenths,
        free_transfers,
        signals,
        neural_policy,
        profile.model_risk_aversion,
        candidate_width=candidate_width,
    )
    hold_key = ("hold", "", "")
    hold_value = model_values.get(hold_key, (0.0, 0.0))[0]
    model_advantage = {
        key: value[0] - hold_value for key, value in model_values.items()
    }
    signal_by_id = {signal.player_id: signal for signal in signals}

    # Union the robust heuristic candidates and model candidates.  A raw
    # neural policy is otherwise constrained by whichever heuristic happened
    # to generate its narrow candidate set during training.
    recommendation_by_key = {
        key: recommendation
        for key, recommendation in model_recommendations.items()
        if profile.allow_hits or recommendation.hit_cost <= 0.0
    }
    if not recommendation_by_key:
        return []

    # Standing gap is intentionally a small component.  It should change the
    # objective near the margin, not cause a manager to abandon the underlying
    # points/value forecasts after one bad week.
    deficit = max(0.0, leader_net_points - manager_net_points)
    behind = float(np.clip(deficit / max(20.0, 10.0 * max(1, league_size)), 0.0, 1.0))
    leading = float(np.clip(max(0.0, manager_net_points - leader_net_points) / 30.0, 0.0, 1.0))

    scored = []
    for key, recommendation in recommendation_by_key.items():
        if recommendation.hit_cost > 0.0:
            model_edge = model_advantage.get(key, -99.0)
            # Hits are an upside option for a manager behind, never the default
            # response to merely being a few points off the leader.
            hit_gate = profile.hit_model_advantage - 1.25 * behind + 0.75 * leading
            if model_edge < hit_gate:
                continue
        heuristic_score = float(np.tanh(recommendation.combined_score / 6.0))
        model_edge = model_advantage.get(key, 0.0)
        model_score = float(np.tanh(model_edge / 6.0))
        spread = model_values.get(key, (0.0, 0.0))[1]
        in_signal = signal_by_id.get(recommendation.player_in_id)
        leverage = (
            float(in_signal.ownership_leverage - signal_by_id[recommendation.player_out_id].ownership_leverage)
            if in_signal is not None and recommendation.player_out_id in signal_by_id
            else 0.0
        )
        # A behind manager values differentiated upside; a manager leading the
        # league pays a small price for uncertainty instead.
        game_theory = profile.aggression * behind * leverage
        risk_penalty = profile.risk_penalty * float(spread)
        score = (
            profile.model_weight * model_score
            + (1.0 - profile.model_weight) * heuristic_score
            + game_theory
            - risk_penalty
        )
        scored.append((score, model_edge, recommendation))

    if not scored:
        return []
    scored.sort(key=lambda row: (row[0], row[1], row[2].combined_score), reverse=True)
    best_score, best_model_edge, best_recommendation = scored[0]
    if best_score < profile.minimum_score:
        return []
    if best_model_edge < profile.min_model_advantage:
        # The heuristic is still allowed to win when it is decisively better;
        # this is the key distinction between a hybrid and a model veto.
        if best_recommendation.combined_score < 1.0:
            return []
    return [best_recommendation]


def _initial_runtime(
    current: SeasonData,
    rules,
    spec: LeagueManagerSpec,
    starting_squad_ids: Iterable[str],
) -> _ManagerRuntime:
    squad_ids = list(starting_squad_ids)
    gw1 = current.snapshots_by_gw[1]
    purchase_prices = {player_id: gw1[player_id].price_tenths for player_id in squad_ids}
    return _ManagerRuntime(
        spec=spec,
        squad_ids=squad_ids,
        purchase_prices=purchase_prices,
        bank_tenths=rules.budget_tenths - sum(purchase_prices.values()),
        free_transfers=0,
    )


def _standings(runtimes: list[_ManagerRuntime]) -> dict[str, tuple[int, float, float]]:
    """Return rank, leader points, and rank percentile before a deadline."""

    if not runtimes:
        return {}
    leader = max(runtime.net_points for runtime in runtimes)
    size = len(runtimes)
    output = {}
    for runtime in runtimes:
        rank = 1 + sum(other.net_points > runtime.net_points for other in runtimes)
        percentile = 1.0 if size == 1 else 1.0 - (rank - 1) / (size - 1)
        output[runtime.spec.manager_id] = (rank, leader, percentile)
    return output


def simulate_league(
    current: SeasonData,
    previous: SeasonData | None,
    starting_squad_ids: Iterable[str],
    manager_specs: Iterable[LeagueManagerSpec],
    signal_cache: dict[int, list[PlayerSignal]],
    neural_policy: ActionValueMLP | ActionValueEnsemble | None = None,
    context_store=None,
    candidate_width: int = 18,
) -> list[LeagueManagerResult]:
    """Simulate a set of managers head-to-head over one historical season."""

    specs = list(manager_specs)
    if not specs:
        raise ValueError("at least one league manager is required")
    if len({spec.manager_id for spec in specs}) != len(specs):
        raise ValueError("manager_id values must be unique")
    for spec in specs:
        if spec.strategy not in LEAGUE_STRATEGIES:
            raise ValueError(f"unknown league strategy: {spec.strategy}")
        if spec.strategy.startswith("hybrid_") or spec.strategy == "neural":
            if neural_policy is None:
                raise ValueError(f"strategy={spec.strategy} requires a neural policy")
    rules = season_rules(current.season)
    common_squad = list(starting_squad_ids)
    if len(common_squad) != 15 or len(set(common_squad)) != 15:
        raise ValueError("league starting squad must contain 15 unique players")
    runtimes = [_initial_runtime(current, rules, spec, common_squad) for spec in specs]
    available_gameweeks = sorted(current.snapshots_by_gw)

    for gameweek in available_gameweeks:
        if gameweek < 1:
            continue
        signals = signal_cache.get(gameweek)
        if signals is None:
            raise ValueError(f"missing signal cache for GW{gameweek}")
        signal_by_id = {signal.player_id: signal for signal in signals}
        standings = _standings(runtimes)
        for runtime in runtimes:
            pre_squad_ids = list(runtime.squad_ids)
            pre_purchase_prices = dict(runtime.purchase_prices)
            pre_bank_tenths = runtime.bank_tenths
            pre_free_transfers = runtime.free_transfers
            manager_rank, leader_points, rank_percentile = standings[runtime.spec.manager_id]
            transfer_bundle: list[TransferRecommendation] = []
            strategy = runtime.spec.strategy
            if gameweek >= 2 and strategy != "hold":
                if strategy == "neural":
                    transfer_bundle = _choose_neural_transfer(
                        current,
                        gameweek,
                        runtime.squad_ids,
                        runtime.purchase_prices,
                        runtime.bank_tenths,
                        runtime.free_transfers,
                        signals,
                        neural_policy,
                    )
                elif strategy.startswith("hybrid_"):
                    transfer_bundle = _hybrid_transfer(
                        current,
                        gameweek,
                        runtime.squad_ids,
                        runtime.purchase_prices,
                        runtime.bank_tenths,
                        runtime.free_transfers,
                        signals,
                        neural_policy,
                        runtime.net_points,
                        leader_points,
                        strategy,
                        len(runtimes),
                        candidate_width=candidate_width,
                    )
                else:
                    transfer_bundle = _choose_transfer_bundle(
                        current,
                        gameweek,
                        runtime.squad_ids,
                        runtime.purchase_prices,
                        runtime.bank_tenths,
                        runtime.free_transfers,
                        signals,
                        strategy,
                        3,
                        8,
                    )
            if transfer_bundle:
                for transfer in transfer_bundle:
                    runtime.squad_ids, runtime.purchase_prices, runtime.bank_tenths = _apply_transfer(
                        transfer,
                        runtime.squad_ids,
                        runtime.purchase_prices,
                        current,
                        gameweek,
                        runtime.bank_tenths,
                    )
                runtime.transfers += len(transfer_bundle)
                paid_this_week = max(0, len(transfer_bundle) - runtime.free_transfers)
                runtime.paid_transfers += paid_this_week
                runtime.hit_points += paid_this_week * rules.hit_cost

            squad_snapshots = {
                player_id: _scoring_snapshot(current, player_id, gameweek)
                for player_id in runtime.squad_ids
            }
            squad_signal_map = {
                player_id: signal_by_id[player_id]
                for player_id in squad_snapshots
                if player_id in signal_by_id
            }
            for player_id in squad_snapshots:
                squad_signal_map.setdefault(
                    player_id,
                    PlayerSignal(player_id=player_id, short_expected_points=0.0, long_expected_points=0.0),
                )
            starting, bench, captain, vice = choose_lineup(
                list(squad_snapshots), squad_snapshots, squad_signal_map
            )
            week_points, scoring = score_gameweek(
                list(squad_snapshots), starting, bench, captain, vice, squad_snapshots
            )
            runtime.gross_points += week_points
            if not transfer_bundle:
                runtime.free_transfers = min(rules.free_transfer_cap, runtime.free_transfers + 1)
            else:
                runtime.free_transfers = min(
                    rules.free_transfer_cap,
                    max(0, runtime.free_transfers - len(transfer_bundle)) + 1,
                )
            runtime.log.append(
                {
                    "gameweek": gameweek,
                    "manager_rank_before": manager_rank,
                    "leader_net_points_before": leader_points,
                    "manager_net_points_before": runtime.net_points - week_points,
                    "rank_percentile_before": rank_percentile,
                    "decision_time": current.decision_times.get(gameweek).isoformat()
                    if current.decision_times.get(gameweek)
                    else None,
                    "pre_squad_ids": pre_squad_ids,
                    "pre_purchase_prices": pre_purchase_prices,
                    "pre_bank_tenths": pre_bank_tenths,
                    "pre_free_transfers": pre_free_transfers,
                    "post_squad_ids": list(runtime.squad_ids),
                    "transfer_out": [transfer.player_out for transfer in transfer_bundle],
                    "transfer_in": [transfer.player_in for transfer in transfer_bundle],
                    "transfer_score": sum(transfer.combined_score for transfer in transfer_bundle)
                    if transfer_bundle
                    else None,
                    "free_transfers_after": runtime.free_transfers,
                    "bank": runtime.bank_tenths / 10.0,
                    "squad_value": sum(snapshot.price for snapshot in squad_snapshots.values()),
                    "gross_points": week_points,
                    "net_points_after_week": runtime.net_points,
                    "captain": scoring["captain"],
                    "starting": scoring["final_lineup"],
                }
            )

    final_net_points = {runtime.spec.manager_id: runtime.net_points for runtime in runtimes}
    size = len(runtimes)
    leader = max(final_net_points.values())
    results = []
    for runtime in runtimes:
        rank = 1 + sum(value > runtime.net_points for value in final_net_points.values())
        percentile = 1.0 if size == 1 else 1.0 - (rank - 1) / (size - 1)
        last_gameweek = max(available_gameweeks)
        final_value = sum(
            current.snapshot(player_id, last_gameweek).price
            for player_id in runtime.squad_ids
            if current.snapshot(player_id, last_gameweek) is not None
        )
        results.append(
            LeagueManagerResult(
                manager_id=runtime.spec.manager_id,
                strategy=runtime.spec.strategy,
                season=current.season,
                gross_points=runtime.gross_points,
                hit_points=runtime.hit_points,
                net_points=runtime.net_points,
                transfers=runtime.transfers,
                paid_transfers=runtime.paid_transfers,
                final_bank=runtime.bank_tenths / 10.0,
                final_squad_value=final_value,
                gameweeks=len(runtime.log),
                rank=rank,
                rank_percentile=percentile,
                win=rank == 1 and runtime.net_points == leader,
                podium=rank <= min(3, size),
                log=runtime.log,
            )
        )
    return results
