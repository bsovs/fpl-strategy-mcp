"""Counterfactual action-value training data for the strategic policy.

The target is a short-horizon downstream utility from the legal simulator, not
the next fixture's player points. For each observed pre-deadline state, the
module evaluates hold and a diverse set of legal alternatives by replaying the
season with the first action forced and a fixed, non-learned continuation
policy. This makes the label measure the value of choosing an action now while
still allowing the manager to make ordinary future decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .context import ContextStore
from .policy import PolicyAction, PolicyState
from .simulator import (
    CHIP_KINDS,
    PolicyActionCandidate,
    SeasonData,
    build_neural_action_candidates,
    policy_state_from_runtime,
    simulate_season,
)


@dataclass(frozen=True)
class CounterfactualExample:
    season: str
    scenario: str
    behavior_policy: str
    gameweek: int
    decision_time: str | None
    action_id: str
    state: PolicyState
    action: PolicyAction
    realized_value: float
    future_net_points: float
    hit_points: float
    ending_bank: float
    ending_squad_value: float
    horizon_gameweeks: int

    def to_record(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "scenario": self.scenario,
            "behavior_policy": self.behavior_policy,
            "gameweek": self.gameweek,
            "decision_time": self.decision_time,
            "action_id": self.action_id,
            "state": asdict(self.state),
            "action": asdict(self.action),
            "realized_value": self.realized_value,
            "future_net_points": self.future_net_points,
            "hit_points": self.hit_points,
            "ending_bank": self.ending_bank,
            "ending_squad_value": self.ending_squad_value,
            "horizon_gameweeks": self.horizon_gameweeks,
        }


def counterfactual_target_values(
    examples: Iterable[CounterfactualExample],
    chip_opportunity_cost: float = 0.65,
) -> list[float]:
    """Return resource-aware within-state action advantages.

    A chip's immediate counterfactual points are not its full strategic value:
    spending it now also gives up the option to spend that chip later. The
    same-season, same-trajectory best future label is a simple offline optimal-
    stopping proxy. It is used only to shape development targets; future
    observed outcomes are never sent as runtime features.
    """

    if chip_opportunity_cost < 0.0:
        raise ValueError("chip_opportunity_cost cannot be negative")
    rows = list(examples)
    hold_values = {
        (example.season, example.scenario, example.behavior_policy, example.gameweek): example.realized_value
        for example in rows
        if example.action_id == "hold"
    }
    raw_by_trajectory: dict[tuple[str, str, str, str], list[tuple[int, float]]] = {}
    for example in rows:
        key = (example.season, example.scenario, example.behavior_policy, example.action_id)
        state_key = (example.season, example.scenario, example.behavior_policy, example.gameweek)
        if state_key not in hold_values:
            raise ValueError("every counterfactual state must contain a hold label")
        raw_by_trajectory.setdefault(key, []).append(
            (example.gameweek, example.realized_value - hold_values[state_key])
        )
    for values in raw_by_trajectory.values():
        values.sort(key=lambda pair: pair[0])

    targets: list[float] = []
    for example in rows:
        state_key = (example.season, example.scenario, example.behavior_policy, example.gameweek)
        raw_advantage = example.realized_value - hold_values[state_key]
        if example.action_id in CHIP_KINDS:
            future_values = [
                value
                for future_gameweek, value in raw_by_trajectory.get(
                    (example.season, example.scenario, example.behavior_policy, example.action_id),
                    [],
                )
                if future_gameweek > example.gameweek
            ]
            raw_advantage -= chip_opportunity_cost * max(future_values, default=0.0)
        targets.append(float(raw_advantage))
    return targets


def _action_id(action: PolicyAction) -> str:
    if action.kind == "hold":
        return "hold"
    if action.kind in CHIP_KINDS:
        return action.kind
    return f"transfer:{action.player_out_id}>{action.player_in_id}"


def _counterfactual_value(
    current: SeasonData,
    previous: SeasonData | None,
    row: dict[str, Any],
    candidate: PolicyActionCandidate,
    continuation_policy: str,
    signal_cache: dict[int, list],
    context_store: ContextStore | None,
    horizon_gameweeks: int,
) -> tuple[float, float, float, float, float, int]:
    gameweek = int(row["gameweek"])
    squad_ids = [str(player_id) for player_id in row["pre_squad_ids"]]
    purchase_prices = {str(player_id): int(price) for player_id, price in row["pre_purchase_prices"].items()}
    bank_tenths = int(row["pre_bank_tenths"])
    free_transfers = int(row["pre_free_transfers"])
    ending_gameweek = min(gameweek + horizon_gameweeks - 1, max(current.snapshots_by_gw))
    starting_squad_value = sum(
        current.snapshot(player_id, gameweek).price
        for player_id in squad_ids
        if current.snapshot(player_id, gameweek) is not None
    )
    result = simulate_season(
        current,
        previous,
        # The first action is forced below; afterward use the behavior policy
        # that generated this state. A hold-only continuation made every
        # action compete in an unrealistically frozen future and especially
        # distorted chip timing.
        policy=continuation_policy,
        initial_squad_ids=squad_ids,
        initial_purchase_prices=purchase_prices,
        initial_bank_tenths=bank_tenths,
        initial_free_transfers=free_transfers,
        initial_chips_available=row.get("pre_chips_available", list(CHIP_KINDS)),
        start_gameweek=gameweek,
        end_gameweek=ending_gameweek,
        signal_cache=signal_cache,
        context_store=context_store,
        forced_first_bundle=list(candidate.transfer_bundle),
        forced_first_chip=candidate.chip,
        # Keep counterfactual continuation computationally bounded. The
        # candidate action itself is still fully evaluated; only future
        # behavior uses the same shallow beam as the trajectory generator.
        max_transfer_depth=1,
        transfer_beam_width=3,
        transfer_candidate_width=6,
    )
    # A small, fixed financial term keeps the target strategic without letting
    # price movement overwhelm realized FPL points. The coefficient is held
    # constant across seasons and is not tuned on the holdout.
    value_change = result.final_squad_value - starting_squad_value
    realized_value = result.net_points + 0.15 * value_change + 0.05 * result.final_bank
    return (
        float(realized_value),
        float(result.net_points),
        float(result.hit_points),
        float(result.final_bank),
        float(result.final_squad_value),
        int(result.gameweeks),
    )


def collect_counterfactual_examples(
    current: SeasonData,
    previous: SeasonData | None,
    scenario: str,
    behavior_policy: str,
    trajectory_log: Iterable[dict[str, Any]],
    signal_cache: dict[int, list],
    context_store: ContextStore | None = None,
    horizon_gameweeks: int = 3,
    candidate_width: int = 12,
    max_states: int | None = None,
    continuation_policy: str = "hold",
) -> list[CounterfactualExample]:
    """Turn a legal simulator trajectory into action-value examples."""

    examples: list[CounterfactualExample] = []
    eligible_rows = [row for row in trajectory_log if int(row["gameweek"]) >= 2]
    if max_states is not None and max_states < 1:
        raise ValueError("max_states must be positive when supplied")
    if max_states is not None and len(eligible_rows) > max_states:
        # Uniformly sample the season rather than taking only GW2-GW12. Chip
        # timing and transfer scarcity are late-season phenomena, so temporal
        # coverage is part of action-label quality.
        positions = (
            [
                round(index * (len(eligible_rows) - 1) / (max_states - 1))
                for index in range(max_states)
            ]
            if max_states > 1
            else [0]
        )
        eligible_rows = [eligible_rows[position] for position in positions]

    for row in eligible_rows:
        gameweek = int(row["gameweek"])
        squad_ids = [str(player_id) for player_id in row["pre_squad_ids"]]
        purchase_prices = {str(player_id): int(price) for player_id, price in row["pre_purchase_prices"].items()}
        signals = signal_cache[gameweek]
        chips_available = row.get("pre_chips_available", list(CHIP_KINDS))
        candidates = build_neural_action_candidates(
            current,
            gameweek,
            squad_ids,
            purchase_prices,
            int(row["pre_bank_tenths"]),
            int(row["pre_free_transfers"]),
            signals,
            candidate_width=candidate_width,
            chips_available=chips_available,
        )
        state = policy_state_from_runtime(
            current,
            gameweek,
            squad_ids,
            int(row["pre_bank_tenths"]),
            int(row["pre_free_transfers"]),
            chips_available=chips_available,
        )
        for candidate in candidates:
            (
                realized_value,
                future_net_points,
                hit_points,
                ending_bank,
                ending_squad_value,
                horizon,
            ) = _counterfactual_value(
                current,
                previous,
                row,
                candidate,
                continuation_policy,
                signal_cache,
                context_store,
                horizon_gameweeks,
            )
            examples.append(
                CounterfactualExample(
                    season=current.season,
                    scenario=scenario,
                    behavior_policy=behavior_policy,
                    gameweek=gameweek,
                    decision_time=row.get("decision_time"),
                    action_id=_action_id(candidate.action),
                    state=state,
                    action=candidate.action,
                    realized_value=realized_value,
                    future_net_points=future_net_points,
                    hit_points=hit_points,
                    ending_bank=ending_bank,
                    ending_squad_value=ending_squad_value,
                    horizon_gameweeks=horizon,
                )
            )
    return examples
