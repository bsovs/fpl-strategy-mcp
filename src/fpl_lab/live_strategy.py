"""Live current-squad adapter for the selected FPL hybrid strategy.

This module is deliberately separate from the historical SeasonData simulator.
The MCP tool receives a current squad, a buyable pool, and point-in-time
signals directly, then applies the same legal/action-value/rank-leverage logic
used by the synthetic strategy league.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

import numpy as np

from .decision import (
    DecisionConfig,
    PlayerSignal,
    PlayerState,
    TransferRecommendation,
    recommend_transfers,
    recommendation_to_dict,
)
from .league import HYBRID_PROFILES
from .policy import ActionValueEnsemble, ActionValueMLP, PolicyAction, PolicyState
from .simulator import CHIP_KINDS, recommendation_to_policy_action


NEURAL_TYPES = (ActionValueMLP, ActionValueEnsemble)

# FPL permits these eight outfield shapes for a legal starting XI. The live
# adapter uses forecast points to choose both the shape and the players inside
# it; this is deliberately separate from transfer selection because a transfer
# can be good for the squad while still leaving a different player on the bench
# this week.
LIVE_FORMATIONS = (
    ("3-4-3", 3, 4, 3),
    ("3-5-2", 3, 5, 2),
    ("4-3-3", 4, 3, 3),
    ("4-4-2", 4, 4, 2),
    ("4-5-1", 4, 5, 1),
    ("5-2-3", 5, 2, 3),
    ("5-3-2", 5, 3, 2),
    ("5-4-1", 5, 4, 1),
)


def _lineup_position(player: PlayerState) -> str:
    """Normalize the API's GKP label to the simulator's GK label."""

    position = str(player.position).upper()
    return "GK" if position in {"GK", "GKP"} else position


def _live_expected_points(signal: PlayerSignal) -> float:
    """Return the one-GW forecast used for lineup selection."""

    return float(signal.next_expected_points or signal.short_expected_points)


def _lineup_player_row(player: PlayerState, signal: PlayerSignal) -> dict[str, Any]:
    return {
        "player_id": player.player_id,
        "name": player.name,
        "position": "GKP" if _lineup_position(player) == "GK" else _lineup_position(player),
        "team": player.team,
        "price": float(player.price),
        "expected_points": round(_live_expected_points(signal), 3),
        "minutes_probability": round(float(signal.short_minutes_probability), 3),
        "news_risk": round(float(signal.news_risk), 3),
    }


def _compact_lineup_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Keep per-move lineup effects readable without repeating all player rows."""

    return {
        "formation": plan["formation"],
        "starting_ids": [row["player_id"] for row in plan["starting_xi"]],
        "bench_order_ids": [row["player_id"] for row in plan["bench_order"]],
        "captain_id": plan["captain"]["player_id"],
        "vice_captain_id": plan["vice_captain"]["player_id"],
        "projected_total_with_captain": plan["projected_total_with_captain"],
        "bench_boost_increment": plan["bench_boost_increment"],
    }


def _post_transfer_lineup(
    current_squad: list[PlayerState],
    buyable_players: list[PlayerState],
    signals: list[PlayerSignal],
    recommendation: TransferRecommendation,
) -> dict[str, Any]:
    """Re-optimize the XI after a legal transfer candidate is applied."""

    player_out = next(
        (player for player in current_squad if player.player_id == recommendation.player_out_id),
        None,
    )
    player_in = next(
        (player for player in buyable_players if player.player_id == recommendation.player_in_id),
        None,
    )
    if player_out is None or player_in is None:
        raise ValueError("transfer lineup evaluation could not find both players")
    after_transfer = [
        player for player in current_squad if player.player_id != player_out.player_id
    ]
    after_transfer.append(replace(player_in, selling_price=player_in.price))
    return choose_live_lineup(after_transfer, signals)


def _transfer_lineup_effect(
    current_plan: dict[str, Any],
    current_squad: list[PlayerState],
    buyable_players: list[PlayerState],
    signals: list[PlayerSignal],
    recommendation: TransferRecommendation,
) -> dict[str, Any]:
    after = _post_transfer_lineup(current_squad, buyable_players, signals, recommendation)
    return {
        "plan": after,
        "compact_plan": _compact_lineup_plan(after),
        "normal_week_delta": round(
            float(after["projected_total_with_captain"])
            - float(current_plan["projected_total_with_captain"]),
            3,
        ),
        "bench_boost_delta": round(
            float(after["projected_total_with_bench_boost"])
            - float(current_plan["projected_total_with_bench_boost"]),
            3,
        ),
    }


def choose_live_lineup(
    current_squad: Iterable[PlayerState],
    signals: Iterable[PlayerSignal],
) -> dict[str, Any]:
    """Choose the exact legal live XI, bench order, shape and captaincy.

    This is an optimizer over the legal FPL formations, not a four-player
    bench approximation. It uses the point-in-time forecast supplied to the
    MCP; realized minutes and autosubs remain a historical-simulation concern.
    """

    squad = list(current_squad)
    if len(squad) != 15:
        raise ValueError(f"current_squad must contain exactly 15 players; received {len(squad)}")
    signal_by_id = {signal.player_id: signal for signal in signals}
    missing = sorted({player.player_id for player in squad} - set(signal_by_id))
    if missing:
        raise ValueError(f"missing signals for lineup player IDs: {missing[:10]}")
    players_by_position: dict[str, list[PlayerState]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for player in squad:
        position = _lineup_position(player)
        if position not in players_by_position:
            raise ValueError(f"unsupported lineup position {player.position!r} for {player.player_id}")
        players_by_position[position].append(player)

    def sort_key(player: PlayerState) -> tuple[float, float, float]:
        signal = signal_by_id[player.player_id]
        return (
            _live_expected_points(signal),
            float(signal.short_minutes_probability),
            float(signal.captain_upside),
        )

    best: dict[str, Any] | None = None
    for formation, defender_count, midfielder_count, forward_count in LIVE_FORMATIONS:
        counts = {
            "GK": 1,
            "DEF": defender_count,
            "MID": midfielder_count,
            "FWD": forward_count,
        }
        if any(len(players_by_position[position]) < count for position, count in counts.items()):
            continue
        starting_players: list[PlayerState] = []
        for position, count in counts.items():
            starting_players.extend(sorted(players_by_position[position], key=sort_key, reverse=True)[:count])
        starting_ids = {player.player_id for player in starting_players}
        bench_players = [player for player in squad if player.player_id not in starting_ids]
        bench_goalkeepers = [player for player in bench_players if _lineup_position(player) == "GK"]
        bench_outfield = [player for player in bench_players if _lineup_position(player) != "GK"]
        bench_order = bench_goalkeepers[:1] + sorted(bench_outfield, key=sort_key, reverse=True)
        starting_sorted = sorted(starting_players, key=sort_key, reverse=True)
        captain = starting_sorted[0]
        vice_captain = starting_sorted[1]
        base_points = sum(_live_expected_points(signal_by_id[player.player_id]) for player in starting_players)
        captain_bonus = _live_expected_points(signal_by_id[captain.player_id])
        projected_total = base_points + captain_bonus
        bench_points = sum(_live_expected_points(signal_by_id[player.player_id]) for player in bench_players)
        candidate = {
            "formation": formation,
            "starting_players": starting_players,
            "bench_players": bench_players,
            "bench_order_players": bench_order,
            "captain_player": captain,
            "vice_captain_player": vice_captain,
            "projected_start_points": base_points,
            "projected_bench_points": bench_points,
            "projected_total": projected_total,
        }
        if best is None or candidate["projected_total"] > best["projected_total"]:
            best = candidate
    if best is None:
        raise ValueError("current_squad cannot produce a legal starting XI")

    starting_players = best["starting_players"]
    bench_players = best["bench_players"]
    bench_order_players = best["bench_order_players"]
    captain = best["captain_player"]
    vice_captain = best["vice_captain_player"]
    bench_boost_increment = float(best["projected_bench_points"])
    return {
        "formation": best["formation"],
        "starting_xi": [
            _lineup_player_row(player, signal_by_id[player.player_id])
            for player in starting_players
        ],
        "bench_order": [
            _lineup_player_row(player, signal_by_id[player.player_id])
            for player in bench_order_players
        ],
        "captain": _lineup_player_row(captain, signal_by_id[captain.player_id]),
        "vice_captain": _lineup_player_row(vice_captain, signal_by_id[vice_captain.player_id]),
        "projected_start_points": round(float(best["projected_start_points"]), 3),
        "projected_bench_points": round(bench_boost_increment, 3),
        "projected_total_with_captain": round(float(best["projected_total"]), 3),
        "projected_total_with_bench_boost": round(float(best["projected_total"]) + bench_boost_increment, 3),
        "bench_boost_increment": round(bench_boost_increment, 3),
        "method": "legal_formation_forecast_optimizer",
        "note": "Only the starting XI scores normally; the four bench players add their forecast points only when Bench Boost is active.",
    }


def _key(recommendation: TransferRecommendation) -> tuple[str, str]:
    return recommendation.player_out_id, recommendation.player_in_id


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -50.0, 50.0))
    return 1.0 / (1.0 + float(np.exp(-value)))


def _model_scores(
    model: ActionValueMLP | ActionValueEnsemble,
    state: PolicyState,
    actions: list[PolicyAction],
    risk_aversion: float,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(model, ActionValueEnsemble):
        means, uncertainty = model.predict_with_uncertainty([state] * len(actions), actions)
    else:
        means = model.predict([state] * len(actions), actions)
        uncertainty = np.zeros(len(actions), dtype=float)
    return means - risk_aversion * uncertainty, uncertainty


def _candidate_union(
    current_squad: list[PlayerState],
    buyable_players: list[PlayerState],
    signals: list[PlayerSignal],
    config: DecisionConfig,
    strategy: str,
) -> list[TransferRecommendation]:
    """Generate both neutral and differentiated candidate support."""

    permissive = replace(config, min_move_score=-10.0, rank_mode="neutral")
    neutral = recommend_transfers(current_squad, buyable_players, signals, permissive)
    if strategy == "hybrid_win":
        chase_config = replace(config, min_move_score=-10.0, rank_mode="chase")
        chase = recommend_transfers(current_squad, buyable_players, signals, chase_config)
    else:
        chase = []
    by_key: dict[tuple[str, str], TransferRecommendation] = {}
    for recommendation in neutral + chase:
        key = _key(recommendation)
        current = by_key.get(key)
        if current is None or recommendation.combined_score > current.combined_score:
            by_key[key] = recommendation
    return sorted(
        by_key.values(),
        key=lambda recommendation: recommendation.combined_score,
        reverse=True,
    )


def _live_chip_bundle(
    current_squad: list[PlayerState],
    buyable_players: list[PlayerState],
    signals: list[PlayerSignal],
    config: DecisionConfig,
    recommendations: list[TransferRecommendation],
    max_depth: int = 2,
) -> tuple[TransferRecommendation, ...]:
    """Build a small sequentially legal wildcard/Free Hit plan.

    The live adapter does not have the historical snapshot object used by the
    simulator, so it recomputes the second move after applying the first move
    to the supplied current squad and bank. The plan is intentionally capped;
    the action model decides whether the chip is worth using, while the
    returned plan remains inspectable and easy to verify.
    """

    selected: list[TransferRecommendation] = []
    live_current = list(current_squad)
    live_buyable = list(buyable_players)
    live_bank = float(config.bank)
    next_recommendations = list(recommendations)
    for depth in range(max_depth):
        used_out = {item.player_out_id for item in selected}
        used_in = {item.player_in_id for item in selected}
        recommendation = next(
            (
                item
                for item in next_recommendations
                if item.player_out_id not in used_out and item.player_in_id not in used_in
            ),
            None,
        )
        if recommendation is None:
            break
        selected.append(recommendation)
        player_out = next(
            (player for player in live_current if player.player_id == recommendation.player_out_id),
            None,
        )
        player_in = next(
            (player for player in live_buyable if player.player_id == recommendation.player_in_id),
            None,
        )
        if player_out is None or player_in is None:
            break
        live_bank += player_out.sell_value - player_in.price
        live_current = [player for player in live_current if player.player_id != player_out.player_id]
        live_current.append(replace(player_in, selling_price=player_in.price))
        live_buyable = [player for player in live_buyable if player.player_id != player_in.player_id]
        if all(player.player_id != player_out.player_id for player in live_buyable):
            live_buyable.append(replace(player_out, can_buy=True))
        if depth + 1 >= max_depth:
            break
        recompute_config = replace(
            config,
            bank=live_bank,
            free_transfers=15,
            min_move_score=-10.0,
            rank_mode="neutral",
        )
        next_recommendations = _candidate_union(
            live_current,
            live_buyable,
            signals,
            recompute_config,
            "hybrid_win",
        )
    return tuple(selected)


def _live_chip_action(
    chip: str,
    current_squad: list[PlayerState],
    buyable_players: list[PlayerState],
    signals: list[PlayerSignal],
    config: DecisionConfig,
    recommendations: list[TransferRecommendation],
) -> tuple[PolicyAction, dict[str, Any]]:
    """Create an action-model candidate for a live chip decision."""

    signal_by_id = {signal.player_id: signal for signal in signals}
    bundle = (
        _live_chip_bundle(current_squad, buyable_players, signals, config, recommendations)
        if chip in {"wildcard", "free_hit"}
        else ()
    )
    if chip == "bench_boost":
        lineup = choose_live_lineup(current_squad, signals)
        bench_ids = {row["player_id"] for row in lineup["bench_order"]}
        short_points = lineup["bench_boost_increment"]
        long_points = sum(
            signal.long_expected_points / 8.0
            for signal in signals
            if signal.player_id in bench_ids
        )
    elif chip == "triple_captain":
        current_signals = [signal_by_id[player.player_id] for player in current_squad]
        captain = max(
            current_signals,
            key=lambda signal: signal.next_expected_points or signal.short_expected_points,
        )
        short_points = captain.next_expected_points or captain.short_expected_points
        long_points = captain.long_expected_points / 8.0
    else:
        short_points = sum(item.short_gain for item in bundle)
        long_points = (
            sum(item.long_gain for item in bundle)
            if chip != "free_hit"
            else 0.0
        )
    short_price = sum(
        signal_by_id[item.player_in_id].short_price_signal
        - signal_by_id[item.player_out_id].short_price_signal
        for item in bundle
    )
    long_price = (
        sum(
            signal_by_id[item.player_in_id].long_price_signal
            - signal_by_id[item.player_out_id].long_price_signal
            for item in bundle
        )
        if chip != "free_hit"
        else 0.0
    )
    action = PolicyAction(
        kind=chip,
        short_points_delta=short_points,
        long_points_delta=long_points,
        short_price_delta=short_price,
        long_price_delta=long_price,
        ownership_leverage_delta=sum(
            signal_by_id[item.player_in_id].ownership_leverage
            - signal_by_id[item.player_out_id].ownership_leverage
            for item in bundle
        ),
        legal=True,
    )
    return action, {
        "chip": chip,
        "bundle": bundle,
        "short_points": float(short_points),
        "long_points": float(long_points),
        "short_price": float(short_price),
        "long_price": float(long_price),
    }


def recommend_live_moves(
    current_squad: Iterable[PlayerState],
    buyable_players: Iterable[PlayerState],
    signals: Iterable[PlayerSignal],
    config: DecisionConfig,
    gameweek: int,
    strategy: str = "hybrid_win",
    my_points: float | None = None,
    leader_points: float | None = None,
    league_size: int = 10,
    model: ActionValueMLP | ActionValueEnsemble | None = None,
    limit: int = 10,
    chips_available: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return legal current-week moves under a hybrid strategy.

    ``hybrid_win`` is the title-seeking challenger. ``hybrid_safe`` and
    ``hybrid_balanced`` remain available for comparisons. If no model is
    supplied, the function falls back to the transparent legal heuristic and
    marks that fact in the result.
    """

    if strategy == "champion":
        # The frozen tournament champion is intentionally a points-first,
        # free-transfer-only policy.  It returns no hit/chip recommendation;
        # learned action variants remain available under their explicit names.
        current = list(current_squad)
        buyable = list(buyable_players)
        signal_rows = list(signals)
        signal_by_id = {signal.player_id: signal for signal in signal_rows}
        if len(current) != 15:
            raise ValueError(f"current_squad must contain exactly 15 players; received {len(current)}")
        lineup_plan = choose_live_lineup(current, signal_rows)
        if int(config.free_transfers) < 1:
            return {
                "strategy": "champion",
                "gameweek": int(gameweek),
                "action": "hold",
                "hold_reason": "The frozen champion does not recommend paid hits; no free transfer is available.",
                "moves": [],
                "state_summary": {
                    "current_squad_size": len(current),
                    "buyable_pool_size": len(buyable),
                    "free_transfers": config.free_transfers,
                    "chips_available": sorted(set(CHIP_KINDS if chips_available is None else chips_available)),
                    "bank": config.bank,
                },
                "lineup_plan": lineup_plan,
                "model_loaded": model is not None,
                "warnings": [
                    "Frozen champion selected by temporal tournament; it does not spend paid hits or chips.",
                    "Learned action policies are challengers until they clear the same paired holdout guardrail.",
                ],
            }
        champion_config = replace(
            config,
            short_weight=0.70,
            long_weight=0.30,
            price_weight=0.0,
            ownership_weight=0.0,
            min_move_score=max(0.15, config.min_move_score),
            rank_mode="neutral",
        )
        recommendations = recommend_transfers(current, buyable, signal_rows, champion_config)
        recommendations = [item for item in recommendations if item.hit_cost <= 0.0]
        lineup_effects = {
            _key(item): _transfer_lineup_effect(
                current_plan=lineup_plan,
                current_squad=current,
                buyable_players=buyable,
                signals=signal_rows,
                recommendation=item,
            )
            for item in recommendations
        }
        recommendations.sort(
            key=lambda item: item.combined_score
            + champion_config.lineup_weight * lineup_effects[_key(item)]["normal_week_delta"],
            reverse=True,
        )
        recommendations = recommendations[: int(limit)]
        rows = []
        for recommendation in recommendations:
            row = recommendation_to_dict(recommendation)
            lineup_effect = lineup_effects[_key(recommendation)]
            row.update(
                {
                    "action": "make_transfer",
                    "decision": "make",
                    "strategy_score": round(
                        float(
                            np.tanh(
                                (
                                    recommendation.combined_score
                                    + champion_config.lineup_weight
                                    * lineup_effect["normal_week_delta"]
                                )
                                / 6.0
                            )
                        ),
                        4,
                    ),
                    "model_advantage": None,
                    "model_uncertainty": None,
                    "why": list(recommendation.why),
                    "normal_week_lineup_delta": lineup_effect["normal_week_delta"],
                    "bench_boost_lineup_delta": lineup_effect["bench_boost_delta"],
                    "post_transfer_lineup": lineup_effect["compact_plan"],
                }
            )
            rows.append(row)
        if not rows:
            return {
                "strategy": "champion",
                "gameweek": int(gameweek),
                "action": "hold",
                "hold_reason": "No free-transfer move cleared the champion's points-first threshold.",
                "moves": [],
                "state_summary": {
                    "current_squad_size": len(current),
                    "buyable_pool_size": len(buyable),
                    "free_transfers": config.free_transfers,
                    "chips_available": sorted(set(CHIP_KINDS if chips_available is None else chips_available)),
                    "bank": config.bank,
                },
                "lineup_plan": lineup_plan,
                "model_loaded": model is not None,
                "warnings": [
                    "Frozen champion selected by temporal tournament; hold is an intentional action.",
                    "The champion is validated for points/risk robustness, not guaranteed to win every mini-league.",
                ],
            }
        return {
            "strategy": "champion",
            "gameweek": int(gameweek),
            "action": "make_transfer",
            "recommended_move": rows[0],
            "moves": rows,
            "state_summary": {
                "current_squad_size": len(current),
                "buyable_pool_size": len(buyable),
                "free_transfers": config.free_transfers,
                "chips_available": sorted(set(CHIP_KINDS if chips_available is None else chips_available)),
                "bank": config.bank,
                "my_points": my_points,
                "leader_points": leader_points,
            },
            "lineup_plan": lineup_plan,
            "model_loaded": model is not None,
            "warnings": [
                "Frozen champion selected by temporal tournament; it uses points, legality, prices, and the free-transfer constraint.",
                "It does not spend paid hits or chips because those overrides were not robustly validated on held-out seasons.",
                "Learned action policies remain available as explicit challengers.",
            ],
        }

    if strategy not in HYBRID_PROFILES:
        raise ValueError(f"strategy must be one of ['champion', *sorted(HYBRID_PROFILES)]")
    if not 1 <= int(gameweek) <= 38:
        raise ValueError("gameweek must be between 1 and 38")
    current = list(current_squad)
    buyable = list(buyable_players)
    signal_rows = list(signals)
    if len(current) != 15:
        raise ValueError(f"current_squad must contain exactly 15 players; received {len(current)}")
    available_chips = set(CHIP_KINDS if chips_available is None else chips_available)
    unknown_chips = available_chips - set(CHIP_KINDS)
    if unknown_chips:
        raise ValueError(f"unknown chips: {sorted(unknown_chips)}")
    signal_by_id = {signal.player_id: signal for signal in signal_rows}
    missing_signals = sorted(
        {player.player_id for player in current + buyable} - set(signal_by_id)
    )
    if missing_signals:
        raise ValueError(f"missing signals for player IDs: {missing_signals[:10]}")
    lineup_plan = choose_live_lineup(current, signal_rows)
    if int(limit) < 1:
        raise ValueError("limit must be positive")
    profile = HYBRID_PROFILES[strategy]
    recommendations = _candidate_union(current, buyable, signal_rows, config, strategy)
    # The live tool intentionally caps model evaluation. The transparent
    # heuristic has already sorted all legal pairs; 120 candidates is enough to
    # preserve diverse same-position options without making a current-week
    # request slow.
    recommendations = recommendations[:120]
    lineup_effects = {
        _key(item): _transfer_lineup_effect(
            current_plan=lineup_plan,
            current_squad=current,
            buyable_players=buyable,
            signals=signal_rows,
            recommendation=item,
        )
        for item in recommendations
    }

    if my_points is None:
        my_points = 0.0
    if leader_points is None:
        leader_points = float(my_points)
    league_size = max(1, int(league_size))
    deficit = max(0.0, float(leader_points) - float(my_points))
    behind = float(np.clip(deficit / max(20.0, 10.0 * league_size), 0.0, 1.0))
    leading = float(
        np.clip(max(0.0, float(my_points) - float(leader_points)) / 30.0, 0.0, 1.0)
    )

    state = PolicyState(
        gameweek=int(gameweek),
        weeks_remaining=max(0, 38 - int(gameweek) + 1),
        bank=float(config.bank),
        free_transfers=int(config.free_transfers),
        squad_value=sum(player.price for player in current),
        # The neural model was trained with neutral rank-mode encoding. The
        # league gap is applied below as a controlled decision overlay instead
        # of sending an out-of-distribution rank-mode value into the network.
        rank_percentile=0.5,
        target_rank_percentile=0.5,
        rank_mode="neutral",
        chip_flexibility=len(available_chips) / len(CHIP_KINDS),
        wildcard_available="wildcard" in available_chips,
        free_hit_available="free_hit" in available_chips,
        bench_boost_available="bench_boost" in available_chips,
        triple_captain_available="triple_captain" in available_chips,
    )
    hold_action = PolicyAction(kind="hold")
    action_rows: list[tuple[PolicyAction, TransferRecommendation | None, dict[str, Any] | None]] = [
        (hold_action, None, None),
        *[
            (recommendation_to_policy_action(recommendation, signal_by_id), recommendation, None)
            for recommendation in recommendations
        ],
    ]
    for chip in CHIP_KINDS:
        if chip not in available_chips:
            continue
        chip_action, chip_meta = _live_chip_action(
            chip,
            current,
            buyable,
            signal_rows,
            config,
            recommendations,
        )
        action_rows.append((chip_action, None, chip_meta))
    if model is not None and not isinstance(model, NEURAL_TYPES):
        raise TypeError("model must be ActionValueMLP, ActionValueEnsemble, or None")
    if model is None:
        model_scores = np.zeros(len(action_rows), dtype=float)
        model_uncertainty = np.zeros(len(action_rows), dtype=float)
    else:
        model_scores, model_uncertainty = _model_scores(
            model,
            state,
            [action for action, _recommendation, _chip_meta in action_rows],
            profile.model_risk_aversion,
        )
    hold_score = float(model_scores[0])
    moves = []
    for index, (action, recommendation, chip_meta) in enumerate(action_rows[1:], start=1):
        if recommendation is None:
            if chip_meta is None or model is None:
                # The transparent fallback can rank transfers, but should not
                # pretend it knows optimal chip timing without the trained
                # action-value model.
                continue
            model_edge = float(model_scores[index] - hold_score)
            uncertainty = float(model_uncertainty[index])
            if model_edge < profile.min_model_advantage:
                continue
            heuristic_score = float(
                np.tanh(
                    (chip_meta["short_points"] + 0.35 * chip_meta["long_points"])
                    / 6.0
                )
            )
            score = (
                profile.model_weight * float(np.tanh(model_edge / 6.0))
                + (1.0 - profile.model_weight) * heuristic_score
                - profile.risk_penalty * uncertainty
            )
            if score < profile.minimum_score:
                continue
            plan = [recommendation_to_dict(item) for item in chip_meta["bundle"]]
            moves.append(
                {
                    "action": "use_chip",
                    "chip": chip_meta["chip"],
                    "chip_plan": plan,
                    "chip_forecast_short": round(chip_meta["short_points"], 3),
                    "chip_forecast_long": round(chip_meta["long_points"], 3),
                    "strategy_score": round(score, 4),
                    "model_advantage": round(model_edge, 4),
                    "model_uncertainty": round(uncertainty, 4),
                    "ownership_leverage_delta": round(action.ownership_leverage_delta, 4),
                    "league_deficit": round(deficit, 2),
                    "behind_factor": round(behind, 4),
                    "combined_score": round(
                        chip_meta["short_points"] + chip_meta["long_points"], 4
                    ),
                    "decision": "use_chip",
                    "why": (
                        "action model prefers this chip to holding",
                        "chip value is evaluated alongside transfer and hold actions",
                    ),
                }
            )
            continue
        model_edge = float(model_scores[index] - hold_score)
        if recommendation.hit_cost > 0.0:
            hit_gate = profile.hit_model_advantage - 1.25 * behind + 0.75 * leading
            if model is None or model_edge < hit_gate:
                continue
        heuristic_score = float(np.tanh(recommendation.combined_score / 6.0))
        model_component = float(np.tanh(model_edge / 6.0)) if model is not None else 0.0
        lineup_effect = lineup_effects[_key(recommendation)]
        lineup_overlay = config.lineup_weight * float(
            np.tanh(lineup_effect["normal_week_delta"] / 6.0)
        )
        in_signal = signal_by_id[recommendation.player_in_id]
        out_signal = signal_by_id[recommendation.player_out_id]
        leverage = float(in_signal.ownership_leverage - out_signal.ownership_leverage)
        game_theory = profile.aggression * behind * leverage
        uncertainty = float(model_uncertainty[index])
        score = (
            profile.model_weight * model_component
            + (1.0 - profile.model_weight) * heuristic_score
            + game_theory
            + lineup_overlay
            - profile.risk_penalty * uncertainty
        )
        if score < profile.minimum_score:
            continue
        if model is not None and model_edge < profile.min_model_advantage:
            # Preserve a clearly strong transparent move even if the learned
            # model is not enthusiastic; otherwise require model confirmation.
            if recommendation.combined_score < 1.0:
                continue
        row = recommendation_to_dict(recommendation)
        row.update(
            {
                "strategy_score": round(score, 4),
                "model_advantage": round(model_edge, 4),
                "model_uncertainty": round(uncertainty, 4),
                "ownership_leverage_delta": round(leverage, 4),
                "league_deficit": round(deficit, 2),
                "behind_factor": round(behind, 4),
                "action": "make_transfer",
                "decision": "make",
                "normal_week_lineup_delta": lineup_effect["normal_week_delta"],
                "bench_boost_lineup_delta": lineup_effect["bench_boost_delta"],
                "post_transfer_lineup": lineup_effect["compact_plan"],
            }
        )
        moves.append(row)
    moves.sort(
        key=lambda row: (
            row["strategy_score"],
            row["model_advantage"],
            row.get("combined_score", row.get("chip_forecast_short", 0.0)),
        ),
        reverse=True,
    )
    warnings = [
        "This is the development-selected hybrid_win challenger, not a proven global-winning policy.",
        "The historical benchmark uses synthetic rival managers; it is not a claim of dominance over real FPL winners.",
        "The live lineup plan uses the legal formation optimizer; actual autosubs still depend on confirmed minutes and late team news.",
    ]
    if model is None:
        warnings.append("No action-value model was loaded; scores are heuristic/rank-leverage fallback scores.")
    if not moves:
        hold_reason = "No legal move cleared the strategy threshold after hit cost, uncertainty, and league-gap checks."
        if config.free_transfers <= 0:
            hold_reason += " You have no free transfer, so a hit would need unusually strong evidence."
        return {
            "strategy": strategy,
            "gameweek": int(gameweek),
            "action": "hold",
            "hold_reason": hold_reason,
            "moves": [],
            "state_summary": {
                "current_squad_size": len(current),
                "buyable_pool_size": len(buyable),
                "free_transfers": config.free_transfers,
                "chips_available": sorted(available_chips),
                "bank": config.bank,
                "my_points": my_points,
                "leader_points": leader_points,
                "league_deficit": deficit,
                "behind_factor": behind,
                "leading_factor": leading,
            },
            "lineup_plan": lineup_plan,
            "model_loaded": model is not None,
            "warnings": warnings,
        }
    return {
        "strategy": strategy,
        "gameweek": int(gameweek),
        "action": "use_chip" if moves[0].get("action") == "use_chip" else "make_transfer",
        "recommended_move": moves[0],
        "moves": moves[: int(limit)],
        "state_summary": {
            "current_squad_size": len(current),
            "buyable_pool_size": len(buyable),
            "free_transfers": config.free_transfers,
            "chips_available": sorted(available_chips),
            "bank": config.bank,
            "my_points": my_points,
            "leader_points": leader_points,
            "league_deficit": deficit,
            "behind_factor": behind,
            "leading_factor": leading,
        },
        "lineup_plan": lineup_plan,
        "model_loaded": model is not None,
        "warnings": warnings,
    }
