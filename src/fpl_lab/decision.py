"""Decision layer for strategic FPL transfer recommendations.

Player-point projections are inputs here, not the final objective.  The
decision layer combines short- and long-horizon projections with expected
minutes, fixture outlook, form, value, role security, uncertainty and risk,
then applies FPL-style budget, position, transfer-cost and team-cap rules.

The module intentionally accepts model outputs as an explicit signal table so
additional models (expected minutes, fixture strength, xPoints, news or
sentiment) can be added without rewriting the transfer logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PlayerState:
    player_id: str
    name: str
    position: str
    team: str
    price: float
    selling_price: float | None = None
    can_buy: bool = True
    # Original purchase price is optional because live callers may not know
    # it. When present, it lets the action model distinguish a normal sale
    # from realizing a loss on a declining asset.
    purchase_price: float | None = None

    @property
    def unrealized_loss(self) -> float:
        if self.purchase_price is None:
            return 0.0
        return max(0.0, float(self.purchase_price) - float(self.price))

    @property
    def sell_value(self) -> float:
        return self.price if self.selling_price is None else self.selling_price


@dataclass(frozen=True)
class PlayerSignal:
    """Aggregated model signals for one player over two decision horizons.

    ``short_expected_points`` and ``long_expected_points`` should be ensemble
    outputs, not a single model's prediction. Other fields are deliberately
    separate so the recommendation can explain *why* a move is attractive.
    Scores are normalized as documented in ``docs/decision-system.md``.
    """

    player_id: str
    short_expected_points: float
    long_expected_points: float
    next_expected_points: float = 0.0
    short_fixture_delta: float = 0.0
    long_fixture_delta: float = 0.0
    short_minutes_probability: float = 1.0
    long_minutes_probability: float = 1.0
    form_signal: float = 0.0
    value_signal: float = 0.0
    role_security: float = 0.5
    injury_risk: float = 0.0
    rotation_risk: float = 0.0
    price_change_risk: float = 0.0
    uncertainty: float = 0.0
    captain_upside: float = 0.0
    short_price_signal: float = 0.0
    long_price_signal: float = 0.0
    ownership_leverage: float = 0.0
    news_risk: float = 0.0
    news_sentiment: float = 0.0
    social_sentiment: float = 0.0
    set_piece_signal: float = 0.0
    transfer_role_signal: float = 0.0
    context_reliability: float = 0.0
    context_event_count: int = 0
    trigger: str = ""
    note: str = ""


@dataclass(frozen=True)
class DecisionConfig:
    short_horizon_gameweeks: int = 3
    long_horizon_gameweeks: int = 8
    short_weight: float = 0.45
    long_weight: float = 0.55
    hit_cost: float = 4.0
    free_transfers: int = 1
    bank: float = 0.0
    max_players_per_team: int = 3
    min_move_score: float = 0.15
    now_threshold: float = 0.75
    risk_aversion: float = 1.0
    lineup_weight: float = 0.35
    price_weight: float = 0.75
    ownership_weight: float = 0.25
    rank_mode: str = "neutral"


@dataclass(frozen=True)
class TransferRecommendation:
    player_out_id: str
    player_out: str
    player_in_id: str
    player_in: str
    position: str
    price_delta: float
    hit_cost: float
    short_gain: float
    long_gain: float
    combined_score: float
    timing: str
    why: tuple[str, ...]
    risk_flags: tuple[str, ...]
    trigger: str
    unrealized_loss: float = 0.0


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _signal(signal_by_id: dict[str, PlayerSignal], player_id: str) -> PlayerSignal:
    try:
        return signal_by_id[player_id]
    except KeyError as exc:
        raise ValueError(f"missing model signal for player_id={player_id}") from exc


def _player_score(
    signal: PlayerSignal,
    horizon: str,
    risk_aversion: float,
    price_weight: float = 0.75,
    ownership_weight: float = 0.25,
    rank_mode: str = "neutral",
) -> tuple[float, dict[str, float]]:
    if horizon == "short":
        points = signal.short_expected_points
        fixture = signal.short_fixture_delta
        minutes = signal.short_minutes_probability
        price_signal = signal.short_price_signal
    elif horizon == "long":
        points = signal.long_expected_points
        fixture = signal.long_fixture_delta
        minutes = signal.long_minutes_probability
        price_signal = signal.long_price_signal
    else:
        raise ValueError(f"unknown horizon: {horizon}")
    if rank_mode not in {"neutral", "chase", "defend"}:
        raise ValueError(f"unknown rank mode: {rank_mode}")
    ownership_leverage = signal.ownership_leverage if rank_mode == "chase" else 0.0
    if rank_mode == "defend":
        ownership_leverage = -signal.ownership_leverage

    # Signals are on deliberately small, documented scales. The result is a
    # decision score, not a claim that every factor is measured in FPL points.
    components = {
        "expected_points": float(points),
        "fixture": float(fixture),
        "minutes": 1.0 * _clip(minutes, 0.0, 1.0),
        "form": 0.75 * _clip(signal.form_signal, -1.0, 1.0),
        "value": 0.50 * _clip(signal.value_signal, -1.0, 1.0),
        "role_security": 0.75 * _clip(signal.role_security, 0.0, 1.0),
        "captain_upside": 0.50 * _clip(signal.captain_upside, 0.0, 1.0),
        "price_economics": price_weight * _clip(price_signal, -1.0, 1.0),
        "game_theory": ownership_weight * _clip(ownership_leverage, -1.0, 1.0),
        "injury_risk": -1.25 * _clip(signal.injury_risk, 0.0, 1.0) * risk_aversion,
        "news_risk": -1.50 * _clip(signal.news_risk, 0.0, 1.0) * risk_aversion,
        "news_sentiment": 0.20 * _clip(signal.news_sentiment, -1.0, 1.0),
        # Social data is intentionally lower-weight than official/team news.
        "social_sentiment": 0.08 * _clip(signal.social_sentiment, -1.0, 1.0),
        "set_piece_role": 0.35 * _clip(signal.set_piece_signal, -1.0, 1.0),
        "transfer_role": 0.15 * _clip(signal.transfer_role_signal, -1.0, 1.0),
        "rotation_risk": -0.75 * _clip(signal.rotation_risk, 0.0, 1.0) * risk_aversion,
        "price_change_risk": -0.50 * _clip(signal.price_change_risk, 0.0, 1.0) * risk_aversion,
        "uncertainty": -0.50 * _clip(signal.uncertainty, 0.0, 1.0) * risk_aversion,
    }
    return sum(components.values()), components


def _legal_after_transfer(
    current_squad: list[PlayerState],
    player_out: PlayerState,
    player_in: PlayerState,
    config: DecisionConfig,
) -> bool:
    if not player_in.can_buy or player_in.player_id in {p.player_id for p in current_squad}:
        return False
    if player_in.position != player_out.position:
        return False
    if player_in.price > player_out.sell_value + config.bank + 1e-9:
        return False
    team_counts: dict[str, int] = {}
    for player in current_squad:
        if player.player_id != player_out.player_id:
            team_counts[player.team] = team_counts.get(player.team, 0) + 1
    team_counts[player_in.team] = team_counts.get(player_in.team, 0) + 1
    return team_counts[player_in.team] <= config.max_players_per_team


def _why(
    short_components: dict[str, float],
    long_components: dict[str, float],
    signal_in: PlayerSignal,
    signal_out: PlayerSignal,
    config: DecisionConfig,
) -> tuple[str, ...]:
    out_short_components = _player_score(
        signal_out,
        "short",
        config.risk_aversion,
        config.price_weight,
        config.ownership_weight,
        config.rank_mode,
    )[1]
    out_long_components = _player_score(
        signal_out,
        "long",
        config.risk_aversion,
        config.price_weight,
        config.ownership_weight,
        config.rank_mode,
    )[1]
    differences = {
        "short-term projection": short_components["expected_points"] - out_short_components["expected_points"],
        "long-term projection": long_components["expected_points"] - out_long_components["expected_points"],
        "fixture run": (short_components["fixture"] + long_components["fixture"])
        - (out_short_components["fixture"] + out_long_components["fixture"]),
        "availability/role": (short_components["minutes"] + short_components["role_security"])
        - (out_short_components["minutes"] + out_short_components["role_security"]),
        "value/form": (short_components["value"] + short_components["form"])
        - (out_short_components["value"] + out_short_components["form"]),
        "price economics": (short_components["price_economics"] + long_components["price_economics"])
        - (out_short_components["price_economics"] + out_long_components["price_economics"]),
        "rank leverage": (short_components["game_theory"] + long_components["game_theory"])
            - (out_short_components["game_theory"] + out_long_components["game_theory"]),
        "news context": (short_components["news_risk"] + short_components["news_sentiment"])
            - (out_short_components["news_risk"] + out_short_components["news_sentiment"]),
        "set-piece/role context": (short_components["set_piece_role"] + short_components["transfer_role"])
            - (out_short_components["set_piece_role"] + out_short_components["transfer_role"]),
    }
    labels = {
        "short-term projection": "better short-term projection",
        "long-term projection": "better long-term projection",
        "fixture run": "stronger fixture run",
        "availability/role": "better minutes or role security",
        "value/form": "better form or value signal",
        "price economics": "better price or bank-value outlook",
        "rank leverage": "better ownership/rank leverage",
        "news context": "better current news context",
        "set-piece/role context": "better set-piece or transfer-role context",
    }
    positives = sorted(((value, labels[key]) for key, value in differences.items()), reverse=True)
    return tuple(label for value, label in positives if value > 0.05)[:3] or ("no strong factor advantage",)


def recommend_transfers(
    current_squad: Iterable[PlayerState],
    buyable_players: Iterable[PlayerState],
    signals: Iterable[PlayerSignal],
    config: DecisionConfig,
) -> list[TransferRecommendation]:
    """Return legal, scored one-transfer moves ordered by combined utility."""

    squad = list(current_squad)
    buyable = list(buyable_players)
    signal_by_id = {signal.player_id: signal for signal in signals}
    recommendations: list[TransferRecommendation] = []
    transfer_hit = 0.0 if config.free_transfers > 0 else config.hit_cost

    for player_out in squad:
        signal_out = _signal(signal_by_id, player_out.player_id)
        out_short, _ = _player_score(
            signal_out,
            "short",
            config.risk_aversion,
            config.price_weight,
            config.ownership_weight,
            config.rank_mode,
        )
        out_long, _ = _player_score(
            signal_out,
            "long",
            config.risk_aversion,
            config.price_weight,
            config.ownership_weight,
            config.rank_mode,
        )
        for player_in in buyable:
            if not _legal_after_transfer(squad, player_out, player_in, config):
                continue
            signal_in = _signal(signal_by_id, player_in.player_id)
            in_short, in_short_components = _player_score(
                signal_in,
                "short",
                config.risk_aversion,
                config.price_weight,
                config.ownership_weight,
                config.rank_mode,
            )
            in_long, in_long_components = _player_score(
                signal_in,
                "long",
                config.risk_aversion,
                config.price_weight,
                config.ownership_weight,
                config.rank_mode,
            )
            short_gain = in_short - out_short
            long_gain = in_long - out_long
            combined = config.short_weight * short_gain + config.long_weight * long_gain - transfer_hit
            if combined < config.min_move_score:
                continue
            if combined >= config.now_threshold and short_gain > 0:
                timing = "now"
            elif long_gain > 0:
                timing = "watch"
            else:
                timing = "avoid"
            risk_flags = []
            if signal_in.injury_risk >= 0.4:
                risk_flags.append("injury/news risk")
            if signal_in.news_risk >= 0.4:
                risk_flags.append("time-sensitive news risk")
            if signal_in.rotation_risk >= 0.4:
                risk_flags.append("rotation risk")
            if signal_in.uncertainty >= 0.5:
                risk_flags.append("high model uncertainty")
            if player_out.unrealized_loss > 0:
                risk_flags.append(f"realizing {player_out.unrealized_loss:.1f}m loss")
            recommendations.append(
                TransferRecommendation(
                    player_out_id=player_out.player_id,
                    player_out=player_out.name,
                    player_in_id=player_in.player_id,
                    player_in=player_in.name,
                    position=player_in.position,
                    price_delta=round(player_in.price - player_out.sell_value, 2),
                    hit_cost=transfer_hit,
                    short_gain=round(short_gain, 3),
                    long_gain=round(long_gain, 3),
                    combined_score=round(combined, 3),
                    timing=timing,
                    why=tuple(
                        list(_why(in_short_components, in_long_components, signal_in, signal_out, config))
                        + (["accepting a loss for a stronger forward outlook"] if player_out.unrealized_loss > 0 else [])
                    )[:4],
                    risk_flags=tuple(risk_flags),
                    trigger=signal_in.trigger,
                    unrealized_loss=round(player_out.unrealized_loss, 2),
                )
            )
    return sorted(recommendations, key=lambda recommendation: recommendation.combined_score, reverse=True)


def load_decision_input(path: str | Path) -> tuple[list[PlayerState], list[PlayerState], list[PlayerSignal], DecisionConfig]:
    """Load the user-facing JSON contract used by the recommendation script."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    current = [PlayerState(**row) for row in payload.get("current_squad", [])]
    buyable = [PlayerState(**row) for row in payload.get("buyable_players", [])]
    signals = [PlayerSignal(**row) for row in payload.get("signals", [])]
    config = DecisionConfig(
        **{key: value for key, value in payload.get("config", {}).items() if key in DecisionConfig.__dataclass_fields__}
    )
    return current, buyable, signals, config


def recommendation_to_dict(recommendation: TransferRecommendation) -> dict[str, Any]:
    return asdict(recommendation)


def assess_current_squad(
    current_squad: Iterable[PlayerState], recommendations: Iterable[TransferRecommendation]
) -> list[dict[str, Any]]:
    """Make the implicit hold decision explicit for every current player."""

    squad = list(current_squad)
    ranked = sorted(recommendations, key=lambda recommendation: recommendation.combined_score, reverse=True)
    best_by_out: dict[str, TransferRecommendation] = {}
    for recommendation in ranked:
        best_by_out.setdefault(recommendation.player_out_id, recommendation)
    assessments = []
    for player in squad:
        best = best_by_out.get(player.player_id)
        if best is None:
            assessments.append(
                {
                    "player_id": player.player_id,
                    "player": player.name,
                    "action": "hold",
                    "reason": "No legal transfer clears the net-value threshold.",
                }
            )
        else:
            assessments.append(
                {
                    "player_id": player.player_id,
                    "player": player.name,
                    "action": "transfer" if best.timing == "now" else "watch",
                    "best_replacement": best.player_in,
                    "timing": best.timing,
                    "combined_score": best.combined_score,
                    "reason": "; ".join(best.why),
                }
            )
    return assessments
