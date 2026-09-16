"""Historical FPL squad and transfer-policy simulator.

This is the executable policy backtest. It models the parts of FPL that
directly affect strategic decisions: a 15-player squad, position quotas, the
three-player club cap, the budget, historical prices, selling-price rules,
free-transfer carryover, four-point hits, formation, captaincy, automatic
substitutions, chips, and a bounded legal transfer-bundle search.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from itertools import combinations
from math import floor
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .context import ContextStore
from .decision import DecisionConfig, PlayerSignal, PlayerState, TransferRecommendation, recommend_transfers
from .policy import ActionValueEnsemble, ActionValueMLP, CocktailConfig, PolicyAction, PolicyState


POSITIONS = ("GK", "DEF", "MID", "FWD")
SQUAD_QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_MIN = {"DEF": 3, "MID": 2, "FWD": 1}
INITIAL_SQUAD_MODES = ("points", "value", "template", "randomized_points")
CHIP_KINDS = ("wildcard", "free_hit", "bench_boost", "triple_captain")


@dataclass(frozen=True)
class PlayerSnapshot:
    player_id: str
    name: str
    position: str
    team: str
    gameweek: int
    price_tenths: int
    points: float
    minutes: float
    starts: float
    selected: float
    transfers_in: float
    transfers_out: float
    fixture_count: int

    @property
    def price(self) -> float:
        return self.price_tenths / 10.0


@dataclass
class SeasonData:
    season: str
    snapshots_by_gw: dict[int, dict[str, PlayerSnapshot]]
    history_by_player: dict[str, list[PlayerSnapshot]]
    schedule: dict[tuple[str, int], int]
    names_to_ids: dict[str, str]
    decision_times: dict[int, datetime] = field(default_factory=dict)

    def snapshot(self, player_id: str, gameweek: int) -> PlayerSnapshot | None:
        current = self.snapshots_by_gw.get(gameweek, {}).get(player_id)
        if current is not None:
            return current
        history = self.history_by_player.get(player_id, [])
        prior = [row for row in history if row.gameweek <= gameweek]
        return prior[-1] if prior else None


@dataclass(frozen=True)
class SeasonRules:
    free_transfer_cap: int
    hit_cost: float = 4.0
    budget_tenths: int = 1000
    max_players_per_team: int = 3


@dataclass
class PolicySeasonResult:
    season: str
    policy: str
    gross_points: float
    hit_points: float
    net_points: float
    transfers: int
    paid_transfers: int
    final_bank: float
    final_squad_value: float
    gameweeks: int
    log: list[dict]
    chip_uses: dict[str, int] = field(default_factory=dict)
    chips_remaining: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PolicyActionCandidate:
    """A model action plus the legal transfer/chip realization it represents."""

    action: PolicyAction
    transfer_bundle: tuple[TransferRecommendation, ...] = ()
    chip: str | None = None


def season_rules(season: str) -> SeasonRules:
    """Return the historical transfer-cap rule used by the season."""

    # FPL raised the banked-transfer cap from two to five for 2024/25.
    return SeasonRules(free_transfer_cap=5 if season >= "2024-25" else 2)


def _position(value: object) -> str:
    return "GK" if str(value) == "GKP" else str(value)


def build_season_data(raw: pd.DataFrame, season: str) -> SeasonData:
    """Aggregate raw fixture rows to one player snapshot per gameweek."""

    frame = raw[raw["season"].astype(str) == season].copy()
    if frame.empty:
        raise ValueError(f"no rows found for season {season}")
    frame["position"] = frame["position"].map(_position)
    frame = frame[frame["position"].isin(POSITIONS)].copy()
    frame["player_id"] = (
        pd.to_numeric(frame["element"], errors="coerce")
        .round()
        .astype("Int64")
        .astype(str)
    )
    frame["team"] = frame["team"].astype(str)
    for column in ("value", "total_points", "minutes", "selected", "transfers_in", "transfers_out"):
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    if "starts" not in frame:
        frame["starts"] = (frame["minutes"] > 0).astype(float)
    frame["starts"] = pd.to_numeric(frame["starts"], errors="coerce").fillna(0.0)

    grouped = (
        frame.groupby(["player_id", "name", "position", "team", "gameweek"], as_index=False, sort=False)
        .agg(
            price_tenths=("value", "first"),
            points=("total_points", "sum"),
            minutes=("minutes", "sum"),
            starts=("starts", "sum"),
            selected=("selected", "first"),
            transfers_in=("transfers_in", "sum"),
            transfers_out=("transfers_out", "sum"),
            fixture_count=("fixture", "nunique"),
        )
    )
    grouped["price_tenths"] = grouped["price_tenths"].round().astype(int)
    snapshots_by_gw: dict[int, dict[str, PlayerSnapshot]] = {}
    history_by_player: dict[str, list[PlayerSnapshot]] = {}
    names_to_ids: dict[str, str] = {}
    for row in grouped.itertuples(index=False):
        snapshot = PlayerSnapshot(
            player_id=str(row.player_id),
            name=str(row.name),
            position=str(row.position),
            team=str(row.team),
            gameweek=int(row.gameweek),
            price_tenths=int(row.price_tenths),
            points=float(row.points),
            minutes=float(row.minutes),
            starts=float(row.starts),
            selected=float(row.selected),
            transfers_in=float(row.transfers_in),
            transfers_out=float(row.transfers_out),
            fixture_count=max(1, int(row.fixture_count)),
        )
        snapshots_by_gw.setdefault(snapshot.gameweek, {})[snapshot.player_id] = snapshot
        history_by_player.setdefault(snapshot.player_id, []).append(snapshot)
        names_to_ids[snapshot.name.strip().lower()] = snapshot.player_id
    for rows in history_by_player.values():
        rows.sort(key=lambda snapshot: snapshot.gameweek)

    # Fixture schedule is known before a GW deadline. Derive it from distinct
    # team/GW/fixture combinations, not from whether a particular player later
    # appeared, so this does not leak player availability.
    schedule = (
        frame.drop_duplicates(["team", "gameweek", "fixture"])
        .groupby(["team", "gameweek"])
        .size()
        .astype(int)
        .to_dict()
    )
    decision_times: dict[int, datetime] = {}
    if "kickoff_time" in frame:
        kickoff_times = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
        for gameweek, kickoff in kickoff_times.groupby(frame["gameweek"]).min().items():
            if pd.isna(kickoff):
                continue
            # FPL deadlines are normally 90 minutes before the first fixture.
            # Using that conservative cutoff prevents a post-deadline report
            # from entering a historical signal cache.
            decision_times[int(gameweek)] = (kickoff.to_pydatetime() - timedelta(minutes=90))
    return SeasonData(season, snapshots_by_gw, history_by_player, schedule, names_to_ids, decision_times)


def _history_for_player(
    current: SeasonData,
    previous: SeasonData | None,
    player_id: str,
    name: str,
    gameweek: int,
) -> list[PlayerSnapshot]:
    prior = [row for row in current.history_by_player.get(player_id, []) if row.gameweek < gameweek]
    if prior:
        return prior
    if previous is None:
        return []
    previous_id = previous.names_to_ids.get(name.strip().lower(), player_id)
    return list(previous.history_by_player.get(previous_id, []))


def _mean_points(history: list[PlayerSnapshot], window: int) -> float:
    if not history:
        return 2.0
    return float(np.mean([row.points for row in history[-window:]]))


def _mean_minutes_probability(history: list[PlayerSnapshot], window: int) -> float:
    if not history:
        return 0.70
    return float(np.clip(np.mean([min(1.0, row.minutes / 90.0) for row in history[-window:]]), 0.0, 1.0))


def _price_trend(history: list[PlayerSnapshot], window: int) -> float:
    prices = [row.price for row in history[-(window + 1) :]]
    if len(prices) < 2:
        return 0.0
    return float(np.mean(np.diff(prices)))


def build_forecast_signals(
    current: SeasonData,
    previous: SeasonData | None,
    gameweek: int,
    short_horizon: int = 3,
    long_horizon: int = 8,
    extra_player_ids: Iterable[str] = (),
    context_store: ContextStore | None = None,
) -> list[PlayerSignal]:
    """Build point/price/ownership/context signals using only pre-GW data."""

    current_snapshots = current.snapshots_by_gw.get(gameweek, {})
    if not current_snapshots:
        raise ValueError(f"no player snapshots for {current.season} GW{gameweek}")
    snapshots = dict(current_snapshots)
    for player_id in extra_player_ids:
        if player_id not in snapshots:
            prior_snapshot = current.snapshot(player_id, gameweek)
            if prior_snapshot is not None:
                snapshots[player_id] = prior_snapshot
    histories = {
        player_id: _history_for_player(current, previous, player_id, snapshot.name, gameweek)
        for player_id, snapshot in snapshots.items()
    }
    ownership_reference = {
        player_id: (history[-1].selected if history else 0.0)
        for player_id, history in histories.items()
    }
    # Ownership must come from the last completed GW (or the prior season at
    # GW1), never from the current GW snapshot downloaded after the deadline.
    max_selected = max(
        (ownership_reference[player_id] for player_id in current_snapshots if ownership_reference[player_id] > 0),
        default=1.0,
    )
    raw_signals = []
    for player_id, snapshot in snapshots.items():
        history = histories[player_id]
        mean_3 = _mean_points(history, 3)
        mean_5 = _mean_points(history, 5)
        mean_10 = _mean_points(history, 10)
        per_fixture = 0.55 * mean_5 + 0.30 * mean_3 + 0.15 * mean_10
        minutes_probability = _mean_minutes_probability(history, 5)
        trend = _price_trend(history, 5)
        context_features = (
            context_store.features_for_player(
                player_id=player_id,
                player_name=snapshot.name,
                team=snapshot.team,
                as_of=current.decision_times.get(gameweek),
            )
            if context_store is not None
            else None
        )
        if context_features is not None:
            minutes_probability = float(
                np.clip(minutes_probability + context_features.availability_delta, 0.0, 1.0)
            )
            context_projection_multiplier = float(
                np.clip(1.0 + 0.20 * context_features.availability_delta, 0.65, 1.10)
            )
        else:
            context_projection_multiplier = 1.0
        short_fixtures = sum(current.schedule.get((snapshot.team, gw), 0) for gw in range(gameweek, min(38, gameweek + short_horizon - 1) + 1))
        long_fixtures = sum(current.schedule.get((snapshot.team, gw), 0) for gw in range(gameweek, min(38, gameweek + long_horizon - 1) + 1))
        short_points = max(0.0, per_fixture * short_fixtures * context_projection_multiplier)
        long_points = max(0.0, per_fixture * long_fixtures * context_projection_multiplier)
        form = 0.0 if mean_10 == 0 else float(np.clip((mean_3 - mean_10) / max(1.0, mean_10), -1.0, 1.0))
        short_price = float(np.clip(trend * short_horizon / 0.3, -1.0, 1.0))
        long_price = float(np.clip(trend * long_horizon / 0.6, -1.0, 1.0))
        ownership_leverage = float(np.clip(0.5 - ownership_reference[player_id] / max_selected, -1.0, 1.0))
        news_risk = context_features.news_risk if context_features is not None else 0.0
        news_sentiment = context_features.news_sentiment if context_features is not None else 0.0
        social_sentiment = context_features.social_sentiment if context_features is not None else 0.0
        context_reliability = context_features.reliability if context_features is not None else 0.0
        context_event_count = context_features.event_count if context_features is not None else 0
        role_security = float(np.clip(minutes_probability, 0.0, 1.0))
        injury_risk = float(np.clip(1.0 - minutes_probability, 0.0, 1.0))
        rotation_risk = float(np.clip(1.0 - minutes_probability, 0.0, 1.0) * 0.5)
        if context_features is not None:
            role_security = float(
                np.clip(role_security + context_features.role_security_delta, 0.0, 1.0)
            )
            injury_risk = float(np.clip(injury_risk + context_features.news_risk, 0.0, 1.0))
            rotation_risk = float(
                np.clip(rotation_risk + 0.35 * max(0.0, -context_features.role_security_delta), 0.0, 1.0)
            )
            short_price = float(np.clip(short_price + 0.25 * context_features.price_pressure, -1.0, 1.0))
            long_price = float(np.clip(long_price + 0.25 * context_features.price_pressure, -1.0, 1.0))
        note = "Historical pre-GW rolling baseline"
        if context_features is not None and context_features.event_count:
            note = (
                f"{note}; {context_features.event_count} as-of context event(s), "
                f"{context_features.news_count} news/{context_features.social_count} social"
            )
        raw_signals.append(
            {
                "player_id": player_id,
                "short_expected_points": short_points,
                "long_expected_points": long_points,
                "next_expected_points": max(0.0, per_fixture * current.schedule.get((snapshot.team, gameweek), 0)),
                "short_fixture_delta": (short_fixtures - short_horizon) * 0.10,
                "long_fixture_delta": (long_fixtures - long_horizon) * 0.10,
                "short_minutes_probability": minutes_probability,
                "long_minutes_probability": minutes_probability,
                "form_signal": form,
                "value_signal": 0.0,
                "role_security": role_security,
                "injury_risk": injury_risk,
                "rotation_risk": rotation_risk,
                "price_change_risk": float(np.clip(abs(trend) / 0.3, 0.0, 1.0)),
                "uncertainty": float(1.0 / np.sqrt(max(1, len(history)))),
                "captain_upside": float(np.clip(per_fixture / 8.0, 0.0, 1.0)),
                "short_price_signal": short_price,
                "long_price_signal": long_price,
                "ownership_leverage": ownership_leverage,
                "news_risk": news_risk,
                "news_sentiment": news_sentiment,
                "social_sentiment": social_sentiment,
                "context_reliability": context_reliability,
                "context_event_count": context_event_count,
                "trigger": "Reassess after the next deadline",
                "note": note,
            }
        )

    # Value signal is relative to same-position projected output, not raw
    # points, so cheap players are compared with realistic positional peers.
    by_position: dict[str, list[dict]] = {}
    for signal in raw_signals:
        position = snapshots[signal["player_id"]].position
        by_position.setdefault(position, []).append(signal)
    for position, rows in by_position.items():
        ratios = [row["short_expected_points"] / max(0.1, snapshots[row["player_id"]].price) for row in rows]
        median = float(np.median(ratios)) if ratios else 0.0
        for row, ratio in zip(rows, ratios):
            row["value_signal"] = float(np.clip((ratio / max(0.1, median)) - 1.0, -1.0, 1.0))
    return [PlayerSignal(**row) for row in raw_signals]


def build_signal_cache(
    current: SeasonData,
    previous: SeasonData | None,
    short_horizon: int = 3,
    long_horizon: int = 8,
    extra_player_ids: Iterable[str] = (),
    context_store: ContextStore | None = None,
) -> dict[int, list[PlayerSignal]]:
    """Precompute point-in-time signals once for repeated starting squads."""

    # A player bought in one GW can have no row in a later GW (for example
    # after an injury or a data-source omission). Include the complete
    # season-player universe so repeated simulations cannot depend on which
    # opening squad happened to own that player.
    extra = tuple(sorted(set(current.history_by_player).union(extra_player_ids)))
    return {
        gameweek: build_forecast_signals(
            current,
            previous,
            gameweek,
            short_horizon,
            long_horizon,
            extra_player_ids=extra,
            context_store=context_store,
        )
        for gameweek in sorted(current.snapshots_by_gw)
    }


def build_model_signal_cache(
    current: SeasonData,
    forecast_rows: pd.DataFrame,
    short_horizon: int = 3,
    long_horizon: int = 8,
) -> dict[int, list[PlayerSignal]]:
    """Convert point-in-time forecast rows into simulator signals.

    The adapter keeps the forecast layer separate from the action policy. A
    row is expected to contain the held-out player's predicted next-fixture
    points and, when available, predicted price change. Missing players fall
    back to a zero-information signal rather than borrowing a later outcome.
    """

    required = {"gameweek", "element", "extended_selected"}
    missing = required - set(forecast_rows.columns)
    if missing:
        raise ValueError(f"forecast rows missing columns: {sorted(missing)}")
    frame = forecast_rows.copy()
    frame["player_id"] = (
        pd.to_numeric(frame["element"], errors="coerce")
        .round()
        .astype("Int64")
        .astype(str)
    )
    frame["gameweek"] = pd.to_numeric(frame["gameweek"], errors="coerce").astype(int)
    frame["extended_selected"] = pd.to_numeric(frame["extended_selected"], errors="coerce").fillna(0.0)
    if "future_price_change_ridge" not in frame:
        frame["future_price_change_ridge"] = 0.0
    frame["future_price_change_ridge"] = pd.to_numeric(
        frame["future_price_change_ridge"], errors="coerce"
    ).fillna(0.0)
    if frame.duplicated(["gameweek", "player_id"]).any():
        frame = frame.drop_duplicates(["gameweek", "player_id"], keep="last")

    result: dict[int, list[PlayerSignal]] = {}
    for gameweek in sorted(current.snapshots_by_gw):
        rows = frame[frame["gameweek"] == gameweek]
        by_id = {str(row.player_id): row for row in rows.itertuples(index=False)}
        signals: list[PlayerSignal] = []
        # Include the full season universe because a squad can still own a
        # player whose source row is missing in this GW; ``current.snapshot``
        # then supplies the last known price/identity for that player.
        player_ids = set(current.history_by_player).union(current.snapshots_by_gw[gameweek])
        for player_id in sorted(player_ids):
            row = by_id.get(str(player_id))
            if row is None:
                signals.append(PlayerSignal(player_id=player_id, short_expected_points=0.0, long_expected_points=0.0))
                continue
            next_points = max(0.0, float(row.extended_selected))
            price_change = float(row.future_price_change_ridge)
            minutes_share = float(getattr(row, "minutes_share_5", 0.0) or 0.0)
            form_acceleration = float(getattr(row, "form_acceleration", 0.0) or 0.0)
            news_risk = float(getattr(row, "news_risk", 0.0) or 0.0)
            social_sentiment = float(getattr(row, "social_sentiment", 0.0) or 0.0)
            short_fixtures = float(getattr(row, "fixtures_next_3", short_horizon) or short_horizon)
            long_fixtures = float(getattr(row, "fixtures_next_5", long_horizon) or long_horizon)
            signals.append(
                PlayerSignal(
                    player_id=player_id,
                    short_expected_points=next_points * max(1.0, min(float(short_horizon), short_fixtures)),
                    long_expected_points=next_points * max(1.0, min(float(long_horizon), long_fixtures)),
                    next_expected_points=next_points,
                    short_minutes_probability=float(np.clip(minutes_share, 0.0, 1.0)),
                    long_minutes_probability=float(np.clip(minutes_share, 0.0, 1.0)),
                    form_signal=float(np.clip(form_acceleration / 3.0, -1.0, 1.0)),
                    role_security=float(np.clip(minutes_share, 0.0, 1.0)),
                    injury_risk=float(np.clip(1.0 - minutes_share, 0.0, 1.0)),
                    rotation_risk=float(np.clip(0.5 * (1.0 - minutes_share), 0.0, 1.0)),
                    price_change_risk=float(np.clip(abs(price_change) / 2.0, 0.0, 1.0)),
                    uncertainty=float(np.clip(1.0 / max(1.0, np.sqrt(5.0)), 0.0, 1.0)),
                    captain_upside=float(np.clip(next_points / 8.0, 0.0, 1.0)),
                    short_price_signal=float(np.clip(price_change / 2.0, -1.0, 1.0)),
                    long_price_signal=float(np.clip(price_change / 2.0, -1.0, 1.0)),
                    news_risk=float(np.clip(news_risk, 0.0, 1.0)),
                    social_sentiment=float(np.clip(social_sentiment, -1.0, 1.0)),
                    trigger="extended historical forecast",
                    note="forecast layer bridged into the legal simulator; news/social are zero without timestamped context",
                )
            )
        result[gameweek] = signals
    return result


def selling_price_tenths(purchase_price_tenths: int, current_price_tenths: int) -> int:
    """Apply FPL's half-profit, rounded-down selling-price rule."""

    if current_price_tenths <= purchase_price_tenths:
        return current_price_tenths
    return purchase_price_tenths + (current_price_tenths - purchase_price_tenths) // 2


def _formation_valid(ids: Iterable[str], snapshots: dict[str, PlayerSnapshot]) -> bool:
    counts = {position: 0 for position in POSITIONS}
    for player_id in ids:
        counts[snapshots[player_id].position] += 1
    return (
        counts["GK"] == 1
        and 3 <= counts["DEF"] <= 5
        and 2 <= counts["MID"] <= 5
        and 1 <= counts["FWD"] <= 3
        and sum(counts.values()) == 11
    )


def choose_lineup(squad_ids: list[str], snapshots: dict[str, PlayerSnapshot], signals: dict[str, PlayerSignal]) -> tuple[list[str], list[str], str, str]:
    """Choose a legal XI, bench order, captain and vice from pre-GW signals."""

    best_ids: tuple[str, ...] | None = None
    best_score = -float("inf")
    for candidate in combinations(squad_ids, 11):
        if not _formation_valid(candidate, snapshots):
            continue
        score = sum(signals[player_id].next_expected_points or signals[player_id].short_expected_points for player_id in candidate)
        if score > best_score:
            best_score = score
            best_ids = candidate
    if best_ids is None:
        raise ValueError("squad cannot produce a legal starting XI")
    starting = list(best_ids)
    bench_ids = [player_id for player_id in squad_ids if player_id not in best_ids]
    bench_gk = [player_id for player_id in bench_ids if snapshots[player_id].position == "GK"]
    bench_outfield = [player_id for player_id in bench_ids if snapshots[player_id].position != "GK"]
    key = lambda player_id: signals[player_id].next_expected_points or signals[player_id].short_expected_points
    bench = bench_gk[:1] + sorted(bench_outfield, key=key, reverse=True)
    ordered = sorted(starting, key=key, reverse=True)
    return starting, bench, ordered[0], ordered[1]


def score_gameweek(
    squad_ids: list[str],
    starting: list[str],
    bench: list[str],
    captain: str,
    vice: str,
    snapshots: dict[str, PlayerSnapshot],
    bench_boost: bool = False,
    triple_captain: bool = False,
) -> tuple[float, dict]:
    """Score a Gameweek with captaincy, autosubs, and optional chips."""

    active = list(starting)
    used_bench: set[str] = set()
    non_playing = [player_id for player_id in starting if snapshots[player_id].minutes <= 0]
    for player_id in non_playing:
        position = snapshots[player_id].position
        for substitute in bench:
            if substitute in used_bench or snapshots[substitute].minutes <= 0:
                continue
            if position == "GK" and snapshots[substitute].position != "GK":
                continue
            if position != "GK" and snapshots[substitute].position == "GK":
                continue
            proposed = [candidate for candidate in active if candidate != player_id] + [substitute]
            if _formation_valid(proposed, snapshots):
                active.remove(player_id)
                active.append(substitute)
                used_bench.add(substitute)
                break

    captain_played = captain in active and snapshots[captain].minutes > 0
    multiplier_player = captain if captain_played else vice
    multiplier = 1
    if multiplier_player in active and snapshots[multiplier_player].minutes > 0:
        # If the triple-captain does not play, the vice-captain is a normal
        # captain and receives two times, matching the FPL fallback rule.
        multiplier = 3 if triple_captain and captain_played else 2
    scored_ids = list(squad_ids) if bench_boost else list(active)
    base_points = sum(snapshots[player_id].points for player_id in scored_ids)
    captain_points = snapshots[multiplier_player].points * (multiplier - 1) if multiplier > 1 else 0.0
    total = base_points + captain_points
    return total, {
        "starting": starting,
        "final_lineup": active,
        "bench": bench,
        "captain": multiplier_player if multiplier > 1 else None,
        "captain_multiplier": multiplier,
        "bench_boost": bool(bench_boost),
        "triple_captain": bool(triple_captain),
        "points": total,
    }


def _scoring_snapshot(current: SeasonData, player_id: str, gameweek: int) -> PlayerSnapshot:
    """Return a zero-point snapshot when a player has no row in this GW."""

    direct = current.snapshots_by_gw.get(gameweek, {}).get(player_id)
    if direct is not None:
        return direct
    prior = current.snapshot(player_id, gameweek)
    if prior is None:
        raise ValueError(f"no historical snapshot for player {player_id}")
    return PlayerSnapshot(
        player_id=prior.player_id,
        name=prior.name,
        position=prior.position,
        team=prior.team,
        gameweek=gameweek,
        price_tenths=prior.price_tenths,
        points=0.0,
        minutes=0.0,
        starts=0.0,
        selected=prior.selected,
        transfers_in=0.0,
        transfers_out=0.0,
        fixture_count=0,
    )


def _prior_score(current: PlayerSnapshot, previous: SeasonData | None) -> float:
    if previous is None:
        return 1.0
    previous_id = previous.names_to_ids.get(current.name.strip().lower(), current.player_id)
    rows = previous.history_by_player.get(previous_id, [])
    if not rows:
        return 1.0
    return sum(row.points for row in rows)


def _prior_selected(current: PlayerSnapshot, previous: SeasonData | None) -> float:
    if previous is None:
        return 0.0
    previous_id = previous.names_to_ids.get(current.name.strip().lower(), current.player_id)
    rows = previous.history_by_player.get(previous_id, [])
    return rows[-1].selected if rows else 0.0


def _initial_squad_scores(
    players: list[PlayerSnapshot],
    current: SeasonData,
    previous: SeasonData | None,
    mode: str,
    seed: int | None,
) -> np.ndarray:
    if mode not in INITIAL_SQUAD_MODES:
        raise ValueError(f"unknown initial squad mode: {mode}")
    prior_points = np.asarray([_prior_score(player, previous) for player in players], dtype=float)
    prices = np.asarray([max(1.0, player.price) for player in players], dtype=float)
    value_scores = prior_points / prices
    points_rank = prior_points / max(1.0, float(prior_points.max(initial=0.0)))
    value_rank = value_scores / max(1.0, float(value_scores.max(initial=0.0)))
    prior_selected = np.asarray([_prior_selected(player, previous) for player in players], dtype=float)
    ownership = np.log1p(prior_selected)
    ownership_rank = ownership / max(1.0, float(ownership.max(initial=0.0)))

    if mode == "points":
        # The original deterministic opening squad objective.
        return prior_points + 0.25 * prior_points / prices
    if mode == "value":
        # A genuinely different legal start: prioritize points per budget while
        # retaining a small absolute-points term so it does not become a bench
        # full of cheap, low-minute players.
        return 100.0 * value_rank + 20.0 * points_rank
    if mode == "template":
        # Ownership is measured at the end of the previous season only. It is
        # a reproducible template proxy, not current-season information.
        return 70.0 * points_rank + 30.0 * ownership_rank

    rng = np.random.default_rng(0 if seed is None else seed)
    noise_scale = max(1.0, float(prior_points.std()) * 0.20)
    return prior_points + 0.25 * prior_points / prices + rng.normal(0.0, noise_scale, len(players))


def select_initial_squad(
    current: SeasonData,
    previous: SeasonData | None,
    budget_tenths: int = 1000,
    mode: str = "points",
    seed: int | None = None,
) -> list[str]:
    """Select a legal initial 15 from information available before GW1.

    ``mode`` is intentionally a scenario generator, not a tuned test-season
    choice. It allows the benchmark to compare the same transfer policy from
    several plausible openings without using future outcomes.
    """

    snapshots = current.snapshots_by_gw.get(1, {})
    players = list(snapshots.values())
    if not players:
        raise ValueError("no GW1 players available")
    scores = _initial_squad_scores(players, current, previous, mode, seed)
    n = len(players)
    rows = []
    lower = []
    upper = []
    rows.append(np.ones(n))
    lower.append(15)
    upper.append(15)
    for position, quota in SQUAD_QUOTAS.items():
        rows.append(np.asarray([float(player.position == position) for player in players]))
        lower.append(quota)
        upper.append(quota)
    teams = sorted({player.team for player in players})
    for team in teams:
        rows.append(np.asarray([float(player.team == team) for player in players]))
        lower.append(-np.inf)
        upper.append(3)
    rows.append(np.asarray([player.price_tenths for player in players], dtype=float))
    lower.append(-np.inf)
    upper.append(budget_tenths)
    result = milp(
        c=-scores,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 30},
    )
    if not result.success or result.x is None:
        raise ValueError(f"initial squad optimization failed: {result.message}")
    return [players[index].player_id for index, value in enumerate(result.x) if value > 0.5]


def select_initial_squad_from_signals(
    current: SeasonData,
    signals: Iterable[PlayerSignal],
    budget_tenths: int = 1000,
) -> list[str]:
    """Select a legal opening squad from point-in-time forecast signals.

    This mode is useful for separating opening-squad quality from transfer
    policy quality.  Signals must be available before GW1, and the optimizer
    still enforces the ordinary 15-player quotas, budget, and three-player
    club cap.  It is intentionally not used by the legacy points/value modes.
    """

    snapshots = current.snapshots_by_gw.get(1, {})
    players = list(snapshots.values())
    signal_by_id = {signal.player_id: signal for signal in signals}
    if not players:
        raise ValueError("no GW1 players available")
    fallback = PlayerSignal("", 0.0, 0.0)
    scores = np.asarray(
        [
            (
                0.50 * (signal := signal_by_id.get(player.player_id, fallback)).short_expected_points
                + 0.30 * signal.long_expected_points
                + 0.20 * signal.next_expected_points
                + 0.10 * signal.short_price_signal
            )
            for player in players
        ],
        dtype=float,
    )
    n = len(players)
    rows = [np.ones(n)]
    lower = [15]
    upper = [15]
    for position, quota in SQUAD_QUOTAS.items():
        rows.append(np.asarray([float(player.position == position) for player in players]))
        lower.append(quota)
        upper.append(quota)
    teams = sorted({player.team for player in players})
    for team in teams:
        rows.append(np.asarray([float(player.team == team) for player in players]))
        lower.append(-np.inf)
        upper.append(3)
    rows.append(np.asarray([player.price_tenths for player in players], dtype=float))
    lower.append(-np.inf)
    upper.append(budget_tenths)
    result = milp(
        c=-scores,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 30},
    )
    if not result.success or result.x is None:
        raise ValueError(f"forecast initial squad optimization failed: {result.message}")
    return [players[index].player_id for index, value in enumerate(result.x) if value > 0.5]


def validate_initial_squad(
    current: SeasonData,
    squad_ids: Iterable[str],
    budget_tenths: int = 1000,
) -> list[str]:
    """Validate an explicit historical or user-supplied opening squad."""

    ids = list(squad_ids)
    snapshots = current.snapshots_by_gw.get(1, {})
    if len(ids) != 15 or len(set(ids)) != 15:
        raise ValueError("initial squad must contain 15 unique players")
    missing = sorted(set(ids) - set(snapshots))
    if missing:
        raise ValueError(f"initial squad contains players unavailable at GW1: {missing}")
    position_counts = {position: 0 for position in POSITIONS}
    team_counts: dict[str, int] = {}
    total_cost = 0
    for player_id in ids:
        player = snapshots[player_id]
        position_counts[player.position] += 1
        team_counts[player.team] = team_counts.get(player.team, 0) + 1
        total_cost += player.price_tenths
    if position_counts != SQUAD_QUOTAS:
        raise ValueError(f"initial squad position quotas are invalid: {position_counts}")
    if max(team_counts.values(), default=0) > 3:
        raise ValueError("initial squad exceeds the three-player club cap")
    if total_cost > budget_tenths:
        raise ValueError("initial squad exceeds the opening budget")
    return ids


def _states_for_decision(
    current: SeasonData,
    gameweek: int,
    squad_ids: list[str],
    purchase_prices: dict[str, int],
    bank_tenths: int,
) -> tuple[list[PlayerState], list[PlayerState]]:
    snapshots = current.snapshots_by_gw.get(gameweek, {})
    current_players = []
    for player_id in squad_ids:
        snapshot = current.snapshot(player_id, gameweek)
        if snapshot is None:
            continue
        current_players.append(
            PlayerState(
                player_id=player_id,
                name=snapshot.name,
                position=snapshot.position,
                team=snapshot.team,
                price=snapshot.price,
                selling_price=selling_price_tenths(purchase_prices[player_id], snapshot.price_tenths) / 10.0,
            )
        )
    buyable = [
        PlayerState(
            player_id=player_id,
            name=snapshot.name,
            position=snapshot.position,
            team=snapshot.team,
            price=snapshot.price,
        )
        for player_id, snapshot in snapshots.items()
    ]
    return current_players, buyable


def _apply_transfer(
    recommendation,
    squad_ids: list[str],
    purchase_prices: dict[str, int],
    current: SeasonData,
    gameweek: int,
    bank_tenths: int,
) -> tuple[list[str], dict[str, int], int]:
    snapshots = current.snapshots_by_gw[gameweek]
    player_out = recommendation.player_out_id
    player_in = recommendation.player_in_id
    out_snapshot = current.snapshot(player_out, gameweek)
    in_snapshot = snapshots[player_in]
    sell_tenths = selling_price_tenths(purchase_prices[player_out], out_snapshot.price_tenths)
    new_bank = bank_tenths + sell_tenths - in_snapshot.price_tenths
    if new_bank < 0:
        raise ValueError("transfer application created a negative bank")
    new_squad = [player_id for player_id in squad_ids if player_id != player_out] + [player_in]
    new_purchase_prices = {player_id: price for player_id, price in purchase_prices.items() if player_id != player_out}
    new_purchase_prices[player_in] = in_snapshot.price_tenths
    return new_squad, new_purchase_prices, new_bank


def _policy_config(
    policy: str,
    short_horizon: int,
    long_horizon: int,
    free_transfers: int,
    bank_tenths: int,
) -> DecisionConfig:
    common = {
        "short_horizon_gameweeks": short_horizon,
        "long_horizon_gameweeks": long_horizon,
        "free_transfers": free_transfers,
        "bank": bank_tenths / 10.0,
        "min_move_score": -1.0,
    }
    if policy.startswith("points_only"):
        common.update({"short_weight": 0.70, "long_weight": 0.30, "price_weight": 0.0, "ownership_weight": 0.0})
    elif policy == "chase":
        common.update({"rank_mode": "chase"})
    return DecisionConfig(**common)


def _choose_transfer_bundle(
    current: SeasonData,
    gameweek: int,
    squad_ids: list[str],
    purchase_prices: dict[str, int],
    bank_tenths: int,
    free_transfers: int,
    signals: list[PlayerSignal],
    policy: str,
    short_horizon: int,
    long_horizon: int,
    max_transfers: int = 3,
    beam_width: int = 12,
    candidate_width: int = 18,
) -> list:
    """Search a small legal transfer beam, including optional hits."""

    signal_by_id = {signal.player_id: signal for signal in signals}
    base_short_weight = 0.70 if policy.startswith("points_only") else 0.45
    base_long_weight = 0.30 if policy.startswith("points_only") else 0.55
    allow_hits = not policy.endswith("_free_only")
    max_depth = min(max_transfers, max(1, free_transfers + 1) if allow_hits else free_transfers)
    if max_depth == 0:
        return []
    # (squad, purchase prices, bank, recommendations, short gain, long gain, utility)
    beam = [(list(squad_ids), dict(purchase_prices), bank_tenths, [], 0.0, 0.0, 0.0)]
    best = beam[0]
    for _depth in range(1, max_depth + 1):
        expanded = []
        for state_squad, state_purchase, state_bank, selected, short_gain, long_gain, _utility in beam:
            current_states, buyable = _states_for_decision(
                current, gameweek, state_squad, state_purchase, state_bank
            )
            remaining_free = max(0, free_transfers - len(selected))
            config = _policy_config(
                policy,
                short_horizon,
                long_horizon,
                remaining_free,
                state_bank,
            )
            candidates = recommend_transfers(current_states, buyable, signals, config)[:candidate_width]
            previous_out = {recommendation.player_out_id for recommendation in selected}
            previous_in = {recommendation.player_in_id for recommendation in selected}
            for recommendation in candidates:
                if recommendation.player_in_id in previous_out or recommendation.player_out_id in previous_in:
                    continue
                new_squad, new_purchase, new_bank = _apply_transfer(
                    recommendation,
                    state_squad,
                    state_purchase,
                    current,
                    gameweek,
                    state_bank,
                )
                new_short = short_gain + recommendation.short_gain
                new_long = long_gain + recommendation.long_gain
                new_count = len(selected) + 1
                hit = max(0, new_count - free_transfers) * 4.0
                utility = base_short_weight * new_short + base_long_weight * new_long - hit
                expanded.append(
                    (new_squad, new_purchase, new_bank, selected + [recommendation], new_short, new_long, utility)
                )
        if not expanded:
            break
        expanded.sort(key=lambda state: state[-1], reverse=True)
        beam = expanded[:beam_width]
        if beam[0][-1] > best[-1]:
            best = beam[0]
    if best[-1] <= 0.25:
        return []
    return best[3]


def policy_state_from_runtime(
    current: SeasonData,
    gameweek: int,
    squad_ids: Iterable[str],
    bank_tenths: int,
    free_transfers: int,
    chips_available: Iterable[str] | None = None,
) -> PolicyState:
    """Encode the observable simulator state at a transfer deadline."""

    ids = list(squad_ids)
    squad_value = sum(
        current.snapshot(player_id, gameweek).price
        for player_id in ids
        if current.snapshot(player_id, gameweek) is not None
    )
    last_gameweek = max(current.snapshots_by_gw)
    available = set(CHIP_KINDS if chips_available is None else chips_available)
    return PolicyState(
        gameweek=gameweek,
        weeks_remaining=max(0, last_gameweek - gameweek + 1),
        bank=bank_tenths / 10.0,
        free_transfers=free_transfers,
        squad_value=squad_value,
        chip_flexibility=len(available) / len(CHIP_KINDS),
        wildcard_available="wildcard" in available,
        free_hit_available="free_hit" in available,
        bench_boost_available="bench_boost" in available,
        triple_captain_available="triple_captain" in available,
    )


def recommendation_to_policy_action(
    recommendation: TransferRecommendation,
    signals: dict[str, PlayerSignal],
) -> PolicyAction:
    """Convert a scored legal transfer into the neural policy feature type."""

    signal_out = signals[recommendation.player_out_id]
    signal_in = signals[recommendation.player_in_id]
    return PolicyAction(
        kind="transfer",
        player_out_id=recommendation.player_out_id,
        player_in_id=recommendation.player_in_id,
        hit_cost=recommendation.hit_cost,
        short_points_delta=recommendation.short_gain,
        long_points_delta=recommendation.long_gain,
        short_price_delta=signal_in.short_price_signal - signal_out.short_price_signal,
        long_price_delta=signal_in.long_price_signal - signal_out.long_price_signal,
        ownership_leverage_delta=signal_in.ownership_leverage - signal_out.ownership_leverage,
        legal=True,
    )


def _chip_transfer_plan(
    current: SeasonData,
    gameweek: int,
    squad_ids: list[str],
    purchase_prices: dict[str, int],
    bank_tenths: int,
    signals: list[PlayerSignal],
    recommendations: Iterable[TransferRecommendation],
    max_depth: int = 15,
) -> tuple[TransferRecommendation, ...]:
    """Create a sequentially legal wildcard/free-hit plan.

    These chips can replace more than one player.  Reusing the normal transfer
    beam means every step rechecks selling price, bank, team caps, positions,
    and the already-selected in/out set.  The default depth is the full
    15-player squad; callers can lower it for faster counterfactual sweeps.
    """

    # ``recommendations`` seeds the first-step candidate pool indirectly via
    # the same current-state scorer used by ordinary transfers.  The helper
    # recomputes each subsequent step, which is essential after the first
    # sale changes both bank and squad composition.
    del recommendations
    bundle = _choose_transfer_bundle(
        current,
        gameweek,
        squad_ids,
        purchase_prices,
        bank_tenths,
        free_transfers=15,
        signals=signals,
        policy="points_only_free_only",
        short_horizon=3,
        long_horizon=8,
        max_transfers=max_depth,
        beam_width=12,
        candidate_width=18,
    )
    return tuple(bundle)


def _chip_forecast_deltas(
    chip: str,
    squad_ids: list[str],
    current: SeasonData,
    gameweek: int,
    signals: dict[str, PlayerSignal],
) -> tuple[float, float]:
    """Approximate the forecast-side value of a no-transfer chip."""

    snapshots = {
        player_id: _scoring_snapshot(current, player_id, gameweek)
        for player_id in squad_ids
    }
    squad_signals = {
        player_id: signals.get(player_id, PlayerSignal(player_id, 0.0, 0.0))
        for player_id in squad_ids
    }
    starting, bench, captain, _vice = choose_lineup(squad_ids, snapshots, squad_signals)
    expected = {
        player_id: squad_signals[player_id].next_expected_points
        or squad_signals[player_id].short_expected_points
        for player_id in squad_ids
    }
    if chip == "bench_boost":
        return sum(expected[player_id] for player_id in bench), sum(
            squad_signals[player_id].long_expected_points / 8.0 for player_id in bench
        )
    if chip == "triple_captain":
        return expected.get(captain, 0.0), squad_signals[captain].long_expected_points / 8.0
    return 0.0, 0.0


def build_neural_action_candidates(
    current: SeasonData,
    gameweek: int,
    squad_ids: list[str],
    purchase_prices: dict[str, int],
    bank_tenths: int,
    free_transfers: int,
    signals: list[PlayerSignal],
    candidate_width: int = 12,
    chips_available: Iterable[str] | None = None,
    chip_transfer_depth: int = 15,
) -> list[PolicyActionCandidate]:
    """Build legal transfer, hold, and chip actions for the neural policy."""

    current_states, buyable = _states_for_decision(
        current, gameweek, squad_ids, purchase_prices, bank_tenths
    )
    configs = [
        _policy_config("points_only", 3, 8, free_transfers, bank_tenths),
        _policy_config("price_aware", 3, 8, free_transfers, bank_tenths),
        _policy_config("chase", 3, 8, free_transfers, bank_tenths),
    ]
    signal_by_id = {signal.player_id: signal for signal in signals}
    recommendations: dict[tuple[str, str], TransferRecommendation] = {}
    for config in configs:
        for recommendation in recommend_transfers(current_states, buyable, signals, config)[:candidate_width]:
            recommendations.setdefault(
                (recommendation.player_out_id, recommendation.player_in_id), recommendation
            )
    available_chips = set(CHIP_KINDS if chips_available is None else chips_available)
    candidates: list[PolicyActionCandidate] = [PolicyActionCandidate(action=PolicyAction(kind="hold"))]
    for recommendation in sorted(
        recommendations.values(), key=lambda item: (item.player_out_id, item.player_in_id)
    ):
        candidates.append(
            PolicyActionCandidate(
                action=recommendation_to_policy_action(recommendation, signal_by_id),
                transfer_bundle=(recommendation,),
            )
        )
    chip_bundle = _chip_transfer_plan(
        current,
        gameweek,
        squad_ids,
        purchase_prices,
        bank_tenths,
        signals,
        recommendations.values(),
        max_depth=chip_transfer_depth,
    ) if available_chips.intersection({"wildcard", "free_hit"}) else ()
    for chip in CHIP_KINDS:
        if chip not in available_chips:
            continue
        bundle = chip_bundle if chip in {"wildcard", "free_hit"} else ()
        short_delta = sum(recommendation.short_gain for recommendation in bundle)
        # Free Hit transfers revert after the gameweek, so their projected
        # long-horizon squad benefit must not be presented as persistent.
        long_delta = (
            sum(recommendation.long_gain for recommendation in bundle)
            if chip != "free_hit"
            else 0.0
        )
        short_chip, long_chip = _chip_forecast_deltas(
            chip,
            squad_ids,
            current,
            gameweek,
            signal_by_id,
        )
        action = PolicyAction(
            kind=chip,
            short_points_delta=short_delta + short_chip,
            long_points_delta=long_delta + long_chip,
            short_price_delta=sum(
                signal_by_id[recommendation.player_in_id].short_price_signal
                - signal_by_id[recommendation.player_out_id].short_price_signal
                for recommendation in bundle
            ),
            long_price_delta=(
                sum(
                    signal_by_id[recommendation.player_in_id].long_price_signal
                    - signal_by_id[recommendation.player_out_id].long_price_signal
                    for recommendation in bundle
                )
                if chip != "free_hit"
                else 0.0
            ),
            ownership_leverage_delta=sum(
                signal_by_id[recommendation.player_in_id].ownership_leverage
                - signal_by_id[recommendation.player_out_id].ownership_leverage
                for recommendation in bundle
            ),
            legal=True,
        )
        candidates.append(
            PolicyActionCandidate(action=action, transfer_bundle=tuple(bundle), chip=chip)
        )
    return candidates


def _choose_neural_action(
    current: SeasonData,
    gameweek: int,
    squad_ids: list[str],
    purchase_prices: dict[str, int],
    bank_tenths: int,
    free_transfers: int,
    signals: list[PlayerSignal],
    neural_policy: ActionValueMLP | ActionValueEnsemble,
    chips_available: Iterable[str] | None = None,
    chip_transfer_depth: int = 15,
) -> PolicyActionCandidate | None:
    candidates = build_neural_action_candidates(
        current,
        gameweek,
        squad_ids,
        purchase_prices,
        bank_tenths,
        free_transfers,
        signals,
        chips_available=chips_available,
        chip_transfer_depth=chip_transfer_depth,
    )
    state = policy_state_from_runtime(
        current,
        gameweek,
        squad_ids,
        bank_tenths,
        free_transfers,
        chips_available=chips_available,
    )
    actions = [candidate.action for candidate in candidates]
    ranked = neural_policy.rank_actions(state, actions, risk_aversion=0.20)
    if not ranked:
        return None
    hold_score = next(score for action, score in ranked if action.kind == "hold")
    best_action, best_score = ranked[0]
    # The hold action is explicit. A learned transfer must beat it after the
    # ensemble's uncertainty penalty, rather than merely having a positive
    # point forecast in isolation.
    if best_action.kind == "hold" or best_score <= hold_score:
        return None
    return next((candidate for candidate in candidates if candidate.action == best_action), None)


def _choose_neural_transfer(
    current: SeasonData,
    gameweek: int,
    squad_ids: list[str],
    purchase_prices: dict[str, int],
    bank_tenths: int,
    free_transfers: int,
    signals: list[PlayerSignal],
    neural_policy: ActionValueMLP | ActionValueEnsemble,
) -> list[TransferRecommendation]:
    """Backward-compatible transfer-only adapter for league code."""

    candidate = _choose_neural_action(
        current,
        gameweek,
        squad_ids,
        purchase_prices,
        bank_tenths,
        free_transfers,
        signals,
        neural_policy,
        chips_available=(),
    )
    return list(candidate.transfer_bundle) if candidate is not None else []


def _choose_cocktail_action(
    current: SeasonData,
    gameweek: int,
    squad_ids: list[str],
    purchase_prices: dict[str, int],
    bank_tenths: int,
    free_transfers: int,
    signals: list[PlayerSignal],
    neural_policy: ActionValueMLP | ActionValueEnsemble | None,
    cocktail_config: CocktailConfig | None = None,
    chips_available: Iterable[str] | None = None,
    chip_transfer_depth: int = 15,
) -> PolicyActionCandidate | None:
    """Choose an anchored, gated model action.

    This is intentionally more conservative than a raw argmax over the
    neural action model.  A free-transfer heuristic bundle is always the
    fallback.  A model transfer must beat hold by the configured edge and a
    chip has a higher hurdle because it consumes a scarce season resource.
    """

    config = cocktail_config or CocktailConfig()
    anchor_bundle = _choose_transfer_bundle(
        current,
        gameweek,
        squad_ids,
        purchase_prices,
        bank_tenths,
        free_transfers,
        signals,
        config.anchor_policy,
        3,
        8,
        max_transfers=3,
        beam_width=6,
        candidate_width=12,
    )
    signal_by_id = {signal.player_id: signal for signal in signals}
    anchor_utility = (
        float(np.tanh(sum(item.combined_score for item in anchor_bundle) / 6.0))
        if anchor_bundle
        else 0.0
    )
    anchor_candidate = None
    if anchor_bundle:
        anchor_candidate = PolicyActionCandidate(
            action=recommendation_to_policy_action(anchor_bundle[0], signal_by_id),
            transfer_bundle=tuple(anchor_bundle),
        )

    # With no loaded model the transparent anchor still remains usable.  This
    # also makes the policy safe for local dry-runs where the model artifact
    # has not yet been installed.
    if neural_policy is None:
        return anchor_candidate

    available_chips = set(CHIP_KINDS if chips_available is None else chips_available)
    candidates = build_neural_action_candidates(
        current,
        gameweek,
        squad_ids,
        purchase_prices,
        bank_tenths,
        free_transfers,
        signals,
        candidate_width=12,
        chips_available=available_chips,
        chip_transfer_depth=chip_transfer_depth,
    )
    if not candidates:
        return anchor_candidate
    state = policy_state_from_runtime(
        current,
        gameweek,
        squad_ids,
        bank_tenths,
        free_transfers,
        chips_available=available_chips,
    )
    actions = [candidate.action for candidate in candidates]
    if isinstance(neural_policy, ActionValueEnsemble):
        means, uncertainty = neural_policy.predict_with_uncertainty([state] * len(actions), actions)
    else:
        means = neural_policy.predict([state] * len(actions), actions)
        uncertainty = np.zeros(len(actions), dtype=float)
    adjusted = means - config.model_risk_aversion * uncertainty
    hold_index = next(
        (index for index, candidate in enumerate(candidates) if candidate.action.kind == "hold"),
        None,
    )
    if hold_index is None:
        return anchor_candidate
    hold_score = float(adjusted[hold_index])
    scored: list[tuple[float, float, PolicyActionCandidate]] = []
    for index, candidate in enumerate(candidates):
        action = candidate.action
        if action.kind == "hold":
            continue
        if action.kind in CHIP_KINDS:
            if not config.allow_chips or action.kind not in available_chips:
                continue
            if not config.min_chip_gameweek <= gameweek <= config.max_chip_gameweek:
                continue
            gate = config.chip_gate
            heuristic = float(
                np.tanh((action.short_points_delta + 0.35 * action.long_points_delta) / 6.0)
            )
        else:
            if action.hit_cost > 0.0 and not config.allow_hits:
                continue
            gate = config.hit_gate if action.hit_cost > 0.0 else config.transfer_gate
            recommendation = candidate.transfer_bundle[0] if candidate.transfer_bundle else None
            heuristic = (
                float(np.tanh(recommendation.combined_score / 6.0))
                if recommendation is not None
                else 0.0
            )
        edge = float(adjusted[index] - hold_score)
        if edge < gate:
            continue
        score = (
            config.model_weight * float(np.tanh(edge / 6.0))
            + (1.0 - config.model_weight) * heuristic
            - config.risk_penalty * float(uncertainty[index])
        )
        scored.append((score, edge, candidate))
    if not scored:
        return anchor_candidate
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, best_edge, best_candidate = scored[0]
    # Do not let a one-transfer model action displace a materially stronger
    # multi-free-transfer anchor unless the model has a clear edge. Chips are
    # allowed to beat the anchor only through their higher opportunity-cost
    # gate above.
    if best_candidate.action.kind == "transfer" and anchor_candidate is not None:
        if best_score + config.anchor_tolerance < anchor_utility:
            return anchor_candidate
    return best_candidate


def simulate_season(
    current: SeasonData,
    previous: SeasonData | None,
    policy: str,
    short_horizon: int = 3,
    long_horizon: int = 8,
    initial_squad_ids: Iterable[str] | None = None,
    initial_squad_mode: str = "points",
    initial_squad_seed: int | None = None,
    signal_cache: dict[int, list[PlayerSignal]] | None = None,
    context_store: ContextStore | None = None,
    neural_policy: ActionValueMLP | ActionValueEnsemble | None = None,
    start_gameweek: int = 1,
    end_gameweek: int | None = None,
    initial_purchase_prices: dict[str, int] | None = None,
    initial_bank_tenths: int | None = None,
    initial_free_transfers: int = 0,
    initial_chips_available: Iterable[str] | None = None,
    forced_first_bundle: list[TransferRecommendation] | None = None,
    forced_first_chip: str | None = None,
    max_transfer_depth: int = 3,
    transfer_beam_width: int = 12,
    transfer_candidate_width: int = 18,
    cocktail_config: CocktailConfig | None = None,
    chip_transfer_depth: int = 15,
) -> PolicySeasonResult:
    """Simulate a season or counterfactual from a legal pre-deadline state.

    The optional state/forced-action arguments are used by the counterfactual
    trainer. The ordinary benchmark path keeps the original full-season
    behavior when they are omitted.
    """

    if policy not in {"hold", "points_only", "points_only_free_only", "price_aware", "chase", "neural", "cocktail"}:
        raise ValueError(f"unknown policy: {policy}")
    available_gameweeks = sorted(current.snapshots_by_gw)
    if start_gameweek not in current.snapshots_by_gw:
        raise ValueError(f"start_gameweek {start_gameweek} is not available for {current.season}")
    last_gameweek = max(available_gameweeks) if end_gameweek is None else min(end_gameweek, max(available_gameweeks))
    if last_gameweek < start_gameweek:
        raise ValueError("end_gameweek must be at or after start_gameweek")
    if policy == "neural" and neural_policy is None:
        raise ValueError("policy='neural' requires a fitted neural_policy")
    rules = season_rules(current.season)
    if initial_squad_ids is None:
        if start_gameweek != 1:
            raise ValueError("an explicit initial_squad_ids state is required after GW1")
        if initial_squad_mode == "forecast":
            if signal_cache is None or 1 not in signal_cache:
                raise ValueError("initial_squad_mode='forecast' requires GW1 signal_cache")
            squad_ids = select_initial_squad_from_signals(
                current,
                signal_cache[1],
                rules.budget_tenths,
            )
        else:
            squad_ids = select_initial_squad(
                current,
                previous,
                rules.budget_tenths,
                mode=initial_squad_mode,
                seed=initial_squad_seed,
            )
    else:
        squad_ids = list(initial_squad_ids)
        if start_gameweek == 1:
            squad_ids = validate_initial_squad(current, squad_ids, rules.budget_tenths)
        elif len(squad_ids) != 15 or len(set(squad_ids)) != 15:
            raise ValueError("counterfactual squad must contain 15 unique players")
        missing = [player_id for player_id in squad_ids if current.snapshot(player_id, start_gameweek) is None]
        if missing:
            raise ValueError(f"counterfactual squad has no historical snapshot: {missing}")
    if initial_purchase_prices is None:
        if start_gameweek != 1:
            raise ValueError("initial_purchase_prices is required after GW1")
        gw1 = current.snapshots_by_gw[1]
        purchase_prices = {player_id: gw1[player_id].price_tenths for player_id in squad_ids}
    else:
        purchase_prices = {str(player_id): int(price) for player_id, price in initial_purchase_prices.items()}
        if set(purchase_prices) != set(squad_ids):
            raise ValueError("initial_purchase_prices must contain exactly the counterfactual squad")
    if initial_bank_tenths is None:
        if start_gameweek != 1:
            raise ValueError("initial_bank_tenths is required after GW1")
        bank_tenths = rules.budget_tenths - sum(purchase_prices.values())
    else:
        bank_tenths = int(initial_bank_tenths)
    if bank_tenths < 0:
        raise ValueError("initial bank cannot be negative")
    # The initial squad is selected before GW1. The first free transfer is
    # awarded after that deadline, so GW2 starts with one free transfer.
    free_transfers = int(initial_free_transfers)
    if not 0 <= free_transfers <= rules.free_transfer_cap:
        raise ValueError("initial_free_transfers is outside the historical rule cap")
    chips_available = set(CHIP_KINDS if initial_chips_available is None else initial_chips_available)
    unknown_chips = chips_available - set(CHIP_KINDS)
    if unknown_chips:
        raise ValueError(f"unknown chips: {sorted(unknown_chips)}")
    if forced_first_chip is not None and forced_first_chip not in chips_available:
        raise ValueError(f"forced chip {forced_first_chip} is unavailable")
    gross_points = 0.0
    hit_points = 0.0
    transfers = 0
    paid_transfers = 0
    chip_uses: dict[str, int] = {}
    log: list[dict] = []

    for gameweek in available_gameweeks:
        if gameweek < start_gameweek or gameweek > last_gameweek:
            continue
        pre_squad_ids = list(squad_ids)
        pre_purchase_prices = dict(purchase_prices)
        pre_bank_tenths = bank_tenths
        pre_free_transfers = free_transfers
        pre_chips_available = tuple(sorted(chips_available))
        if signal_cache is not None and gameweek in signal_cache:
            signals = signal_cache[gameweek]
        else:
            signals = build_forecast_signals(
                current,
                previous,
                gameweek,
                short_horizon,
                long_horizon,
                extra_player_ids=squad_ids,
                context_store=context_store,
            )
        signal_by_id = {signal.player_id: signal for signal in signals}
        transfer_bundle: list[TransferRecommendation] = []
        chip_action: str | None = None
        if gameweek == start_gameweek and forced_first_bundle is not None:
            transfer_bundle = list(forced_first_bundle)
            chip_action = forced_first_chip
        elif gameweek >= 2 and policy not in {"hold"}:
            if policy == "neural":
                neural_action = _choose_neural_action(
                    current,
                    gameweek,
                    squad_ids,
                    purchase_prices,
                    bank_tenths,
                    free_transfers,
                    signals,
                    neural_policy,
                    chips_available=chips_available,
                    chip_transfer_depth=chip_transfer_depth,
                )
                if neural_action is not None:
                    transfer_bundle = list(neural_action.transfer_bundle)
                    chip_action = neural_action.chip
            elif policy == "cocktail":
                cocktail_action = _choose_cocktail_action(
                    current,
                    gameweek,
                    squad_ids,
                    purchase_prices,
                    bank_tenths,
                    free_transfers,
                    signals,
                    neural_policy,
                    cocktail_config=cocktail_config,
                    chips_available=chips_available,
                    chip_transfer_depth=chip_transfer_depth,
                )
                if cocktail_action is not None:
                    transfer_bundle = list(cocktail_action.transfer_bundle)
                    chip_action = cocktail_action.chip
            else:
                transfer_bundle = _choose_transfer_bundle(
                    current,
                    gameweek,
                    squad_ids,
                    purchase_prices,
                    bank_tenths,
                    free_transfers,
                    signals,
                    policy,
                    short_horizon,
                    long_horizon,
                    max_transfers=max_transfer_depth,
                    beam_width=transfer_beam_width,
                    candidate_width=transfer_candidate_width,
                )
        if chip_action is not None:
            if chip_action not in chips_available:
                raise ValueError(f"chip {chip_action} was selected after being consumed")
            if chip_action not in {"wildcard", "free_hit", "bench_boost", "triple_captain"}:
                raise ValueError(f"unknown chip action: {chip_action}")
            chips_available.remove(chip_action)
            chip_uses[chip_action] = chip_uses.get(chip_action, 0) + 1
        if transfer_bundle:
            for transfer in transfer_bundle:
                squad_ids, purchase_prices, bank_tenths = _apply_transfer(
                    transfer, squad_ids, purchase_prices, current, gameweek, bank_tenths
                )
            transfers += len(transfer_bundle)
            chip_transfer = chip_action in {"wildcard", "free_hit"}
            paid_this_week = 0 if chip_transfer else max(0, len(transfer_bundle) - free_transfers)
            paid_transfers += paid_this_week
            hit_points += paid_this_week * rules.hit_cost

        squad_snapshots = {player_id: _scoring_snapshot(current, player_id, gameweek) for player_id in squad_ids}
        squad_signal_map = {player_id: signal_by_id[player_id] for player_id in squad_snapshots if player_id in signal_by_id}
        for player_id in squad_snapshots:
            squad_signal_map.setdefault(
                player_id,
                PlayerSignal(player_id=player_id, short_expected_points=0.0, long_expected_points=0.0),
            )
        starting, bench, captain, vice = choose_lineup(list(squad_snapshots), squad_snapshots, squad_signal_map)
        week_points, scoring = score_gameweek(
            list(squad_snapshots),
            starting,
            bench,
            captain,
            vice,
            squad_snapshots,
            bench_boost=chip_action == "bench_boost",
            triple_captain=chip_action == "triple_captain",
        )
        gross_points += week_points
        if chip_action == "free_hit":
            # Free Hit changes the current GW squad only. Bank, purchase prices
            # and the original squad return at the next deadline.
            squad_ids = pre_squad_ids
            purchase_prices = pre_purchase_prices
            bank_tenths = pre_bank_tenths
            free_transfers = min(rules.free_transfer_cap, pre_free_transfers + 1)
        elif chip_action == "wildcard":
            # A wildcard gives a clean free-transfer state after its deadline;
            # the exact historical bank-cap rules remain season-specific.
            free_transfers = 1
        elif not transfer_bundle:
            free_transfers = min(rules.free_transfer_cap, free_transfers + 1)
        else:
            free_transfers = min(rules.free_transfer_cap, max(0, free_transfers - len(transfer_bundle)) + 1)
        squad_value = sum(snapshot.price for snapshot in squad_snapshots.values())
        log.append(
            {
                "gameweek": gameweek,
                "decision_time": current.decision_times.get(gameweek).isoformat()
                if current.decision_times.get(gameweek)
                else None,
                "pre_squad_ids": pre_squad_ids,
                "pre_purchase_prices": pre_purchase_prices,
                "pre_bank_tenths": pre_bank_tenths,
                "pre_free_transfers": pre_free_transfers,
                "pre_chips_available": list(pre_chips_available),
                "post_squad_ids": list(squad_ids),
                "transfer_out": [transfer.player_out for transfer in transfer_bundle],
                "transfer_in": [transfer.player_in for transfer in transfer_bundle],
                "transfer_score": sum(transfer.combined_score for transfer in transfer_bundle) if transfer_bundle else None,
                "free_transfers_after": free_transfers,
                "chip": chip_action,
                "chips_available_after": sorted(chips_available),
                "bank": bank_tenths / 10.0,
                "squad_value": squad_value,
                "gross_points": week_points,
                "captain": scoring["captain"],
                "starting": scoring["final_lineup"],
            }
        )

    return PolicySeasonResult(
        season=current.season,
        policy=policy,
        gross_points=gross_points,
        hit_points=hit_points,
        net_points=gross_points - hit_points,
        transfers=transfers,
        paid_transfers=paid_transfers,
        final_bank=bank_tenths / 10.0,
        final_squad_value=sum(
            current.snapshot(player_id, last_gameweek).price
            for player_id in squad_ids
            if current.snapshot(player_id, last_gameweek)
        ),
        gameweeks=len(log),
        log=log,
        chip_uses=chip_uses,
        chips_remaining=tuple(sorted(chips_available)),
    )
