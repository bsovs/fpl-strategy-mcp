"""Point-in-time official FPL bootstrap snapshot features.

The live bootstrap endpoint is a current-state document.  An archive of those
documents becomes a historical feature source only after every row is joined
to the time the snapshot was observed.  This module keeps that join explicit
and returns neutral defaults when no snapshot existed before a deadline.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from typing import Any, Iterable

from .context import parse_timestamp


def _number(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@dataclass(frozen=True)
class OfficialPlayerSnapshot:
    observed_at: object
    player_id: str
    team: str = ""
    status: str = ""
    chance_of_playing_this_round: float | None = None
    chance_of_playing_next_round: float | None = None
    ep_this: float = 0.0
    ep_next: float = 0.0
    form: float = 0.0
    points_per_game: float = 0.0
    selected_by_percent: float = 0.0
    transfers_in_event: float = 0.0
    transfers_out_event: float = 0.0
    value: float = 0.0
    news_present: float = 0.0

    @property
    def availability_probability(self) -> float:
        chance = self.chance_of_playing_next_round
        if chance is None:
            chance = self.chance_of_playing_this_round
        if chance is not None:
            return max(0.0, min(1.0, float(chance) / 100.0))
        return {"a": 1.0, "d": 0.5, "i": 0.0, "s": 0.0, "u": 0.0}.get(self.status, 0.70)


class OfficialSnapshotStore:
    """Latest archived official player snapshot available at a cutoff."""

    def __init__(self, rows: Iterable[OfficialPlayerSnapshot] = ()):
        grouped: dict[str, list[OfficialPlayerSnapshot]] = {}
        for row in rows:
            grouped.setdefault(str(row.player_id), []).append(row)
        self._rows = {
            player_id: sorted(values, key=lambda row: row.observed_at)
            for player_id, values in grouped.items()
        }
        self._times = {
            player_id: [row.observed_at for row in values]
            for player_id, values in self._rows.items()
        }

    @classmethod
    def from_path(cls, path: str | Path) -> "OfficialSnapshotStore":
        source = Path(path)
        opener = gzip.open if source.suffix == ".gz" else open
        rows: list[OfficialPlayerSnapshot] = []
        with opener(source, "rt", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)
                observed_at = parse_timestamp(record.get("observed_at"))
                player_id = str(record.get("player_id") or record.get("id") or "").strip()
                if observed_at is None or not player_id:
                    raise ValueError(f"official snapshot record {index} is missing observed_at/player_id")
                rows.append(
                    OfficialPlayerSnapshot(
                        observed_at=observed_at,
                        player_id=player_id,
                        team=str(record.get("team") or ""),
                        status=str(record.get("status") or "").lower(),
                        chance_of_playing_this_round=(
                            _number(record["chance_of_playing_this_round"])
                            if record.get("chance_of_playing_this_round") is not None
                            else None
                        ),
                        chance_of_playing_next_round=(
                            _number(record["chance_of_playing_next_round"])
                            if record.get("chance_of_playing_next_round") is not None
                            else None
                        ),
                        ep_this=_number(record.get("ep_this")),
                        ep_next=_number(record.get("ep_next")),
                        form=_number(record.get("form")),
                        points_per_game=_number(record.get("points_per_game")),
                        selected_by_percent=_number(record.get("selected_by_percent")),
                        transfers_in_event=_number(record.get("transfers_in_event")),
                        transfers_out_event=_number(record.get("transfers_out_event")),
                        # Older fplcache exports used ``now_cost`` while the
                        # history rows use ``value``. Accept both so official
                        # price snapshots are not silently flattened to zero.
                        value=_number(
                            record.get("value")
                            if record.get("value") is not None
                            else record.get("now_cost")
                        ),
                        news_present=float(bool(str(record.get("news") or "").strip())),
                    )
                )
        return cls(rows)

    def latest(self, player_id: str, as_of: object) -> OfficialPlayerSnapshot | None:
        cutoff = parse_timestamp(as_of)
        if cutoff is None:
            return None
        key = str(player_id)
        rows = self._rows.get(key)
        if not rows:
            return None
        index = bisect_right(self._times[key], cutoff) - 1
        return rows[index] if index >= 0 else None

    def features_for_player(self, player_id: str, as_of: object) -> dict[str, float]:
        row = self.latest(player_id, as_of)
        if row is None:
            return {
                "official_snapshot_available": 0.0,
                "official_availability_probability": 0.0,
                "official_chance_this_round": 0.0,
                "official_chance_next_round": 0.0,
                "official_ep_this": 0.0,
                "official_ep_next": 0.0,
                "official_form": 0.0,
                "official_points_per_game": 0.0,
                "official_selected_by_percent": 0.0,
                "official_transfers_in_event": 0.0,
                "official_transfers_out_event": 0.0,
                "official_value": 0.0,
                "official_news_present": 0.0,
                "official_snapshot_age_days": 0.0,
            }
        cutoff = parse_timestamp(as_of)
        age_days = max(0.0, (cutoff - row.observed_at).total_seconds() / 86400.0) if cutoff else 0.0
        return {
            "official_snapshot_available": 1.0,
            "official_availability_probability": row.availability_probability,
            "official_chance_this_round": (row.chance_of_playing_this_round or 0.0) / 100.0,
            "official_chance_next_round": (row.chance_of_playing_next_round or 0.0) / 100.0,
            "official_ep_this": row.ep_this,
            "official_ep_next": row.ep_next,
            "official_form": row.form,
            "official_points_per_game": row.points_per_game,
            "official_selected_by_percent": row.selected_by_percent,
            "official_transfers_in_event": row.transfers_in_event,
            "official_transfers_out_event": row.transfers_out_event,
            "official_value": row.value,
            "official_news_present": row.news_present,
            "official_snapshot_age_days": min(age_days / 7.0, 1.0),
        }


__all__ = ["OfficialPlayerSnapshot", "OfficialSnapshotStore"]
