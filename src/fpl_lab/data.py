"""Validated match and team-observation inputs for the maintained model."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = {
    "match_id",
    "gameweek",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
}


@dataclass(frozen=True)
class Match:
    match_id: str
    gameweek: int
    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int


@dataclass(frozen=True)
class TeamObservation:
    match_id: str
    gameweek: int
    date: str
    team: str
    opponent: str
    home: int
    goals: int


def _integer(value: str, field: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: {field} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"row {row_number}: {field} must be non-negative")
    return parsed


def load_matches(path: str | Path) -> list[Match]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source}: missing required columns: {sorted(missing)}")
        matches: list[Match] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            match_id = (row.get("match_id") or "").strip()
            home_team = (row.get("home_team") or "").strip()
            away_team = (row.get("away_team") or "").strip()
            date = (row.get("date") or "").strip()
            if not match_id or not date or not home_team or not away_team:
                raise ValueError(f"row {row_number}: match_id, date, and team names are required")
            if match_id in seen:
                raise ValueError(f"row {row_number}: duplicate match_id {match_id!r}")
            if home_team == away_team:
                raise ValueError(f"row {row_number}: a team cannot play itself")
            seen.add(match_id)
            matches.append(
                Match(
                    match_id=match_id,
                    gameweek=_integer(row.get("gameweek", ""), "gameweek", row_number),
                    date=date,
                    home_team=home_team,
                    away_team=away_team,
                    home_goals=_integer(row.get("home_goals", ""), "home_goals", row_number),
                    away_goals=_integer(row.get("away_goals", ""), "away_goals", row_number),
                )
            )
    if not matches:
        raise ValueError(f"{source}: no rows found")
    return sorted(matches, key=lambda match: (match.gameweek, match.date, match.match_id))


def to_team_observations(matches: list[Match]) -> list[TeamObservation]:
    observations: list[TeamObservation] = []
    for match in matches:
        observations.extend(
            [
                TeamObservation(match.match_id, match.gameweek, match.date, match.home_team, match.away_team, 1, match.home_goals),
                TeamObservation(match.match_id, match.gameweek, match.date, match.away_team, match.home_team, 0, match.away_goals),
            ]
        )
    return observations
