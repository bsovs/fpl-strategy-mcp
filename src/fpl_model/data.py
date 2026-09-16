"""CSV loading and validation for match-level FPL model inputs."""

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
    is_home: int
    goals: int


def _parse_nonnegative_int(value: str, field: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: {field} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"row {row_number}: {field} must be non-negative")
    return parsed


def load_matches(path: str | Path) -> list[Match]:
    """Load and validate one match per CSV row.

    Expected columns are match_id, gameweek, date, home_team, away_team,
    home_goals, and away_goals. The loader intentionally rejects duplicate
    match ids and same-team fixtures so downstream grain errors are visible.
    """

    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"{source}: missing required columns: {sorted(missing)}")

        matches: list[Match] = []
        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            match_id = (row.get("match_id") or "").strip()
            home_team = (row.get("home_team") or "").strip()
            away_team = (row.get("away_team") or "").strip()
            date = (row.get("date") or "").strip()
            if not match_id or not home_team or not away_team or not date:
                raise ValueError(f"row {row_number}: match_id, date, and team names are required")
            if match_id in seen_ids:
                raise ValueError(f"row {row_number}: duplicate match_id {match_id!r}")
            if home_team == away_team:
                raise ValueError(f"row {row_number}: home_team and away_team must differ")
            gameweek = _parse_nonnegative_int(row.get("gameweek", ""), "gameweek", row_number)
            home_goals = _parse_nonnegative_int(row.get("home_goals", ""), "home_goals", row_number)
            away_goals = _parse_nonnegative_int(row.get("away_goals", ""), "away_goals", row_number)
            seen_ids.add(match_id)
            matches.append(
                Match(
                    match_id=match_id,
                    gameweek=gameweek,
                    date=date,
                    home_team=home_team,
                    away_team=away_team,
                    home_goals=home_goals,
                    away_goals=away_goals,
                )
            )

    if not matches:
        raise ValueError(f"{source}: no match rows found")
    return sorted(matches, key=lambda match: (match.gameweek, match.date, match.match_id))


def to_team_observations(matches: list[Match]) -> list[TeamObservation]:
    """Represent each fixture as two team-level goal observations."""

    observations: list[TeamObservation] = []
    for match in matches:
        observations.extend(
            [
                TeamObservation(
                    match_id=match.match_id,
                    gameweek=match.gameweek,
                    date=match.date,
                    team=match.home_team,
                    opponent=match.away_team,
                    is_home=1,
                    goals=match.home_goals,
                ),
                TeamObservation(
                    match_id=match.match_id,
                    gameweek=match.gameweek,
                    date=match.date,
                    team=match.away_team,
                    opponent=match.home_team,
                    is_home=0,
                    goals=match.away_goals,
                ),
            ]
        )
    return observations
