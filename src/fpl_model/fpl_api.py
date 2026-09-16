"""Adapters for the public official Fantasy Premier League API."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .data import Match


BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"


def fetch_json(url: str, timeout: int = 30) -> Any:
    request = Request(url, headers={"User-Agent": "fpl-regression-local/0.1"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_official_data(output_dir: str | Path, timeout: int = 30) -> dict[str, Any]:
    """Download official bootstrap and fixture JSON plus a provenance manifest."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat()
    paths = {
        "bootstrap": directory / "bootstrap-static.json",
        "fixtures": directory / "fixtures.json",
    }
    payloads = {
        "bootstrap": fetch_json(BOOTSTRAP_URL, timeout=timeout),
        "fixtures": fetch_json(FIXTURES_URL, timeout=timeout),
    }
    for name, path in paths.items():
        path.write_text(json.dumps(payloads[name], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "fetched_at_utc": fetched_at,
        "sources": {
            "bootstrap": {"url": BOOTSTRAP_URL, "path": str(paths["bootstrap"]), "sha256": _sha256(paths["bootstrap"])},
            "fixtures": {"url": FIXTURES_URL, "path": str(paths["fixtures"]), "sha256": _sha256(paths["fixtures"])},
        },
        "bootstrap_counts": {
            "events": len(payloads["bootstrap"].get("events", [])),
            "teams": len(payloads["bootstrap"].get("teams", [])),
            "elements": len(payloads["bootstrap"].get("elements", [])),
        },
        "fixture_counts": {
            "all": len(payloads["fixtures"]),
            "finished": sum(1 for fixture in payloads["fixtures"] if fixture.get("finished")),
        },
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def official_fixtures_to_matches(bootstrap: dict[str, Any], fixtures: list[dict[str, Any]]) -> list[Match]:
    """Convert finished official fixtures into the model's match CSV grain."""

    team_names = {int(team["id"]): str(team["name"]) for team in bootstrap.get("teams", [])}
    matches: list[Match] = []
    for fixture in fixtures:
        if not fixture.get("finished"):
            continue
        required = ("id", "event", "kickoff_time", "team_h", "team_a", "team_h_score", "team_a_score")
        if any(fixture.get(field) is None for field in required):
            continue
        home_id = int(fixture["team_h"])
        away_id = int(fixture["team_a"])
        if home_id not in team_names or away_id not in team_names:
            raise ValueError(f"fixture {fixture['id']}: team id not found in bootstrap data")
        kickoff = str(fixture["kickoff_time"])
        matches.append(
            Match(
                match_id=f"fixture-{int(fixture['id'])}",
                gameweek=int(fixture["event"]),
                date=kickoff[:10],
                home_team=team_names[home_id],
                away_team=team_names[away_id],
                home_goals=int(fixture["team_h_score"]),
                away_goals=int(fixture["team_a_score"]),
            )
        )
    if not matches:
        raise ValueError("no completed fixtures with scores were found")
    return sorted(matches, key=lambda match: (match.gameweek, match.date, match.match_id))


def write_matches_csv(matches: list[Match], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["match_id", "gameweek", "date", "home_team", "away_team", "home_goals", "away_goals"],
        )
        writer.writeheader()
        for match in matches:
            writer.writerow(match.__dict__)
