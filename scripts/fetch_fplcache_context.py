#!/usr/bin/env python3
"""Download sampled point-in-time official FPL bootstrap data from fplcache.

Randdalf/fplcache stores compressed bootstrap-static snapshots.  This adapter
turns selected snapshots into the local context JSONL contract and preserves
both publication time and the time the snapshot was observed.  It is designed
for historical availability replay; it does not invent press-conference,
lineup, or set-piece events that are absent from the official bootstrap.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from dataclasses import replace
import json
import lzma
from pathlib import Path
import re
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from fpl_lab.context import ContextStore, bootstrap_news_events


UTC = timezone.utc
API_ROOT = "https://api.github.com/repos/Randdalf/fplcache"
RAW_ROOT = "https://raw.githubusercontent.com/Randdalf/fplcache/main"


def _get_bytes(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "fpl-strategy-mcp-context/1.0",
        },
    )
    with urlopen(request, timeout=60) as response:
        return response.read()


def _get_json(url: str) -> dict[str, Any]:
    return json.loads(_get_bytes(url).decode("utf-8"))


def _snapshot_time(path: str) -> datetime | None:
    """Parse cache/YYYY/MM/DD/{HH or HHMM}.json.xz paths."""

    parts = path.split("/")
    if len(parts) < 5 or parts[0] != "cache":
        return None
    try:
        year, month, day = (int(parts[1]), int(parts[2]), int(parts[3]))
    except ValueError:
        return None
    stem = parts[-1].split(".", 1)[0]
    digits = re.sub(r"[^0-9]", "", stem)
    if len(digits) >= 4:
        hour, minute = int(digits[:2]), int(digits[2:4])
    elif len(digits) == 2:
        hour, minute = int(digits), 0
    else:
        return None
    try:
        return datetime(year, month, day, hour, minute, tzinfo=UTC)
    except ValueError:
        return None


def _select_paths(
    paths: list[str],
    start: datetime,
    end: datetime,
    sample_hours: float,
    max_files: int,
) -> list[tuple[str, datetime]]:
    candidates = []
    for path in paths:
        if not path.endswith(".json.xz"):
            continue
        timestamp = _snapshot_time(path)
        if timestamp is not None and start <= timestamp <= end:
            candidates.append((path, timestamp))
    candidates.sort(key=lambda item: item[1])
    selected: list[tuple[str, datetime]] = []
    last: datetime | None = None
    spacing = timedelta(hours=max(0.0, sample_hours))
    for path, timestamp in candidates:
        if last is not None and spacing and timestamp - last < spacing:
            continue
        selected.append((path, timestamp))
        last = timestamp
        if max_files > 0 and len(selected) >= max_files:
            break
    return selected


def _parse_date(value: str, *, end: bool = False) -> datetime:
    parsed = date.fromisoformat(value)
    return datetime.combine(parsed, datetime.max.time() if end else datetime.min.time(), tzinfo=UTC)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2021-04-01")
    parser.add_argument("--end-date", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--sample-hours", type=float, default=24.0, help="minimum spacing between downloaded snapshots; use 0 for every snapshot")
    parser.add_argument("--max-files", type=int, default=0, help="optional download cap; 0 means no cap")
    parser.add_argument("--out", default="data/context/fplcache-news.jsonl")
    parser.add_argument("--manifest-out", default="data/context/fplcache-manifest.json")
    parser.add_argument(
        "--snapshot-out",
        default=None,
        help="optional JSONL archive of point-in-time official player fields",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start = _parse_date(args.start_date)
    end = _parse_date(args.end_date, end=True)
    tree = _get_json(f"{API_ROOT}/git/trees/main?recursive=1")
    if tree.get("truncated"):
        raise RuntimeError("fplcache git tree was truncated; narrow the date range or use an authenticated GitHub API")
    paths = [str(item.get("path")) for item in tree.get("tree", []) if item.get("type") == "blob"]
    selected = _select_paths(paths, start, end, args.sample_hours, args.max_files)
    print(f"Selected {len(selected):,} snapshots from {start.isoformat()} to {end.isoformat()}", flush=True)
    if args.dry_run:
        print(json.dumps({"selected": len(selected), "first": selected[0][1].isoformat() if selected else None, "last": selected[-1][1].isoformat() if selected else None}, indent=2))
        return

    events_by_id = {}
    active_roles: dict[str, tuple[str, str]] = {}
    failures: list[dict[str, str]] = []
    snapshot_handle = None
    snapshot_rows = 0
    if args.snapshot_out:
        snapshot_path = Path(args.snapshot_out)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_handle = snapshot_path.open("w", encoding="utf-8")
    for index, (path, snapshot_time) in enumerate(selected, start=1):
        try:
            payload = json.loads(lzma.decompress(_get_bytes(f"{RAW_ROOT}/{path}")).decode("utf-8"))
            if snapshot_handle is not None:
                for player in payload.get("elements", []):
                    snapshot_handle.write(
                        json.dumps(
                            {
                                "observed_at": snapshot_time.isoformat(),
                                "player_id": player.get("id"),
                                "team": player.get("team"),
                                "status": player.get("status"),
                                "chance_of_playing_this_round": player.get("chance_of_playing_this_round"),
                                "chance_of_playing_next_round": player.get("chance_of_playing_next_round"),
                                "ep_this": player.get("ep_this"),
                                "ep_next": player.get("ep_next"),
                                "form": player.get("form"),
                                "points_per_game": player.get("points_per_game"),
                                "selected_by_percent": player.get("selected_by_percent"),
                                "transfers_in_event": player.get("transfers_in_event"),
                                "transfers_out_event": player.get("transfers_out_event"),
                                "value": (
                                    player.get("value")
                                    if player.get("value") is not None
                                    else player.get("now_cost")
                                ),
                                "now_cost": player.get("now_cost"),
                                "news": player.get("news"),
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    snapshot_rows += 1
            snapshot_events = bootstrap_news_events(payload, fetched_at=snapshot_time)
            current_roles: dict[str, str] = {}
            for event in snapshot_events:
                if event.event_type == "set_piece":
                    role_key = str(event.player_id or event.player_name or "")
                    current_roles[role_key] = event.body
                    active = active_roles.get(role_key)
                    if active is not None and active[1] == event.body:
                        # The same official role is still active; keep one
                        # interval instead of one event per API snapshot.
                        continue
                    if active is not None:
                        previous = events_by_id.get(active[0])
                        if previous is not None:
                            events_by_id[active[0]] = replace(previous, expires_at=event.published_at)
                    events_by_id[event.event_id] = event
                    active_roles[role_key] = (event.event_id, event.body)
                    continue
                previous = events_by_id.get(event.event_id)
                if previous is None:
                    events_by_id[event.event_id] = event
                elif previous.observed_at is None or (
                    event.observed_at is not None and event.observed_at < previous.observed_at
                ):
                    # A news_added timestamp identifies the same provider
                    # event across snapshots.  Preserve the first snapshot
                    # that exposed it; keeping the last one would create
                    # artificial look-ahead in earlier gameweeks.
                    events_by_id[event.event_id] = event
            for role_key in list(active_roles):
                if role_key in current_roles:
                    continue
                event_id, _ = active_roles.pop(role_key)
                previous = events_by_id.get(event_id)
                if previous is not None:
                    events_by_id[event_id] = replace(previous, expires_at=snapshot_time)
        except Exception as exc:  # noqa: BLE001 - preserve partial archive progress
            failures.append({"path": path, "error": str(exc)})
        if index % 25 == 0 or index == len(selected):
            print(f"Processed {index:,}/{len(selected):,} snapshots; {len(events_by_id):,} unique events", flush=True)

    if snapshot_handle is not None:
        snapshot_handle.close()

    archive_end = selected[-1][1] + timedelta(hours=36) if selected else datetime.now(UTC)
    for event_id, _ in active_roles.values():
        previous = events_by_id.get(event_id)
        if previous is not None and previous.expires_at is None:
            events_by_id[event_id] = replace(previous, expires_at=archive_end)
    store = ContextStore(events_by_id.values())
    output = Path(args.out)
    store.write_jsonl(output)
    manifest = {
        "source": "https://github.com/Randdalf/fplcache",
        "api_tree": f"{API_ROOT}/git/trees/main?recursive=1",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "sample_hours": args.sample_hours,
        "selected_snapshots": len(selected),
        "download_failures": failures,
        "context": store.summary(),
        "dedupe_rule": "retain earliest observed_at for repeated event_id",
        "set_piece_rule": "retain one interval per player role state and close it when the official role changes or disappears",
        "observed_at_rule": "official bootstrap events are eligible only at or after the cache snapshot that contained them",
        "official_snapshot_rows": snapshot_rows,
    }
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
