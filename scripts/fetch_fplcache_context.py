#!/usr/bin/env python3
"""Download sampled point-in-time official FPL news from fplcache.

Randdalf/fplcache stores compressed bootstrap-static snapshots.  This adapter
turns selected snapshots into the local context JSONL contract and preserves
both publication time and the time the snapshot was observed.  It is designed
for historical availability replay; it does not invent press-conference,
lineup, or set-piece events that are absent from the official bootstrap.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
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
    failures: list[dict[str, str]] = []
    for index, (path, snapshot_time) in enumerate(selected, start=1):
        try:
            payload = json.loads(lzma.decompress(_get_bytes(f"{RAW_ROOT}/{path}")).decode("utf-8"))
            for event in bootstrap_news_events(payload, fetched_at=snapshot_time):
                events_by_id[event.event_id] = event
        except Exception as exc:  # noqa: BLE001 - preserve partial archive progress
            failures.append({"path": path, "error": str(exc)})
        if index % 25 == 0 or index == len(selected):
            print(f"Processed {index:,}/{len(selected):,} snapshots; {len(events_by_id):,} unique events", flush=True)

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
        "observed_at_rule": "official bootstrap events are eligible only at or after the cache snapshot that contained them",
    }
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
