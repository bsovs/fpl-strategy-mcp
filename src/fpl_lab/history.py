"""Download and normalize public Vaastav FPL historical gameweek files."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


VAASTAV_RAW = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download_seasons(
    seasons: list[str],
    output_dir: str | Path,
    max_gameweek: int = 38,
    timeout: int = 30,
) -> dict:
    """Fetch GW CSVs for selected seasons and write a source manifest.

    The files are kept as immutable raw inputs. Existing files are reused and
    their hashes are recorded, so reruns do not silently replace a snapshot.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"fetched_at_utc": datetime.now(timezone.utc).isoformat(), "files": []}
    for season in seasons:
        season_dir = root / season / "gws"
        season_dir.mkdir(parents=True, exist_ok=True)
        for gameweek in range(1, max_gameweek + 1):
            relative = f"{season}/gws/gw{gameweek}.csv"
            url = f"{VAASTAV_RAW}/{relative}"
            destination = root / relative
            if destination.exists():
                payload = destination.read_bytes()
            else:
                request = Request(url, headers={"User-Agent": "fpl-regression-local/0.1"})
                with urlopen(request, timeout=timeout) as response:
                    payload = response.read()
                destination.write_bytes(payload)
            manifest["files"].append(
                {
                    "season": season,
                    "gameweek": gameweek,
                    "url": url,
                    "path": str(destination),
                    "sha256": _sha256_bytes(payload),
                    "bytes": len(payload),
                }
            )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
