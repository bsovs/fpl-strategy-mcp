"""Official API snapshot adapters exposed by the maintained package."""

from fpl_model.fpl_api import (
    BOOTSTRAP_URL,
    FIXTURES_URL,
    fetch_json,
    official_fixtures_to_matches,
    snapshot_official_data,
    write_matches_csv,
)

__all__ = [
    "BOOTSTRAP_URL",
    "FIXTURES_URL",
    "fetch_json",
    "official_fixtures_to_matches",
    "snapshot_official_data",
    "write_matches_csv",
]
