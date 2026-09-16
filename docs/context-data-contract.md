# Backdated FPL context data contract

News and sentiment can improve the strategic model, but only if the replay
knows when the information became available. The context layer therefore
stores an event publication time, source, entity match, analyzed sentiment,
reliability, and expiry. It never joins an article merely because it mentions
a player who appeared later in the season.

## Required fields

```json
{
  "event_id": "club-2025-08-14-player-7",
  "published_at": "2025-08-14T09:15:00Z",
  "source": "official FPL",
  "title": "Player named in the confirmed starting XI",
  "body": "Optional source text or analyst summary",
  "player_id": "7",
  "player_name": "Player",
  "team": "1",
  "event_type": "lineup_confirmed",
  "sentiment": 0.9,
  "reliability": 1.0,
  "expires_at": "2025-08-15T12:00:00Z"
}
```

`published_at`, `source`, and either a player or team entity are the important
audit fields. `sentiment` can be supplied by an external classifier or analyst;
if omitted, the local deterministic lexicon provides a conservative fallback.
The raw text should still be retained so a later sentiment model can be
retrained and audited without changing event timing.

## Event types

Use explicit `event_type` values whenever the provider supplies them:

- `injury`, `availability`, and `suspension` — fitness and eligibility risk;
- `rotation`, `lineup_predicted`, `lineup_confirmed`, and `lineup_benched` —
  expected-minutes and starting-role evidence;
- `set_piece` — penalty, direct free-kick, corner, or dead-ball responsibility;
- `transfer` — arrival, departure, contract, or role-change evidence;
- `role_positive` and `price` — other structured positive-role or price news.

The feature layer keeps set-piece and transfer-role signals separate from
generic sentiment. Official/team reports receive higher default reliability
than social posts, and team-level events are downweighted relative to direct
player matches.

## Leakage rules

For historical gameweek `g`, the cutoff is 90 minutes before the first fixture
of `g`. An event is eligible only when:

1. `published_at <= cutoff`;
2. `expires_at` is absent or `expires_at >= cutoff`;
3. the player/team match is valid at that deadline; and
4. the source record existed in the archived feed before the replay was run.

This matters for confirmed lineups: a lineup published after the FPL deadline
may explain the outcome but cannot be used to choose the squad. The test suite
contains a regression test for an injury published between the deadline and
kickoff.

## Ingestion

Normalize provider exports with:

```sh
PYTHONPATH=src python scripts/ingest_context.py \
  --bootstrap data/fpl_snapshot/bootstrap-static.json \
  --news-file data/context/raw-news.jsonl \
  --social-file data/context/raw-social.jsonl \
  --news-out data/context/news.jsonl \
  --social-out data/context/social.jsonl
```

Then pass the normalized files to either trainer:

```sh
PYTHONPATH=src python scripts/train_action_policy.py \
  --history-root data/vaastav \
  --news-context data/context/news.jsonl \
  --social-context data/context/social.jsonl
```

The current public historical archive does not contain a complete backdated
news/social feed. Until one is supplied, those columns remain zero in the
walk-forward benchmark. Current bootstrap news is suitable for live decisions,
but must not be backfilled into historical seasons.

## Official snapshot archive

The public [Randdalf/fplcache archive](https://github.com/Randdalf/fplcache)
keeps compressed historical `bootstrap-static` snapshots. The adapter in this
repository samples those snapshots, converts the official `news`, status, and
chance-of-playing fields into the contract above, and records the snapshot
time as `observed_at`:

```sh
PYTHONPATH=src python scripts/fetch_fplcache_context.py \
  --start-date 2021-04-01 \
  --end-date 2025-08-01 \
  --sample-hours 6 \
  --out data/context/fplcache-news.jsonl \
  --manifest-out data/context/fplcache-manifest.json
```

Use `--dry-run` first and keep the manifest with the generated JSONL. The
archive is valuable for historical availability and official-news replay, but
it does not provide every press-conference, predicted-lineup, or social signal.
Those sources need separate timestamped feeds and should be joined under the
same deadline and `observed_at` rules.
