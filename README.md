# FPL Strategy MCP

FPL Strategy MCP is a local, rules-aware Fantasy Premier League decision engine. Give it your current 15-player squad, the players you can buy, prices/selling values, free transfers, chips, and current signals. It returns legal hold/transfer/chip options with the reason, risk, short-term outlook, long-term outlook, and price economics.

It is an action policy, not a promise that one player will score the most points. The shipped champion is deliberately conservative: it uses the validated free-transfer anchor unless a learned action clears the temporal guardrail. See the [research paper draft](docs/research-paper.md) for the evidence and limitations.

## Install

macOS or Linux:

```sh
curl -fsSL https://raw.githubusercontent.com/bsovs/fpl-strategy-mcp/main/install.sh | sh -s -- --clients all
```

Windows PowerShell:

```powershell
$env:FPL_STRATEGY_CLIENTS="all"; irm https://raw.githubusercontent.com/bsovs/fpl-strategy-mcp/main/install.ps1 | iex
```

The installers download the latest release binary, register it with the selected
clients, and run a fast health check. Use `--clients claude`,
`--clients claude-code`, `--clients codex`, or `--clients none` to narrow the
setup. Existing Claude JSON and Codex TOML are backed up before they are
changed. Each GitHub release also publishes SHA-256 checksums. To pin a
version, set `FPL_STRATEGY_VERSION=0.1.7` before running the installer.

## Connect a client

The default command is a stdio MCP server:

```sh
fpl-strategy-mcp
```

The server writes one readiness line to stderr so MCP protocol stdout stays
clean. To inspect the installation later:

```sh
fpl-strategy-mcp status
fpl-strategy-mcp status --json
fpl-strategy-mcp status --deep
```

The default status check is fast and only verifies installed assets and client
configuration. `--deep` additionally loads the bundled model.

To register an already-installed binary:

```sh
fpl-strategy-mcp setup --clients all
```

In Claude Desktop, add the installed command to `claude_desktop_config.json` under `mcpServers`. Use the absolute path to the binary:

```json
{
  "mcpServers": {
    "fpl-strategy": {
      "command": "/absolute/path/to/fpl-strategy-mcp"
    }
  }
}
```

On macOS the file is `~/Library/Application Support/Claude/claude_desktop_config.json`; on Windows it is `%APPDATA%\Claude\claude_desktop_config.json`. Fully restart Claude after editing it. The same stdio command works with Claude Code and Codex; the installer can register both automatically.

For ChatGPT or Claude web, start the optional remote transport:

```sh
FPL_MCP_BEARER_TOKEN="choose-a-long-random-token" \
  fpl-strategy-mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

Expose `http://127.0.0.1:8000/mcp` through an HTTPS tunnel or authenticated reverse proxy, then add that HTTPS MCP URL as a custom connector/remote MCP server. Do not expose an unauthenticated listener. OpenAI’s API can call remote MCP servers through the Responses API; Claude web also expects a reachable remote connector. The default stdio mode remains the safer local option.

For HTTP mode, `http://127.0.0.1:8000/health` returns a JSON readiness report
and is protected by `FPL_MCP_BEARER_TOKEN` when that variable is set.

## MCP tools

The server exposes these tools. `fpl_recommend_moves` is the primary decision
tool; the others make the forecasts, weights, player universe, and evaluation
loop inspectable and tunable.

| Tool | Purpose |
| --- | --- |
| `fpl_recommend_moves` | Return the recommended hold, transfer, or model-backed chip action under FPL legality, prices, short/long forecasts, projected XI/bench effects, uncertainty, news, and league context. `buyable_players` is optional; omit it to load the full official player pool. |
| `fpl_lineup_plan` | Choose the legal formation, starting XI, bench order, captain, and vice-captain. Reports the projected four-player Bench Boost increment; normally only the XI scores. |
| `fpl_search_players` | Search the cached official pool by name, team, position, price, or availability. Useful for inspecting candidates or constructing a smaller request payload. |
| `fpl_forecast_signals` | Inspect each player’s short/long expected points, future price signals, minutes/role, risk, news/social context, ownership leverage, and uncertainty before making a decision. |
| `fpl_score_moves` | Rank legal one-transfer moves under explicit `weight_overrides` such as `short_weight`, `long_weight`, `lineup_weight`, `price_weight`, `ownership_weight`, `risk_aversion`, and `rank_mode`. |
| `fpl_strategy_info` | Return the shipped champion, benchmark summaries, research sources, and limitations. |
| `fpl_strategy_catalog` | Return available strategies, action kinds, default weights/gates, signal components, and the fields that can be tuned. |
| `fpl_backtest_strategy` | Compare candidate weights on supplied point-in-time scenarios, or replay the legal simulator over Vaastav-format historical GW files. |

For a normal recommendation, provide `gameweek` and the exact 15-player
`current_squad`. You can provide `buyable_players` and trained `signals`, or
set `auto_official_signals: true` and let the server load the current official
bootstrap pool and transparent fallback signals. Per-request weight changes go
under `weight_overrides` (also accepted as `weights`); they do not modify the
bundled model or the shipped champion.

For loss-aware decisions, include each owned player’s current `price`,
`selling_price`, and original `purchase_price` when available. The latter is
optional; without it, the recommendation can still score the move but cannot
explain or learn the cost of realizing a loss.

## Input

Call `fpl_recommend_moves` with `gameweek` and `current_squad`, optionally
adding `buyable_players`, `config`, `weight_overrides`, and either point-in-time
`signals` or `auto_official_signals: true`. If `buyable_players` or `signals`
are omitted, the same official fallback is used automatically. Add
`league_context` when rank/leader information should influence risk. Call `fpl_strategy_info` or
`fpl_strategy_catalog` to inspect the served strategy and tuning contract.

`fpl_backtest_strategy` accepts either:

- `scenarios` or `scenarios_path`: point-in-time states with realized
  `action_outcomes` such as `hold` and `transfer:out_id>in_id`; or
- `history_root` and `season`: a full legal replay using Vaastav-format
  `season/gws/gw*.csv` files. Use separate development and held-out seasons and
starting-squad modes when tuning.

The action learner explicitly carries short/long fixture-window deltas,
recent form, value, role security, price-change risk, and any unrealized loss
on the player being sold. That lets it distinguish a tactical
three-gameweek punt from a season-long core hold and learn when a declining
player is worth selling at a loss because the forward upgrade is stronger.

See [`examples/`](examples/) for client configuration and a protocol smoke test. The official FPL bootstrap endpoint is used only when requested: `https://fantasy.premierleague.com/api/bootstrap-static/`.

## Development

```sh
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev,remote]"
python -m unittest discover -s tests -q
```

The model and bootstrap snapshot are bundled under `assets/`. Historical training and evaluation artifacts are documented in the paper rather than required to run the server.

## Train the local historical forecast layer

The research trainer keeps a complete final season untouched. The current
protocol uses 2016/17–2023/24 for development, 2024/25 for model selection,
and 2025/26 as the final test season. Download the raw Vaastav archive into a
local data directory, then run:

```sh
python - <<'PY'
from fpl_lab.history import download_seasons
download_seasons(
    ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
     "2022-23", "2023-24", "2024-25", "2025-26"],
    "data/vaastav",
)
PY

PYTHONPATH=src python scripts/train_player_models.py \
  --history-root data/vaastav \
  --validation-season 2024-25 \
  --evaluation-season 2025-26 \
  --output-dir runs/player-models
```

The trainer builds 200+ numeric point-in-time features plus categorical context:
lagged form and volatility, minutes/start security, ownership and transfer
momentum, price movement, team/opponent and prior-matchup form, schedule shape,
cross-season player history and breakout signals, and optional timestamped
news/social context. It writes the selected player
forecast model, a future-price model, evaluation predictions, metrics, and a
data audit under `runs/player-models/`. News/social columns remain zero unless
an auditable `ContextStore` is supplied; modern articles are never backfilled
into old seasons. These forecast artifacts are research inputs and are not
promoted to the shipped action-policy champion until the legal season
simulator shows a robust strategy-level improvement.

The downloader also fetches one `players_raw.csv` roster snapshot per season.
This restores the missing position/team fields in the oldest Vaastav GW files;
the resulting `metadata_*_imputed` flags are retained in the audit because a
season-level roster snapshot is not a point-in-time transfer history.

To test decision value on the held-out season, bridge the forecast CSV into
the legal simulator:

```sh
PYTHONPATH=src python scripts/backtest_forecast_strategy.py \
  --history-root data/vaastav \
  --forecast-csv runs/player-models/evaluation-predictions.csv \
  --season 2025-26 \
  --previous-season 2024-25
```

This compares the legacy signals and expanded forecasts under the same
rules-aware free-transfer policy. It is a strategy smoke test. For the
walk-forward action-value layer, run:

```sh
PYTHONPATH=src python scripts/train_action_policy.py \
  --history-root data/vaastav \
  --validation-season 2024-25 \
  --evaluation-season 2025-26 \
  --output-dir runs/action-policy \
  --max-states 4 \
  --candidate-width 6 \
  --horizon-gameweeks 3 \
  --training-chip-depth 5 \
  --test-chip-depth 15 \
  --forecast-model neural \
  --label-policy points_only
```

When backdated archives are available, add `--news-context PATH` and/or
`--social-context PATH` to that command.

This generates legal counterfactual action labels, fits the action ensemble,
and evaluates neural actions against the free-transfer anchor over points,
value, template, and randomized opening squads. The forecast bridge is
walk-forward: it fits point forecasts, direct 3/8-gameweek totals, a
next-gameweek price-change model, and a separate expected-minutes model using
only earlier seasons. The current
career-feature baseline produced 1,476 development and 186 validation
examples. After fixing two temporal-grain defects—calendar-window horizon
labels and double-gameweek lag aggregation—the clean untouched 2025/26 replay
scored 2,076.25 points on average across four opening families, with a best
opening of 2,156. The free-transfer anchor averaged 2,008.5 and peaked at
2,073; the anchored cocktail averaged 2,020.75. These are improvements over
the anchor in this replay, but the best result is still 257 points below the
2,413 research target. This remains a research artifact, not a promoted
champion.

The latest corrected-rules replay is saved under
`runs/action-policy-official-snapshot-v4-season-chips/`. It uses development
seasons 2016/17–2023/24, validation on 2024/25, and a completely untouched
2025/26 test. It fixes a major simulator defect by using the season-specific
2025/26 eight-token chip inventory and half-season chip gates. The corrected
2025/26 neural policy averaged 1,969.5 points across the four opening families
(best 2,053), versus 1,983.75 for the no-hit free-transfer anchor (best 2,129).
The cocktail averaged 1,997.5 (best 2,069). The corrected run is now the
authoritative corrected-rules baseline; it is below the 2,413 target and is
not a promoted champion. The later external-style candidate is reported
separately below. The earlier v3 hit-aware result was an ablation under the
old one-copy chip inventory and must not be used as the final 2025/26 score.

The public [fpl-luck-or-skill challenger](https://github.com/zakariae-boui/fpl-luck-or-skill)
reports 2,431 points from a patient, no-hit, use-it-or-lose-it TC/BB policy.
The local simulator now contains that policy as `patient_chips`, but the
neural/context replication scored 1,928.25 on average on the untouched
2025/26 openings (best 1,955). The external number is therefore a useful
benchmark and hypothesis, not a locally verified result; see the data-quality
audit for the exact reproducibility limitation and artifact path.

### External-style forecast and forecast-optimized opening

The repo now includes a Mac-compatible `external_hgb` forecast family. It
reproduces the public challenger's minutes-plus-conditional-points design with
53 leakage-safe features: player form, minutes/start security, xG/xA, price,
ownership, transfer momentum, true fixture team, opponent/team form, and
previous-season production. It uses histogram gradient boosting locally, so it
does not require the external LightGBM/OpenMP runtime.

Run the strict walk-forward candidate with:

```sh
PYTHONPATH=src python scripts/train_action_policy.py \
  --history-root /path/to/fpl-history \
  --validation-season 2024-25 \
  --evaluation-season 2025-26 \
  --forecast-model external_hgb \
  --starting-modes forecast \
  --label-policy points_only \
  --output-dir runs/action-policy-external-hgb-forecast-v1
```

The resulting artifact is `runs/action-policy-external-hgb-forecast-v1/metrics.json`.
On the untouched 2025/26 test it scored:

| Policy | Points | Transfers | Hits |
|---|---:|---:|---:|
| Neural action policy | 2,160 | 62 | 0 |
| Free-transfer anchor | 2,338 | 37 | 0 |
| Cocktail | 2,442 | 53 | 0 |
| **Patient chips candidate** | **2,486** | **37** | **0** |

The patient candidate clears the 2,413 target in this strict held-out replay.
Its 2024/25 validation score was 2,357, so the opening rule was checked on a
prior season before the final test was read. This is the strongest current
research candidate. It is now exposed through the installed MCP's
`fpl_backtest_strategy` tool; it is not silently made the live default because
the published score uses one forecast-optimized opening family rather than a
distribution of random starts.

Run the same candidate through the MCP season simulator with:

```json
{
  "history_root": "/path/to/data/vaastav",
  "season": "2025-26",
  "previous_season": "2024-25",
  "forecast_model": "external_hgb",
  "candidates": [
    {
      "name": "published_research_candidate",
      "policy": "patient_chips",
      "initial_squad_modes": ["forecast"]
    }
  ]
}
```

The binary rebuilds the forecast walk-forward from all seasons before the test
season, so this path is part of the MCP rather than a results-only artifact.
The exact historical elite-manager alternative archive remains incomplete.

The context files are optional. Each event must carry a publication timestamp;
archived events also carry the snapshot `observed_at` timestamp. The live
official API exposes current cumulative `minutes`/`starts`, current
`chance_of_playing_this_round`/`chance_of_playing_next_round`, `news`,
`scout_risks`, price projections, and set-piece order fields. Its player
history exposes realized minutes and starts, not historical probability
snapshots; use `scripts/fetch_fplcache_context.py` to reconstruct those
point-in-time beliefs. The feature builder cuts context off at the simulated
gameweek deadline (90 minutes before the first fixture), not at kickoff.
Supported structured event types
include `injury`, `availability`, `suspension`, `rotation`,
`lineup_predicted`, `lineup_confirmed`, `lineup_benched`, `set_piece`, and
`transfer`. Store the analyzed sentiment, source, reliability, player/team
entity, and expiry alongside the text so the backtest can audit what was known.
See `docs/context-data-contract.md`.

The validated official archive replay sampled 1,722 snapshots through 1 August
2026, produced 9,366 interval/news events, improved validation action RMSE from
9.016 to 8.670, but reduced the held-out 2025/26 neural policy to 2,007.5 mean
points. It is therefore an inspectable context ablation, not part of the
shipped champion until the action layer gates and calibrates these signals.

For a model-family ablation, add `--forecast-model ridge`. The expanded ridge
run reached 2,189 points in an earlier replay, but that result used the
pre-fix temporal grain and is not comparable to the clean result above. The
direct-horizon neural and price-aware variants are retained as inspectable
research outputs; they are not evidence of a winning strategy by themselves.

The optional `--starting-modes ... forecast` mode adds a legal
forecast-optimized opening squad. Under the older neural/ridge bridge it
reached 2,162 points on 2025/26 and was not promoted. The newer
`external_hgb` replay above is a separate, validation-approved forecast
candidate and should not be conflated with that older ablation.

The hit-aware replay is saved under
`runs/action-policy-official-snapshot-v3-hit-aware/`. It generated 165 paid-hit
and 453 multi-transfer counterfactual actions. On untouched 2025/26, its
neural policy averaged 1,986 points (best opening 2,114), versus 1,983.75
(best 2,129) for the no-hit anchor. This is a useful coverage fix and a
negative strategy result under the pre-chip-fix simulator. The corrected-rule
follow-up is v4 above.

## Observed elite-manager benchmark

The repo also includes a provenance-tracked aggregate benchmark from a public
archive of 24,041 complete 2025/26 manager seasons under
`data/elite_managers/`. It includes transfer gain, transfer count, hits,
captain agreement, chip usage, and transfer timing by manager rank band. To
inspect the observed behavior profile locally:

```sh
PYTHONPATH=src python scripts/analyze_elite_managers.py \
  --autopsy data/elite_managers/autopsy_all.csv \
  --output runs/elite-manager-benchmark/summary.json
```

To inspect the pre-deadline conditions around observed transfers, including
recent points/minutes, price, ownership, transfer momentum, and a separate
forward-outcome audit:

```sh
PYTHONPATH=src python scripts/analyze_observed_manager_decisions.py \
  --history-root /path/to/fpl-history \
  --output runs/elite-manager-benchmark/decision-audit.json
```

The top-100 band has a median 325-point net transfer gain, 62 transfers, 4
hits, 0 unused chips, 71.1% captain agreement, and 17 hours' median timing;
the top-10k band has a median 302-point transfer gain, 63 transfers, 4 hits,
0 unused chips, and 79.0% captain agreement. These are descriptive
benchmarks, not causal labels. The shipped loader deliberately excludes final
rank and final points from behavior features. The aggregate table cannot yet
identify exact weekly alternatives. The detailed 2025/26 weekly archive is now present under
`data/elite_managers/season_winners_2025-26/` and is held out from training.
Pre-2025/26 weekly manager archives are still needed for leakage-safe direct
imitation or inverse-decision modeling. The local 2025/26 decision audit found
that top-100 managers bought players with a higher prior three-gameweek point
rate (13.15 versus 12.61 for players sold), slightly lower prior three-
gameweek minutes (216.9 versus 223.7), lower prior ownership, and stronger
positive transfer momentum. The following-three-gameweek points are retained
only as a quarantined outcome audit, never as model features.

To fit descriptive, leakage-audited behavior heads for transfer/hold,
bundles, paid hits, chips, and incoming-versus-outgoing player signals:

```sh
PYTHONPATH=src python scripts/reverse_engineer_observed_actions.py \
  --history-root /path/to/fpl-history \
  --output runs/elite-manager-benchmark/action-heads.json
```

See [observed action heads](docs/observed-action-heads.md) for the measured
behavior metrics and the data needed before these heads can train on older
seasons. The `patient_chips` simulator challenger is also documented there;
it is not the promoted policy.

## Data coverage and missing signals

The archive is not missing the basic FPL history: it contains 247,896 raw
player-fixture rows across ten seasons (2016/17 through 2025/26), including
gameweek points, minutes, starts, form, ownership, transfers, prices, team
scores, opponents, and fixture timing. The model turns this into 286
point-in-time features and keeps the 2025/26 season completely out of fitting
and model selection.

The important gaps are contextual rather than raw player rows. The public
snapshot archive now supplies official news/availability and set-piece
intervals, and `data/elite_managers/` supplies an aggregate observed-manager
benchmark. We still do not have each elite manager's exact point-in-time
alternative set wired into action labels, nor a complete timestamped
expected-minutes history, press-conference/predicted-lineup/social stream, or
richer historical fixture-strength feed. The pipeline has a leakage-safe
expected-minutes model and a full official-context ablation, but neither is
currently a validated strategy improvement. The oldest gameweek
files also need season-level roster snapshots to fill team/position metadata;
those rows are flagged and are not treated as point-in-time transfer history.
News/social signals require an explicitly supplied timestamped context archive
in the historical trainer.

The temporal audit found and fixed 293 three-gameweek label mismatches caused
by skipping blank calendar gameweeks, plus inconsistent lag values in 416
double-gameweek player groups. The current run uses calendar-window labels
and one aggregated player/gameweek grain for lags and horizon/price models.
Details and the remediation plan are in `docs/data-quality-audit.md`.

## License

MIT. This is an independent research tool and is not affiliated with the Premier League or Fantasy Premier League.
