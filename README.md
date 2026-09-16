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
  --forecast-model neural
```

This generates legal counterfactual action labels, fits the action ensemble,
and evaluates neural actions against the free-transfer anchor over points,
value, template, and randomized opening squads. The forecast bridge is
walk-forward: it fits point forecasts, direct 3/8-gameweek totals, and a
next-gameweek price-change model using only earlier seasons. The current
career-feature baseline produced 1,441 development and 177 validation
examples; on the untouched 2025/26 replay its best opening family scored
2,079 points, below the 2,413 research target. It is therefore a research
artifact, not a promoted champion. The same corrected run's anchored cocktail
averaged 1,975.75 on the untouched test, so the current public fallback
remains the transparent anchor until the gate is tuned on additional held-out
seasons.

For a model-family ablation, add `--forecast-model ridge`. The expanded ridge
run reached 2,189 points in its best 2025/26 opening-squad replay (2,113 mean
across four openings), still 224 points below the target. The direct-horizon
neural and price-aware variants are retained as inspectable research outputs;
they are not evidence of a winning strategy by themselves.

The optional `--starting-modes ... forecast` stress test adds a legal
forecast-optimized opening squad. It reached 2,162 points on 2025/26, below
the four-family ridge result, so it is not part of the default benchmark.

## License

MIT. This is an independent research tool and is not affiliated with the Premier League or Fantasy Premier League.
