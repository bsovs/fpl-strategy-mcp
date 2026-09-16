# FPL Strategy MCP

FPL Strategy MCP is a local, rules-aware Fantasy Premier League decision engine. Give it your current 15-player squad, the players you can buy, prices/selling values, free transfers, chips, and current signals. It returns legal hold/transfer/chip options with the reason, risk, short-term outlook, long-term outlook, and price economics.

It is an action policy, not a promise that one player will score the most points. The shipped champion is deliberately conservative: it uses the validated free-transfer anchor unless a learned action clears the temporal guardrail. See the [research paper draft](docs/research-paper.md) for the evidence and limitations.

## Install

macOS or Linux:

```sh
curl -fsSL https://raw.githubusercontent.com/bsovs/fpl-strategy-mcp/main/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/bsovs/fpl-strategy-mcp/main/install.ps1 | iex
```

The installers download the latest release binary; each GitHub release also publishes SHA-256 checksums. To pin a version, set `FPL_STRATEGY_VERSION=0.1.1` before running the installer.

## Connect a client

The default command is a stdio MCP server:

```sh
fpl-strategy-mcp
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

On macOS the file is `~/Library/Application Support/Claude/claude_desktop_config.json`; on Windows it is `%APPDATA%\Claude\claude_desktop_config.json`. Fully restart Claude after editing it. The same stdio command works with Claude Code and other local MCP hosts.

For ChatGPT or Claude web, start the optional remote transport:

```sh
FPL_MCP_BEARER_TOKEN="choose-a-long-random-token" \
  fpl-strategy-mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

Expose `http://127.0.0.1:8000/mcp` through an HTTPS tunnel or authenticated reverse proxy, then add that HTTPS MCP URL as a custom connector/remote MCP server. Do not expose an unauthenticated listener. OpenAI’s API can call remote MCP servers through the Responses API; Claude web also expects a reachable remote connector. The default stdio mode remains the safer local option.

## Input

Call `fpl_recommend_moves` with `gameweek`, `current_squad`, `buyable_players`, `config`, and either point-in-time `signals` or `auto_official_signals: true`. Add `league_context` when rank/leader information should influence risk. Call `fpl_strategy_info` to inspect the served champion and benchmark status.

See [`examples/`](examples/) for client configuration and a protocol smoke test. The official FPL bootstrap endpoint is used only when requested: `https://fantasy.premierleague.com/api/bootstrap-static/`.

## Development

```sh
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev,remote]"
python -m unittest discover -s tests -q
```

The model and bootstrap snapshot are bundled under `assets/`. Historical training and evaluation artifacts are documented in the paper rather than required to run the server.

## License

MIT. This is an independent research tool and is not affiliated with the Premier League or Fantasy Premier League.
