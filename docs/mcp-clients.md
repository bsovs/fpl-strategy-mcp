# MCP client setup

The binary has two transports. Use stdio when the MCP host is on the same
computer. Use Streamable HTTP only when a client needs a URL.

## One-command setup

macOS/Linux:

```sh
curl -fsSL https://raw.githubusercontent.com/bsovs/fpl-strategy-mcp/main/install.sh | sh -s -- --clients all
```

Windows PowerShell:

```powershell
$env:FPL_STRATEGY_CLIENTS="all"; irm https://raw.githubusercontent.com/bsovs/fpl-strategy-mcp/main/install.ps1 | iex
```

The installer downloads the matching release binary, configures the selected
clients, backs up existing Claude/Codex config, and runs a fast
`fpl-strategy-mcp status`. To configure an existing installation later, run
`fpl-strategy-mcp setup --clients all`. Use `fpl-strategy-mcp status --deep`
when you want to load-test the bundled model.

## Claude Desktop and Claude Code

Start the local server through the installed binary:

```json
{
  "mcpServers": {
    "fpl-strategy": {
      "command": "/absolute/path/to/fpl-strategy-mcp"
    }
  }
}
```

Claude Desktop config paths:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

For Claude Code:

```sh
claude mcp add fpl-strategy -- /absolute/path/to/fpl-strategy-mcp
```

Restart Claude Desktop after editing its file. MCP protocol messages remain the
only stdout output. The readiness line goes to stderr, and
`fpl-strategy-mcp status` provides a human-readable or JSON diagnostic report.

## ChatGPT or Claude web

Start a local HTTP endpoint:

```sh
export FPL_MCP_BEARER_TOKEN="choose-a-long-random-token"
fpl-strategy-mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

The endpoint is `http://127.0.0.1:8000/mcp`. A web client cannot reach that
private address directly. Put it behind an HTTPS tunnel or reverse proxy with
authentication, then register the resulting URL as a custom connector/remote
MCP server. Treat the URL and token as credentials.

While running, check `http://127.0.0.1:8000/health`; it returns the same JSON
readiness report and inherits the bearer-token guard when configured.

OpenAI’s Responses API documents remote MCP tools in its [MCP tool
reference](https://developers.openai.com/api/reference/cli/resources/beta/subresources/responses).
The MCP Python SDK documents the [local stdio host
flow](https://py.sdk.modelcontextprotocol.io/get-started/real-host/) and the
[Streamable HTTP server
flow](https://py.sdk.modelcontextprotocol.io/run/).

## Manual smoke test

The repository includes a newline-delimited initialize/list-tools request in
[`examples/stdio-initialize.jsonl`](../examples/stdio-initialize.jsonl). From a
source checkout:

```sh
python -m fpl_strategy_mcp < examples/stdio-initialize.jsonl
```
