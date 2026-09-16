"""Optional Streamable HTTP transport for remote MCP hosts.

The default package stays dependency-light and uses stdio.  Installing the
``remote`` extra adds the official MCP Python SDK and enables this module.
Keep the HTTP listener on localhost unless it is placed behind HTTPS and an
authentication layer.
"""

from __future__ import annotations

import os
from typing import Any

try:
    from mcp.server import MCPServer
    from starlette.responses import JSONResponse
    import uvicorn
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise RuntimeError(
        "remote transport is optional; install fpl-strategy-mcp[remote] first"
    ) from exc

from .server import SERVER_NAME, _recommend, _strategy_info
from .server import _status_payload


mcp = MCPServer(SERVER_NAME)


@mcp.tool()
def fpl_recommend_moves(
    gameweek: int,
    current_squad: list[dict[str, Any]],
    buyable_players: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    signals: list[dict[str, Any]] | None = None,
    strategy: str = "champion",
    league_context: dict[str, Any] | None = None,
    chips_available: list[str] | None = None,
    chips_used: list[str] | None = None,
    auto_official_signals: bool = False,
    bootstrap_path: str | None = None,
    fetch_official: bool = False,
    limit: int = 10,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Return legal FPL moves for the supplied current squad and gameweek."""

    payload: dict[str, Any] = {
        "gameweek": gameweek,
        "current_squad": current_squad,
        "buyable_players": buyable_players,
        "config": config or {},
        "signals": signals,
        "strategy": strategy,
        "league_context": league_context or {},
        "chips_available": chips_available,
        "chips_used": chips_used,
        "auto_official_signals": auto_official_signals,
        "bootstrap_path": bootstrap_path,
        "fetch_official": fetch_official,
        "limit": limit,
        "model_path": model_path,
    }
    # The stdio handler treats an omitted key differently from an explicit
    # null for a few optional values, so remove nulls before dispatching.
    return _recommend({key: value for key, value in payload.items() if value is not None})


@mcp.tool()
def fpl_strategy_info() -> dict[str, Any]:
    """Return the served strategy, benchmark status, and limitations."""

    return _strategy_info()


class _BearerTokenMiddleware:
    """Small optional bearer-token guard for a directly exposed endpoint."""

    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        supplied = headers.get("authorization", "")
        if supplied != f"Bearer {self.token}":
            response = JSONResponse({"error": "missing or invalid bearer token"}, status_code=401)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


async def _health(request):
    """Small human- and machine-readable readiness endpoint."""

    return JSONResponse(_status_payload())


def run_http(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the MCP endpoint at ``/mcp`` using Streamable HTTP."""

    app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        json_response=True,
    )
    app.add_route("/health", _health, methods=["GET"])
    token = os.environ.get("FPL_MCP_BEARER_TOKEN")
    if token:
        app = _BearerTokenMiddleware(app, token)
    uvicorn.run(app, host=host, port=port, log_level="info")
