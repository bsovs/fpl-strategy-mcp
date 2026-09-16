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

from .server import (
    SERVER_NAME,
    _backtest_strategy,
    _forecast_signals,
    _lineup_plan,
    _recommend,
    _score_moves,
    _search_players,
    _status_payload,
    _strategy_catalog,
    _strategy_info,
)


mcp = MCPServer(SERVER_NAME)


@mcp.tool()
def fpl_recommend_moves(
    gameweek: int,
    current_squad: list[dict[str, Any]],
    buyable_players: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    signals: list[dict[str, Any]] | None = None,
    strategy: str = "champion",
    weight_overrides: dict[str, Any] | None = None,
    league_context: dict[str, Any] | None = None,
    chips_available: list[str] | None = None,
    chips_used: list[str] | None = None,
    auto_official_signals: bool = True,
    bootstrap_path: str | None = None,
    fetch_official: bool = False,
    limit: int = 10,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Return legal FPL moves for the supplied current squad and gameweek."""

    payload: dict[str, Any] = {
        "gameweek": gameweek,
        "current_squad": current_squad,
        "config": config or {},
        "strategy": strategy,
        "weight_overrides": weight_overrides or {},
        "league_context": league_context or {},
        "chips_available": chips_available,
        "chips_used": chips_used,
        "auto_official_signals": auto_official_signals,
        "bootstrap_path": bootstrap_path,
        "fetch_official": fetch_official,
        "limit": limit,
        "model_path": model_path,
    }
    if buyable_players is not None:
        payload["buyable_players"] = buyable_players
    if signals is not None:
        payload["signals"] = signals
    # The stdio handler treats an omitted key differently from an explicit
    # null for a few optional values, so remove nulls before dispatching.
    return _recommend({key: value for key, value in payload.items() if value is not None})


@mcp.tool()
def fpl_lineup_plan(
    gameweek: int,
    current_squad: list[dict[str, Any]],
    buyable_players: list[dict[str, Any]] | None = None,
    signals: list[dict[str, Any]] | None = None,
    auto_official_signals: bool = True,
    bootstrap_path: str | None = None,
    fetch_official: bool = False,
) -> dict[str, Any]:
    """Return the legal live formation, XI, bench and captaincy plan."""

    payload: dict[str, Any] = {
        "gameweek": gameweek,
        "current_squad": current_squad,
        "auto_official_signals": auto_official_signals,
        "bootstrap_path": bootstrap_path,
        "fetch_official": fetch_official,
    }
    if buyable_players is not None:
        payload["buyable_players"] = buyable_players
    if signals is not None:
        payload["signals"] = signals
    return _lineup_plan({key: value for key, value in payload.items() if value is not None})


@mcp.tool()
def fpl_strategy_info() -> dict[str, Any]:
    """Return the served strategy, benchmark status, and limitations."""

    return _strategy_info()


@mcp.tool()
def fpl_search_players(
    query: str = "",
    position: str | None = None,
    team: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    available_only: bool = False,
    limit: int = 50,
    bootstrap_path: str | None = None,
    fetch_official: bool = False,
) -> dict[str, Any]:
    """Search the official current-season player pool."""

    return _search_players(
        {
            "query": query,
            "position": position or "",
            "team": team or "",
            "min_price": min_price,
            "max_price": max_price,
            "available_only": available_only,
            "limit": limit,
            "bootstrap_path": bootstrap_path,
            "fetch_official": fetch_official,
        }
    )


@mcp.tool()
def fpl_forecast_signals(
    gameweek: int,
    current_squad: list[dict[str, Any]],
    buyable_players: list[dict[str, Any]] | None = None,
    signals: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    auto_official_signals: bool = True,
    bootstrap_path: str | None = None,
    fetch_official: bool = False,
) -> dict[str, Any]:
    """Inspect point-in-time signal inputs without choosing an action."""

    payload: dict[str, Any] = {
        "gameweek": gameweek,
        "current_squad": current_squad,
        "config": config or {},
        "auto_official_signals": auto_official_signals,
        "bootstrap_path": bootstrap_path,
        "fetch_official": fetch_official,
    }
    if buyable_players is not None:
        payload["buyable_players"] = buyable_players
    if signals is not None:
        payload["signals"] = signals
    return _forecast_signals(payload)


@mcp.tool()
def fpl_score_moves(
    gameweek: int,
    current_squad: list[dict[str, Any]],
    buyable_players: list[dict[str, Any]] | None = None,
    signals: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    weight_overrides: dict[str, Any] | None = None,
    auto_official_signals: bool = True,
    bootstrap_path: str | None = None,
    fetch_official: bool = False,
    limit: int = 10,
) -> dict[str, Any]:
    """Rank transparent legal transfers under explicit weights."""

    payload: dict[str, Any] = {
        "gameweek": gameweek,
        "current_squad": current_squad,
        "config": config or {},
        "weight_overrides": weight_overrides or {},
        "auto_official_signals": auto_official_signals,
        "bootstrap_path": bootstrap_path,
        "fetch_official": fetch_official,
        "limit": limit,
    }
    if buyable_players is not None:
        payload["buyable_players"] = buyable_players
    if signals is not None:
        payload["signals"] = signals
    return _score_moves(payload)


@mcp.tool()
def fpl_strategy_catalog() -> dict[str, Any]:
    """Return strategy profiles and tuning fields."""

    return _strategy_catalog()


@mcp.tool()
def fpl_backtest_strategy(
    scenarios: list[dict[str, Any]] | None = None,
    scenarios_path: str | None = None,
    candidates: list[dict[str, Any]] | None = None,
    base_config: dict[str, Any] | None = None,
    objective: str = "net_points",
    include_details: bool = True,
    history_root: str | None = None,
    season: str | None = None,
    previous_season: str | None = None,
    policy: str | None = None,
    initial_squad_modes: list[str] | None = None,
    cocktail_config: dict[str, Any] | None = None,
    start_gameweek: int | None = None,
    end_gameweek: int | None = None,
    seed: int = 0,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Replay supplied scenarios or a historical season under candidate strategies."""

    payload: dict[str, Any] = {
        "objective": objective,
        "include_details": include_details,
        "base_config": base_config or {},
        "seed": seed,
    }
    optional = {
        "scenarios": scenarios,
        "scenarios_path": scenarios_path,
        "candidates": candidates,
        "history_root": history_root,
        "season": season,
        "previous_season": previous_season,
        "policy": policy,
        "initial_squad_modes": initial_squad_modes,
        "cocktail_config": cocktail_config,
        "start_gameweek": start_gameweek,
        "end_gameweek": end_gameweek,
        "model_path": model_path,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    return _backtest_strategy(payload)


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
