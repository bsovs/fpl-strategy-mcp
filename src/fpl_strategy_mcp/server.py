#!/usr/bin/env python3
"""Minimal dependency-free MCP server for the local FPL strategy.

The server speaks JSON-RPC over stdin/stdout, so it can be launched by any MCP
client without installing a separate web framework.  The main tool is
``fpl_recommend_moves``: pass the current 15-player squad, buyable players,
point-in-time signals, bank/free transfers, gameweek, unused chips, and optional
mini-league context. It returns legal transfers or a hold decision under the
frozen champion by default; explicit hybrid strategies remain available.

Run directly:

    fpl-strategy-mcp

The default model and bootstrap snapshot are bundled with the distribution.
Override the model with the ``FPL_NEURAL_MODEL`` environment variable or a
per-call ``model_path``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import joblib

if os.environ.get("FPL_STRATEGY_HOME"):
    ROOT = Path(os.environ["FPL_STRATEGY_HOME"]).expanduser().resolve()
elif getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(ROOT))

from fpl_lab.decision import DecisionConfig, PlayerSignal, PlayerState
from fpl_lab.fpl_api import BOOTSTRAP_URL, fetch_json
from fpl_lab.live_strategy import recommend_live_moves
from fpl_lab.league import HYBRID_PROFILES
from fpl_lab.policy import ActionValueEnsemble, ActionValueMLP
from fpl_lab.simulator import CHIP_KINDS


SERVER_NAME = "fpl-strategy"
SERVER_VERSION = "0.1.2"
PROTOCOL_VERSION = "2024-11-05"
PACKAGE_ASSETS = Path(__file__).resolve().parent / "assets"
DEFAULT_MODEL = ROOT / "assets" / "action-policy-model.joblib"
CHAMPION_PATH = ROOT / "assets" / "champion.json"
_MODEL_CACHE: dict[str, object] = {}
POSITION_BY_ELEMENT_TYPE = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def _asset_path(filename: str) -> Path:
    """Resolve an installed, bundled, or source-checkout asset."""

    candidates = [
        ROOT / "assets" / filename,
        PACKAGE_ASSETS / filename,
        Path(getattr(sys, "_MEIPASS", "")) / "assets" / filename,
    ]
    for candidate in candidates:
        if str(candidate) and candidate.exists():
            return candidate
    return candidates[0]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_model(path_value: str | None = None):
    raw = path_value or os.environ.get("FPL_NEURAL_MODEL") or str(_asset_path("action-policy-model.joblib"))
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    key = str(path.resolve())
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key], key, None
    if not path.exists():
        return None, key, f"model file does not exist: {path}"
    try:
        model = joblib.load(path)
    except Exception as exc:  # pragma: no cover - defensive server boundary
        return None, key, f"could not load model {path}: {exc}"
    if not isinstance(model, (ActionValueMLP, ActionValueEnsemble)):
        return None, key, f"model at {path} is not an action-value policy model"
    _MODEL_CACHE[key] = model
    return model, key, None


def _dataclass_rows(rows: Any, cls):
    if not isinstance(rows, list):
        raise ValueError(f"expected a list of {cls.__name__} records")
    allowed = set(cls.__dataclass_fields__)
    output = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"record {index} must be an object")
        values = {key: value for key, value in row.items() if key in allowed}
        if "player_id" in values:
            values["player_id"] = str(values["player_id"])
        output.append(cls(**values))
    return output


def _number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _load_bootstrap(path_value: str | None, fetch_official: bool) -> tuple[dict[str, Any], str]:
    """Load a local snapshot by default, or fetch the official API explicitly."""

    if fetch_official:
        return fetch_json(BOOTSTRAP_URL), BOOTSTRAP_URL
    path = Path(path_value) if path_value else _asset_path("bootstrap-static.snapshot.json")
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise ValueError(
            f"bootstrap snapshot does not exist: {path}; set fetch_official=true or provide bootstrap_path"
        )
    return json.loads(path.read_text(encoding="utf-8")), str(path.resolve())


def _bootstrap_lookup(bootstrap: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    elements = {
        str(element["id"]): element
        for element in bootstrap.get("elements", [])
        if isinstance(element, dict) and element.get("id") is not None
    }
    teams = {
        str(team["id"]): str(team.get("name", team.get("short_name", team["id"])))
        for team in bootstrap.get("teams", [])
        if isinstance(team, dict) and team.get("id") is not None
    }
    return elements, teams


def _hydrate_player_row(row: dict[str, Any], elements: dict[str, dict[str, Any]], teams: dict[str, str]) -> dict[str, Any]:
    """Allow compact ``{"player_id": 123}`` rows when a bootstrap is supplied."""

    hydrated = dict(row)
    if hydrated.get("player_id") is None and hydrated.get("id") is not None:
        hydrated["player_id"] = str(hydrated["id"])
    player_id = str(hydrated.get("player_id", ""))
    element = elements.get(player_id)
    if element is None:
        return hydrated
    if not hydrated.get("name"):
        hydrated["name"] = str(
            element.get("web_name")
            or " ".join(part for part in (element.get("first_name"), element.get("second_name")) if part)
            or player_id
        )
    if not hydrated.get("position"):
        hydrated["position"] = POSITION_BY_ELEMENT_TYPE.get(int(element.get("element_type", 0)), "")
    if not hydrated.get("team"):
        hydrated["team"] = teams.get(str(element.get("team")), str(element.get("team", "")))
    if hydrated.get("price") is None:
        hydrated["price"] = _number(element.get("now_cost")) / 10.0
    if hydrated.get("selling_price") is None and hydrated.get("selling_value") is not None:
        hydrated["selling_price"] = hydrated["selling_value"]
    if hydrated.get("can_buy") is None:
        hydrated["can_buy"] = bool(element.get("can_transact", True) and element.get("can_select", True))
    return hydrated


def _official_signal_rows(
    player_rows: list[dict[str, Any]],
    elements: dict[str, dict[str, Any]],
    gameweek: int,
) -> list[PlayerSignal]:
    """Build a transparent, explicitly weaker signal fallback from bootstrap fields."""

    signals: list[PlayerSignal] = []
    remaining = max(1, 38 - int(gameweek) + 1)
    long_window = min(8, remaining)
    for row in player_rows:
        player_id = str(row.get("player_id", ""))
        element = elements.get(player_id)
        if element is None:
            raise ValueError(f"player_id={player_id} is not present in the bootstrap snapshot")
        ep_next = _number(element.get("ep_next"))
        ep_this = _number(element.get("ep_this"))
        form = _number(element.get("form"))
        points_per_game = _number(element.get("points_per_game"))
        short_points = ep_next or ep_this or form or points_per_game
        long_points = max(points_per_game, short_points) * long_window
        chance = element.get("chance_of_playing_next_round")
        status = str(element.get("status", "a"))
        if chance is None:
            minutes_probability = {"a": 1.0, "d": 0.5}.get(status, 0.0)
        else:
            minutes_probability = _clip(_number(chance) / 100.0, 0.0, 1.0)
        selected = _number(element.get("selected_by_percent"))
        ownership_leverage = _clip((50.0 - selected) / 50.0, -1.0, 1.0)
        net_transfers = _number(element.get("transfers_in_event")) - _number(element.get("transfers_out_event"))
        price_flow = _clip(net_transfers / 150_000.0, -1.0, 1.0)
        value_season = _number(element.get("value_season"))
        news = str(element.get("news", "") or "")
        risk = 1.0 - minutes_probability
        signals.append(
            PlayerSignal(
                player_id=player_id,
                short_expected_points=short_points,
                long_expected_points=long_points,
                next_expected_points=short_points,
                short_minutes_probability=minutes_probability,
                long_minutes_probability=minutes_probability,
                form_signal=_clip((form - 3.5) / 3.5, -1.0, 1.0),
                value_signal=_clip((value_season - 4.0) / 4.0, -1.0, 1.0),
                role_security=minutes_probability,
                injury_risk=risk,
                rotation_risk=_clip(0.25 * risk, 0.0, 1.0),
                price_change_risk=_clip(0.5 * risk, 0.0, 1.0),
                captain_upside=_clip(short_points / 10.0, 0.0, 1.0),
                short_price_signal=price_flow,
                long_price_signal=_clip(_number(element.get("cost_change_start")) / 5.0, -1.0, 1.0),
                ownership_leverage=ownership_leverage,
                news_risk=risk if status != "a" or news else 0.0,
                news_sentiment=0.0,
                context_event_count=1 if news else 0,
                trigger=news or f"official status: {status}",
                note="official bootstrap fallback; replace with the full point-in-time signal ensemble when available",
            )
        )
    return signals


def _state_from_arguments(arguments: dict[str, Any]):
    payload = arguments.get("state", arguments)
    if not isinstance(payload, dict):
        raise ValueError("state must be an object")
    current_rows = payload.get("current_squad") or payload.get("current_team")
    buyable_rows = payload.get("buyable_players") or payload.get("available_players")
    signals_rows = payload.get("signals")
    if current_rows is None or buyable_rows is None or signals_rows is None:
        if current_rows is None or buyable_rows is None:
            raise ValueError(
                "state requires current_squad/current_team and buyable_players/available_players"
            )
        if signals_rows is not None:
            raise ValueError("signals must be a list when supplied")
    gameweek = payload.get("gameweek")
    if gameweek is None:
        raise ValueError("gameweek is required")
    auto_official = bool(payload.get("auto_official_signals", False))
    bootstrap = None
    bootstrap_source = None
    if auto_official:
        bootstrap, bootstrap_source = _load_bootstrap(
            payload.get("bootstrap_path"), bool(payload.get("fetch_official", False))
        )
        elements, teams = _bootstrap_lookup(bootstrap)
        current_rows = [_hydrate_player_row(row, elements, teams) for row in current_rows]
        buyable_rows = [_hydrate_player_row(row, elements, teams) for row in buyable_rows]
    current = _dataclass_rows(current_rows, PlayerState)
    buyable = _dataclass_rows(buyable_rows, PlayerState)
    if signals_rows is None:
        if not auto_official or bootstrap is None:
            raise ValueError(
                "signals is required unless auto_official_signals=true; the fallback uses the local or official bootstrap"
            )
        signals = _official_signal_rows(
            [*current_rows, *buyable_rows],
            _bootstrap_lookup(bootstrap)[0],
            int(gameweek),
        )
        signal_source = f"official_bootstrap:{bootstrap_source}"
        state_warnings = [
            "Automatic bootstrap signals are a transparent fallback, not the full leakage-safe future point/price/news ensemble.",
            "Use supplied signals from the trained forecasting pipeline for production decisions.",
        ]
    else:
        signals = _dataclass_rows(signals_rows, PlayerSignal)
        signal_source = "provided"
        state_warnings = []
    config_payload = payload.get("config", {})
    if not isinstance(config_payload, dict):
        raise ValueError("config must be an object")
    allowed_config = set(DecisionConfig.__dataclass_fields__)
    config = DecisionConfig(
        **{key: value for key, value in config_payload.items() if key in allowed_config}
    )
    league_context = payload.get("league_context", {})
    if not isinstance(league_context, dict):
        raise ValueError("league_context must be an object")
    strategy = str(payload.get("strategy", "champion"))
    limit = int(payload.get("limit", 10))
    model_path = payload.get("model_path")
    chips_payload = payload.get("chips_available")
    if chips_payload is None:
        used_chips = payload.get("chips_used", [])
        if not isinstance(used_chips, list):
            raise ValueError("chips_used must be a list when supplied")
        chips_payload = sorted(set(CHIP_KINDS) - {str(chip) for chip in used_chips})
    if not isinstance(chips_payload, list):
        raise ValueError("chips_available must be a list when supplied")
    chips_available = [str(chip) for chip in chips_payload]
    unknown_chips = set(chips_available) - set(CHIP_KINDS)
    if unknown_chips:
        raise ValueError(f"unknown chips: {sorted(unknown_chips)}")
    return {
        "current": current,
        "buyable": buyable,
        "signals": signals,
        "config": config,
        "gameweek": int(gameweek),
        "strategy": strategy,
        "limit": limit,
        "my_points": league_context.get("my_points", payload.get("my_points")),
        "leader_points": league_context.get("leader_points", payload.get("leader_points")),
        "league_size": int(league_context.get("league_size", payload.get("league_size", 10))),
        "chips_available": chips_available,
        "model_path": str(model_path) if model_path else None,
        "signal_source": signal_source,
        "state_warnings": state_warnings,
    }


def _recommend(arguments: dict[str, Any]) -> dict[str, Any]:
    state = _state_from_arguments(arguments)
    model, model_path, model_error = _load_model(state["model_path"])
    result = recommend_live_moves(
        state["current"],
        state["buyable"],
        state["signals"],
        state["config"],
        gameweek=state["gameweek"],
        strategy=state["strategy"],
        my_points=state["my_points"],
        leader_points=state["leader_points"],
        league_size=state["league_size"],
        model=model,
        limit=state["limit"],
        chips_available=state["chips_available"],
    )
    result["model_path"] = model_path
    result["model_error"] = model_error
    result["input_contract"] = {
        "current_squad": "exactly 15 PlayerState records",
        "buyable_players": "same-position legal alternatives with prices",
        "signals": "one PlayerSignal record for every current and buyable player; optional with auto_official_signals=true",
        "league_context": "optional my_points, leader_points, league_size",
        "chips_available": "optional list of unused chips; chips_used may be supplied instead",
    }
    result["signal_source"] = state["signal_source"]
    result["state_warnings"] = state["state_warnings"]
    result.setdefault("warnings", []).extend(state["state_warnings"])
    return result


def _strategy_info() -> dict[str, Any]:
    metrics_path = _asset_path("league-benchmark.json")
    league = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    selected = league.get("development_best_hybrid", {})
    cocktail_path = _asset_path("action-cocktail-metrics.json")
    cocktail = json.loads(cocktail_path.read_text(encoding="utf-8")) if cocktail_path.exists() else {}
    champion = cocktail.get("champion", {})
    return {
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "default_strategy": "champion",
        "available_strategies": ["champion", *sorted(HYBRID_PROFILES)],
        "development_selected_strategy": selected.get("strategy", "hybrid_win"),
        "action_cocktail_champion": champion,
        "action_cocktail_holdout_summary": cocktail.get("holdout_summary", []),
        "holdout_summary": next(
            (row for row in league.get("holdout_summary_by_strategy", []) if row.get("strategy") == "hybrid_win"),
            None,
        ),
        "scope": [
            "legal transfer recommendations",
            "hold versus transfer versus Wildcard/Free Hit/Bench Boost/Triple Captain action ranking",
            "short/long points and price signals",
            "model uncertainty",
            "standings deficit and ownership leverage",
        ],
        "limitations": [
            "synthetic strategy league, not real rival-manager behavior",
            "live Bench Boost uses a proxy lineup; historical backtests use the exact legal lineup",
            "the frozen champion is currently the free-transfer points anchor because no learned override cleared the paired temporal guardrail",
            "model quality is not proven against global FPL winners",
        ],
        "research_sources": [
            "https://www.premierleague.com/en/news/2632794",
            "https://www.premierleague.com/en/news/3444858",
            "https://www.premierleague.com/en/news/3495479",
            "https://www.premierleague.com/en/news/4022717",
            "https://www.premierleague.com/en/news/4316375",
            "https://www.premierleague.com/en/news/4317606",
            "https://fplgod.com/blog/what-we-learned-from-ten-years-of-fpl-winners",
        ],
    }


TOOLS = [
    {
        "name": "fpl_recommend_moves",
        "description": (
            "Return legal FPL transfer moves for the current gameweek using the "
            "frozen champion strategy by default. Give a full 15-player "
            "current squad, buyable players, point-in-time signals or enable "
            "auto_official_signals, bank/free transfers, unused chips, and optional "
            "mini-league standings context. The default champion ranks hold and "
            "legal free transfers; explicit hybrid challengers can rank Wildcard, "
            "Free Hit, Bench Boost, and Triple Captain when model-backed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "gameweek": {"type": "integer", "minimum": 1, "maximum": 38},
                "strategy": {
                    "type": "string",
                    "enum": ["champion", "hybrid_win", "hybrid_balanced", "hybrid_safe"],
                    "default": "champion",
                },
                "current_squad": {"type": "array", "items": {"type": "object"}},
                "buyable_players": {"type": "array", "items": {"type": "object"}},
                "signals": {"type": "array", "items": {"type": "object"}},
                "config": {"type": "object"},
                "league_context": {"type": "object"},
                "chips_available": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["wildcard", "free_hit", "bench_boost", "triple_captain"]},
                    "description": "Unused chips at this deadline. If omitted, all four are assumed available.",
                },
                "chips_used": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["wildcard", "free_hit", "bench_boost", "triple_captain"]},
                    "description": "Alternative to chips_available: chips already used this season.",
                },
                "auto_official_signals": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, signals may be omitted and are built from a local bootstrap snapshot or the official API.",
                },
                "bootstrap_path": {"type": "string"},
                "fetch_official": {
                    "type": "boolean",
                    "default": False,
                    "description": "When auto_official_signals is true, fetch the current official bootstrap instead of using the local snapshot.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                "model_path": {"type": "string"},
            },
            "required": ["gameweek", "current_squad", "buyable_players", "config"],
        },
    },
    {
        "name": "fpl_strategy_info",
        "description": "Return the selected strategy, benchmark status, research sources, and known limitations.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _tool_result(payload: Any, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2, ensure_ascii=False)}],
        "structuredContent": payload,
        "isError": is_error,
    }


def _dispatch(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    if method in {"notifications/initialized", "notifications/cancelled", "notifications/progress"}:
        return None
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "instructions": "Use fpl_recommend_moves with a current 15-player squad, point-in-time signals, and unused chips when known.",
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
    if method == "resources/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"resources": []}}
    if method == "prompts/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"prompts": []}}
    if method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        try:
            if name == "fpl_recommend_moves":
                payload = _recommend(arguments)
            elif name == "fpl_strategy_info":
                payload = _strategy_info()
            else:
                raise ValueError(f"unknown tool: {name}")
            return {"jsonrpc": "2.0", "id": request_id, "result": _tool_result(payload)}
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": _tool_result({"error": str(exc)}, is_error=True),
            }
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def _stdio_main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("JSON-RPC request must be an object")
            response = _dispatch(request)
            if response is not None:
                sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
                sys.stdout.flush()
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": str(exc)},
            }
            sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            sys.stdout.flush()


def main(argv: list[str] | None = None) -> None:
    """Start the local stdio server or the optional Streamable HTTP server."""

    import argparse

    parser = argparse.ArgumentParser(description="FPL Strategy MCP server")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport; stdio is the default for local desktop clients",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    args = parser.parse_args(argv)
    if args.transport == "stdio":
        _stdio_main()
        return
    try:
        from .http_server import run_http
    except ImportError:
        from fpl_strategy_mcp.http_server import run_http

    run_http(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
