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
from dataclasses import asdict
from pathlib import Path
import re
import shutil
import sys
import subprocess
import time
from typing import Any

if os.environ.get("FPL_STRATEGY_HOME"):
    ROOT = Path(os.environ["FPL_STRATEGY_HOME"]).expanduser().resolve()
elif getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(ROOT))

SERVER_NAME = "fpl-strategy"
SERVER_VERSION = "0.1.6"
PROTOCOL_VERSION = "2024-11-05"
PACKAGE_ASSETS = Path(__file__).resolve().parent / "assets"
DEFAULT_MODEL = ROOT / "assets" / "action-policy-model.joblib"
CHAMPION_PATH = ROOT / "assets" / "champion.json"
_MODEL_CACHE: dict[str, object] = {}
_BOOTSTRAP_CACHE: dict[str, tuple[dict[str, Any], str]] = {}
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


def _server_command() -> list[str]:
    """Return the command a local MCP client should launch."""

    if getattr(sys, "frozen", False):
        return [str(Path(sys.executable).resolve())]
    installed = shutil.which("fpl-strategy-mcp")
    if installed:
        return [installed]
    return [sys.executable, "-m", "fpl_strategy_mcp"]


def _claude_desktop_config_path() -> Path:
    """Return the conventional Claude Desktop config path for this platform."""

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if os.name == "nt":
        appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(appdata) / "Claude" / "claude_desktop_config.json"
    config_home = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(config_home) / "Claude" / "claude_desktop_config.json"


def _codex_config_path() -> Path:
    codex_home = os.environ.get("CODEX_HOME") or str(Path.home() / ".codex")
    return Path(codex_home) / "config.toml"


def _backup_file(path: Path) -> str | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.bak-{int(time.time())}")
    shutil.copy2(path, backup)
    return str(backup)


def _write_text_file(path: Path, text: str) -> str | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = _backup_file(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return backup


def _write_json_file(path: Path, payload: dict[str, Any]) -> str | None:
    return _write_text_file(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _configure_claude_desktop(command: list[str]) -> dict[str, Any]:
    path = _claude_desktop_config_path()
    if path.exists() and path.read_text(encoding="utf-8").strip():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Claude Desktop config must be a JSON object: {path}")
    else:
        payload = {}
    servers = payload.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError(f"Claude Desktop mcpServers must be an object: {path}")
    entry: dict[str, Any] = {"command": command[0]}
    if len(command) > 1:
        entry["args"] = command[1:]
    servers[SERVER_NAME] = entry
    backup = _write_json_file(path, payload)
    return {"status": "configured", "path": str(path), "backup": backup, "entry": entry}


def _toml_has_server(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(re.search(r"(?m)^\[mcp_servers\.fpl-strategy(?:\.[^\]]+)?\]\s*$", text))


def _configure_codex_config(command: list[str]) -> dict[str, Any]:
    """Register the stdio server without requiring the Codex CLI executable."""

    path = _codex_config_path()
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    skipping = False
    for line in lines:
        match = re.match(r"^\s*\[([^\]]+)\]\s*$", line)
        if match:
            skipping = match.group(1) == "mcp_servers.fpl-strategy" or match.group(1).startswith("mcp_servers.fpl-strategy.")
        if not skipping:
            kept.append(line)
    cleaned = "".join(kept).rstrip() + ("\n\n" if kept else "")
    command_literal = json.dumps(command[0], ensure_ascii=False)
    args_literal = json.dumps(command[1:], ensure_ascii=False)
    updated = (
        cleaned
        + "[mcp_servers.fpl-strategy]\n"
        + f"command = {command_literal}\nargs = {args_literal}\nstartup_timeout_sec = 120\n"
    )
    backup = _write_text_file(path, updated)
    return {
        "status": "configured",
        "client": "codex",
        "path": str(path),
        "backup": backup,
        "entry": {"command": command[0], "args": command[1:], "startup_timeout_sec": 120},
    }


def _replace_cli_mcp_server(cli_name: str, command: list[str]) -> dict[str, Any]:
    executable = shutil.which(cli_name)
    if not executable:
        return {"status": "not_found", "client": cli_name}
    remove = subprocess.run(
        [executable, "mcp", "remove", SERVER_NAME],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
        timeout=10,
        check=False,
    )
    add = subprocess.run(
        [executable, "mcp", "add", SERVER_NAME, "--", *command],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
        timeout=20,
        check=False,
    )
    if add.returncode != 0:
        detail = (add.stderr or add.stdout or "unknown error").strip()
        return {"status": "error", "client": cli_name, "error": detail}
    return {
        "status": "configured",
        "client": cli_name,
        "removed_existing": remove.returncode == 0,
        "output": (add.stdout or "").strip(),
    }


def _parse_clients(value: str) -> set[str]:
    requested = {item.strip().lower().replace("_", "-") for item in value.split(",") if item.strip()}
    if not requested or requested == {"none"}:
        return set()
    if "all" in requested:
        requested |= {"claude", "codex"}
        requested.discard("all")
    aliases = {
        "claude-desktop": "claude",
    }
    requested = {aliases.get(item, item) for item in requested}
    unknown = requested - {"claude", "claude-code", "codex"}
    if unknown:
        raise ValueError(f"unknown client(s): {sorted(unknown)}; use claude, claude-code, codex, or all")
    return requested


def _setup_clients(value: str) -> dict[str, Any]:
    command = _server_command()
    clients = _parse_clients(value)
    results: dict[str, Any] = {"command": command, "requested": sorted(clients)}
    if "claude" in clients:
        print("Configuring Claude Desktop...", file=sys.stderr, flush=True)
        results["claude_desktop"] = _configure_claude_desktop(command)
    if "claude-code" in clients:
        print("Configuring Claude Code CLI...", file=sys.stderr, flush=True)
        results["claude_code"] = _replace_cli_mcp_server("claude", command)
    if "codex" in clients:
        print("Configuring Codex...", file=sys.stderr, flush=True)
        # The same config is consumed by the Codex desktop app and CLI. Write
        # it directly so desktop-only users do not need a second executable.
        results["codex"] = _configure_codex_config(command)
    return results


def _config_has_server(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(payload, dict) and SERVER_NAME in (payload.get("mcpServers") or {})
    except (OSError, ValueError, TypeError):
        return False


def _json_contains_server(value: Any) -> bool:
    if isinstance(value, dict):
        servers = value.get("mcpServers")
        if isinstance(servers, dict) and SERVER_NAME in servers:
            return True
        return any(_json_contains_server(item) for item in value.values())
    if isinstance(value, list):
        return any(_json_contains_server(item) for item in value)
    return False


def _json_config_has_server(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return _json_contains_server(payload)


def _cli_status(cli_name: str) -> dict[str, Any]:
    executable = shutil.which(cli_name)
    if not executable:
        return {"available": False, "configured": False}
    try:
        result = subprocess.run(
            [executable, "mcp", "get", SERVER_NAME],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"available": True, "configured": False}
    return {"available": True, "configured": result.returncode == 0}


def _status_payload(deep: bool = False) -> dict[str, Any]:
    bootstrap_path = _asset_path("bootstrap-static.snapshot.json")
    model_asset_path = _asset_path("action-policy-model.joblib")
    model = None
    model_path = str(model_asset_path.resolve())
    model_error = None
    if deep:
        model, model_path, model_error = _load_model()
    claude_path = _claude_desktop_config_path()
    claude_code_path = Path.home() / ".claude.json"
    codex_path = _codex_config_path()
    return {
        "status": "ready" if model_asset_path.exists() and bootstrap_path.exists() and (not deep or model is not None) else "degraded",
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "default_transport": "stdio",
        "protocol_version": PROTOCOL_VERSION,
        "model_present": model_asset_path.exists(),
        "model_checked": deep,
        "model_loaded": model is not None if deep else None,
        "model_path": model_path,
        "model_error": model_error,
        "bootstrap_snapshot": str(bootstrap_path),
        "bootstrap_present": bootstrap_path.exists(),
        "clients": {
            "claude_desktop": {
                "configured": _config_has_server(claude_path),
                "config_path": str(claude_path),
            },
            "claude_code_cli": {
                "available": shutil.which("claude") is not None,
                "configured": _json_config_has_server(claude_code_path),
                "config_path": str(claude_code_path),
            },
            "codex_cli": {
                "available": shutil.which("codex") is not None,
                "configured": _toml_has_server(codex_path),
                "config_path": str(codex_path),
            },
        },
    }


def _print_status(payload: dict[str, Any]) -> None:
    print(f"FPL Strategy MCP {payload['version']}")
    print(f"Status: {str(payload['status']).upper()}")
    if payload["model_checked"]:
        print(f"Model: {'loaded' if payload['model_loaded'] else 'not loaded'}")
    else:
        print(f"Model asset: {'present' if payload['model_present'] else 'missing'} (not loaded; use --deep to verify)")
    print(f"Bootstrap snapshot: {'present' if payload['bootstrap_present'] else 'missing'}")
    clients = payload["clients"]
    claude = clients["claude_desktop"]
    print(f"Claude Desktop: {'configured' if claude['configured'] else 'not configured'}")
    claude_code = clients["claude_code_cli"]
    codex = clients["codex_cli"]
    print(
        "Claude Code CLI: "
        + ("configured" if claude_code["configured"] else "available, not configured" if claude_code["available"] else "not found")
    )
    print(
        "Codex CLI: "
        + ("configured" if codex["configured"] else "available, not configured" if codex["available"] else "not found")
    )
    if payload.get("model_error"):
        print(f"Model error: {payload['model_error']}")


def _print_setup(payload: dict[str, Any]) -> None:
    print(f"Configured command: {' '.join(payload['command'])}")
    for name in ("claude_desktop", "claude_code", "codex"):
        result = payload.get(name)
        if not result:
            continue
        status = result.get("status", "unknown")
        suffix = result.get("path") or result.get("error") or result.get("client", "")
        print(f"{name}: {status}{f' ({suffix})' if suffix else ''}")
        if result.get("backup"):
            print(f"  backup: {result['backup']}")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_model(path_value: str | None = None):
    import joblib

    from fpl_lab.policy import ActionValueEnsemble, ActionValueMLP

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

    from fpl_lab.fpl_api import BOOTSTRAP_URL, fetch_json

    if fetch_official:
        cache_key = BOOTSTRAP_URL
        if cache_key not in _BOOTSTRAP_CACHE:
            _BOOTSTRAP_CACHE[cache_key] = (fetch_json(BOOTSTRAP_URL), BOOTSTRAP_URL)
        return _BOOTSTRAP_CACHE[cache_key]
    path = Path(path_value) if path_value else _asset_path("bootstrap-static.snapshot.json")
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise ValueError(
            f"bootstrap snapshot does not exist: {path}; set fetch_official=true or provide bootstrap_path"
        )
    cache_key = str(path.resolve())
    if cache_key not in _BOOTSTRAP_CACHE:
        _BOOTSTRAP_CACHE[cache_key] = (json.loads(path.read_text(encoding="utf-8")), cache_key)
    return _BOOTSTRAP_CACHE[cache_key]


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

    from fpl_lab.decision import PlayerSignal

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
    from fpl_lab.decision import DecisionConfig, PlayerSignal, PlayerState
    from fpl_lab.simulator import CHIP_KINDS

    payload = arguments.get("state", arguments)
    if not isinstance(payload, dict):
        raise ValueError("state must be an object")
    current_rows = payload.get("current_squad") or payload.get("current_team")
    buyable_rows = payload.get("buyable_players") or payload.get("available_players")
    signals_rows = payload.get("signals")
    if current_rows is None:
        raise ValueError("state requires current_squad or current_team")
    if not isinstance(current_rows, list):
        raise ValueError("current_squad/current_team must be a list")
    gameweek = payload.get("gameweek")
    if gameweek is None:
        raise ValueError("gameweek is required")
    # The official bootstrap is the source of truth for the current player
    # universe. If the caller omits buyable_players, load the full pool rather
    # than silently narrowing the decision to whatever players were pasted in.
    pool_was_omitted = buyable_rows is None
    auto_official = (
        bool(payload.get("auto_official_signals", False))
        or pool_was_omitted
        or signals_rows is None
    )
    bootstrap = None
    bootstrap_source = None
    if auto_official:
        bootstrap, bootstrap_source = _load_bootstrap(
            payload.get("bootstrap_path"), bool(payload.get("fetch_official", False))
        )
        elements, teams = _bootstrap_lookup(bootstrap)
        if pool_was_omitted:
            current_ids = {
                str(row.get("player_id", row.get("id", "")))
                for row in current_rows
                if isinstance(row, dict)
            }
            buyable_rows = [
                {"player_id": player_id}
                for player_id in elements
                if player_id not in current_ids
            ]
        current_rows = [_hydrate_player_row(row, elements, teams) for row in current_rows]
        buyable_rows = [_hydrate_player_row(row, elements, teams) for row in buyable_rows]
    if buyable_rows is None or not isinstance(buyable_rows, list):
        raise ValueError(
            "buyable_players/available_players must be a list, or omit it with official bootstrap data enabled"
        )
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
        if auto_official and bootstrap is not None:
            # A caller may provide high-quality signals for a subset while the
            # official pool supplies the rest. Fill only missing IDs with the
            # transparent fallback so one omitted player cannot vanish from
            # the legal action search.
            fallback = _official_signal_rows(
                [*current_rows, *buyable_rows],
                _bootstrap_lookup(bootstrap)[0],
                int(gameweek),
            )
            supplied_ids = {signal.player_id for signal in signals}
            signals.extend(signal for signal in fallback if signal.player_id not in supplied_ids)
            signal_source = f"provided+official_bootstrap:{bootstrap_source}"
            if len(supplied_ids) < len(fallback):
                state_warnings.append(
                    "Official bootstrap fallback filled signal rows missing from the supplied signal table."
                )
    if pool_was_omitted:
        state_warnings.append(
            f"Loaded the full official player pool ({len(buyable)} buyable rows) because buyable_players was omitted."
        )
    config_payload = payload.get("config", {})
    if not isinstance(config_payload, dict):
        raise ValueError("config must be an object")
    weight_overrides = payload.get("weight_overrides", payload.get("weights", {}))
    if not isinstance(weight_overrides, dict):
        raise ValueError("weight_overrides/weights must be an object when supplied")
    allowed_config = set(DecisionConfig.__dataclass_fields__)
    unknown_overrides = set(weight_overrides) - allowed_config
    if unknown_overrides:
        raise ValueError(f"unknown decision weight/config fields: {sorted(unknown_overrides)}")
    merged_config = {
        key: value
        for key, value in config_payload.items()
        if key in allowed_config
    }
    merged_config.update(weight_overrides)
    config = DecisionConfig(
        **merged_config
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


def _decision_config_payload(config: Any) -> dict[str, Any]:
    return asdict(config)


def _player_signal_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    players = {
        player.player_id: {
            "player_id": player.player_id,
            "name": player.name,
            "position": player.position,
            "team": player.team,
            "price": player.price,
            "selling_price": player.selling_price,
            "can_buy": player.can_buy,
            "in_current_squad": False,
            "buyable": False,
        }
        for player in [*state["current"], *state["buyable"]]
    }
    current_ids = {player.player_id for player in state["current"]}
    buyable_ids = {player.player_id for player in state["buyable"]}
    for player_id, row in players.items():
        row["in_current_squad"] = player_id in current_ids
        row["buyable"] = player_id in buyable_ids
    signals = {signal.player_id: asdict(signal) for signal in state["signals"]}
    missing = sorted(set(players) - set(signals))
    if missing:
        raise ValueError(f"missing signals for player IDs: {missing[:10]}")
    return [
        {**players[player_id], **signals[player_id]}
        for player_id in sorted(players, key=lambda item: (players[item]["position"], players[item]["name"]))
    ]


def _recommend(arguments: dict[str, Any]) -> dict[str, Any]:
    from fpl_lab.live_strategy import recommend_live_moves

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
        "weight_overrides": "optional DecisionConfig fields such as short_weight, long_weight, price_weight, ownership_weight, risk_aversion, and rank_mode",
    }
    result["effective_decision_config"] = _decision_config_payload(state["config"])
    result["signal_source"] = state["signal_source"]
    result["state_warnings"] = state["state_warnings"]
    result.setdefault("warnings", []).extend(state["state_warnings"])
    return result


def _lineup_plan(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return the legal live formation, XI, bench order and captaincy."""

    from fpl_lab.live_strategy import choose_live_lineup

    state = _state_from_arguments(arguments)
    lineup = choose_live_lineup(state["current"], state["signals"])
    return {
        "gameweek": state["gameweek"],
        "lineup_plan": lineup,
        "signal_source": state["signal_source"],
        "warnings": state["state_warnings"]
        + [
            "Only the starting XI scores normally; the four bench players score only when Bench Boost is active.",
            "The forecast lineup cannot know final autosubs until confirmed minutes and late team news are available.",
        ],
    }


def _forecast_signals(arguments: dict[str, Any]) -> dict[str, Any]:
    """Expose the point-in-time signal table without making a decision."""

    state = _state_from_arguments(arguments)
    return {
        "gameweek": state["gameweek"],
        "signal_source": state["signal_source"],
        "decision_config": _decision_config_payload(state["config"]),
        "players": _player_signal_rows(state),
        "warnings": state["state_warnings"]
        + [
            "These are the supplied or bootstrap-derived ensemble inputs; this tool does not invent future information.",
            "Use point-in-time news/social records and trained model outputs when tuning or backtesting.",
        ],
    }


def _search_players(arguments: dict[str, Any]) -> dict[str, Any]:
    """Search the cached official player pool for candidate construction."""

    bootstrap, source = _load_bootstrap(
        arguments.get("bootstrap_path"), bool(arguments.get("fetch_official", False))
    )
    elements, teams = _bootstrap_lookup(bootstrap)
    query = str(arguments.get("query", "")).strip().lower()
    position = str(arguments.get("position", "")).strip().upper()
    if position == "GK":
        position = "GKP"
    team_query = str(arguments.get("team", "")).strip().lower()
    min_price = arguments.get("min_price")
    max_price = arguments.get("max_price")
    available_only = bool(arguments.get("available_only", False))
    limit = max(1, min(200, int(arguments.get("limit", 50))))
    rows = []
    for player_id, element in elements.items():
        team_id = str(element.get("team", ""))
        team_name = teams.get(team_id, team_id)
        player_position = POSITION_BY_ELEMENT_TYPE.get(int(element.get("element_type", 0)), "")
        price = _number(element.get("now_cost")) / 10.0
        name = str(element.get("web_name") or "")
        haystack = f"{name} {element.get('first_name', '')} {element.get('second_name', '')} {team_name}".lower()
        if query and query not in haystack and query != player_id:
            continue
        if position and player_position != position:
            continue
        if team_query and team_query not in team_name.lower() and team_query != team_id:
            continue
        if min_price is not None and price < float(min_price):
            continue
        if max_price is not None and price > float(max_price):
            continue
        can_buy = bool(element.get("can_transact", True) and element.get("can_select", True))
        if available_only and not can_buy:
            continue
        rows.append(
            {
                "player_id": player_id,
                "name": name,
                "position": player_position,
                "team_id": team_id,
                "team": team_name,
                "price": price,
                "status": str(element.get("status", "")),
                "can_buy": can_buy,
                "news": str(element.get("news", "")),
                "form": _number(element.get("form")),
                "ep_next": _number(element.get("ep_next")),
                "ep_this": _number(element.get("ep_this")),
                "points_per_game": _number(element.get("points_per_game")),
                "selected_by_percent": _number(element.get("selected_by_percent")),
                "chance_of_playing_next_round": element.get("chance_of_playing_next_round"),
                "total_points": _number(element.get("total_points")),
                "value_season": _number(element.get("value_season")),
                "transfers_in_event": _number(element.get("transfers_in_event")),
                "transfers_out_event": _number(element.get("transfers_out_event")),
                "cost_change_event": _number(element.get("cost_change_event")),
                "cost_change_start": _number(element.get("cost_change_start")),
            }
        )
    rows.sort(key=lambda row: (-row["total_points"], -row["points_per_game"], row["name"]))
    return {
        "source": source,
        "pool_size": len(elements),
        "matched": len(rows),
        "filters": {
            "query": query or None,
            "position": position or None,
            "team": team_query or None,
            "min_price": min_price,
            "max_price": max_price,
            "available_only": available_only,
        },
        "players": rows[:limit],
        "truncated": len(rows) > limit,
    }


def _score_moves(arguments: dict[str, Any]) -> dict[str, Any]:
    """Rank legal one-transfer moves under explicit user-supplied weights."""

    from fpl_lab.decision import recommendation_to_dict, recommend_transfers

    state = _state_from_arguments(arguments)
    if len(state["current"]) != 15:
        raise ValueError(f"current_squad must contain exactly 15 players; received {len(state['current'])}")
    recommendations = recommend_transfers(
        state["current"], state["buyable"], state["signals"], state["config"]
    )
    actions = [
        {
            "action": "hold",
            "decision": "hold",
            "combined_score": 0.0,
            "why": ["hold is the zero-cost baseline for this transfer-only scorecard"],
        }
    ]
    for recommendation in recommendations[: int(state["limit"])]:
        row = recommendation_to_dict(recommendation)
        row.update({"action": "make_transfer", "decision": "make"})
        actions.append(row)
    best = actions[1] if len(actions) > 1 and actions[1]["combined_score"] > 0 else actions[0]
    return {
        "gameweek": state["gameweek"],
        "action": best["action"],
        "recommended_move": best if best["action"] != "hold" else None,
        "actions": actions,
        "legal_move_count": len(recommendations),
        "effective_decision_config": _decision_config_payload(state["config"]),
        "signal_source": state["signal_source"],
        "warnings": state["state_warnings"]
        + [
            "This scorecard tunes the transparent transfer layer only; chip actions remain in fpl_recommend_moves and the action-value model.",
        ],
    }


def _strategy_catalog() -> dict[str, Any]:
    from fpl_lab.decision import DecisionConfig
    from fpl_lab.league import HYBRID_PROFILES
    from fpl_lab.policy import ACTION_KINDS, CocktailConfig

    return {
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "primary_tool": "fpl_recommend_moves",
        "strategies": {
            "champion": {
                "description": "Frozen points-first, free-transfer anchor selected by the temporal tournament.",
                "uses_action_model": False,
                "allows_paid_hits": False,
                "allows_chips": False,
            },
            **{
                name: {
                    **asdict(profile),
                    "description": f"Hybrid challenger profile: {name.replace('_', ' ')}.",
                    "uses_action_model": True,
                }
                for name, profile in HYBRID_PROFILES.items()
            },
        },
        "action_kinds": list(ACTION_KINDS),
        "default_decision_config": asdict(DecisionConfig()),
        "default_cocktail_config": asdict(CocktailConfig()),
        "tunable_fields": {
            "short_weight": "relative emphasis on the next few gameweeks",
            "long_weight": "relative emphasis on the medium-term horizon",
            "price_weight": "future price/bank-value signal weight",
            "ownership_weight": "rank leverage weight; rank_mode must be chase or defend to activate it",
            "risk_aversion": "penalty on injury, rotation, news, price-change, and model uncertainty",
            "rank_mode": "neutral, chase, or defend",
            "min_move_score": "minimum transfer score before a move is considered",
            "now_threshold": "score threshold for labeling a move now rather than watch",
            "free_transfers": "current free transfers available",
            "bank": "current bank in millions",
            "weight_overrides": "the same fields can be passed per request without changing the installed model",
            "cocktail_config": "action-model gates and model/risk blending for historical simulator backtests",
        },
        "signal_components": {
            "points": "short_expected_points, long_expected_points, next_expected_points",
            "price": "short_price_signal, long_price_signal, price_change_risk",
            "availability": "short_minutes_probability, long_minutes_probability, role_security, injury_risk, rotation_risk",
            "context": "news_risk, news_sentiment, social_sentiment, context_reliability, context_event_count",
            "game_theory": "ownership_leverage and league_context",
            "uncertainty": "uncertainty from the underlying forecast ensemble",
        },
        "model_components": [
            {
                "name": "historical_rolling_points",
                "role": "Leakage-safe short/long player-point baseline built from prior gameweeks and fixture counts.",
                "runtime_field": "short_expected_points / long_expected_points",
            },
            {
                "name": "price_economics",
                "role": "Short/long future-price and bank-value signals, including selling-price constraints.",
                "runtime_field": "short_price_signal / long_price_signal / price_change_risk",
            },
            {
                "name": "availability_and_role",
                "role": "Expected minutes, role security, injury and rotation risk.",
                "runtime_field": "short_minutes_probability / role_security / injury_risk / rotation_risk",
            },
            {
                "name": "news_and_social_context",
                "role": "As-of official/team news and social context adjustments when supplied by the caller/context pipeline.",
                "runtime_field": "news_risk / news_sentiment / social_sentiment / context_reliability",
            },
            {
                "name": "action_value_policy",
                "role": "Neural/ensemble value of legal actions after points, prices, hits, chips, and rank state are combined.",
                "runtime_field": "model_advantage / model_uncertainty",
            },
        ],
        "backtest_contracts": {
            "scenario_replay": "Pass point-in-time states plus realized action_outcomes to tune decision weights without future leakage.",
            "season_simulator": "Pass history_root and season to replay the legal FPL simulator over Vaastav-format GW CSVs.",
        },
    }


def _load_scenarios(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = arguments.get("scenarios")
    path_value = arguments.get("scenarios_path")
    if scenarios is None and path_value:
        path = Path(str(path_value)).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        scenarios = payload.get("scenarios") if isinstance(payload, dict) else payload
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("backtest requires a non-empty scenarios list or scenarios_path")
    if not all(isinstance(row, dict) for row in scenarios):
        raise ValueError("every backtest scenario must be an object")
    return scenarios


def _candidate_config(candidate: dict[str, Any], base_config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base_config)
    nested = candidate.get("config", {})
    if not isinstance(nested, dict):
        raise ValueError("backtest candidate config must be an object")
    merged.update(nested)
    weights = candidate.get("weights", candidate.get("weight_overrides", {}))
    if not isinstance(weights, dict):
        raise ValueError("backtest candidate weights must be an object")
    merged.update(weights)
    return merged


def _outcome_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, (int, float)):
        return {"net_points": float(value)}
    if not isinstance(value, dict):
        raise ValueError("each action outcome must be a number or object")
    return value


def _outcome_utility(outcome: dict[str, Any], objective: str) -> float:
    if objective == "net_points":
        value = outcome.get("net_points", outcome.get("points"))
    elif objective == "rank_utility":
        value = outcome.get("rank_utility")
    elif objective == "composite":
        value = (
            outcome.get("net_points", outcome.get("points", 0.0))
            + 0.15 * outcome.get("squad_value_delta", 0.0)
            + outcome.get("rank_utility", 0.0)
        )
    else:
        raise ValueError("objective must be net_points, rank_utility, or composite")
    if value is None:
        raise ValueError(f"action outcome is missing the field required for objective={objective}")
    return float(value)


def _find_outcome(outcomes: dict[str, Any], action_key: str) -> dict[str, Any]:
    aliases = [action_key]
    if action_key.startswith("transfer:"):
        aliases.append(action_key.removeprefix("transfer:"))
    for alias in aliases:
        if alias in outcomes:
            return _outcome_payload(outcomes[alias])
    raise ValueError(
        f"scenario is missing realized outcome for selected action {action_key!r}; "
        f"available keys: {sorted(outcomes)[:12]}"
    )


def _scenario_backtest(arguments: dict[str, Any]) -> dict[str, Any]:
    from fpl_lab.decision import recommend_transfers

    scenarios = _load_scenarios(arguments)
    objective = str(arguments.get("objective", "net_points"))
    base_config = arguments.get("base_config", {})
    if not isinstance(base_config, dict):
        raise ValueError("base_config must be an object")
    candidates = arguments.get("candidates") or [{"name": "provided_config", "config": {}}]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list")
    results = []
    for index, raw_candidate in enumerate(candidates):
        if not isinstance(raw_candidate, dict):
            raise ValueError("each backtest candidate must be an object")
        name = str(raw_candidate.get("name", f"candidate_{index + 1}"))
        details = []
        action_counts: dict[str, int] = {}
        utility_values = []
        net_points = []
        regrets = []
        for scenario_index, scenario in enumerate(scenarios):
            payload = dict(scenario.get("state", scenario))
            scenario_config = payload.get("config", {})
            if not isinstance(scenario_config, dict):
                raise ValueError(f"scenario {scenario_index} config must be an object")
            payload["config"] = _candidate_config(raw_candidate, {**base_config, **scenario_config})
            state = _state_from_arguments(payload)
            recommendations = recommend_transfers(
                state["current"], state["buyable"], state["signals"], state["config"]
            )
            if recommendations and recommendations[0].combined_score > 0.0:
                recommendation = recommendations[0]
                action_key = f"transfer:{recommendation.player_out_id}>{recommendation.player_in_id}"
                predicted_score = recommendation.combined_score
            else:
                action_key = "hold"
                predicted_score = 0.0
            outcomes = scenario.get("action_outcomes")
            if not isinstance(outcomes, dict):
                raise ValueError(f"scenario {scenario_index} requires an action_outcomes object")
            chosen_outcome = _find_outcome(outcomes, action_key)
            chosen_utility = _outcome_utility(chosen_outcome, objective)
            oracle_utility = max(
                _outcome_utility(_outcome_payload(value), objective)
                for value in outcomes.values()
            )
            action_counts[action_key] = action_counts.get(action_key, 0) + 1
            utility_values.append(chosen_utility)
            net_points.append(float(chosen_outcome.get("net_points", chosen_outcome.get("points", 0.0))))
            regrets.append(oracle_utility - chosen_utility)
            if bool(arguments.get("include_details", True)):
                details.append(
                    {
                        "scenario": scenario.get("id", scenario_index),
                        "gameweek": state["gameweek"],
                        "action": action_key,
                        "predicted_score": round(float(predicted_score), 4),
                        "realized_utility": round(float(chosen_utility), 4),
                        "realized_net_points": round(float(net_points[-1]), 4),
                        "oracle_utility": round(float(oracle_utility), 4),
                        "regret": round(float(regrets[-1]), 4),
                    }
                )
        results.append(
            {
                "name": name,
                "config": _candidate_config(raw_candidate, base_config),
                "scenarios": len(utility_values),
                "mean_utility": round(float(sum(utility_values) / len(utility_values)), 4),
                "total_utility": round(float(sum(utility_values)), 4),
                "mean_net_points": round(float(sum(net_points) / len(net_points)), 4),
                "hold_rate": round(float(action_counts.get("hold", 0) / len(utility_values)), 4),
                "mean_regret_to_oracle": round(float(sum(regrets) / len(regrets)), 4),
                "action_counts": action_counts,
                "details": details,
            }
        )
    results.sort(key=lambda row: (row["mean_utility"], -row["mean_regret_to_oracle"]), reverse=True)
    return {
        "kind": "scenario_replay",
        "objective": objective,
        "candidate_count": len(results),
        "scenario_count": len(scenarios),
        "winner": results[0]["name"],
        "results": results,
        "warnings": [
            "This replay is only as honest as the supplied point-in-time signals and realized action outcomes.",
            "Do not tune candidates and report the same season as an untouched test result; reserve a final season and starting squads.",
        ],
    }


def _season_backtest(arguments: dict[str, Any]) -> dict[str, Any]:
    from fpl_lab.player_models import load_vaastav_gameweeks
    from fpl_lab.simulator import build_season_data, build_signal_cache, simulate_season
    from fpl_lab.policy import CocktailConfig

    history_root = arguments.get("history_root")
    season = arguments.get("season")
    if not history_root or not season:
        raise ValueError("season backtest requires history_root and season")
    previous_season = arguments.get("previous_season")
    seasons = [str(season)] + ([str(previous_season)] if previous_season else [])
    raw = load_vaastav_gameweeks(history_root, seasons)
    current = build_season_data(raw, str(season))
    previous = build_season_data(raw, str(previous_season)) if previous_season else None
    signal_cache = build_signal_cache(current, previous)
    candidates = arguments.get("candidates") or [{"name": str(arguments.get("policy", "points_only_free_only"))}]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list")
    rows = []
    for index, raw_candidate in enumerate(candidates):
        if not isinstance(raw_candidate, dict):
            raise ValueError("each season backtest candidate must be an object")
        policy = str(raw_candidate.get("policy", arguments.get("policy", "points_only_free_only")))
        model = None
        if policy in {"neural", "cocktail"}:
            model, model_path, model_error = _load_model(raw_candidate.get("model_path", arguments.get("model_path")))
            if model is None:
                raise ValueError(model_error or "action-value model could not be loaded")
        cocktail_payload = raw_candidate.get("cocktail_config", arguments.get("cocktail_config", {}))
        if not isinstance(cocktail_payload, dict):
            raise ValueError("cocktail_config must be an object")
        cocktail_config = CocktailConfig(
            **{key: value for key, value in cocktail_payload.items() if key in CocktailConfig.__dataclass_fields__}
        )
        modes = raw_candidate.get("initial_squad_modes", arguments.get("initial_squad_modes", ["points"]))
        if not isinstance(modes, list) or not modes:
            raise ValueError("initial_squad_modes must be a non-empty list")
        for mode_index, mode in enumerate(modes):
            result = simulate_season(
                current,
                previous,
                policy=policy,
                initial_squad_mode=str(mode),
                initial_squad_seed=int(arguments.get("seed", 0)) + mode_index,
                signal_cache=signal_cache,
                neural_policy=model,
                start_gameweek=int(arguments.get("start_gameweek", 1)),
                end_gameweek=arguments.get("end_gameweek"),
                initial_free_transfers=int(arguments.get("initial_free_transfers", 0)),
                cocktail_config=cocktail_config,
            )
            rows.append(
                {
                    "candidate": str(raw_candidate.get("name", f"candidate_{index + 1}")),
                    "policy": policy,
                    "initial_squad_mode": str(mode),
                    "season": result.season,
                    "gross_points": result.gross_points,
                    "hit_points": result.hit_points,
                    "net_points": result.net_points,
                    "transfers": result.transfers,
                    "paid_transfers": result.paid_transfers,
                    "final_bank": result.final_bank,
                    "final_squad_value": result.final_squad_value,
                    "gameweeks": result.gameweeks,
                    "chip_uses": result.chip_uses,
                    "chips_remaining": list(result.chips_remaining),
                }
            )
    rows.sort(key=lambda row: row["net_points"], reverse=True)
    return {
        "kind": "season_simulator",
        "season": str(season),
        "history_root": str(history_root),
        "player_snapshot_gameweeks": len(current.snapshots_by_gw),
        "results": rows,
        "warnings": [
            "This is a historical simulator, not a guarantee of future performance.",
            "Use separate development and final test seasons/starting squads when tuning cocktail_config.",
            "Vaastav-format data is expected to be point-in-time player GW history; context/news ingestion is not inferred from future rows.",
        ],
    }


def _backtest_strategy(arguments: dict[str, Any]) -> dict[str, Any]:
    if arguments.get("history_root") or arguments.get("season"):
        return _season_backtest(arguments)
    return _scenario_backtest(arguments)


def _strategy_info() -> dict[str, Any]:
    from fpl_lab.league import HYBRID_PROFILES

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
            "live lineup selection is forecast-based; final autosubs depend on confirmed minutes and late team news",
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
            "current squad. buyable_players is optional: when omitted, the server "
            "loads the full current official player pool from the cached bootstrap "
            "snapshot/API. Give point-in-time signals or enable auto_official_signals, "
            "bank/free transfers, unused chips, and optional mini-league standings "
            "context. The default champion ranks hold and "
            "legal free transfers; explicit hybrid challengers can rank Wildcard, "
            "Free Hit, Bench Boost, and Triple Captain when model-backed. Use "
            "weight_overrides to tune the transparent decision layer per request."
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
                "buyable_players": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional. Omit to load every player from the official bootstrap pool.",
                },
                "signals": {"type": "array", "items": {"type": "object"}},
                "config": {"type": "object"},
                "weight_overrides": {
                    "type": "object",
                    "description": "Per-request DecisionConfig overrides, e.g. short_weight, long_weight, price_weight, ownership_weight, risk_aversion, rank_mode.",
                },
                "weights": {"type": "object", "description": "Alias for weight_overrides."},
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
                    "default": True,
                    "description": "If true, omitted signals are built from a local bootstrap snapshot or the official API. The server also falls back automatically when signals are omitted.",
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
            "required": ["gameweek", "current_squad"],
        },
    },
    {
        "name": "fpl_lineup_plan",
        "description": "Choose the legal current-week formation, starting XI, bench order, captain and vice-captain from the supplied 15-player squad. It also reports the projected bench points that Bench Boost would add; normally only the XI scores.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "gameweek": {"type": "integer", "minimum": 1, "maximum": 38},
                "current_squad": {"type": "array", "items": {"type": "object"}},
                "buyable_players": {"type": "array", "items": {"type": "object"}, "description": "Optional; omit to use the official bootstrap universe."},
                "signals": {"type": "array", "items": {"type": "object"}},
                "auto_official_signals": {"type": "boolean", "default": True},
                "bootstrap_path": {"type": "string"},
                "fetch_official": {"type": "boolean", "default": False},
            },
            "required": ["gameweek", "current_squad"],
        },
    },
    {
        "name": "fpl_search_players",
        "description": "Search the full cached official FPL player pool by name, position, team, price, and availability. Use this to construct or inspect buyable candidates; it returns current price, official expected points, form, ownership, status, and transfer signals.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Name, player ID, or team text."},
                "position": {"type": "string", "enum": ["GKP", "DEF", "MID", "FWD"]},
                "team": {"type": "string", "description": "Team ID or team name fragment."},
                "min_price": {"type": "number"},
                "max_price": {"type": "number"},
                "available_only": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                "bootstrap_path": {"type": "string"},
                "fetch_official": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "fpl_forecast_signals",
        "description": "Return the point-in-time signal table for the supplied squad and buyable pool, including short/long expected points, future price signals, minutes, risk, news/social context, ownership leverage, and uncertainty. Use it to inspect the underlying inputs before asking for a decision.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "gameweek": {"type": "integer", "minimum": 1, "maximum": 38},
                "current_squad": {"type": "array", "items": {"type": "object"}},
                "buyable_players": {"type": "array", "items": {"type": "object"}, "description": "Optional; omit to include the full official pool."},
                "signals": {"type": "array", "items": {"type": "object"}},
                "auto_official_signals": {"type": "boolean", "default": True},
                "bootstrap_path": {"type": "string"},
                "fetch_official": {"type": "boolean", "default": False},
                "config": {"type": "object"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            "required": ["gameweek", "current_squad"],
        },
    },
    {
        "name": "fpl_score_moves",
        "description": "Rank legal one-transfer moves under explicit transparent weights. This is the tuning and explanation tool for short/long points, price economics, ownership leverage, availability, news risk, uncertainty, hit cost, and risk aversion; use fpl_recommend_moves for the final action including chips and the learned policy.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "gameweek": {"type": "integer", "minimum": 1, "maximum": 38},
                "current_squad": {"type": "array", "items": {"type": "object"}},
                "buyable_players": {"type": "array", "items": {"type": "object"}, "description": "Optional; omit to load the full official pool."},
                "signals": {"type": "array", "items": {"type": "object"}},
                "config": {"type": "object"},
                "weight_overrides": {"type": "object"},
                "auto_official_signals": {"type": "boolean", "default": True},
                "bootstrap_path": {"type": "string"},
                "fetch_official": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            "required": ["gameweek", "current_squad"],
        },
    },
    {
        "name": "fpl_strategy_info",
        "description": "Return the selected strategy, benchmark status, research sources, and known limitations.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "fpl_strategy_catalog",
        "description": "Return all available strategy profiles, action kinds, default decision/cocktail configs, signal components, and the fields that can be tuned per request or in a backtest.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "fpl_backtest_strategy",
        "description": "Backtest and compare tunable strategies. Use scenarios/scenarios_path for point-in-time replay with realized action_outcomes, or history_root plus season for the legal Vaastav-format season simulator. Returns mean utility, points, hold rate, regret to the supplied oracle, action counts, and per-scenario details.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scenarios": {"type": "array", "items": {"type": "object"}},
                "scenarios_path": {"type": "string"},
                "candidates": {"type": "array", "items": {"type": "object"}},
                "base_config": {"type": "object"},
                "objective": {"type": "string", "enum": ["net_points", "rank_utility", "composite"], "default": "net_points"},
                "include_details": {"type": "boolean", "default": True},
                "history_root": {"type": "string", "description": "Root of Vaastav-format season folders when running the full simulator."},
                "season": {"type": "string"},
                "previous_season": {"type": "string"},
                "policy": {"type": "string", "enum": ["hold", "points_only", "points_only_free_only", "price_aware", "chase", "neural", "cocktail"]},
                "initial_squad_modes": {"type": "array", "items": {"type": "string", "enum": ["points", "value", "template", "randomized_points"]}},
                "cocktail_config": {"type": "object"},
                "start_gameweek": {"type": "integer", "minimum": 1, "maximum": 38},
                "end_gameweek": {"type": "integer", "minimum": 1, "maximum": 38},
                "seed": {"type": "integer", "default": 0},
                "model_path": {"type": "string"},
            },
        },
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
                "instructions": "Use fpl_recommend_moves with a current 15-player squad; buyable_players and signals are optional because the server can load the full official pool. Use fpl_strategy_catalog and fpl_backtest_strategy to inspect and tune strategies.",
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
            elif name == "fpl_lineup_plan":
                payload = _lineup_plan(arguments)
            elif name == "fpl_search_players":
                payload = _search_players(arguments)
            elif name == "fpl_forecast_signals":
                payload = _forecast_signals(arguments)
            elif name == "fpl_score_moves":
                payload = _score_moves(arguments)
            elif name == "fpl_strategy_info":
                payload = _strategy_info()
            elif name == "fpl_strategy_catalog":
                payload = _strategy_catalog()
            elif name == "fpl_backtest_strategy":
                payload = _backtest_strategy(arguments)
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
    if os.environ.get("FPL_MCP_QUIET") != "1":
        print(
            f"FPL Strategy MCP {SERVER_VERSION} ready on stdio "
            f"(model asset: {'present' if _asset_path('action-policy-model.joblib').exists() else 'missing'})",
            file=sys.stderr,
            flush=True,
        )
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
    """Start the server, inspect health, or configure local MCP clients."""

    import argparse

    parser = argparse.ArgumentParser(description="FPL Strategy MCP server")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("serve", "status", "setup", "version"),
        default="serve",
        help="serve the MCP (default), print status, configure clients, or print the version",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport; stdio is the default for local desktop clients",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument(
        "--clients",
        default=os.environ.get("FPL_STRATEGY_CLIENTS", "all"),
        help="comma-separated clients for setup: claude, claude-code, codex, or all",
    )
    parser.add_argument("--json", action="store_true", help="print status/setup output as JSON")
    parser.add_argument("--deep", action="store_true", help="load and validate the bundled model during status")
    parser.add_argument("--quiet", action="store_true", help="suppress the stdio readiness line")
    args = parser.parse_args(argv)
    if args.command == "version":
        print(SERVER_VERSION)
        return
    if args.command == "status":
        if args.deep:
            print("Checking bundled model, data snapshot, and client configuration...", file=sys.stderr, flush=True)
        else:
            print("Checking installed assets and client configuration...", file=sys.stderr, flush=True)
        payload = _status_payload(deep=args.deep)
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            _print_status(payload)
        if payload["status"] != "ready":
            raise SystemExit(1)
        return
    if args.command == "setup":
        try:
            print("Starting MCP client setup...", file=sys.stderr, flush=True)
            payload = _setup_clients(args.clients)
        except Exception as exc:
            if args.json:
                print(json.dumps({"status": "error", "error": str(exc)}))
            else:
                print(f"Setup failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            _print_setup(payload)
        if any(result.get("status") == "error" for key, result in payload.items() if isinstance(result, dict) and key != "command"):
            raise SystemExit(1)
        return
    if args.quiet:
        os.environ["FPL_MCP_QUIET"] = "1"
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
