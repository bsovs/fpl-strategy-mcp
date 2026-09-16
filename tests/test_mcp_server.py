import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fpl_strategy_mcp.server import (
    _configure_claude_desktop,
    _configure_codex_config,
    _dispatch,
    _load_model,
    _parse_clients,
    _search_players,
    _state_from_arguments,
    _strategy_catalog,
    _strategy_info,
)


class MCPServerTests(unittest.TestCase):
    def test_initialize_and_tools_list(self):
        initialized = _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertEqual(initialized["result"]["serverInfo"]["name"], "fpl-strategy")

        tools = _dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        names = {tool["name"] for tool in tools["result"]["tools"]}
        self.assertEqual(
            names,
            {
                "fpl_recommend_moves",
                "fpl_search_players",
                "fpl_forecast_signals",
                "fpl_score_moves",
                "fpl_strategy_info",
                "fpl_strategy_catalog",
                "fpl_backtest_strategy",
            },
        )

    def test_bundled_strategy_metadata_and_model(self):
        info = _strategy_info()
        self.assertEqual(info["default_strategy"], "champion")
        self.assertEqual(info["action_cocktail_champion"]["candidate"], "baseline_free")
        model, path, error = _load_model()
        self.assertIsNotNone(model)
        self.assertTrue(path.endswith("action-policy-model.joblib"))
        self.assertIsNone(error)

    def test_setup_preserves_claude_config_and_creates_backup(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "claude_desktop_config.json"
            path.write_text('{"preferences": {"theme": "dark"}}\n', encoding="utf-8")
            with patch("fpl_strategy_mcp.server._claude_desktop_config_path", return_value=path):
                result = _configure_claude_desktop(["/tmp/fpl-strategy-mcp"])
            self.assertEqual(result["status"], "configured")
            self.assertTrue(result["backup"])
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["preferences"]["theme"], "dark")
            self.assertEqual(payload["mcpServers"]["fpl-strategy"]["command"], "/tmp/fpl-strategy-mcp")

    def test_client_aliases(self):
        self.assertEqual(_parse_clients("all"), {"claude", "codex"})
        self.assertEqual(_parse_clients("claude-desktop,codex"), {"claude", "codex"})

    def test_player_search_uses_full_official_pool(self):
        result = _search_players({"limit": 3})
        self.assertGreater(result["pool_size"], 100)
        self.assertLessEqual(len(result["players"]), 3)

    def test_omitted_buyable_pool_is_hydrated_from_bootstrap(self):
        players = _search_players({"limit": 15})["players"]
        state = _state_from_arguments(
            {
                "gameweek": 1,
                "current_squad": [{"player_id": row["player_id"]} for row in players],
                "auto_official_signals": True,
                "config": {"free_transfers": 1},
            }
        )
        self.assertEqual(len(state["current"]), 15)
        self.assertGreater(len(state["buyable"]), 100)
        self.assertEqual(len(state["signals"]), len(state["current"]) + len(state["buyable"]))
        self.assertTrue(any("full official player pool" in warning for warning in state["state_warnings"]))

    def test_strategy_catalog_exposes_tuning_contract(self):
        catalog = _strategy_catalog()
        self.assertEqual(catalog["primary_tool"], "fpl_recommend_moves")
        self.assertIn("short_weight", catalog["tunable_fields"])
        self.assertIn("hybrid_win", catalog["strategies"])
        self.assertIn("model_weight", catalog["default_cocktail_config"])

    def test_scenario_backtest_compares_weight_candidates(self):
        from fpl_strategy_mcp.server import _backtest_strategy

        positions = ["GK", "GK", *(["DEF"] * 5), *(["MID"] * 5), *(["FWD"] * 3)]
        current = [
            {
                "player_id": f"p{index}",
                "name": f"Player {index}",
                "position": position,
                "team": "A",
                "price": 5.0,
                "selling_price": 5.0,
            }
            for index, position in enumerate(positions)
        ]
        buyable = [{"player_id": "in", "name": "In", "position": "MID", "team": "B", "price": 5.0}]
        signals = [
            {
                "player_id": player["player_id"],
                "short_expected_points": 1.0,
                "long_expected_points": 1.0,
            }
            for player in current
        ]
        signals.append({"player_id": "in", "short_expected_points": 5.0, "long_expected_points": 8.0})
        result = _backtest_strategy(
            {
                "scenarios": [
                    {
                        "id": "gw1",
                        "gameweek": 1,
                        "current_squad": current,
                        "buyable_players": buyable,
                        "signals": signals,
                        "config": {"free_transfers": 1},
                        "action_outcomes": {
                            "hold": {"net_points": 1.0},
                            "transfer:p7>in": {"net_points": 4.0},
                        },
                    }
                ],
                "candidates": [
                    {"name": "short", "weights": {"short_weight": 1.0, "long_weight": 0.0}},
                    {"name": "long", "weights": {"short_weight": 0.0, "long_weight": 1.0}},
                ],
            }
        )
        self.assertEqual(result["kind"], "scenario_replay")
        self.assertEqual(result["winner"], "short")
        self.assertEqual(len(result["results"]), 2)

    def test_setup_writes_codex_toml_and_replaces_old_entry(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            config.write_text(
                '[mcp_servers.other]\ncommand = "other"\n\n'
                '[mcp_servers.fpl-strategy]\ncommand = "old"\n',
                encoding="utf-8",
            )
            with patch("fpl_strategy_mcp.server._codex_config_path", return_value=config):
                result = _configure_codex_config(["/tmp/fpl-strategy-mcp"])
            self.assertEqual(result["status"], "configured")
            content = config.read_text(encoding="utf-8")
            self.assertIn('[mcp_servers.other]', content)
            self.assertIn('command = "/tmp/fpl-strategy-mcp"', content)
            self.assertNotIn('command = "old"', content)


if __name__ == "__main__":
    unittest.main()
