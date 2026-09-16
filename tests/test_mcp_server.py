import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fpl_strategy_mcp.server import (
    _configure_claude_desktop,
    _dispatch,
    _load_model,
    _parse_clients,
    _strategy_info,
)


class MCPServerTests(unittest.TestCase):
    def test_initialize_and_tools_list(self):
        initialized = _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertEqual(initialized["result"]["serverInfo"]["name"], "fpl-strategy")

        tools = _dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        names = {tool["name"] for tool in tools["result"]["tools"]}
        self.assertEqual(names, {"fpl_recommend_moves", "fpl_strategy_info"})

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


if __name__ == "__main__":
    unittest.main()
