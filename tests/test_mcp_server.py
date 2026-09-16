import unittest

from fpl_strategy_mcp.server import _dispatch, _load_model, _strategy_info


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


if __name__ == "__main__":
    unittest.main()

