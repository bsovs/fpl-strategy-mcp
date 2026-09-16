import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_lab.backtest import paired_match_bootstrap, run_temporal_backtest
from fpl_lab.context import ContextStore
from fpl_lab.data import Match, load_matches, to_team_observations
from fpl_lab.decision import DecisionConfig, PlayerSignal, PlayerState, recommend_transfers
from fpl_lab.model import PoissonTeamGoalsModel
from fpl_lab.player_models import build_extended_player_feature_table, build_player_feature_table
from fpl_lab.policy import ActionValueEnsemble, ActionValueMLP, PolicyAction, PolicyState, encode_state_action
from fpl_lab.simulator import season_rules, selling_price_tenths


def synthetic_matches() -> list[Match]:
    rows = []
    fixtures = [
        (1, "A", "B", 2, 0),
        (1, "C", "D", 1, 1),
        (1, "E", "F", 0, 1),
        (2, "A", "C", 1, 2),
        (2, "B", "E", 2, 1),
        (2, "D", "F", 0, 0),
        (3, "A", "D", 3, 1),
        (3, "B", "F", 1, 1),
        (3, "C", "E", 2, 0),
    ]
    for number, (gameweek, home, away, home_goals, away_goals) in enumerate(fixtures, start=1):
        rows.append(Match(f"m{number}", gameweek, f"2026-08-{number:02d}", home, away, home_goals, away_goals))
    return rows


class FplLabTests(unittest.TestCase):
    def test_observation_grain_is_two_rows_per_match(self):
        observations = to_team_observations(synthetic_matches())
        self.assertEqual(len(observations), 18)
        self.assertEqual({row.home for row in observations}, {0, 1})

    def test_model_fits_and_exports_coefficients(self):
        model = PoissonTeamGoalsModel(alpha=1.0).fit(to_team_observations(synthetic_matches()[:6]))
        predictions = model.predict(to_team_observations(synthetic_matches()[:3]))
        self.assertEqual(predictions.shape, (6,))
        self.assertTrue(np.all(predictions > 0))
        self.assertEqual(len(model.to_dict()["coefficients"]), 13)

    def test_backtest_is_temporal_and_deterministic(self):
        matches = synthetic_matches()
        first = run_temporal_backtest(matches, test_gameweek=3, n_bootstrap=1000)
        second = run_temporal_backtest(matches, test_gameweek=3, n_bootstrap=1000)
        self.assertEqual(first.selected_alpha, second.selected_alpha)
        self.assertEqual(first.bootstrap, second.bootstrap)
        self.assertEqual(first.test_gameweek, 3)

    def test_csv_loader_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matches.csv"
            path.write_text(
                "match_id,gameweek,date,home_team,away_team,home_goals,away_goals\n"
                "m1,1,2026-08-01,A,B,1,0\n"
                "m1,1,2026-08-02,C,D,0,0\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_matches(path)

    def test_player_history_loader_falls_back_to_latin1(self):
        from fpl_lab.player_models import load_vaastav_gameweeks

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2020-21" / "gws"
            path.mkdir(parents=True)
            (path / "gw1.csv").write_bytes(
                b"name,position,team,opponent_team,kickoff_time,was_home,total_points\n"
                b"Jos\xe9, MID, 1, 2, 2020-09-01T12:00:00Z, True, 2\n"
            )
            loaded = load_vaastav_gameweeks(directory, ["2020-21"])
            self.assertEqual(loaded.iloc[0]["name"], "José")

    def test_player_history_loader_restores_early_metadata_with_flags(self):
        from fpl_lab.player_models import load_vaastav_gameweeks

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2016-17" / "gws"
            path.mkdir(parents=True)
            (path / "gw1.csv").write_text(
                "name,element,opponent_team,kickoff_time,was_home,total_points\n"
                "Player,7,2,2016-08-01T12:00:00Z,True,6\n",
                encoding="utf-8",
            )
            (Path(directory) / "2016-17" / "players_raw.csv").write_text(
                "id,element_type,team\n7,3,11\n",
                encoding="utf-8",
            )
            loaded = load_vaastav_gameweeks(directory, ["2016-17"])
            self.assertEqual(loaded.iloc[0]["position"], "MID")
            self.assertEqual(loaded.iloc[0]["team"], 11)
            self.assertEqual(loaded.iloc[0]["metadata_position_imputed"], 1.0)
            self.assertEqual(loaded.iloc[0]["metadata_team_imputed"], 1.0)

    def test_player_features_are_lagged_and_archive_labels_are_normalized(self):
        raw = pd.DataFrame(
            [
                {"name": "Keeper", "position": "GKP", "team": 1, "opponent_team": 2, "kickoff_time": "2024-08-01T12:00:00Z", "was_home": True, "total_points": 2, "season": "2024-25", "season_order": 2024, "gameweek": 1},
                {"name": "Keeper", "position": "GKP", "team": 1, "opponent_team": 3, "kickoff_time": "2024-08-08T12:00:00Z", "was_home": False, "total_points": 5, "season": "2024-25", "season_order": 2024, "gameweek": 2},
                {"name": "Manager", "position": "AM", "team": 1, "opponent_team": 2, "kickoff_time": "2024-08-01T12:00:00Z", "was_home": True, "total_points": 0, "season": "2024-25", "season_order": 2024, "gameweek": 1},
            ]
        )
        features = build_player_feature_table(raw)
        self.assertEqual(set(features["position"]), {"GK"})
        self.assertTrue(pd.isna(features.iloc[0]["total_points_last"]))
        self.assertEqual(features.iloc[1]["total_points_last"], 2)

    def test_extended_features_block_double_gameweek_result_leakage(self):
        raw = pd.DataFrame(
            [
                {"name": "Player", "position": "MID", "team": 1, "opponent_team": 2, "kickoff_time": "2024-08-01T12:00:00Z", "was_home": True, "total_points": 15, "team_h_score": 3, "team_a_score": 0, "season": "2024-25", "season_order": 2024, "gameweek": 1, "element": 7},
                {"name": "Player", "position": "MID", "team": 1, "opponent_team": 3, "kickoff_time": "2024-08-04T12:00:00Z", "was_home": False, "total_points": 1, "team_h_score": 0, "team_a_score": 0, "season": "2024-25", "season_order": 2024, "gameweek": 1, "element": 7},
                {"name": "Player", "position": "MID", "team": 1, "opponent_team": 4, "kickoff_time": "2024-08-10T12:00:00Z", "was_home": True, "total_points": 4, "team_h_score": 1, "team_a_score": 0, "season": "2024-25", "season_order": 2024, "gameweek": 2, "element": 7},
            ]
        )
        features = build_extended_player_feature_table(raw)
        gw1 = features[features["gameweek"] == 1]
        self.assertTrue(gw1["points_mean_5"].isna().all())
        self.assertTrue(gw1["total_points_last"].isna().all())
        self.assertEqual(features.iloc[-1]["total_points_last"], 1)

    def test_extended_features_carry_prior_season_player_history_without_leakage(self):
        raw = pd.DataFrame(
            [
                {"name": "Player", "element": 7, "position": "MID", "team": 1, "opponent_team": 2, "kickoff_time": "2024-05-01T12:00:00Z", "was_home": True, "total_points": 9, "season": "2023-24", "season_order": 2023, "gameweek": 38},
                {"name": "Player", "element": 7, "position": "MID", "team": 1, "opponent_team": 3, "kickoff_time": "2024-08-01T12:00:00Z", "was_home": False, "total_points": 2, "season": "2024-25", "season_order": 2024, "gameweek": 1},
            ]
        )
        features = build_extended_player_feature_table(raw)
        opening = features[features["season"] == "2024-25"].iloc[0]
        self.assertTrue(pd.isna(opening["total_points_last"]))
        self.assertEqual(opening["career_total_points_last"], 9)
        self.assertEqual(opening["career_games_before"], 1)

    def test_decision_layer_scores_legal_same_position_move(self):
        current = [PlayerState("out", "Out", "MID", "A", 7.0, 6.9)]
        buyable = [PlayerState("in", "In", "MID", "B", 6.8)]
        signals = [
            PlayerSignal("out", 3.5, 24.0, role_security=0.5),
            PlayerSignal("in", 5.5, 34.0, short_fixture_delta=0.8, long_fixture_delta=0.5, role_security=0.9),
        ]
        recommendations = recommend_transfers(
            current,
            buyable,
            signals,
            DecisionConfig(free_transfers=1, bank=0.0, min_move_score=0.1),
        )
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0].timing, "now")
        self.assertIn("stronger fixture run", recommendations[0].why)

    def test_decision_layer_rejects_unaffordable_candidate(self):
        current = [PlayerState("out", "Out", "MID", "A", 7.0, 6.9)]
        buyable = [PlayerState("in", "In", "MID", "B", 8.0)]
        signals = [PlayerSignal("out", 3.0, 20.0), PlayerSignal("in", 8.0, 45.0)]
        recommendations = recommend_transfers(
            current,
            buyable,
            signals,
            DecisionConfig(free_transfers=1, bank=0.0),
        )
        self.assertEqual(recommendations, [])

    def test_action_value_policy_uses_state_and_action_features(self):
        state = PolicyState(gameweek=10, weeks_remaining=28, bank=1.0, free_transfers=1, squad_value=100.0)
        hold = PolicyAction(kind="hold")
        transfer = PolicyAction(kind="transfer", hit_cost=0.0, short_points_delta=1.0, long_points_delta=3.0)
        self.assertGreater(len(encode_state_action(state, transfer)), 20)
        policy = ActionValueMLP(hidden_layer_sizes=(8,))
        policy.fit([state, state, state], [hold, transfer, hold], [0.0, 2.0, 0.1])
        ranked = policy.rank_actions(state, [hold, transfer])
        self.assertEqual(ranked[0][0].kind, "transfer")

    def test_action_value_ensemble_fits_and_ranks(self):
        state = PolicyState(gameweek=10, weeks_remaining=28, bank=1.0, free_transfers=1, squad_value=100.0)
        hold = PolicyAction(kind="hold")
        transfer = PolicyAction(kind="transfer", short_points_delta=1.0, long_points_delta=3.0)
        model = ActionValueEnsemble(n_models=2, hidden_layer_sizes=(8,), random_state=7)
        model.fit([state, state, state, state], [hold, transfer, hold, transfer], [0.0, 2.0, 0.1, 2.1])
        ranked = model.rank_actions(state, [hold, transfer])
        self.assertEqual(ranked[0][0].kind, "transfer")
        means, uncertainty = model.predict_with_uncertainty([state], [transfer])
        self.assertEqual(means.shape, (1,))
        self.assertEqual(uncertainty.shape, (1,))

    def test_context_features_respect_as_of_and_expiry(self):
        store = ContextStore.from_records(
            [
                {
                    "event_id": "past-injury",
                    "published_at": "2026-09-01T10:00:00Z",
                    "expires_at": "2026-09-05T10:00:00Z",
                    "source": "official FPL",
                    "title": "Hamstring injury",
                    "player_id": "7",
                    "player_name": "Player",
                    "team": "1",
                    "event_type": "injury",
                },
                {
                    "event_id": "future-return",
                    "published_at": "2026-09-10T10:00:00Z",
                    "source": "official FPL",
                    "title": "Back in training",
                    "player_id": "7",
                    "player_name": "Player",
                    "team": "1",
                    "event_type": "role_positive",
                },
            ],
            kind="news",
        )
        before = store.features_for_player("7", "Player", "1", "2026-09-02T12:00:00Z")
        after_expiry = store.features_for_player("7", "Player", "1", "2026-09-06T12:00:00Z")
        before_future_event = store.features_for_player("7", "Player", "1", "2026-09-08T12:00:00Z")
        self.assertGreater(before.news_risk, 0.0)
        self.assertEqual(after_expiry.event_count, 0)
        self.assertEqual(before_future_event.event_count, 0)

    def test_selling_price_rounds_profit_down(self):
        self.assertEqual(selling_price_tenths(75, 78), 76)
        self.assertEqual(selling_price_tenths(75, 79), 77)
        self.assertEqual(selling_price_tenths(75, 74), 74)

    def test_historical_free_transfer_caps(self):
        self.assertEqual(season_rules("2023-24").free_transfer_cap, 2)
        self.assertEqual(season_rules("2024-25").free_transfer_cap, 5)


if __name__ == "__main__":
    unittest.main()
