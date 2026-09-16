#!/usr/bin/env python3
"""Score legal FPL transfer moves from a current-team decision input."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpl_lab.decision import assess_current_squad, load_decision_input, recommendation_to_dict, recommend_transfers
from fpl_lab.policy import PolicyAction, PolicyState
from fpl_lab.simulator import recommendation_to_policy_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON file containing current squad, buyable pool and model signals")
    parser.add_argument("--out", default="runs/transfer-strategy/strategy.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--neural-model", default=None, help="optional trained ActionValueEnsemble joblib path")
    parser.add_argument("--gameweek", type=int, default=1, help="current GW used by the neural state encoder")
    args = parser.parse_args()

    current, buyable, signals, config = load_decision_input(args.input)
    all_recommendations = recommend_transfers(current, buyable, signals, config)
    recommendations = all_recommendations[: args.limit]
    result = {
        "input": str(Path(args.input)),
        "current_squad_size": len(current),
        "buyable_pool_size": len(buyable),
        "signal_count": len(signals),
        "config": config.__dict__,
        "recommendations": [recommendation_to_dict(recommendation) for recommendation in recommendations],
        "hold_assessment": assess_current_squad(current, all_recommendations),
        "note": "Point forecasts are one factor inside a constrained short-/long-horizon decision score.",
    }
    if args.neural_model:
        neural_model = joblib.load(args.neural_model)
        # Rebuild the legal candidate set with a permissive threshold. The
        # ordinary output remains the transparent heuristic ranking; the
        # learned ranking is reported alongside it until it beats the frozen
        # benchmark in a future retraining cycle.
        candidate_config = replace(config, min_move_score=-10.0)
        candidates = recommend_transfers(current, buyable, signals, candidate_config)
        signal_by_id = {signal.player_id: signal for signal in signals}
        state = PolicyState(
            gameweek=args.gameweek,
            weeks_remaining=max(0, 38 - args.gameweek + 1),
            bank=config.bank,
            free_transfers=config.free_transfers,
            squad_value=sum(player.price for player in current),
            rank_mode=config.rank_mode,
        )
        action_rows = [(PolicyAction(kind="hold"), None)]
        for recommendation in candidates:
            action_rows.append(
                (recommendation_to_policy_action(recommendation, signal_by_id), recommendation)
            )
        ranked = neural_model.rank_actions(state, [action for action, _ in action_rows], risk_aversion=0.20)
        recommendation_by_action = {action: recommendation for action, recommendation in action_rows}
        ranked_actions = []
        for action, score in ranked[: args.limit]:
            recommendation = recommendation_by_action.get(action)
            ranked_actions.append(
                {
                    "policy_score": score,
                    "action": action.__dict__,
                    "recommendation": recommendation_to_dict(recommendation)
                    if recommendation is not None
                    else None,
                }
            )
        result["neural_policy"] = {
            "model": str(Path(args.neural_model)),
            "gameweek": args.gameweek,
            "candidate_count": len(candidates),
            "ranked_actions": ranked_actions,
            "model_scope_note": "trained on full legal squads and historical simulator states; verify the supplied 15-player input before using live",
        }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
