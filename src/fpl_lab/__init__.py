"""Maintained local FPL modeling package reconstructed from the ChatGPT handoff."""

from .backtest import run_temporal_backtest
from .data import Match, TeamObservation, load_matches, to_team_observations
from .model import PoissonTeamGoalsModel

__all__ = [
    "Match",
    "PoissonTeamGoalsModel",
    "TeamObservation",
    "load_matches",
    "run_temporal_backtest",
    "to_team_observations",
]
