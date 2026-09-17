from pathlib import Path

from fpl_lab.elite_managers import (
    elite_behavior_priors,
    load_elite_manager_autopsy,
    load_observed_manager_transitions,
    summarize_manager_behavior,
    summarize_observed_manager_transitions,
)


def test_elite_manager_benchmark_loads_and_exposes_behavior_priors() -> None:
    path = Path("data/elite_managers/autopsy_all.csv")
    frame = load_elite_manager_autopsy(path)
    assert len(frame) >= 24_000
    assert frame["entry"].is_unique

    summary = summarize_manager_behavior(frame)
    priors = elite_behavior_priors(frame)
    assert summary["top100"]["manager_count"] >= 100
    assert priors["manager_count"] >= 10_000
    assert priors["median_hits_per_season"] >= 0.0
    assert priors["median_cap_agreement"] >= 0.0


def test_observed_manager_archive_has_38_point_in_time_rows_per_manager() -> None:
    frame = load_observed_manager_transitions(
        "data/elite_managers/season_winners_2025-26",
        "data/elite_managers/manager_index.csv",
    )
    assert frame["manager_id"].nunique() == 100
    assert len(frame) == 100 * 38
    assert frame.groupby("manager_id")["gameweek"].nunique().eq(38).all()
    assert frame["squad_ids"].map(len).eq(15).all()
    assert frame["season_final_rank"].notna().all()
    summary = summarize_observed_manager_transitions(frame)
    assert summary["observed_action_rows"] == 100 * 37
    assert summary["by_rank_band"]["top100"]["manager_count"] > 0
