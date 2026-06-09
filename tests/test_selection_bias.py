from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optimizer import rank_top_pools
from selection_bias import (
    build_best_pool_matchup_detail,
    build_favorable_selection_summary,
    build_pool_dependency,
    compute_matchup_enrichment,
)


def test_matchup_enrichment_and_favorable_selection_formula() -> None:
    matchups = pd.DataFrame(
        {
            "champion_i": ["A", "A"],
            "champion_j": ["X", "Y"],
            "games_ij": [80.0, 20.0],
        }
    )
    frequencies = pd.DataFrame(
        {"champion_j": ["X", "Y"], "freq_j": [0.5, 0.5]}
    )
    posterior = pd.DataFrame(
        {
            "champion": ["A", "A"],
            "enemy_champion": ["X", "Y"],
            "games": [80.0, 20.0],
            "raw_winrate": [0.6, 0.4],
            "posterior_mean": [0.6, 0.4],
            "posterior_lower_5": [0.5, 0.3],
            "posterior_upper_95": [0.7, 0.5],
        }
    )
    summary = pd.DataFrame(
        {
            "champion_name": ["A"],
            "pickrate": [0.1],
            "winrate": [0.56],
            "total_games": [125.0],
        }
    )

    enrichment = compute_matchup_enrichment(
        matchups, frequencies, posterior, summary
    )
    x_row = enrichment[enrichment["enemy_champion"] == "X"].iloc[0]
    y_row = enrichment[enrichment["enemy_champion"] == "Y"].iloc[0]
    assert np.isclose(x_row["conditional_enemy_frequency"], 0.8)
    assert np.isclose(y_row["conditional_enemy_frequency"], 0.2)
    assert np.isclose(x_row["expected_matchup_games"], 50.0)
    assert np.isclose(x_row["enrichment_ratio"], 1.6)
    assert np.isclose(y_row["enrichment_ratio"], 0.4)

    favorable = build_favorable_selection_summary(enrichment).iloc[0]
    assert np.isclose(favorable["selection_advantage"], 0.06)
    assert np.isclose(favorable["recorded_matchup_coverage"], 0.8)


def test_best_pool_marginal_lift_uses_second_best_champion() -> None:
    frequencies = pd.DataFrame({"champion_j": ["X"], "freq_j": [1.0]})
    lookup = {("A", "X"): 0.60, ("B", "X"): 0.55}
    posterior = pd.DataFrame(
        {
            "champion": ["A", "B"],
            "enemy_champion": ["X", "X"],
            "games": [500.0, 500.0],
            "posterior_mean": [0.60, 0.55],
            "posterior_lower_5": [0.56, 0.51],
            "posterior_upper_95": [0.64, 0.59],
        }
    )

    detail = build_best_pool_matchup_detail(
        ("A", "B"), frequencies, lookup, posterior
    )
    assert detail.loc[0, "champion"] == "A"
    assert np.isclose(detail.loc[0, "best_pool_marginal_lift"], 0.05)


def test_pool_dependency_reports_exact_exclusion_loss() -> None:
    candidates = ["A", "B", "C"]
    frequencies = pd.DataFrame({"champion_j": ["X"], "freq_j": [1.0]})
    lookup = {
        ("A", "X"): 0.60,
        ("B", "X"): 0.55,
        ("C", "X"): 0.50,
    }
    ranked = rank_top_pools(
        candidates,
        1,
        frequencies,
        lookup,
        top_n=3,
    )

    dependency = build_pool_dependency(
        candidates,
        1,
        frequencies,
        lookup,
        ranked,
    )
    a_row = dependency[
        (dependency["scenario"] == "single_champion")
        & (dependency["champion"] == "A")
    ].iloc[0]
    c_row = dependency[
        (dependency["scenario"] == "single_champion")
        & (dependency["champion"] == "C")
    ].iloc[0]
    assert np.isclose(a_row["score_drop"], 0.05)
    assert np.isclose(c_row["score_drop"], 0.0)
    assert a_row["top_pool_appearances"] == 1
