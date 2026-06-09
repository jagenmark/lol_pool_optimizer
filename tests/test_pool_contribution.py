from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pool_contribution import build_pool_contribution_report


def test_pool_contributions_select_best_posterior_matchup_and_sum_to_score() -> None:
    comparison = pd.DataFrame(
        {
            "pool": ["A, B"],
            "raw_rank": pd.Series([1], dtype="Int64"),
            "eb_rank": pd.Series([1], dtype="Int64"),
            "posterior_mean_rank": pd.Series([1], dtype="Int64"),
        }
    )
    frequencies = pd.DataFrame(
        {"champion_j": ["X", "Y"], "freq_j": [0.75, 0.25]}
    )
    posterior = pd.DataFrame(
        {
            "champion": ["A", "B", "A", "B"],
            "enemy_champion": ["X", "X", "Y", "Y"],
            "raw_winrate": [0.55, 0.50, 0.40, 0.60],
            "posterior_mean": [0.54, 0.51, 0.45, 0.58],
            "posterior_lower_5": [0.50, 0.47, 0.40, 0.54],
            "posterior_upper_95": [0.58, 0.55, 0.50, 0.62],
            "games": [100, 100, 100, 100],
        }
    )

    report = build_pool_contribution_report(
        comparison, frequencies, posterior
    )
    assert report["best_pool_champion_against_enemy"].tolist() == ["A", "B"]
    assert np.isclose(report["weighted_contribution"].sum(), 0.75 * 0.54 + 0.25 * 0.58)
    assert np.isclose(report["enemy_frequency"].sum(), 1.0)
