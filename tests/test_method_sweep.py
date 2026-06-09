from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from method_sweep import (
    build_offmeta_penalties,
    build_pool_matrix,
    fixed_policy_analytic_pool_stats,
    score_pool_matrix,
)
from scoring import pool_score


def test_pool_matrix_score_matches_primary_scorer() -> None:
    candidates = ["A", "B", "C"]
    frequencies = pd.DataFrame(
        {"champion_j": ["X", "Y"], "freq_j": [0.7, 0.3]}
    )
    lookup = {
        ("A", "X"): 0.55,
        ("A", "Y"): 0.48,
        ("B", "X"): 0.52,
        ("B", "Y"): 0.57,
        ("C", "X"): 0.51,
    }
    pools = [("A", "B"), ("A", "C")]
    matrix = build_pool_matrix(candidates, 2, frequencies, lookup, pools=pools)

    expected = np.array(
        [pool_score(pool, frequencies, lookup) for pool in pools],
        dtype=float,
    )

    assert np.allclose(score_pool_matrix(matrix), expected)


def test_fixed_policy_analytic_stats_lock_posterior_mean_choice() -> None:
    frequencies = pd.DataFrame(
        {"champion_j": ["X", "Y"], "freq_j": [0.6, 0.4]}
    )
    posterior = pd.DataFrame(
        [
            {
                "champion": "A",
                "enemy_champion": "X",
                "posterior_alpha": 60.0,
                "posterior_beta": 40.0,
                "posterior_mean": 0.60,
            },
            {
                "champion": "B",
                "enemy_champion": "X",
                "posterior_alpha": 55.0,
                "posterior_beta": 45.0,
                "posterior_mean": 0.55,
            },
            {
                "champion": "A",
                "enemy_champion": "Y",
                "posterior_alpha": 45.0,
                "posterior_beta": 55.0,
                "posterior_mean": 0.45,
            },
            {
                "champion": "B",
                "enemy_champion": "Y",
                "posterior_alpha": 58.0,
                "posterior_beta": 42.0,
                "posterior_mean": 0.58,
            },
        ]
    )

    result = fixed_policy_analytic_pool_stats(
        [("A", "B")],
        frequencies,
        posterior,
    ).iloc[0]

    assert np.isclose(result.fixed_policy_mean, 0.6 * 0.60 + 0.4 * 0.58)
    assert result.fixed_policy_sd > 0
    assert result.fixed_policy_lower_5_normal < result.fixed_policy_mean


def test_offmeta_penalty_flags_low_pickrate_high_importance() -> None:
    candidates = ["Common", "Niche"]
    summary = pd.DataFrame(
        {
            "champion_name": candidates,
            "pickrate": [10.0, 0.5],
        }
    )
    inclusion = pd.DataFrame(
        {
            "champion": candidates,
            "top_pool_share": [0.2, 1.0],
        }
    )
    source_stability = pd.DataFrame(
        {
            "champion": candidates,
            "lolalytics_depth": [1.0, 1.4],
            "lolalytics_classification": ["broad", "niche"],
            "specialist_heuristic_flag": [False, True],
        }
    )

    penalties = build_offmeta_penalties(
        candidates,
        summary,
        source_stability,
        inclusion,
        pool_size=2,
    ).set_index("champion")

    assert penalties.loc["Niche", "offmeta_penalty"] > penalties.loc[
        "Common", "offmeta_penalty"
    ]
