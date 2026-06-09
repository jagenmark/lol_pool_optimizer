from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rank_comparison import build_pool_rank_comparison, normalize_pool


def test_pool_normalization_ignores_order_and_whitespace() -> None:
    assert normalize_pool("Viktor, Ahri, Yone") == normalize_pool(
        " Ahri,Yone, Viktor "
    )


def test_rank_comparison_uses_union_and_positive_change_means_riser() -> None:
    raw = pd.DataFrame(
        {
            "pool": [("A", "B"), ("A", "C"), ("B", "C")],
            "pool_score": [0.60, 0.59, 0.58],
        }
    )
    eb = pd.DataFrame(
        {
            "pool": [("B", "C"), ("A", "B"), ("A", "C")],
            "pool_score": [0.61, 0.60, 0.57],
        }
    )
    simulation = pd.DataFrame(
        {
            "pool": ["C, B", "B, A", "A, C"],
            "mean_score": [0.62, 0.60, 0.58],
            "median_score": [0.62, 0.60, 0.58],
            "sd_score": [0.01, 0.01, 0.02],
            "lower_5_score": [0.60, 0.58, 0.54],
            "upper_95_score": [0.64, 0.62, 0.62],
            "probability_of_being_best": [0.7, 0.2, 0.1],
        }
    )

    comparison = build_pool_rank_comparison(raw, eb, simulation, top_n=1)
    assert set(comparison["pool"]) == {"A, B", "B, C"}
    riser = comparison.loc[comparison["pool"] == "B, C"].iloc[0]
    assert riser.raw_rank == 3
    assert riser.posterior_mean_rank == 1
    assert riser.rank_change_raw_to_posterior_mean == 2
