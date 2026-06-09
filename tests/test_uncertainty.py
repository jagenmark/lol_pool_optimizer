from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uncertainty import build_matchup_posteriors, simulate_pool_scores
from optimizer import rank_pools, rank_top_pools


def matchup_frame(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["champion_i", "champion_j", "games_ij", "wins_i"],
    )


def test_posterior_mean_matches_empirical_bayes_formula() -> None:
    frame = matchup_frame([("A", "X", 20, 12)])
    posterior = build_matchup_posteriors(frame, prior_strength=100, prior_mean=0.5)
    expected = (12 + 100 * 0.5) / (20 + 100)
    assert posterior.loc[0, "posterior_mean"] == expected


def test_posterior_uncertainty_decreases_with_more_games() -> None:
    frame = matchup_frame(
        [
            ("Low", "X", 10, 5),
            ("High", "X", 1000, 500),
        ]
    )
    posterior = build_matchup_posteriors(frame, prior_strength=20, prior_mean=0.5)
    low = posterior.iloc[0]
    high = posterior.iloc[1]
    assert high.posterior_sd < low.posterior_sd
    assert (
        high.posterior_upper_95 - high.posterior_lower_5
        < low.posterior_upper_95 - low.posterior_lower_5
    )


def test_zero_game_matchup_returns_prior_distribution() -> None:
    frame = matchup_frame([("A", "X", 0, 0)])
    posterior = build_matchup_posteriors(frame, prior_strength=40)
    row = posterior.iloc[0]
    assert np.isnan(row.raw_winrate)
    assert row.posterior_alpha == 20
    assert row.posterior_beta == 20
    assert row.posterior_mean == 0.5


def test_simulation_is_reproducible_and_best_probabilities_sum_to_one() -> None:
    frame = matchup_frame(
        [
            ("A", "X", 20, 12),
            ("B", "X", 20, 10),
            ("A", "Y", 20, 8),
            ("B", "Y", 20, 13),
        ]
    )
    posterior = build_matchup_posteriors(frame, prior_strength=20, prior_mean=0.5)
    frequencies = pd.DataFrame(
        {"champion_j": ["X", "Y"], "freq_j": [0.6, 0.4]}
    )
    pools = [("A",), ("B",)]

    detail_a, summary_a = simulate_pool_scores(
        pools, frequencies, posterior, sample_count=250, seed=42
    )
    detail_b, summary_b = simulate_pool_scores(
        pools, frequencies, posterior, sample_count=250, seed=42
    )

    pd.testing.assert_frame_equal(detail_a, detail_b)
    pd.testing.assert_frame_equal(summary_a, summary_b)
    assert np.isclose(summary_a["probability_of_being_best"].sum(), 1.0)


def test_oracle_mean_is_not_below_fixed_policy_mean() -> None:
    frame = matchup_frame(
        [
            ("A", "X", 20, 11),
            ("B", "X", 20, 10),
        ]
    )
    posterior = build_matchup_posteriors(frame, prior_strength=20, prior_mean=0.5)
    frequencies = pd.DataFrame({"champion_j": ["X"], "freq_j": [1.0]})
    pools = [("A", "B")]

    _, fixed = simulate_pool_scores(
        pools,
        frequencies,
        posterior,
        sample_count=10_000,
        seed=7,
        simulation_mode="fixed-policy",
    )
    _, oracle = simulate_pool_scores(
        pools,
        frequencies,
        posterior,
        sample_count=10_000,
        seed=7,
        simulation_mode="oracle",
    )

    assert oracle.loc[0, "mean_score"] >= fixed.loc[0, "mean_score"]
    assert fixed.loc[0, "simulation_mode"] == "fixed-policy"
    assert oracle.loc[0, "simulation_mode"] == "oracle"


def test_top_pool_ranking_matches_exact_ranking_for_small_searches() -> None:
    candidates = ["A", "B", "C", "D"]
    frequencies = pd.DataFrame({"champion_j": ["X"], "freq_j": [1.0]})
    lookup = {
        ("A", "X"): 0.51,
        ("B", "X"): 0.55,
        ("C", "X"): 0.53,
        ("D", "X"): 0.52,
    }
    exact = rank_pools(candidates, 2, frequencies, lookup).head(3)
    bounded = rank_top_pools(candidates, 2, frequencies, lookup, top_n=3)
    assert bounded["pool"].tolist() == exact["pool"].tolist()
    assert np.allclose(bounded["pool_score"], exact["pool_score"])
    assert bounded.attrs["search_method"] == "exact_brute_force"
