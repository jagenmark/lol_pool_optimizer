from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution


POSTERIOR_COLUMNS = [
    "champion",
    "enemy_champion",
    "games",
    "wins",
    "losses",
    "raw_winrate",
    "prior_mean",
    "prior_strength",
    "posterior_alpha",
    "posterior_beta",
    "posterior_mean",
    "posterior_sd",
    "posterior_lower_5",
    "posterior_upper_95",
]
SimulationMode = Literal["fixed-policy", "oracle"]


def global_weighted_mean_winrate(
    matchup_df: pd.DataFrame,
    fallback: float = 0.5,
) -> float:
    """Return total wins / total games, falling back when no games are available."""
    games = matchup_df["games_ij"].to_numpy(dtype=float)
    wins = matchup_df["wins_i"].to_numpy(dtype=float)
    total_games = float(games.sum())
    if total_games <= 0:
        return fallback
    return float(wins.sum() / total_games)


def build_matchup_posteriors(
    matchup_df: pd.DataFrame,
    prior_strength: float,
    prior_mean: float | None = None,
) -> pd.DataFrame:
    """Build independent beta-binomial posteriors for all matchup rows."""
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")

    mu = (
        global_weighted_mean_winrate(matchup_df)
        if prior_mean is None
        else float(prior_mean)
    )
    if not 0 < mu < 1:
        raise ValueError("prior_mean must be strictly between 0 and 1")

    games = matchup_df["games_ij"].to_numpy(dtype=float)
    wins = matchup_df["wins_i"].to_numpy(dtype=float)
    if np.any(games < 0):
        raise ValueError("games_ij must be non-negative")
    if np.any((wins < 0) | (wins > games)):
        raise ValueError("wins_i must be between zero and games_ij")

    losses = games - wins
    posterior_alpha = prior_strength * mu + wins
    posterior_beta = prior_strength * (1.0 - mu) + losses
    posterior_total = posterior_alpha + posterior_beta
    posterior_mean = posterior_alpha / posterior_total
    posterior_sd = np.sqrt(
        posterior_alpha
        * posterior_beta
        / (posterior_total**2 * (posterior_total + 1.0))
    )
    raw_winrate = np.divide(
        wins,
        games,
        out=np.full(games.shape, np.nan, dtype=float),
        where=games > 0,
    )

    posterior_df = pd.DataFrame(
        {
            "champion": matchup_df["champion_i"].astype(str).to_numpy(),
            "enemy_champion": matchup_df["champion_j"].astype(str).to_numpy(),
            "games": games,
            "wins": wins,
            "losses": losses,
            "raw_winrate": raw_winrate,
            "prior_mean": mu,
            "prior_strength": prior_strength,
            "posterior_alpha": posterior_alpha,
            "posterior_beta": posterior_beta,
            "posterior_mean": posterior_mean,
            "posterior_sd": posterior_sd,
            "posterior_lower_5": beta_distribution.ppf(
                0.05, posterior_alpha, posterior_beta
            ),
            "posterior_upper_95": beta_distribution.ppf(
                0.95, posterior_alpha, posterior_beta
            ),
        }
    )
    return posterior_df[POSTERIOR_COLUMNS]


def build_posterior_mean_lookup(
    posterior_df: pd.DataFrame,
) -> dict[tuple[str, str], float]:
    return {
        (row.champion, row.enemy_champion): float(row.posterior_mean)
        for row in posterior_df.itertuples(index=False)
    }


def simulate_pool_scores(
    pools: Sequence[tuple[str, ...]],
    enemy_frequencies: pd.DataFrame,
    posterior_df: pd.DataFrame,
    sample_count: int,
    seed: int,
    simulation_mode: SimulationMode = "fixed-policy",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate posterior pool scores using the scorer's skip-and-renormalize rules.

    Processing one enemy at a time keeps memory proportional to samples times
    candidate champions, rather than samples times every matchup row.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not pools:
        raise ValueError("At least one pool is required for simulation")
    if simulation_mode not in {"fixed-policy", "oracle"}:
        raise ValueError(f"Unsupported simulation mode: {simulation_mode}")

    normalized_pools = [tuple(pool) for pool in pools]
    rng = np.random.default_rng(seed)
    weighted_scores = np.zeros((sample_count, len(normalized_pools)), dtype=float)
    usable_weights = np.zeros(len(normalized_pools), dtype=float)

    posterior_lookup = {
        (row.champion, row.enemy_champion): (
            float(row.posterior_alpha),
            float(row.posterior_beta),
            float(row.posterior_mean),
        )
        for row in posterior_df.itertuples(index=False)
    }

    for enemy_row in enemy_frequencies.itertuples(index=False):
        enemy = str(enemy_row.champion_j)
        weight = float(enemy_row.freq_j)
        champions = sorted(
            {
                champion
                for pool in normalized_pools
                for champion in pool
                if champion != enemy and (champion, enemy) in posterior_lookup
            }
        )
        if not champions:
            continue

        alpha = np.array(
            [posterior_lookup[(champion, enemy)][0] for champion in champions],
            dtype=float,
        )
        beta = np.array(
            [posterior_lookup[(champion, enemy)][1] for champion in champions],
            dtype=float,
        )
        posterior_mean = np.array(
            [posterior_lookup[(champion, enemy)][2] for champion in champions],
            dtype=float,
        )
        sampled_matchups = rng.beta(alpha, beta, size=(sample_count, len(champions)))
        champion_index = {champion: index for index, champion in enumerate(champions)}

        for pool_index, pool in enumerate(normalized_pools):
            relevant_indices = [
                champion_index[champion]
                for champion in pool
                if champion in champion_index
            ]
            if not relevant_indices:
                continue
            if simulation_mode == "oracle":
                selected_sample = sampled_matchups[:, relevant_indices].max(axis=1)
            else:
                selected_index = max(
                    relevant_indices,
                    key=lambda index: (posterior_mean[index], champions[index]),
                )
                selected_sample = sampled_matchups[:, selected_index]
            weighted_scores[:, pool_index] += weight * selected_sample
            usable_weights[pool_index] += weight

    if np.any(usable_weights <= 0):
        bad_pools = [
            ", ".join(normalized_pools[index])
            for index in np.flatnonzero(usable_weights <= 0)
        ]
        raise ValueError("No scorable enemies remain for pools: " + "; ".join(bad_pools))

    scores = weighted_scores / usable_weights
    best_pool_indices = np.argmax(scores, axis=1)
    best_counts = np.bincount(best_pool_indices, minlength=len(normalized_pools))

    detail_frames = []
    summary_rows = []
    for pool_index, pool in enumerate(normalized_pools):
        pool_label = ", ".join(pool)
        pool_scores = scores[:, pool_index]
        detail_frames.append(
            pd.DataFrame(
                {
                    "simulation": np.arange(1, sample_count + 1),
                    "pool": pool_label,
                    "pool_size": len(pool),
                    "score": pool_scores,
                    "is_best": best_pool_indices == pool_index,
                    "simulation_mode": simulation_mode,
                }
            )
        )
        summary_rows.append(
            {
                "pool": pool_label,
                "pool_size": len(pool),
                "simulation_mode": simulation_mode,
                "mean_score": float(np.mean(pool_scores)),
                "median_score": float(np.median(pool_scores)),
                "sd_score": float(np.std(pool_scores, ddof=1))
                if sample_count > 1
                else 0.0,
                "lower_5_score": float(np.quantile(pool_scores, 0.05)),
                "upper_95_score": float(np.quantile(pool_scores, 0.95)),
                "probability_of_being_best": float(
                    best_counts[pool_index] / sample_count
                ),
            }
        )

    detail_df = pd.concat(detail_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_score", "pool"], ascending=[False, True]
    ).reset_index(drop=True)
    return detail_df, summary_df


def simulation_summary_path(detail_path: Path) -> Path:
    if detail_path.name == "pool_score_simulation.csv":
        return detail_path.with_name("pool_score_simulation_summary.csv")
    return detail_path.with_name(f"{detail_path.stem}_summary{detail_path.suffix}")
