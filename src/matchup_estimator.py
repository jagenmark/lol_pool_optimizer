from __future__ import annotations

from typing import Literal

import pandas as pd


EstimatorName = Literal["raw", "eb"]
DEFAULT_EB_ALPHA = 100.0
DEFAULT_PRIOR_MEAN = 0.5


def raw_winrate(wins: float, games: float, fallback: float = DEFAULT_PRIOR_MEAN) -> float:
    """Return wins / games, using fallback when no games are available."""
    if games <= 0:
        return fallback
    return wins / games


def empirical_bayes_winrate(
    wins: float,
    games: float,
    alpha: float,
    mu: float,
) -> float:
    """Shrink an observed matchup winrate toward prior mean mu."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if not 0 <= mu <= 1:
        raise ValueError("mu must be between 0 and 1")
    if games < 0:
        raise ValueError("games must be non-negative")

    denominator = games + alpha
    if denominator <= 0:
        return mu
    return (wins + alpha * mu) / denominator


def estimate_global_prior_mean(matchup_df: pd.DataFrame) -> float:
    """Estimate the games-weighted global winrate, with 0.5 as a safe fallback."""
    observed = matchup_df[matchup_df["games_ij"] > 0]
    total_games = float(observed["games_ij"].sum())
    if total_games <= 0:
        return DEFAULT_PRIOR_MEAN
    return float(observed["wins_i"].sum()) / total_games


def apply_matchup_estimator(
    matchup_df: pd.DataFrame,
    estimator: EstimatorName = "raw",
    eb_alpha: float = DEFAULT_EB_ALPHA,
    eb_mu: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Add inspectable raw and shrinkage columns and select the optimizer value.

    The returned prior mean is either the configured value or the weighted
    global mean estimated from the supplied matchup rows.
    """
    if estimator not in {"raw", "eb"}:
        raise ValueError(f"Unsupported estimator: {estimator}")
    if eb_alpha < 0:
        raise ValueError("eb_alpha must be non-negative")

    mu = estimate_global_prior_mean(matchup_df) if eb_mu is None else float(eb_mu)
    if not 0 <= mu <= 1:
        raise ValueError("eb_mu must be between 0 and 1")

    estimated = matchup_df.copy()
    estimated["raw_winrate"] = [
        raw_winrate(float(wins), float(games), fallback=mu)
        for wins, games in zip(estimated["wins_i"], estimated["games_ij"])
    ]
    estimated["shrinked_winrate"] = [
        empirical_bayes_winrate(
            wins=float(wins),
            games=float(games),
            alpha=eb_alpha,
            mu=mu,
        )
        for wins, games in zip(estimated["wins_i"], estimated["games_ij"])
    ]
    estimated["shrinkage_amount"] = (
        estimated["shrinked_winrate"] - estimated["raw_winrate"]
    )
    selected_column = "raw_winrate" if estimator == "raw" else "shrinked_winrate"
    estimated["winrate_ij"] = estimated[selected_column]
    return estimated, mu


def build_shrinkage_comparison(matchup_df: pd.DataFrame) -> pd.DataFrame:
    """Return real matchups ordered by absolute EB adjustment."""
    required = {
        "champion_i",
        "champion_j",
        "wins_i",
        "games_ij",
        "raw_winrate",
        "shrinked_winrate",
        "shrinkage_amount",
    }
    missing = required - set(matchup_df.columns)
    if missing:
        raise ValueError(
            "Matchup data is missing estimator columns: " + ", ".join(sorted(missing))
        )

    comparison = matchup_df.loc[
        matchup_df["champion_i"] != matchup_df["champion_j"],
        [
            "champion_i",
            "champion_j",
            "raw_winrate",
            "shrinked_winrate",
            "games_ij",
            "wins_i",
            "shrinkage_amount",
        ],
    ].copy()
    comparison = comparison.rename(columns={"games_ij": "games", "wins_i": "wins"})
    comparison["absolute_shrinkage"] = comparison["shrinkage_amount"].abs()
    return comparison.sort_values(
        by=["absolute_shrinkage", "games", "champion_i", "champion_j"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
