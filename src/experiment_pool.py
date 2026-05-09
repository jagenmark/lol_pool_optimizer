from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def build_lookup(df: pd.DataFrame, value_column: str) -> dict[tuple[str, str], float]:
    """Convert matchup rows into a fast `(champion_id, enemy_id) -> value` lookup."""
    return {
        (row.champion_id, row.enemy_id): float(getattr(row, value_column))
        for row in df.itertuples(index=False)
    }


def pool_score(
    pool: tuple[str, ...],
    weights_df: pd.DataFrame,
    value_lookup: dict[tuple[str, str], float],
) -> float:
    """Compute the weighted pool score using the pool's best answer into each enemy."""
    weighted_values: list[tuple[float, float]] = []
    for row in weights_df.itertuples(index=False):
        values = [value_lookup[(champion_id, row.enemy_id)] for champion_id in pool if (champion_id, row.enemy_id) in value_lookup]
        if not values:
            continue
        weighted_values.append((float(row.weight), float(max(values))))

    if not weighted_values:
        raise ValueError("No scorable enemies remain for this pool")

    weights = np.array([item[0] for item in weighted_values], dtype=float)
    scores = np.array([item[1] for item in weighted_values], dtype=float)
    return float(np.sum(weights * scores) / np.sum(weights))


def brute_force_best_pools(
    candidate_ids: list[str],
    pool_size: int,
    weights_df: pd.DataFrame,
    value_lookup: dict[tuple[str, str], float],
) -> pd.DataFrame:
    """Enumerate every pool of size `k` and rank them by score."""
    rows = []
    for pool in combinations(candidate_ids, pool_size):
        rows.append({"pool": pool, "score": pool_score(pool, weights_df, value_lookup)})
    ranked = pd.DataFrame(rows).sort_values(["score", "pool"], ascending=[False, True]).reset_index(drop=True)
    return ranked


def weighted_error_metrics(
    estimated_df: pd.DataFrame,
    observed_eval_df: pd.DataFrame,
    eval_weights_df: pd.DataFrame,
) -> tuple[float, float, int]:
    """Compare patch-A estimates against observed patch-B outcomes on shared pairs."""
    joined = estimated_df.merge(
        observed_eval_df[["champion_id", "enemy_id", "matchup_winrate"]],
        on=["champion_id", "enemy_id"],
        how="inner",
        suffixes=("_train", "_eval"),
    ).merge(
        eval_weights_df[["enemy_id", "weight"]],
        on="enemy_id",
        how="inner",
    )
    if joined.empty:
        raise ValueError("No common matchup rows between training estimates and evaluation observations")

    observed_column = "matchup_winrate"
    if observed_column not in joined.columns:
        observed_column = "matchup_winrate_eval"

    abs_error = np.abs(joined["estimated_winrate"] - joined[observed_column])
    sq_error = np.square(joined["estimated_winrate"] - joined[observed_column])
    weights = joined["weight"]
    weighted_mae = float(np.average(abs_error, weights=weights))
    weighted_mse = float(np.average(sq_error, weights=weights))
    return weighted_mae, weighted_mse, int(len(joined))
