from __future__ import annotations

import pandas as pd

from rank_comparison import normalized_pool_tuple


CONTRIBUTION_COLUMNS = [
    "pool",
    "raw_rank",
    "eb_rank",
    "posterior_mean_rank",
    "enemy_champion",
    "enemy_frequency",
    "best_pool_champion_against_enemy",
    "raw_winrate",
    "posterior_mean",
    "posterior_lower_5",
    "posterior_upper_95",
    "games",
    "weighted_contribution",
]


def build_pool_contribution_report(
    comparison_df: pd.DataFrame,
    enemy_frequencies: pd.DataFrame,
    posterior_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Explain each comparison pool's posterior-mean score by enemy matchup.

    Enemy frequencies are renormalized over scorable enemies exactly as in the
    pool scorer. Weighted contributions therefore sum to the pool's score when
    matchup posterior means are used as point estimates.
    """
    posterior_lookup = {
        (row.champion, row.enemy_champion): row
        for row in posterior_df.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []

    ordered_pools = comparison_df.sort_values(
        ["posterior_mean_rank", "pool"], na_position="last"
    )
    for pool_row in ordered_pools.itertuples(index=False):
        pool = normalized_pool_tuple(pool_row.pool)
        selected_matchups: list[tuple[object, object]] = []
        for enemy_row in enemy_frequencies.itertuples(index=False):
            candidates = [
                posterior_lookup[(champion, enemy_row.champion_j)]
                for champion in pool
                if champion != enemy_row.champion_j
                and (champion, enemy_row.champion_j) in posterior_lookup
            ]
            if not candidates:
                continue
            best_matchup = max(
                candidates,
                key=lambda row: (float(row.posterior_mean), row.champion),
            )
            selected_matchups.append((enemy_row, best_matchup))

        total_usable_frequency = sum(
            float(enemy_row.freq_j) for enemy_row, _ in selected_matchups
        )
        if total_usable_frequency <= 0:
            raise ValueError(f"No scorable enemies remain for pool: {pool_row.pool}")

        for enemy_row, matchup in selected_matchups:
            normalized_frequency = (
                float(enemy_row.freq_j) / total_usable_frequency
            )
            rows.append(
                {
                    "pool": pool_row.pool,
                    "raw_rank": pool_row.raw_rank,
                    "eb_rank": pool_row.eb_rank,
                    "posterior_mean_rank": pool_row.posterior_mean_rank,
                    "enemy_champion": enemy_row.champion_j,
                    "enemy_frequency": normalized_frequency,
                    "best_pool_champion_against_enemy": matchup.champion,
                    "raw_winrate": float(matchup.raw_winrate),
                    "posterior_mean": float(matchup.posterior_mean),
                    "posterior_lower_5": float(matchup.posterior_lower_5),
                    "posterior_upper_95": float(matchup.posterior_upper_95),
                    "games": float(matchup.games),
                    "weighted_contribution": (
                        normalized_frequency * float(matchup.posterior_mean)
                    ),
                }
            )

    report = pd.DataFrame(rows, columns=CONTRIBUTION_COLUMNS)
    rank_columns = ["raw_rank", "eb_rank", "posterior_mean_rank"]
    report[rank_columns] = report[rank_columns].astype("Int64")
    return report.sort_values(
        ["posterior_mean_rank", "pool", "enemy_frequency", "enemy_champion"],
        ascending=[True, True, False, True],
        na_position="last",
    ).reset_index(drop=True)
