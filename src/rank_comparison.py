from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


COMPARISON_COLUMNS = [
    "pool",
    "raw_rank",
    "raw_score",
    "eb_rank",
    "eb_score",
    "posterior_mean_rank",
    "mean_score",
    "median_score",
    "sd_score",
    "lower_5_rank",
    "lower_5_score",
    "upper_95_score",
    "prob_best_rank",
    "probability_of_being_best",
    "rank_change_raw_to_eb",
    "rank_change_raw_to_posterior_mean",
    "rank_change_raw_to_lower_5",
    "rank_change_raw_to_prob_best",
]


def normalize_pool(pool: object) -> str:
    """Return a stable pool key regardless of champion ordering or whitespace."""
    if isinstance(pool, (tuple, list)):
        champions = [str(champion).strip() for champion in pool]
    else:
        champions = [
            champion.strip()
            for champion in str(pool).split(",")
            if champion.strip()
        ]
    return ", ".join(sorted(champions, key=str.casefold))


def normalized_pool_tuple(pool: object) -> tuple[str, ...]:
    return tuple(normalize_pool(pool).split(", "))


def unique_normalized_pools(pools: Iterable[object]) -> list[tuple[str, ...]]:
    unique: dict[str, tuple[str, ...]] = {}
    for pool in pools:
        key = normalize_pool(pool)
        unique.setdefault(key, normalized_pool_tuple(pool))
    return list(unique.values())


def _point_ranking(
    ranked_df: pd.DataFrame,
    rank_column: str,
    score_column: str,
) -> pd.DataFrame:
    frame = ranked_df[["pool", "pool_score"]].copy()
    frame["pool"] = frame["pool"].map(normalize_pool)
    frame = frame.sort_values(
        ["pool_score", "pool"], ascending=[False, True]
    ).drop_duplicates("pool")
    frame[rank_column] = range(1, len(frame) + 1)
    return frame.rename(columns={"pool_score": score_column})[
        ["pool", rank_column, score_column]
    ]


def _simulation_ranking(
    summary_df: pd.DataFrame,
    metric: str,
    rank_column: str,
) -> pd.DataFrame:
    frame = summary_df.copy()
    frame["pool"] = frame["pool"].map(normalize_pool)
    frame = frame.sort_values(
        [metric, "pool"], ascending=[False, True]
    ).drop_duplicates("pool")
    frame[rank_column] = range(1, len(frame) + 1)
    return frame


def build_pool_rank_comparison(
    raw_ranked_df: pd.DataFrame,
    eb_ranked_df: pd.DataFrame,
    simulation_summary_df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Join point-estimate and posterior rankings for the union of top-N pools."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    raw = _point_ranking(raw_ranked_df, "raw_rank", "raw_score")
    eb = _point_ranking(eb_ranked_df, "eb_rank", "eb_score")
    posterior_mean = _simulation_ranking(
        simulation_summary_df, "mean_score", "posterior_mean_rank"
    )
    lower_5 = _simulation_ranking(
        simulation_summary_df, "lower_5_score", "lower_5_rank"
    )[["pool", "lower_5_rank"]]
    prob_best = _simulation_ranking(
        simulation_summary_df,
        "probability_of_being_best",
        "prob_best_rank",
    )[["pool", "prob_best_rank"]]

    included_pools = set(raw.head(top_n)["pool"])
    included_pools.update(eb.head(top_n)["pool"])
    included_pools.update(posterior_mean.head(top_n)["pool"])
    included_pools.update(
        lower_5.sort_values("lower_5_rank").head(top_n)["pool"]
    )
    included_pools.update(
        prob_best.sort_values("prob_best_rank").head(top_n)["pool"]
    )

    comparison = pd.DataFrame({"pool": sorted(included_pools)})
    comparison = comparison.merge(raw, on="pool", how="left")
    comparison = comparison.merge(eb, on="pool", how="left")
    comparison = comparison.merge(
        posterior_mean[
            [
                "pool",
                "posterior_mean_rank",
                "mean_score",
                "median_score",
                "sd_score",
                "lower_5_score",
                "upper_95_score",
                "probability_of_being_best",
            ]
        ],
        on="pool",
        how="left",
    )
    comparison = comparison.merge(lower_5, on="pool", how="left")
    comparison = comparison.merge(prob_best, on="pool", how="left")

    comparison["rank_change_raw_to_eb"] = (
        comparison["raw_rank"] - comparison["eb_rank"]
    )
    comparison["rank_change_raw_to_posterior_mean"] = (
        comparison["raw_rank"] - comparison["posterior_mean_rank"]
    )
    comparison["rank_change_raw_to_lower_5"] = (
        comparison["raw_rank"] - comparison["lower_5_rank"]
    )
    comparison["rank_change_raw_to_prob_best"] = (
        comparison["raw_rank"] - comparison["prob_best_rank"]
    )
    rank_columns = [
        "raw_rank",
        "eb_rank",
        "posterior_mean_rank",
        "lower_5_rank",
        "prob_best_rank",
        "rank_change_raw_to_eb",
        "rank_change_raw_to_posterior_mean",
        "rank_change_raw_to_lower_5",
        "rank_change_raw_to_prob_best",
    ]
    comparison[rank_columns] = comparison[rank_columns].astype("Int64")
    return comparison[COMPARISON_COLUMNS].sort_values(
        ["posterior_mean_rank", "pool"], na_position="last"
    ).reset_index(drop=True)


def top_rank_table(
    comparison_df: pd.DataFrame,
    rank_column: str,
    score_column: str,
    top_n: int,
) -> pd.DataFrame:
    return (
        comparison_df.dropna(subset=[rank_column])
        .sort_values([rank_column, "pool"])
        .head(top_n)[["pool", rank_column, score_column]]
        .reset_index(drop=True)
    )
