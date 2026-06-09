from __future__ import annotations

import heapq
from itertools import combinations
from math import comb
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from scoring import pool_score


def generate_pools(candidates: Iterable[str], pool_size: int) -> List[Tuple[str, ...]]:
    return list(combinations(candidates, pool_size))


def rank_pools(
    candidates: Iterable[str],
    pool_size: int,
    enemy_frequencies: pd.DataFrame,
    matchup_lookup: Dict[Tuple[str, str], float],
) -> pd.DataFrame:
    pools = generate_pools(candidates, pool_size)
    rows = []
    for pool in pools:
        rows.append(
            {
                "pool": pool,
                "pool_label": ", ".join(pool),
                "pool_score": pool_score(pool, enemy_frequencies, matchup_lookup),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["pool_score", "pool_label"], ascending=[False, True]
    ).reset_index(drop=True)


def rank_top_pools(
    candidates: Iterable[str],
    pool_size: int,
    enemy_frequencies: pd.DataFrame,
    matchup_lookup: Dict[Tuple[str, str], float],
    top_n: int,
    max_exact_combinations: int = 250_000,
) -> pd.DataFrame:
    """Rank the best N pools exactly for small spaces and by beam search otherwise."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    candidate_list = list(candidates)
    combination_count = comb(len(candidate_list), pool_size)
    if combination_count <= max_exact_combinations:
        ranked = _rank_pool_iterable(
            combinations(candidate_list, pool_size),
            enemy_frequencies,
            matchup_lookup,
            top_n,
        )
        ranked.attrs["search_method"] = "exact_brute_force"
        ranked.attrs["evaluated_pool_count"] = combination_count
        ranked.attrs["scored_candidate_count"] = combination_count
        ranked.attrs["total_combination_count"] = combination_count
        return ranked

    beam_width = max(1_000, top_n * 10)
    candidate_positions = {
        champion: index for index, champion in enumerate(candidate_list)
    }
    beam: list[tuple[str, ...]] = [()]
    evaluated_pool_count = 0
    for _ in range(pool_size):
        expanded = []
        for partial_pool in beam:
            start = (
                candidate_positions[partial_pool[-1]] + 1
                if partial_pool
                else 0
            )
            expanded.extend(
                partial_pool + (candidate,)
                for candidate in candidate_list[start:]
            )
        evaluated_pool_count += len(expanded)
        beam_df = _rank_pool_iterable(
            expanded,
            enemy_frequencies,
            matchup_lookup,
            beam_width,
        )
        beam = beam_df["pool"].tolist()

    ranked = beam_df.head(top_n).reset_index(drop=True)
    ranked.attrs["search_method"] = f"beam_search_width_{beam_width}"
    ranked.attrs["evaluated_pool_count"] = len(expanded)
    ranked.attrs["scored_candidate_count"] = evaluated_pool_count
    ranked.attrs["total_combination_count"] = combination_count
    return ranked


def _rank_pool_iterable(
    pools: Iterable[tuple[str, ...]],
    enemy_frequencies: pd.DataFrame,
    matchup_lookup: Dict[Tuple[str, str], float],
    top_n: int,
) -> pd.DataFrame:
    ranked_heap: list[
        tuple[float, tuple[int, ...], str, tuple[str, ...]]
    ] = []
    for pool in pools:
        label = ", ".join(pool)
        score = pool_score(pool, enemy_frequencies, matchup_lookup)
        reverse_label = tuple(-ord(character) for character in label)
        item = (score, reverse_label, label, pool)
        if len(ranked_heap) < top_n:
            heapq.heappush(ranked_heap, item)
        elif item[:2] > ranked_heap[0][:2]:
            heapq.heapreplace(ranked_heap, item)

    rows = [
        {"pool": pool, "pool_label": label, "pool_score": score}
        for score, _, label, pool in ranked_heap
    ]
    return pd.DataFrame(rows).sort_values(
        by=["pool_score", "pool_label"], ascending=[False, True]
    ).reset_index(drop=True)
