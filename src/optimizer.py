from __future__ import annotations

from itertools import combinations
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
