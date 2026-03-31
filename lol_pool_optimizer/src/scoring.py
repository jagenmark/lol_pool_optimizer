from __future__ import annotations

from typing import Dict, Iterable, Tuple

import pandas as pd


def _weighted_average(score_components: list[tuple[float, float]]) -> float:
    if not score_components:
        raise ValueError("No scorable matchups remain after exclusions")
    weighted_sum = sum(weight * value for weight, value in score_components)
    total_weight = sum(weight for weight, _ in score_components)
    if total_weight <= 0:
        raise ValueError("Total usable enemy frequency is zero")
    return weighted_sum / total_weight


def blind_score(
    champion: str,
    enemy_frequencies: pd.DataFrame,
    matchup_lookup: Dict[Tuple[str, str], float],
) -> float:
    """Compute BlindScore(i), skipping self or missing matchups and renormalizing."""
    components = []
    for row in enemy_frequencies.itertuples(index=False):
        if row.champion_j == champion:
            continue
        matchup_value = matchup_lookup.get((champion, row.champion_j))
        if matchup_value is None:
            continue
        components.append((float(row.freq_j), matchup_value))
    return _weighted_average(components)


def compute_blind_scores(
    candidates: Iterable[str],
    enemy_frequencies: pd.DataFrame,
    matchup_lookup: Dict[Tuple[str, str], float],
) -> pd.DataFrame:
    rows = [
        {
            "champion": champion,
            "blind_score": blind_score(champion, enemy_frequencies, matchup_lookup),
        }
        for champion in candidates
    ]
    return pd.DataFrame(rows).sort_values(
        by=["blind_score", "champion"], ascending=[False, True]
    ).reset_index(drop=True)


def pool_score(
    pool: Tuple[str, ...],
    enemy_frequencies: pd.DataFrame,
    matchup_lookup: Dict[Tuple[str, str], float],
) -> float:
    """Compute Score(S), skipping unscorable enemy rows and renormalizing."""
    components = []
    for row in enemy_frequencies.itertuples(index=False):
        relevant_champions = []
        for champion in pool:
            if champion == row.champion_j:
                continue
            if (champion, row.champion_j) not in matchup_lookup:
                continue
            relevant_champions.append(champion)
        if not relevant_champions:
            continue
        best_matchup = max(
            matchup_lookup[(champion, row.champion_j)] for champion in relevant_champions
        )
        components.append((float(row.freq_j), best_matchup))
    return _weighted_average(components)


def build_counterpick_table(
    pool: Tuple[str, ...],
    enemy_frequencies: pd.DataFrame,
    matchup_lookup: Dict[Tuple[str, str], float],
) -> pd.DataFrame:
    rows = []
    for row in enemy_frequencies.itertuples(index=False):
        relevant_champions = [
            champion
            for champion in pool
            if champion != row.champion_j and (champion, row.champion_j) in matchup_lookup
        ]
        if not relevant_champions:
            continue
        best_pick = max(
            relevant_champions,
            key=lambda champion: matchup_lookup[(champion, row.champion_j)],
        )
        rows.append(
            {
                "enemy_champion": row.champion_j,
                "recommended_pick": best_pick,
                "matchup_value": matchup_lookup[(best_pick, row.champion_j)],
                "enemy_frequency": float(row.freq_j),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["enemy_frequency", "enemy_champion"], ascending=[False, True]
    ).reset_index(drop=True)
