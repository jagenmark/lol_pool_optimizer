from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from data_loader import LoadedInputs, collect_missing_matchup_pairs, load_patch_data
from optimizer import rank_pools
from scoring import build_counterpick_table, blind_score, compute_blind_scores
from utils import canonicalize_champion_name


SUPPORTED_PATCHES = ("16.05", "16.06", "16.07")


@dataclass(frozen=True)
class OptimizerGuiResult:
    loaded: LoadedInputs
    candidates: list[str]
    pool_size: int
    best_pool: tuple[str, ...]
    best_pool_score: float
    best_blind_pick: str
    best_blind_score: float
    blind_scores: pd.DataFrame
    top_pools: pd.DataFrame
    counterpick_table: pd.DataFrame
    heatmap_data: pd.DataFrame
    pool_responsibility: pd.DataFrame
    exclusion_details: dict[str, Any]
    missing_pairs: list[tuple[str, str]]


def get_available_champions(patch: str, data_dir: Path) -> list[str]:
    loaded = load_patch_data(patch, data_dir)
    return sorted(loaded.matchup_df["champion_i"].unique())


def _summary_lookup(summary_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if summary_df.empty:
        return {}
    return {
        row.champion_key: row._asdict()
        for row in summary_df.itertuples(index=False)
        if hasattr(row, "champion_key")
    }


def display_name_for_key(champion: str, summary_df: pd.DataFrame) -> str:
    champion_key = canonicalize_champion_name(champion)
    summary_row = _summary_lookup(summary_df).get(champion_key)
    if summary_row:
        return str(summary_row["champion_name"])
    return champion


def normalize_selected_candidates(
    selected: list[str],
    available_champions: list[str],
    summary_df: pd.DataFrame,
) -> list[str]:
    available_by_key = {
        canonicalize_champion_name(champion): champion for champion in available_champions
    }
    summary_names = {
        canonicalize_champion_name(str(row.champion_name)): str(row.champion_name)
        for row in summary_df.itertuples(index=False)
        if canonicalize_champion_name(str(row.champion_name)) in available_by_key
    }
    candidates = []
    for champion in selected:
        champion_key = canonicalize_champion_name(champion)
        if champion_key in available_by_key:
            candidates.append(available_by_key[champion_key])
        elif champion_key in summary_names:
            candidates.append(summary_names[champion_key])
    return sorted(set(candidates))


def build_heatmap_data(
    pool: tuple[str, ...],
    enemy_frequencies: pd.DataFrame,
    matchup_lookup: dict[tuple[str, str], float],
) -> pd.DataFrame:
    rows = []
    for enemy in enemy_frequencies.sort_values(
        by=["freq_j", "champion_j"], ascending=[False, True]
    ).itertuples(index=False):
        row: dict[str, Any] = {
            "enemy_champion": enemy.champion_j,
            "enemy_frequency": float(enemy.freq_j),
        }
        for champion in pool:
            row[champion] = matchup_lookup.get((champion, enemy.champion_j))
        rows.append(row)
    return pd.DataFrame(rows)


def build_pool_responsibility(
    pool: tuple[str, ...],
    enemy_frequencies: pd.DataFrame,
    matchup_lookup: dict[tuple[str, str], float],
) -> pd.DataFrame:
    coverage = {champion: 0.0 for champion in pool}
    rows = []
    for enemy in enemy_frequencies.itertuples(index=False):
        relevant = [
            champion
            for champion in pool
            if champion != enemy.champion_j and (champion, enemy.champion_j) in matchup_lookup
        ]
        if not relevant:
            continue
        best_pick = max(relevant, key=lambda champion: matchup_lookup[(champion, enemy.champion_j)])
        coverage[best_pick] += float(enemy.freq_j)
        rows.append(
            {
                "enemy_champion": enemy.champion_j,
                "recommended_pick": best_pick,
                "enemy_frequency": float(enemy.freq_j),
                "matchup_value": matchup_lookup[(best_pick, enemy.champion_j)],
            }
        )

    responsibility_df = pd.DataFrame(
        {
            "champion": list(coverage.keys()),
            "weighted_share": list(coverage.values()),
        }
    ).sort_values(by=["weighted_share", "champion"], ascending=[False, True])
    total_covered = responsibility_df["weighted_share"].sum()
    responsibility_df["covered_share_of_scorable_rows"] = (
        responsibility_df["weighted_share"] / total_covered if total_covered > 0 else 0.0
    )
    return responsibility_df.reset_index(drop=True)


def build_exclusion_details(
    pool: tuple[str, ...],
    enemy_frequencies: pd.DataFrame,
    matchup_lookup: dict[tuple[str, str], float],
) -> dict[str, Any]:
    skipped_rows = []
    removed_frequency_mass = 0.0
    enemy_frequency_lookup = {
        row.champion_j: row.freq_j for row in enemy_frequencies.itertuples(index=False)
    }

    for enemy in enemy_frequencies.itertuples(index=False):
        enemy_frequency = enemy_frequency_lookup.get(enemy.champion_j)
        has_missing_frequency = pd.isna(enemy_frequency)
        scorable_champions = []

        for champion in pool:
            reason = None
            notes = ""
            if has_missing_frequency:
                reason = "missing_frequency"
                notes = "Enemy frequency was unavailable for this matchup row."
            elif champion == enemy.champion_j:
                reason = "self_matchup"
                notes = "Pool champion and enemy champion are the same; scorer skips self-matchups."
            elif (champion, enemy.champion_j) not in matchup_lookup:
                reason = "missing_matchup_value"
                notes = "No W(i,j) value was available for this pool champion against this enemy."
            else:
                scorable_champions.append(champion)

            if reason:
                skipped_rows.append(
                    {
                        "pool_champion": champion,
                        "enemy_champion": enemy.champion_j,
                        "reason": reason,
                        "original_frequency": None if has_missing_frequency else float(enemy_frequency),
                        "notes": notes,
                    }
                )

        if has_missing_frequency or not scorable_champions:
            removed_frequency_mass += 0.0 if has_missing_frequency else float(enemy_frequency)

    skipped_df = pd.DataFrame(
        skipped_rows,
        columns=[
            "pool_champion",
            "enemy_champion",
            "reason",
            "original_frequency",
            "notes",
        ],
    )
    reason_counts = skipped_df["reason"].value_counts().to_dict() if not skipped_df.empty else {}
    return {
        "had_exclusions": not skipped_df.empty,
        "total_skipped": int(len(skipped_df)),
        "self_matchups": int(reason_counts.get("self_matchup", 0)),
        "missing_matchups": int(reason_counts.get("missing_matchup_value", 0)),
        "missing_frequencies": int(reason_counts.get("missing_frequency", 0)),
        "other": int(
            len(skipped_df)
            - reason_counts.get("self_matchup", 0)
            - reason_counts.get("missing_matchup_value", 0)
            - reason_counts.get("missing_frequency", 0)
        ),
        "removed_frequency_mass": float(removed_frequency_mass),
        "skipped_rows": skipped_df,
        "renormalized_afterward": True,
    }


def run_optimizer_for_gui(
    patch: str,
    data_dir: Path,
    candidates: list[str],
    pool_size: int,
    top_k: int = 5,
) -> OptimizerGuiResult:
    if patch not in SUPPORTED_PATCHES:
        raise ValueError(f"Unsupported patch: {patch}")
    if not candidates:
        raise ValueError("Select at least one candidate champion.")
    if pool_size <= 0:
        raise ValueError("Pool size must be positive.")
    if pool_size > len(candidates):
        raise ValueError(
            f"Pool size {pool_size} is larger than the number of selected candidates ({len(candidates)})."
        )

    loaded = load_patch_data(patch, data_dir)
    available_champions = sorted(loaded.matchup_df["champion_i"].unique())
    normalized_candidates = normalize_selected_candidates(
        candidates, available_champions, loaded.summary_df
    )
    unknown = sorted(set(candidates) - set(normalized_candidates))
    if unknown:
        raise ValueError("Some candidate champions are not present in the matchup data: " + ", ".join(unknown))

    missing_pairs = collect_missing_matchup_pairs(
        normalized_candidates,
        loaded.frequency_df["champion_j"].tolist(),
        loaded.matchup_lookup,
    )
    blind_scores = compute_blind_scores(
        normalized_candidates, loaded.frequency_df, loaded.matchup_lookup
    )
    ranked_pools = rank_pools(
        normalized_candidates, pool_size, loaded.frequency_df, loaded.matchup_lookup
    )
    if ranked_pools.empty:
        raise ValueError("No pools could be generated from the selected candidates.")

    best_pool = tuple(ranked_pools.iloc[0]["pool"])
    best_pool_score = float(ranked_pools.iloc[0]["pool_score"])
    best_blind_pick = str(blind_scores.iloc[0]["champion"])
    best_blind_score = float(blind_scores.iloc[0]["blind_score"])

    top_pools = ranked_pools.head(top_k).copy()
    top_pools.insert(0, "rank", range(1, len(top_pools) + 1))
    top_pools["difference_from_best"] = best_pool_score - top_pools["pool_score"]
    top_pools["pool_champions"] = top_pools["pool"].apply(lambda pool: ", ".join(pool))

    counterpick_table = build_counterpick_table(
        best_pool, loaded.frequency_df, loaded.matchup_lookup
    )
    heatmap_data = build_heatmap_data(best_pool, loaded.frequency_df, loaded.matchup_lookup)
    pool_responsibility = build_pool_responsibility(
        best_pool, loaded.frequency_df, loaded.matchup_lookup
    )
    exclusion_details = build_exclusion_details(
        best_pool, loaded.frequency_df, loaded.matchup_lookup
    )

    return OptimizerGuiResult(
        loaded=loaded,
        candidates=normalized_candidates,
        pool_size=pool_size,
        best_pool=best_pool,
        best_pool_score=best_pool_score,
        best_blind_pick=best_blind_pick,
        best_blind_score=best_blind_score,
        blind_scores=blind_scores,
        top_pools=top_pools,
        counterpick_table=counterpick_table,
        heatmap_data=heatmap_data,
        pool_responsibility=pool_responsibility,
        exclusion_details=exclusion_details,
        missing_pairs=missing_pairs,
    )


def champion_icon_path(champion: str, project_root: Path) -> Path | None:
    icon_dir = project_root / "assets" / "champion_icons"
    normalized = canonicalize_champion_name(champion)
    manifest_filename = _champion_icon_manifest(project_root).get(normalized)
    if manifest_filename:
        manifest_path = icon_dir / manifest_filename
        if manifest_path.exists():
            return manifest_path

    candidates = [
        icon_dir / f"{normalized}.png",
        icon_dir / f"{normalized}.jpg",
        icon_dir / f"{normalized}.webp",
        icon_dir / f"{champion}.png",
        icon_dir / f"{champion}.jpg",
        icon_dir / f"{champion}.webp",
    ]
    return next((path for path in candidates if path.exists()), None)


@lru_cache(maxsize=8)
def _champion_icon_manifest(project_root: Path) -> dict[str, str]:
    manifest_path = project_root / "assets" / "champion_icons" / "champion_icon_manifest.csv"
    if not manifest_path.exists():
        return {}

    manifest: dict[str, str] = {}
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        for row in csv.DictReader(manifest_file):
            icon_filename = row.get("icon_filename", "")
            if not icon_filename:
                continue
            for value in (row.get("champion_name", ""), row.get("champion_id", "")):
                key = canonicalize_champion_name(value)
                if key:
                    manifest[key] = icon_filename
    return manifest


def build_champion_diagnostics(
    champion: str,
    loaded: LoadedInputs,
    candidates: list[str] | None = None,
) -> dict[str, Any]:
    champion_key = canonicalize_champion_name(champion)
    summary = _summary_lookup(loaded.summary_df).get(champion_key, {})
    champion_name = str(summary.get("champion_name", champion))
    matchup_lookup = loaded.matchup_lookup

    blind = None
    try:
        blind = blind_score(champion_name, loaded.frequency_df, matchup_lookup)
    except ValueError:
        pass

    contribution_rows = []
    for enemy in loaded.frequency_df.itertuples(index=False):
        matchup_value = matchup_lookup.get((champion_name, enemy.champion_j))
        if matchup_value is None or champion_name == enemy.champion_j:
            continue
        contribution_rows.append(
            {
                "enemy_champion": enemy.champion_j,
                "enemy_frequency": float(enemy.freq_j),
                "matchup_value": matchup_value,
                "contribution": float(enemy.freq_j) * matchup_value,
            }
        )
    contributions = pd.DataFrame(contribution_rows)
    if not contributions.empty:
        contributions = contributions.sort_values(
            by=["contribution", "enemy_champion"], ascending=[False, True]
        )

    baseline = build_baseline_profile(
        loaded=loaded,
        selected_champion=champion_name,
        candidates=candidates,
    )

    return {
        "champion": champion_name,
        "blind_score": blind,
        "summary": summary,
        "top_matchup_contributions": contributions.head(10).reset_index(drop=True),
        "profile": baseline,
    }


def build_baseline_profile(
    loaded: LoadedInputs,
    selected_champion: str,
    candidates: list[str] | None = None,
) -> pd.DataFrame:
    champion_pool = candidates or sorted(loaded.matchup_df["champion_i"].unique())
    rows = []
    for enemy in loaded.frequency_df.sort_values(
        by=["freq_j", "champion_j"], ascending=[False, True]
    ).itertuples(index=False):
        values = [
            loaded.matchup_lookup[(champion, enemy.champion_j)]
            for champion in champion_pool
            if champion != enemy.champion_j and (champion, enemy.champion_j) in loaded.matchup_lookup
        ]
        selected_value = loaded.matchup_lookup.get((selected_champion, enemy.champion_j))
        if selected_value is None or not values:
            continue
        baseline_contribution = float(enemy.freq_j) * (sum(values) / len(values))
        champion_contribution = float(enemy.freq_j) * selected_value
        rows.append(
            {
                "enemy_champion": enemy.champion_j,
                "baseline_contribution": baseline_contribution,
                "champion_contribution": champion_contribution,
                "delta": champion_contribution - baseline_contribution,
            }
        )
    return pd.DataFrame(rows)
