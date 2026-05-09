from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from experiment_config import ColumnMapping, MATCHUP_MAPPING, PatchFiles, SUMMARY_MAPPING


@dataclass(frozen=True)
class StandardizedPatchData:
    patch_label: str
    summary_df: pd.DataFrame
    matchup_df: pd.DataFrame
    enemy_weights: pd.DataFrame


def _load_csv(path: str | pd.io.common.FilePath | object) -> pd.DataFrame:
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, required_columns: Iterable[str], file_label: str) -> None:
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{file_label} is missing required columns: {', '.join(missing)}")


def _normalize_rate(series: pd.Series, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"Column {column_name} contains non-numeric values")
    if (numeric > 1).any():
        numeric = numeric / 100.0
    if ((numeric < 0) | (numeric > 1)).any():
        raise ValueError(f"Column {column_name} is outside [0, 1] after normalization")
    return numeric


def _normalize_count(series: pd.Series, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"Column {column_name} contains non-numeric values")
    if (numeric < 0).any():
        raise ValueError(f"Column {column_name} contains negative values")
    return numeric.astype(float)


def standardize_summary(summary_df: pd.DataFrame, mapping: ColumnMapping, file_label: str) -> pd.DataFrame:
    """Map a real champion-summary CSV into a stable internal schema."""
    _require_columns(
        summary_df,
        [
            mapping.champion,
            mapping.champion_normalized,
            mapping.overall_winrate,
            mapping.pickrate,
            mapping.banrate,
            mapping.total_games,
        ],
        file_label,
    )
    standardized = pd.DataFrame(
        {
            "champion_name": summary_df[mapping.champion].astype(str).str.strip(),
            "champion_id": summary_df[mapping.champion_normalized].astype(str).str.strip(),
            "overall_winrate": _normalize_rate(summary_df[mapping.overall_winrate], mapping.overall_winrate),
            "pick_rate": _normalize_rate(summary_df[mapping.pickrate], mapping.pickrate),
            "ban_rate": _normalize_rate(summary_df[mapping.banrate], mapping.banrate),
            "total_games": _normalize_count(summary_df[mapping.total_games], mapping.total_games),
        }
    )
    standardized = standardized.drop_duplicates(subset=["champion_id"]).reset_index(drop=True)
    return standardized


def standardize_matchups(matchup_df: pd.DataFrame, mapping: ColumnMapping, file_label: str) -> pd.DataFrame:
    """Map a real matchup CSV into a stable internal schema."""
    _require_columns(
        matchup_df,
        [
            mapping.champion,
            mapping.champion_normalized,
            mapping.enemy,
            mapping.enemy_normalized,
            mapping.matchup_winrate,
            mapping.matchup_games,
        ],
        file_label,
    )
    standardized = pd.DataFrame(
        {
            "champion_name": matchup_df[mapping.champion].astype(str).str.strip(),
            "champion_id": matchup_df[mapping.champion_normalized].astype(str).str.strip(),
            "enemy_name": matchup_df[mapping.enemy].astype(str).str.strip(),
            "enemy_id": matchup_df[mapping.enemy_normalized].astype(str).str.strip(),
            "matchup_winrate": _normalize_rate(matchup_df[mapping.matchup_winrate], mapping.matchup_winrate),
            "matchup_games": _normalize_count(matchup_df[mapping.matchup_games], mapping.matchup_games),
        }
    )
    standardized = standardized.drop_duplicates(subset=["champion_id", "enemy_id"]).reset_index(drop=True)
    return standardized


def compute_enemy_weights(matchup_df: pd.DataFrame) -> pd.DataFrame:
    """Build normalized enemy-champion weights from matchup game counts."""
    weights = (
        matchup_df.groupby(["enemy_id", "enemy_name"], as_index=False)["matchup_games"]
        .sum()
        .rename(columns={"matchup_games": "enemy_games"})
    )
    weights["weight"] = weights["enemy_games"] / weights["enemy_games"].sum()
    return weights.sort_values(["weight", "enemy_id"], ascending=[False, True]).reset_index(drop=True)


def load_patch_data(files: PatchFiles) -> StandardizedPatchData:
    """Load one patch's summary and matchup files plus derived enemy weights."""
    raw_summary = _load_csv(files.summary_path)
    raw_matchups = _load_csv(files.matchup_path)
    summary_df = standardize_summary(raw_summary, SUMMARY_MAPPING, str(files.summary_path))
    matchup_df = standardize_matchups(raw_matchups, MATCHUP_MAPPING, str(files.matchup_path))
    enemy_weights = compute_enemy_weights(matchup_df)
    return StandardizedPatchData(
        patch_label=files.patch_label,
        summary_df=summary_df,
        matchup_df=matchup_df,
        enemy_weights=enemy_weights,
    )


def intersect_patch_data(
    patch_a: StandardizedPatchData,
    patch_b: StandardizedPatchData,
) -> tuple[StandardizedPatchData, StandardizedPatchData]:
    """Restrict both patches to the shared champion universe before modeling."""
    common_ids = sorted(
        set(patch_a.summary_df["champion_id"]).intersection(set(patch_b.summary_df["champion_id"]))
    )

    def _filter_patch(patch: StandardizedPatchData) -> StandardizedPatchData:
        summary_df = patch.summary_df[patch.summary_df["champion_id"].isin(common_ids)].copy()
        matchup_df = patch.matchup_df[
            patch.matchup_df["champion_id"].isin(common_ids)
            & patch.matchup_df["enemy_id"].isin(common_ids)
        ].copy()
        enemy_weights = compute_enemy_weights(matchup_df)
        return StandardizedPatchData(
            patch_label=patch.patch_label,
            summary_df=summary_df.reset_index(drop=True),
            matchup_df=matchup_df.reset_index(drop=True),
            enemy_weights=enemy_weights,
        )

    return _filter_patch(patch_a), _filter_patch(patch_b)
