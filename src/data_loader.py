from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from matchup_estimator import (
    DEFAULT_EB_ALPHA,
    EstimatorName,
    apply_matchup_estimator,
)
from utils import canonicalize_champion_name


SYNTHETIC_MATCHUP_REQUIRED_COLUMNS = {
    "champion_i",
    "champion_j",
    "games_ij",
    "wins_i",
    "winrate_ij",
}
SYNTHETIC_FREQUENCY_REQUIRED_COLUMNS = {"champion_j", "count_j", "freq_j"}

CLEAN_MATCHUP_REQUIRED_COLUMNS = {
    "champion_i",
    "champion_j",
    "matchup_games",
    "matchup_winrate_i_vs_j",
}
CLEAN_FREQUENCY_REQUIRED_COLUMNS = {
    "champion_j",
    "f_j",
}
CLEAN_SUMMARY_REQUIRED_COLUMNS = {
    "champion_name",
    "pickrate",
    "winrate",
}
OPTIONAL_SUMMARY_NUMERIC_COLUMNS = (
    "winrate",
    "banrate",
    "total_games",
    "depth",
    "worst10_mean",
    "weighted_cvar_10",
)
OPTIONAL_SUMMARY_RATE_COLUMNS = {"winrate", "pickrate", "banrate", "worst10_mean", "weighted_cvar_10"}
OPTIONAL_SUMMARY_TEXT_COLUMNS = (
    "lane",
    "patch",
    "elo",
    "source_url",
    "extraction_date",
)


@dataclass(frozen=True)
class LoadedInputs:
    patch_label: str
    matchup_df: pd.DataFrame
    frequency_df: pd.DataFrame
    summary_df: pd.DataFrame
    matchup_lookup: Dict[Tuple[str, str], float]
    champion_count: int
    matchup_row_count: int
    frequency_status: str
    estimator: EstimatorName
    eb_alpha: float
    eb_mu: float


@dataclass(frozen=True)
class PatchDataPaths:
    patch_label: str
    patch_dir: Path
    matchup_path: Path
    frequency_path: Path
    summary_path: Path


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file from disk with a clearer error message."""
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{name} is missing required columns: {missing_list}")


def _coerce_numeric(series: pd.Series, column_name: str, file_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{file_name} contains non-numeric values in {column_name}")
    return numeric


def _normalize_rate_column(series: pd.Series) -> pd.Series:
    if (series > 1).any():
        return series / 100.0
    return series


def _validate_standardized_matchups(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{file_name} is empty")

    if (df["games_ij"] <= 0).any():
        raise ValueError(f"{file_name} contains non-positive games_ij values")

    if ((df["wins_i"] < 0) | (df["wins_i"] > df["games_ij"])).any():
        raise ValueError(f"{file_name} contains wins_i outside the valid range")

    if ((df["winrate_ij"] < 0) | (df["winrate_ij"] > 1)).any():
        raise ValueError(f"{file_name} contains winrate_ij outside [0, 1]")

    duplicate_rows = df.duplicated(subset=["champion_i", "champion_j"])
    if duplicate_rows.any():
        duplicates = df.loc[duplicate_rows, ["champion_i", "champion_j"]]
        raise ValueError(
            f"{file_name} contains duplicate matchup rows: "
            f"{duplicates.to_dict(orient='records')}"
        )

    return df


def load_synthetic_matchup_data(path: Path) -> pd.DataFrame:
    df = load_csv(path)
    validate_columns(df, SYNTHETIC_MATCHUP_REQUIRED_COLUMNS, "matchup_data.csv")
    return _validate_standardized_matchups(df.copy(), "matchup_data.csv")


def load_synthetic_frequency_data(path: Path) -> pd.DataFrame:
    df = load_csv(path)
    validate_columns(df, SYNTHETIC_FREQUENCY_REQUIRED_COLUMNS, "enemy_frequency.csv")
    return _normalize_frequency_df(df.copy(), "enemy_frequency.csv")


def _normalize_frequency_df(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{file_name} is empty")

    df["count_j"] = _coerce_numeric(df["count_j"], "count_j", file_name)
    df["freq_j"] = _coerce_numeric(df["freq_j"], "freq_j", file_name)

    if (df["count_j"] < 0).any():
        raise ValueError(f"{file_name} contains negative count_j values")

    if (df["freq_j"] < 0).any():
        raise ValueError(f"{file_name} contains negative freq_j values")

    duplicate_rows = df.duplicated(subset=["champion_j"])
    if duplicate_rows.any():
        duplicates = df.loc[duplicate_rows, ["champion_j"]]
        raise ValueError(
            f"{file_name} contains duplicate enemy champions: "
            f"{duplicates.to_dict(orient='records')}"
        )

    total_freq = float(df["freq_j"].sum())
    if total_freq <= 0:
        raise ValueError(f"{file_name} frequency total must be positive")

    normalized_df = df.copy()
    normalized_df["freq_j"] = normalized_df["freq_j"] / total_freq
    return normalized_df


def build_matchup_lookup(matchup_df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Build W(i, j) from the prepared v1 matchup input."""
    return {
        (row.champion_i, row.champion_j): float(row.winrate_ij)
        for row in matchup_df.itertuples(index=False)
    }


def validate_candidate_coverage(
    candidates: Iterable[str],
    enemy_champions: Iterable[str],
    matchup_lookup: Dict[Tuple[str, str], float],
) -> None:
    missing_pairs = collect_missing_matchup_pairs(candidates, enemy_champions, matchup_lookup)
    if missing_pairs:
        preview = ", ".join(f"{i} vs {j}" for i, j in missing_pairs[:10])
        raise ValueError(
            "Missing matchup values for some candidate/enemy pairs. "
            f"First missing entries: {preview}"
        )


def collect_missing_matchup_pairs(
    candidates: Iterable[str],
    enemy_champions: Iterable[str],
    matchup_lookup: Dict[Tuple[str, str], float],
) -> list[tuple[str, str]]:
    missing_pairs = []
    for champion_i in candidates:
        for champion_j in enemy_champions:
            if (champion_i, champion_j) not in matchup_lookup:
                missing_pairs.append((champion_i, champion_j))
    return missing_pairs


def find_complete_coverage_candidates(
    candidates: Iterable[str],
    enemy_champions: Iterable[str],
    matchup_lookup: Dict[Tuple[str, str], float],
) -> tuple[list[str], dict[str, list[str]]]:
    complete_candidates = []
    incomplete_candidates: dict[str, list[str]] = {}

    for champion_i in candidates:
        missing_opponents = [
            champion_j
            for champion_j in enemy_champions
            if (champion_i, champion_j) not in matchup_lookup
        ]
        if missing_opponents:
            incomplete_candidates[champion_i] = missing_opponents
        else:
            complete_candidates.append(champion_i)

    return complete_candidates, incomplete_candidates


def _resolve_clean_path(data_dir: Path, filename: str) -> Path:
    clean_path = data_dir / "clean" / filename
    if clean_path.exists():
        return clean_path
    return data_dir / filename


def resolve_patch_data_paths(data_dir: Path, patch: str) -> PatchDataPaths:
    patch_dir = data_dir / patch
    matchup_path = patch_dir / "opgg_mid_matchups_clean.csv"
    frequency_path = patch_dir / "enemy_freq_df.csv"
    summary_path = patch_dir / "opgg_mid_champion_summary.csv"

    if not patch_dir.exists():
        raise FileNotFoundError(
            f"Patch directory not found: {patch_dir}. Expected data under data/<patch>/"
        )

    if not matchup_path.exists():
        raise FileNotFoundError(
            f"Patch matchup file not found: {matchup_path}"
        )

    if not frequency_path.exists():
        raise FileNotFoundError(
            f"Patch enemy frequency file not found: {frequency_path}"
        )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Patch summary file not found: {summary_path}"
        )

    return PatchDataPaths(
        patch_label=patch,
        patch_dir=patch_dir,
        matchup_path=matchup_path,
        frequency_path=frequency_path,
        summary_path=summary_path,
    )


def load_clean_matchup_data(path: Path) -> pd.DataFrame:
    df = load_csv(path)
    validate_columns(df, CLEAN_MATCHUP_REQUIRED_COLUMNS, path.name)

    standardized_df = pd.DataFrame(
        {
            "champion_i": df["champion_i"].astype(str).str.strip(),
            "champion_j": df["champion_j"].astype(str).str.strip(),
            "games_ij": _coerce_numeric(df["matchup_games"], "matchup_games", path.name),
        }
    )
    standardized_df["winrate_ij"] = _normalize_rate_column(
        _coerce_numeric(df["matchup_winrate_i_vs_j"], "matchup_winrate_i_vs_j", path.name)
    )
    # The source contains a rounded aggregate winrate rather than an exact win
    # count, so wins_i is an inferred (possibly fractional) count.
    standardized_df["wins_i"] = (
        standardized_df["winrate_ij"] * standardized_df["games_ij"]
    )

    self_matchup_rows = []
    all_champions = sorted(
        set(standardized_df["champion_i"]).union(set(standardized_df["champion_j"]))
    )
    existing_pairs = set(
        zip(standardized_df["champion_i"], standardized_df["champion_j"])
    )
    for champion in all_champions:
        if (champion, champion) not in existing_pairs:
            self_matchup_rows.append(
                {
                    "champion_i": champion,
                    "champion_j": champion,
                    "games_ij": 1.0,
                    "wins_i": 0.5,
                    "winrate_ij": 0.5,
                }
            )

    if self_matchup_rows:
        standardized_df = pd.concat(
            [standardized_df, pd.DataFrame(self_matchup_rows)],
            ignore_index=True,
        )

    return _validate_standardized_matchups(standardized_df, path.name)


def load_clean_frequency_data(path: Path) -> tuple[pd.DataFrame, str]:
    df = load_csv(path)
    validate_columns(df, CLEAN_FREQUENCY_REQUIRED_COLUMNS, path.name)

    champion_names = df["champion_j"].astype(str).str.strip()
    duplicate_rows = champion_names.duplicated()
    if duplicate_rows.any():
        duplicates = champion_names[duplicate_rows].tolist()
        raise ValueError(
            f"{path.name} contains duplicate champion_j values: {duplicates[:10]}"
        )

    freq_j = _coerce_numeric(df["f_j"], "f_j", path.name)
    if (freq_j < 0).any():
        raise ValueError(f"{path.name} contains negative f_j values")

    total_freq = float(freq_j.sum())
    if total_freq <= 0:
        raise ValueError(f"{path.name} has a non-positive total f_j sum")

    if abs(total_freq - 1.0) > 1e-6:
        raise ValueError(
            f"{path.name} must already contain normalized enemy weights; "
            f"found sum(f_j)={total_freq:.12f}"
        )

    count_j = (
        _coerce_numeric(df["enemy_total_games"], "enemy_total_games", path.name)
        if "enemy_total_games" in df.columns
        else pd.Series([pd.NA] * len(df))
    )

    frequency_df = pd.DataFrame(
        {
            "champion_j": champion_names,
            "count_j": count_j,
            "freq_j": freq_j,
        }
    )
    normalized_frequency_df = _normalize_frequency_df(frequency_df, path.name)
    return normalized_frequency_df, "present_in_prepared_enemy_frequency_file"


def load_clean_summary_data(path: Path) -> pd.DataFrame:
    df = load_csv(path)
    validate_columns(df, CLEAN_SUMMARY_REQUIRED_COLUMNS, path.name)

    champion_name = df["champion_name"].astype(str).str.strip()
    pickrate = _normalize_rate_column(
        _coerce_numeric(df["pickrate"], "pickrate", path.name)
    )
    winrate = _normalize_rate_column(
        _coerce_numeric(df["winrate"], "winrate", path.name)
    )

    if ((pickrate < 0) | (pickrate > 1)).any():
        raise ValueError(f"{path.name} contains pickrate values outside [0, 1]")

    if ((winrate < 0) | (winrate > 1)).any():
        raise ValueError(f"{path.name} contains winrate values outside [0, 1]")

    normalized_name = (
        df["champion_name_normalized"].astype(str).str.strip()
        if "champion_name_normalized" in df.columns
        else champion_name.map(canonicalize_champion_name)
    )

    summary_df = pd.DataFrame(
        {
            "champion_name": champion_name,
            "champion_name_normalized": normalized_name,
            "pickrate": pickrate,
            "winrate": winrate,
            "champion_key": champion_name.map(canonicalize_champion_name),
        }
    )

    for column in OPTIONAL_SUMMARY_NUMERIC_COLUMNS:
        if column in df.columns and column != "pickrate":
            values = _coerce_numeric(df[column], column, path.name)
            if column in OPTIONAL_SUMMARY_RATE_COLUMNS:
                values = _normalize_rate_column(values)
            summary_df[column] = values
    for column in OPTIONAL_SUMMARY_TEXT_COLUMNS:
        if column in df.columns:
            summary_df[column] = df[column].astype(str).str.strip()

    if summary_df["champion_key"].duplicated().any():
        duplicates = (
            summary_df.loc[summary_df["champion_key"].duplicated(), "champion_name"]
            .tolist()
        )
        raise ValueError(
            f"{path.name} contains duplicate champion mappings after normalization: {duplicates[:10]}"
        )

    return summary_df


def merge_enemy_frequencies_into_matchups(
    matchup_df: pd.DataFrame,
    frequency_df: pd.DataFrame,
) -> pd.DataFrame:
    merged_df = matchup_df.merge(
        frequency_df[["champion_j", "freq_j"]],
        on="champion_j",
        how="left",
    )
    missing_frequencies = merged_df["freq_j"].isna()
    if missing_frequencies.any():
        missing_champions = (
            merged_df.loc[missing_frequencies, "champion_j"].drop_duplicates().tolist()
        )
        raise ValueError(
            "Prepared enemy frequency data is missing weights for some matchup enemies: "
            + ", ".join(missing_champions[:10])
        )
    return merged_df


def load_clean_inputs(
    data_dir: Path,
    estimator: EstimatorName = "raw",
    eb_alpha: float = DEFAULT_EB_ALPHA,
    eb_mu: float | None = None,
) -> LoadedInputs:
    matchup_path = _resolve_clean_path(data_dir, "opgg_mid_matchups_clean.csv")
    frequency_path = _resolve_clean_path(data_dir, "enemy_freq_df.csv")
    summary_path = _resolve_clean_path(data_dir, "opgg_mid_champion_summary.csv")

    matchup_df = load_clean_matchup_data(matchup_path)
    frequency_df, frequency_status = load_clean_frequency_data(frequency_path)
    summary_df = load_clean_summary_data(summary_path)
    matchup_df = merge_enemy_frequencies_into_matchups(matchup_df, frequency_df)
    matchup_df, resolved_mu = apply_matchup_estimator(
        matchup_df,
        estimator=estimator,
        eb_alpha=eb_alpha,
        eb_mu=eb_mu,
    )
    matchup_lookup = build_matchup_lookup(matchup_df)

    return LoadedInputs(
        patch_label="clean",
        matchup_df=matchup_df,
        frequency_df=frequency_df,
        summary_df=summary_df,
        matchup_lookup=matchup_lookup,
        champion_count=int(matchup_df["champion_i"].nunique()),
        matchup_row_count=len(matchup_df),
        frequency_status=frequency_status,
        estimator=estimator,
        eb_alpha=eb_alpha,
        eb_mu=resolved_mu,
    )


def load_synthetic_inputs(
    data_dir: Path,
    estimator: EstimatorName = "raw",
    eb_alpha: float = DEFAULT_EB_ALPHA,
    eb_mu: float | None = None,
) -> LoadedInputs:
    matchup_path = data_dir / "matchup_data.csv"
    frequency_path = data_dir / "enemy_frequency.csv"

    matchup_df = load_synthetic_matchup_data(matchup_path)
    frequency_df = load_synthetic_frequency_data(frequency_path)
    matchup_df, resolved_mu = apply_matchup_estimator(
        matchup_df,
        estimator=estimator,
        eb_alpha=eb_alpha,
        eb_mu=eb_mu,
    )
    matchup_lookup = build_matchup_lookup(matchup_df)

    return LoadedInputs(
        patch_label="synthetic",
        matchup_df=matchup_df,
        frequency_df=frequency_df,
        summary_df=pd.DataFrame(),
        matchup_lookup=matchup_lookup,
        champion_count=int(matchup_df["champion_i"].nunique()),
        matchup_row_count=len(matchup_df),
        frequency_status="present_in_synthetic_frequency_file",
        estimator=estimator,
        eb_alpha=eb_alpha,
        eb_mu=resolved_mu,
    )


def load_patch_data(
    patch: str,
    data_dir: Path,
    estimator: EstimatorName = "raw",
    eb_alpha: float = DEFAULT_EB_ALPHA,
    eb_mu: float | None = None,
) -> LoadedInputs:
    """
    Load a single patch from data/<patch>/.

    This keeps patch loading isolated so we can later compose it into
    train-patch vs eval-patch workflows without changing the scoring code.
    """
    patch_paths = resolve_patch_data_paths(data_dir, patch)
    matchup_df = load_clean_matchup_data(patch_paths.matchup_path)
    frequency_df, frequency_status = load_clean_frequency_data(patch_paths.frequency_path)
    summary_df = load_clean_summary_data(patch_paths.summary_path)
    matchup_df = merge_enemy_frequencies_into_matchups(matchup_df, frequency_df)
    matchup_df, resolved_mu = apply_matchup_estimator(
        matchup_df,
        estimator=estimator,
        eb_alpha=eb_alpha,
        eb_mu=eb_mu,
    )
    matchup_lookup = build_matchup_lookup(matchup_df)

    return LoadedInputs(
        patch_label=patch_paths.patch_label,
        matchup_df=matchup_df,
        frequency_df=frequency_df,
        summary_df=summary_df,
        matchup_lookup=matchup_lookup,
        champion_count=int(matchup_df["champion_i"].nunique()),
        matchup_row_count=len(matchup_df),
        frequency_status=frequency_status,
        estimator=estimator,
        eb_alpha=eb_alpha,
        eb_mu=resolved_mu,
    )


def load_inputs(
    data_dir: Path,
    dataset: str = "clean",
    estimator: EstimatorName = "raw",
    eb_alpha: float = DEFAULT_EB_ALPHA,
    eb_mu: float | None = None,
) -> LoadedInputs:
    if dataset == "clean":
        return load_clean_inputs(data_dir, estimator, eb_alpha, eb_mu)
    if dataset == "synthetic":
        return load_synthetic_inputs(data_dir, estimator, eb_alpha, eb_mu)
    raise ValueError(f"Unsupported dataset mode: {dataset}")
