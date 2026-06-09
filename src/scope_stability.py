from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from data_loader import (
    build_matchup_lookup,
    load_clean_frequency_data,
    load_clean_matchup_data,
    load_clean_summary_data,
)
from matchup_estimator import DEFAULT_EB_ALPHA, apply_matchup_estimator
from optimizer import rank_top_pools
from scoring import pool_score
from utils import canonicalize_champion_name


RANK_ORDER = {
    "plat_plus": 0,
    "emerald_plus": 1,
    "diamond_plus": 2,
    "master_plus": 3,
}
FOCUS_CHAMPIONS = ("Sion", "Pantheon")
AGGREGATE_MATCHUP_PATTERN = re.compile(
    r"^opgg_mid_matchups__(?P<rank>[a-z_]+)__(?P<patch>\d+\.\d+)\.csv$"
)


class UnsafeRankBucketError(ValueError):
    """Raised when cumulative scopes cannot be subtracted exactly."""


@dataclass(frozen=True)
class ScopeFiles:
    patch: str
    rank_scope: str
    matchup_path: Path
    summary_path: Path
    frequency_path: Path | None
    source_format: str

    @property
    def scope_id(self) -> str:
        return f"{self.patch}__{self.rank_scope}__cumulative"


@dataclass(frozen=True)
class LoadedScope:
    files: ScopeFiles
    matchup_df: pd.DataFrame
    summary_df: pd.DataFrame
    frequency_df: pd.DataFrame
    matchup_lookup: dict[tuple[str, str], float]
    eb_mu: float


@dataclass(frozen=True)
class ScopeStabilityArtifacts:
    scope_summary: Path
    best_pools_by_scope: Path
    champion_inclusion_by_scope: Path
    exclusion_loss_by_scope: Path
    sion_pantheon_matchup_stability: Path
    report: Path


def _patch_sort_key(patch: str) -> tuple[int, ...]:
    return tuple(int(part) for part in patch.split("."))


def discover_scope_files(
    prepared_data_dir: Path,
    aggregate_data_dir: Path,
) -> list[ScopeFiles]:
    """
    Discover paired patch/rank inputs.

    Explicit aggregate patch/rank files take precedence over prepared patch
    folders when both represent the same cumulative scope.
    """
    discovered: dict[tuple[str, str], tuple[int, ScopeFiles]] = {}

    if prepared_data_dir.exists():
        for patch_dir in prepared_data_dir.iterdir():
            if not patch_dir.is_dir():
                continue
            matchup_path = patch_dir / "opgg_mid_matchups_clean.csv"
            summary_path = patch_dir / "opgg_mid_champion_summary.csv"
            frequency_path = patch_dir / "enemy_freq_df.csv"
            if not (matchup_path.exists() and summary_path.exists() and frequency_path.exists()):
                continue
            scope = ScopeFiles(
                patch=patch_dir.name,
                rank_scope="plat_plus",
                matchup_path=matchup_path,
                summary_path=summary_path,
                frequency_path=frequency_path,
                source_format="prepared_patch_directory",
            )
            discovered[(scope.patch, scope.rank_scope)] = (1, scope)

    if aggregate_data_dir.exists():
        for matchup_path in aggregate_data_dir.glob("opgg_mid_matchups__*__*.csv"):
            match = AGGREGATE_MATCHUP_PATTERN.match(matchup_path.name)
            if match is None:
                continue
            patch = match.group("patch")
            rank_scope = match.group("rank")
            summary_path = aggregate_data_dir / (
                f"opgg_mid_champion_summary__{rank_scope}__{patch}.csv"
            )
            if not summary_path.exists():
                continue
            scope = ScopeFiles(
                patch=patch,
                rank_scope=rank_scope,
                matchup_path=matchup_path,
                summary_path=summary_path,
                frequency_path=None,
                source_format="aggregate_patch_rank_pair",
            )
            key = (scope.patch, scope.rank_scope)
            if key not in discovered or discovered[key][0] < 2:
                discovered[key] = (2, scope)

        raw_dir = aggregate_data_dir / "raw"
        if raw_dir.exists():
            for matchup_path in raw_dir.glob("opgg_mid_matchups*.csv"):
                summary_path = matchup_path.with_name(
                    matchup_path.name.replace("matchups", "champion_summary")
                )
                if not summary_path.exists():
                    continue
                metadata = pd.read_csv(
                    summary_path,
                    nrows=1,
                    dtype={"patch": str, "elo": str, "lane": str},
                )
                if metadata.empty or not {"patch", "elo"}.issubset(metadata.columns):
                    continue
                scope = ScopeFiles(
                    patch=str(metadata["patch"].iloc[0]).strip(),
                    rank_scope=str(metadata["elo"].iloc[0]).strip(),
                    matchup_path=matchup_path,
                    summary_path=summary_path,
                    frequency_path=None,
                    source_format="dated_raw_patch_rank_pair",
                )
                key = (scope.patch, scope.rank_scope)
                if key not in discovered:
                    discovered[key] = (0, scope)

    return sorted(
        (item[1] for item in discovered.values()),
        key=lambda scope: (
            _patch_sort_key(scope.patch),
            RANK_ORDER.get(scope.rank_scope, 99),
            scope.rank_scope,
        ),
    )


def _validate_scope_metadata(scope: ScopeFiles) -> None:
    if scope.source_format == "prepared_patch_directory":
        return

    for path in (scope.matchup_path, scope.summary_path):
        frame = pd.read_csv(
            path,
            nrows=20,
            dtype={"patch": str, "elo": str, "lane": str},
        )
        for column, expected in (
            ("patch", scope.patch),
            ("elo", scope.rank_scope),
            ("lane", "mid"),
        ):
            if column not in frame.columns:
                raise ValueError(f"{path.name} is missing metadata column {column}")
            values = set(frame[column].astype(str).str.strip())
            if values != {expected}:
                raise ValueError(
                    f"{path.name} has {column}={sorted(values)}, expected {expected}"
                )
        if "source_url" in frame.columns:
            urls = frame["source_url"].dropna().astype(str)
            tier_token = {
                "plat_plus": "platinum_plus",
            }.get(scope.rank_scope, scope.rank_scope)
            expected_fragments = [f"tier={tier_token}"]
            if scope.source_format == "aggregate_patch_rank_pair":
                expected_fragments.append(f"patch={scope.patch}")
            for fragment in expected_fragments:
                if not urls.str.contains(fragment, regex=False).all():
                    raise ValueError(
                        f"{path.name} source URLs do not consistently contain {fragment}"
                    )


def _frequency_from_matchup_counts(matchup_df: pd.DataFrame) -> pd.DataFrame:
    observed = matchup_df[matchup_df["champion_i"] != matchup_df["champion_j"]]
    frequency_df = (
        observed.groupby("champion_j", as_index=False)["games_ij"]
        .sum()
        .rename(columns={"games_ij": "count_j"})
    )
    total = float(frequency_df["count_j"].sum())
    if total <= 0:
        raise ValueError("Matchup-derived enemy frequency total must be positive")
    frequency_df["freq_j"] = frequency_df["count_j"] / total
    return frequency_df


def load_scope(
    scope: ScopeFiles,
    eb_alpha: float = DEFAULT_EB_ALPHA,
    eb_mu: float | None = None,
) -> LoadedScope:
    _validate_scope_metadata(scope)
    matchup_df = load_clean_matchup_data(scope.matchup_path)
    summary_df = load_clean_summary_data(scope.summary_path)
    if scope.frequency_path is None:
        frequency_df = _frequency_from_matchup_counts(matchup_df)
    else:
        frequency_df, _ = load_clean_frequency_data(scope.frequency_path)
    estimated, resolved_mu = apply_matchup_estimator(
        matchup_df,
        estimator="eb",
        eb_alpha=eb_alpha,
        eb_mu=eb_mu,
    )
    return LoadedScope(
        files=scope,
        matchup_df=estimated,
        summary_df=summary_df,
        frequency_df=frequency_df,
        matchup_lookup=build_matchup_lookup(estimated),
        eb_mu=resolved_mu,
    )


def common_candidate_names(scopes: Sequence[LoadedScope]) -> list[str]:
    if not scopes:
        return []
    key_sets = [
        {
            canonicalize_champion_name(champion)
            for champion in scope.matchup_df["champion_i"].unique()
        }
        for scope in scopes
    ]
    common_keys = set.intersection(*key_sets)
    display_lookup = {
        canonicalize_champion_name(champion): str(champion)
        for champion in scopes[0].matchup_df["champion_i"].unique()
    }
    return sorted(display_lookup[key] for key in common_keys)


def subtract_cumulative_rank_bucket(
    lower_scope: pd.DataFrame,
    higher_scope: pd.DataFrame,
    key_columns: Sequence[str] = ("champion_i", "champion_j"),
    count_columns: Sequence[str] = ("games_ij", "wins_i"),
) -> pd.DataFrame:
    """
    Subtract nested cumulative scopes only when exact additive counts exist.

    Rounded rates are deliberately unsupported because inferred wins are not
    mathematically additive.
    """
    required = set(key_columns).union(count_columns)
    for label, frame in (("lower", lower_scope), ("higher", higher_scope)):
        missing = required - set(frame.columns)
        if missing:
            raise UnsafeRankBucketError(
                f"{label} scope lacks exact additive columns: {', '.join(sorted(missing))}"
            )
        if frame.duplicated(subset=list(key_columns)).any():
            raise UnsafeRankBucketError(f"{label} scope contains duplicate keys")

    merged = lower_scope[list(key_columns) + list(count_columns)].merge(
        higher_scope[list(key_columns) + list(count_columns)],
        on=list(key_columns),
        how="outer",
        suffixes=("_lower", "_higher"),
        indicator=True,
    )
    higher_only = merged["_merge"] == "right_only"
    if higher_only.any():
        raise UnsafeRankBucketError(
            "Higher cumulative scope contains keys absent from the lower scope"
        )
    merged = merged.fillna(0.0)

    output = merged[list(key_columns)].copy()
    for column in count_columns:
        difference = merged[f"{column}_lower"] - merged[f"{column}_higher"]
        if (difference < -1e-9).any():
            raise UnsafeRankBucketError(
                f"Higher cumulative scope exceeds lower scope for {column}"
            )
        output[column] = difference.clip(lower=0.0)

    if {"games_ij", "wins_i"}.issubset(output.columns):
        if (output["wins_i"] > output["games_ij"] + 1e-9).any():
            raise UnsafeRankBucketError("Subtracted wins exceed subtracted games")
        output = output[output["games_ij"] > 0].copy()
        output["winrate_ij"] = output["wins_i"] / output["games_ij"]
    return output.reset_index(drop=True)


def assess_disjoint_bucket_safety(scopes: Sequence[ScopeFiles]) -> tuple[bool, str]:
    aggregate_scopes = [
        scope for scope in scopes if scope.source_format == "aggregate_patch_rank_pair"
    ]
    if not aggregate_scopes:
        return False, "No cumulative aggregate patch/rank files were discovered."

    reasons: list[str] = []
    sample_columns = set(pd.read_csv(aggregate_scopes[0].matchup_path, nrows=0).columns)
    if "wins_i" not in sample_columns:
        reasons.append(
            "matchup files contain rounded win rates and matchup games but no "
            "exact win counts, so inferred wins are not exactly additive"
        )

    by_patch: dict[str, dict[str, ScopeFiles]] = {}
    for scope in aggregate_scopes:
        by_patch.setdefault(scope.patch, {})[scope.rank_scope] = scope
    for patch, patch_scopes in by_patch.items():
        ordered = [
            patch_scopes[rank]
            for rank in RANK_ORDER
            if rank in patch_scopes
        ]
        for lower, higher in zip(ordered, ordered[1:]):
            lower_keys = set(
                map(
                    tuple,
                    pd.read_csv(lower.matchup_path)[
                        ["champion_i_normalized", "champion_j_normalized"]
                    ].to_numpy(),
                )
            )
            higher_keys = set(
                map(
                    tuple,
                    pd.read_csv(higher.matchup_path)[
                        ["champion_i_normalized", "champion_j_normalized"]
                    ].to_numpy(),
                )
            )
            if not higher_keys.issubset(lower_keys):
                missing_count = len(higher_keys - lower_keys)
                reasons.append(
                    f"{patch} {higher.rank_scope} contains {missing_count} matchup "
                    f"keys absent from {lower.rank_scope}, consistent with "
                    "minimum-game row censoring"
                )
    if reasons:
        return False, "Unsafe: " + "; ".join(reasons) + "."
    return True, "Exact additive counts and nested matchup keys are available."


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def pairwise_mean_jaccard(sets: Sequence[Iterable[str]]) -> float:
    normalized = [set(values) for values in sets]
    if len(normalized) < 2:
        return np.nan
    return float(
        np.mean(
            [
                jaccard_similarity(left, right)
                for left, right in combinations(normalized, 2)
            ]
        )
    )


def build_inclusion_frequency(
    scope_id: str,
    ranked_pools: pd.DataFrame,
    candidates: Sequence[str],
) -> pd.DataFrame:
    pools = [tuple(pool) for pool in ranked_pools["pool"]]
    rows: list[dict[str, object]] = []
    for champion in candidates:
        ranks = [index + 1 for index, pool in enumerate(pools) if champion in pool]
        rows.append(
            {
                "scope_id": scope_id,
                "champion": champion,
                "top_100_appearances": len(ranks),
                "top_100_inclusion_frequency": len(ranks) / len(pools),
                "best_pool_member": champion in pools[0],
                "best_pool_rank_with_champion": min(ranks) if ranks else np.nan,
                "mean_top_100_rank_with_champion": (
                    float(np.mean(ranks)) if ranks else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def build_exclusion_loss(
    scope: LoadedScope,
    candidates: Sequence[str],
    pool_size: int,
    ranked_pools: pd.DataFrame,
    inclusion_df: pd.DataFrame,
) -> pd.DataFrame:
    baseline_pool = tuple(ranked_pools.iloc[0]["pool"])
    baseline_score = float(ranked_pools.iloc[0]["pool_score"])
    available_by_key = {
        canonicalize_champion_name(champion): champion for champion in candidates
    }
    scenarios = [
        ("exclude_sion", ("sion",)),
        ("exclude_pantheon", ("pantheon",)),
        ("exclude_sion_and_pantheon", ("sion", "pantheon")),
    ]
    rows: list[dict[str, object]] = []
    for scenario, requested_keys in scenarios:
        excluded = tuple(
            available_by_key[key] for key in requested_keys if key in available_by_key
        )
        excluded_set = set(excluded)
        if not excluded or excluded_set.isdisjoint(baseline_pool):
            best_pool = baseline_pool
            best_score = baseline_score
            search_method = ranked_pools.attrs.get("search_method", "")
        else:
            remaining = [
                champion for champion in candidates if champion not in excluded_set
            ]
            reranked = rank_top_pools(
                remaining,
                pool_size,
                scope.frequency_df,
                scope.matchup_lookup,
                top_n=1,
            )
            best_pool = tuple(reranked.iloc[0]["pool"])
            best_score = float(reranked.iloc[0]["pool_score"])
            search_method = reranked.attrs.get("search_method", "")
        loss = baseline_score - best_score
        inclusion_lookup = inclusion_df.set_index(
            inclusion_df["champion"].map(canonicalize_champion_name)
        )
        rows.append(
            {
                "scope_id": scope.files.scope_id,
                "patch": scope.files.patch,
                "rank_scope": scope.files.rank_scope,
                "rank_scope_type": "cumulative",
                "scenario": scenario,
                "excluded_champions": ", ".join(excluded),
                "baseline_best_pool": ", ".join(baseline_pool),
                "baseline_best_score": baseline_score,
                "best_pool_after_exclusion": ", ".join(best_pool),
                "best_score_after_exclusion": best_score,
                "score_loss": loss,
                "relative_score_loss": loss / baseline_score if baseline_score else np.nan,
                "sion_top_100_inclusion_frequency": (
                    float(inclusion_lookup.loc["sion", "top_100_inclusion_frequency"])
                    if "sion" in inclusion_lookup.index
                    else np.nan
                ),
                "pantheon_top_100_inclusion_frequency": (
                    float(
                        inclusion_lookup.loc[
                            "pantheon", "top_100_inclusion_frequency"
                        ]
                    )
                    if "pantheon" in inclusion_lookup.index
                    else np.nan
                ),
                "search_method": search_method,
            }
        )
    return pd.DataFrame(rows)


def build_focus_matchup_contributions(
    scope: LoadedScope,
    best_pool: tuple[str, ...],
    focus_champions: Sequence[str] = FOCUS_CHAMPIONS,
    top_k: int = 10,
) -> pd.DataFrame:
    pool_by_key = {
        canonicalize_champion_name(champion): champion for champion in best_pool
    }
    focus_by_key = {
        canonicalize_champion_name(champion): champion for champion in focus_champions
    }
    rows: list[dict[str, object]] = []
    matchup_lookup = {
        (row.champion_i, row.champion_j): row
        for row in scope.matchup_df.itertuples(index=False)
    }

    for focus_key, requested_name in focus_by_key.items():
        local_name = pool_by_key.get(focus_key)
        if local_name is None:
            rows.append(
                {
                    "scope_id": scope.files.scope_id,
                    "patch": scope.files.patch,
                    "rank_scope": scope.files.rank_scope,
                    "rank_scope_type": "cumulative",
                    "champion": requested_name,
                    "best_pool_member": False,
                    "contribution_rank": np.nan,
                    "enemy_champion": "",
                    "marginal_contribution": 0.0,
                    "weighted_eb_value": 0.0,
                    "enemy_frequency": np.nan,
                    "matchup_games": np.nan,
                    "raw_winrate": np.nan,
                    "eb_winrate": np.nan,
                    "absolute_shrinkage": np.nan,
                }
            )
            continue

        contributions: list[dict[str, object]] = []
        for enemy_row in scope.frequency_df.itertuples(index=False):
            enemy = str(enemy_row.champion_j)
            values = [
                (
                    champion,
                    float(scope.matchup_lookup[(champion, enemy)]),
                )
                for champion in best_pool
                if champion != enemy and (champion, enemy) in scope.matchup_lookup
            ]
            if not values:
                continue
            values.sort(key=lambda item: (item[1], item[0]), reverse=True)
            if values[0][0] != local_name:
                continue
            second_best = values[1][1] if len(values) > 1 else scope.eb_mu
            row = matchup_lookup[(local_name, enemy)]
            frequency = float(enemy_row.freq_j)
            contributions.append(
                {
                    "scope_id": scope.files.scope_id,
                    "patch": scope.files.patch,
                    "rank_scope": scope.files.rank_scope,
                    "rank_scope_type": "cumulative",
                    "champion": local_name,
                    "best_pool_member": True,
                    "enemy_champion": enemy,
                    "marginal_contribution": frequency
                    * max(0.0, float(row.winrate_ij) - second_best),
                    "weighted_eb_value": frequency * float(row.winrate_ij),
                    "enemy_frequency": frequency,
                    "matchup_games": float(row.games_ij),
                    "raw_winrate": float(row.raw_winrate),
                    "eb_winrate": float(row.winrate_ij),
                    "absolute_shrinkage": abs(float(row.shrinkage_amount)),
                }
            )

        contributions.sort(
            key=lambda item: (
                -float(item["marginal_contribution"]),
                str(item["enemy_champion"]),
            )
        )
        for rank, row in enumerate(contributions[:top_k], start=1):
            row["contribution_rank"] = rank
            rows.append(row)
        if not contributions:
            rows.append(
                {
                    "scope_id": scope.files.scope_id,
                    "patch": scope.files.patch,
                    "rank_scope": scope.files.rank_scope,
                    "rank_scope_type": "cumulative",
                    "champion": local_name,
                    "best_pool_member": True,
                    "contribution_rank": np.nan,
                    "enemy_champion": "",
                    "marginal_contribution": 0.0,
                    "weighted_eb_value": 0.0,
                    "enemy_frequency": np.nan,
                    "matchup_games": np.nan,
                    "raw_winrate": np.nan,
                    "eb_winrate": np.nan,
                    "absolute_shrinkage": np.nan,
                }
            )
    return pd.DataFrame(rows)


def add_focus_matchup_stability(
    contribution_df: pd.DataFrame,
) -> pd.DataFrame:
    result = contribution_df.copy()
    result["top_matchup_set"] = ""
    result["mean_pairwise_top_matchup_jaccard"] = np.nan
    result["matchup_stability_label"] = "insufficient"

    for champion, champion_df in result.groupby("champion"):
        scope_sets: dict[str, set[str]] = {}
        for scope_id, scope_df in champion_df.groupby("scope_id"):
            enemies = {
                str(enemy)
                for enemy in scope_df["enemy_champion"]
                if str(enemy).strip()
            }
            if enemies:
                scope_sets[str(scope_id)] = enemies
            result.loc[scope_df.index, "top_matchup_set"] = ", ".join(
                sorted(enemies)
            )
        mean_jaccard = pairwise_mean_jaccard(list(scope_sets.values()))
        if pd.isna(mean_jaccard):
            label = "insufficient"
        elif mean_jaccard >= 0.60:
            label = "stable"
        elif mean_jaccard >= 0.35:
            label = "mixed"
        else:
            label = "unstable"
        result.loc[
            result["champion"] == champion,
            "mean_pairwise_top_matchup_jaccard",
        ] = mean_jaccard
        result.loc[
            result["champion"] == champion,
            "matchup_stability_label",
        ] = label
    return result


def _uncertainty_summary(
    matchup_df: pd.DataFrame,
    eb_alpha: float,
) -> dict[str, object]:
    observed = matchup_df[matchup_df["champion_i"] != matchup_df["champion_j"]]
    games = observed["games_ij"].astype(float)
    shrinkage = observed["shrinkage_amount"].abs().astype(float)
    fraction_below_prior = float((games < eb_alpha).mean())
    if float(games.median()) < 1.5 * eb_alpha or fraction_below_prior >= 0.35:
        warning_level = "high"
    elif float(games.quantile(0.10)) < eb_alpha or fraction_below_prior >= 0.10:
        warning_level = "moderate"
    elif fraction_below_prior >= 0.05:
        warning_level = "watch"
    else:
        warning_level = "low"
    warning = {
        "high": (
            "Many matchup rows have sample sizes near or below the EB prior "
            "strength; rankings are materially shrinkage-dependent."
        ),
        "moderate": (
            "The lower tail of matchup sample sizes is below the EB prior "
            "strength; interpret close pool scores cautiously."
        ),
        "watch": (
            "A noticeable minority of matchup rows are below the EB prior "
            "strength; inspect close recommendations."
        ),
        "low": "Most matchup rows substantially exceed the EB prior strength.",
    }[warning_level]
    return {
        "matchup_games_min": float(games.min()),
        "matchup_games_p10": float(games.quantile(0.10)),
        "matchup_games_median": float(games.median()),
        "fraction_matchups_below_eb_alpha": fraction_below_prior,
        "mean_absolute_shrinkage": float(shrinkage.mean()),
        "p90_absolute_shrinkage": float(shrinkage.quantile(0.90)),
        "uncertainty_warning_level": warning_level,
        "uncertainty_warning": warning,
    }


def _scope_similarity_columns(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    result = summary_df.copy()
    pools = {
        row.scope_id: tuple(str(row.best_pool).split(", "))
        for row in result.itertuples(index=False)
    }
    for index, row in result.iterrows():
        others = [
            jaccard_similarity(pools[row["scope_id"]], pool)
            for scope_id, pool in pools.items()
            if scope_id != row["scope_id"]
        ]
        same_patch = [
            jaccard_similarity(
                pools[row["scope_id"]],
                pools[other.scope_id],
            )
            for other in result.itertuples(index=False)
            if other.scope_id != row["scope_id"] and other.patch == row["patch"]
        ]
        same_rank = [
            jaccard_similarity(
                pools[row["scope_id"]],
                pools[other.scope_id],
            )
            for other in result.itertuples(index=False)
            if other.scope_id != row["scope_id"]
            and other.rank_scope == row["rank_scope"]
        ]
        result.loc[index, "mean_best_pool_jaccard_all_scopes"] = (
            float(np.mean(others)) if others else np.nan
        )
        result.loc[index, "mean_best_pool_jaccard_same_patch"] = (
            float(np.mean(same_patch)) if same_patch else np.nan
        )
        result.loc[index, "mean_best_pool_jaccard_same_rank"] = (
            float(np.mean(same_rank)) if same_rank else np.nan
        )
    return result


def _format_percent(value: object) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.3%}"


def write_scope_stability_report(
    path: Path,
    scope_summary: pd.DataFrame,
    best_pools: pd.DataFrame,
    inclusion: pd.DataFrame,
    exclusion: pd.DataFrame,
    focus_matchups: pd.DataFrame,
    common_candidates: Sequence[str],
    disjoint_safe: bool,
    disjoint_reason: str,
) -> None:
    best_sets = [
        set(str(pool).split(", ")) for pool in scope_summary["best_pool"]
    ]
    overall_jaccard = pairwise_mean_jaccard(best_sets)
    exact_pair_rate = float(
        np.mean(
            [
                left == right
                for left, right in combinations(best_sets, 2)
            ]
        )
    )
    within_patch_jaccards = [
        jaccard_similarity(
            str(left.best_pool).split(", "),
            str(right.best_pool).split(", "),
        )
        for left, right in combinations(
            list(scope_summary.itertuples(index=False)),
            2,
        )
        if left.patch == right.patch
    ]
    same_rank_jaccards = [
        jaccard_similarity(
            str(left.best_pool).split(", "),
            str(right.best_pool).split(", "),
        )
        for left, right in combinations(
            list(scope_summary.itertuples(index=False)),
            2,
        )
        if left.rank_scope == right.rank_scope
    ]
    within_patch_jaccard = float(np.mean(within_patch_jaccards))
    same_rank_jaccard = float(np.mean(same_rank_jaccards))
    focus_lines: list[str] = []
    for champion in FOCUS_CHAMPIONS:
        champion_inclusion = inclusion[inclusion["champion"] == champion]
        champion_exclusion = exclusion[
            exclusion["scenario"] == f"exclude_{champion.lower()}"
        ]
        champion_matchups = focus_matchups[
            focus_matchups["champion"] == champion
        ]
        stability_label = (
            str(champion_matchups["matchup_stability_label"].iloc[0])
            if not champion_matchups.empty
            else "insufficient"
        )
        matchup_jaccard = (
            champion_matchups["mean_pairwise_top_matchup_jaccard"].iloc[0]
            if not champion_matchups.empty
            else np.nan
        )
        focus_lines.append(
            f"- **{champion}:** best-pool member in "
            f"{int(champion_inclusion['best_pool_member'].sum())}/"
            f"{len(champion_inclusion)} scopes; median top-100 inclusion "
            f"{_format_percent(champion_inclusion['top_100_inclusion_frequency'].median())}; "
            f"median exclusion loss "
            f"{_format_percent(champion_exclusion['score_loss'].median())}; "
            f"top-matchup stability `{stability_label}` "
            f"(mean pairwise Jaccard "
            f"{'n/a' if pd.isna(matchup_jaccard) else f'{float(matchup_jaccard):.3f}'})."
        )

    uncertainty_lines = []
    for row in scope_summary.itertuples(index=False):
        if row.uncertainty_warning_level in {"high", "moderate"}:
            uncertainty_lines.append(
                f"- `{row.scope_id}`: **{row.uncertainty_warning_level}** warning; "
                f"median matchup games {row.matchup_games_median:.0f}, "
                f"{row.fraction_matchups_below_eb_alpha:.1%} below alpha."
            )

    patch_rank_lines = [
        f"- `{row.scope_id}`: **{row.best_pool}** "
        f"({_format_percent(row.best_pool_score)})."
        for row in scope_summary.itertuples(index=False)
    ]
    top10_lines = []
    for scope_id, frame in best_pools.groupby("scope_id", sort=False):
        labels = "; ".join(
            f"{int(row.pool_rank)}. {row.pool_label} "
            f"({_format_percent(row.pool_score)})"
            for row in frame.itertuples(index=False)
        )
        top10_lines.append(f"- `{scope_id}`: {labels}")

    concern_direction = (
        "The results weaken the narrow concern that the recommendation is an "
        "artifact of one patch/rank scope, but they do not remove the broader "
        "selection-bias concern."
        if overall_jaccard >= 0.50
        else "The results strengthen the selection-bias concern as a robustness "
        "issue because recommendations change materially with patch/rank scope. "
        "They do not establish that selection bias caused the instability."
    )

    lines = [
        "# Scope Stability Report",
        "",
        "## Technical Summary",
        "",
        f"- Discovered and validated **{len(scope_summary)} cumulative patch/rank scopes**.",
        f"- All scopes use the EB estimator with a fixed common candidate universe "
        f"of **{len(common_candidates)} champions** and deterministic exact search.",
        f"- Mean pairwise Jaccard similarity of best pools is **{overall_jaccard:.3f}**; "
        f"the exact best-pool match rate across scope pairs is **{exact_pair_rate:.1%}**.",
        f"- Mean best-pool Jaccard is **{within_patch_jaccard:.3f} across ranks "
        f"within the same patch** and **{same_rank_jaccard:.3f} across patches "
        "within the same cumulative rank label**.",
        f"- {concern_direction}",
        "- These are descriptive robustness/stability diagnostics. They do not "
        "causally correct selection bias.",
        "",
        "## Best EB Pools Vary Across Cumulative Scopes",
        "",
        *patch_rank_lines,
        "",
        "## Sion And Pantheon Dependence",
        "",
        *focus_lines,
        "",
        "Exclusion loss is the deterministic EB score difference between the "
        "unrestricted best pool and the best pool after removing the named "
        "champion(s). A zero loss means the unrestricted optimum remains feasible.",
        "",
        "## High-Rank Scopes Carry More Shrinkage Dependence",
        "",
        *(
            uncertainty_lines
            or ["- No scope crossed the moderate/high uncertainty thresholds."]
        ),
        "",
        "Warnings compare observed matchup games with the EB prior strength. They "
        "describe sampling support and shrinkage dependence, not total model error.",
        "",
        "## Cumulative Scopes Are Not Disjoint Rank Buckets",
        "",
        f"- Disjoint subtraction status: **{'safe' if disjoint_safe else 'not safe'}**.",
        f"- {disjoint_reason}",
        "- Therefore `Plat+`, `Emerald+`, `Diamond+`, and `Master+` are reported "
        "as overlapping cumulative populations. No Plat-only, Emerald-only, or "
        "Diamond-only estimates were generated.",
        "",
        "## Top 10 Pools Per Scope",
        "",
        *top10_lines,
        "",
        "## Method",
        "",
        "- Scope discovery pairs matchup and champion-summary files by patch and "
        "rank, deduplicating prepared and aggregate representations.",
        f"- The {len(common_candidates)}-champion intersection is used in every "
        "scope so recommendation "
        "changes are not caused by candidate availability.",
        "- Enemy weights come from prepared frequency files when available and "
        "otherwise from normalized aggregate opponent matchup counts.",
        "- Matchup values use empirical Bayes shrinkage "
        "`(wins + alpha * mu) / (games + alpha)` with fractional wins inferred "
        "from the published aggregate win rate.",
        "- Pool stability uses Jaccard similarity of deterministic best-pool sets. "
        "Focus-matchup stability uses Jaccard similarity of each champion's top "
        "marginal-contribution enemy sets while that champion is in the best pool.",
        "",
        "## Limitations And Interpretation",
        "",
        "- OP.GG aggregate rows are observational summaries and may reflect player "
        "specialization, pick timing, counterpick behavior, team composition, "
        "survivorship, and source-specific filtering.",
        "- Rounded win rates imply fractional inferred wins. This is acceptable "
        "for EB robustness diagnostics but is not exact event-level reconstruction.",
        "- Higher-rank scopes have smaller matchup samples and can be more sensitive "
        "to the prior even after shrinkage.",
        "- Stability across overlapping cumulative scopes is not independent "
        "replication, because higher ranks are contained in lower rank scopes.",
        "- No Riot Match-V5 or live Riot API data was used.",
        "",
        "## Recommended Next Steps",
        "",
        "- Treat pools that remain near the top across scopes as robust candidates, "
        "while inspecting close score gaps rather than over-reading rank order.",
        "- Preserve cumulative-scope labels in downstream reporting and avoid "
        "describing them as rank-specific disjoint samples.",
        "- If exact wins/losses and uncensored nested matchup keys become available, "
        "rerun the guarded subtraction path before considering disjoint buckets.",
        "",
        "## Further Questions",
        "",
        "- Are the same recommendations stable under alternative EB prior strengths?",
        "- Do fixed candidate restrictions based on practical champion ownership or "
        "role suitability change Sion/Pantheon dependence?",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scope_stability_workflow(
    prepared_data_dir: Path,
    aggregate_data_dir: Path,
    output_dir: Path,
    pool_size: int = 3,
    top_pool_count: int = 100,
    eb_alpha: float = DEFAULT_EB_ALPHA,
    eb_mu: float | None = None,
) -> ScopeStabilityArtifacts:
    scope_files = discover_scope_files(prepared_data_dir, aggregate_data_dir)
    if not scope_files:
        raise ValueError("No paired patch/rank scopes were discovered")
    loaded_scopes = [
        load_scope(scope, eb_alpha=eb_alpha, eb_mu=eb_mu)
        for scope in scope_files
    ]
    candidates = common_candidate_names(loaded_scopes)
    if len(candidates) < pool_size:
        raise ValueError("Common candidate universe is smaller than pool size")

    summary_rows: list[dict[str, object]] = []
    best_pool_frames: list[pd.DataFrame] = []
    inclusion_frames: list[pd.DataFrame] = []
    exclusion_frames: list[pd.DataFrame] = []
    focus_frames: list[pd.DataFrame] = []

    for scope in loaded_scopes:
        local_by_key = {
            canonicalize_champion_name(champion): champion
            for champion in scope.matchup_df["champion_i"].unique()
        }
        local_candidates = [
            local_by_key[canonicalize_champion_name(champion)]
            for champion in candidates
        ]
        ranked = rank_top_pools(
            local_candidates,
            pool_size,
            scope.frequency_df,
            scope.matchup_lookup,
            top_n=top_pool_count,
        )
        best_pool = tuple(ranked.iloc[0]["pool"])
        top10 = ranked.head(10).copy()
        top10.insert(0, "pool_rank", np.arange(1, len(top10) + 1))
        top10.insert(0, "rank_scope_type", "cumulative")
        top10.insert(0, "rank_scope", scope.files.rank_scope)
        top10.insert(0, "patch", scope.files.patch)
        top10.insert(0, "scope_id", scope.files.scope_id)
        top10["score_loss_vs_best"] = (
            float(ranked.iloc[0]["pool_score"]) - top10["pool_score"]
        )
        best_pool_frames.append(top10.drop(columns=["pool"]))

        inclusion = build_inclusion_frequency(
            scope.files.scope_id,
            ranked,
            local_candidates,
        )
        inclusion.insert(1, "patch", scope.files.patch)
        inclusion.insert(2, "rank_scope", scope.files.rank_scope)
        inclusion.insert(3, "rank_scope_type", "cumulative")
        inclusion_frames.append(inclusion)
        exclusion_frames.append(
            build_exclusion_loss(
                scope,
                local_candidates,
                pool_size,
                ranked,
                inclusion,
            )
        )
        focus_frames.append(
            build_focus_matchup_contributions(scope, best_pool)
        )

        uncertainty = _uncertainty_summary(scope.matchup_df, eb_alpha)
        summary_rows.append(
            {
                "scope_id": scope.files.scope_id,
                "patch": scope.files.patch,
                "rank_scope": scope.files.rank_scope,
                "rank_scope_type": "cumulative",
                "source_format": scope.files.source_format,
                "matchup_file": str(scope.files.matchup_path),
                "summary_file": str(scope.files.summary_path),
                "frequency_source": (
                    str(scope.files.frequency_path)
                    if scope.files.frequency_path is not None
                    else "normalized aggregate opponent matchup counts"
                ),
                "available_candidate_count": int(
                    scope.matchup_df["champion_i"].nunique()
                ),
                "common_candidate_count": len(local_candidates),
                "enemy_count": int(scope.frequency_df["champion_j"].nunique()),
                "matchup_row_count": int(
                    (
                        scope.matchup_df["champion_i"]
                        != scope.matchup_df["champion_j"]
                    ).sum()
                ),
                "pool_size": pool_size,
                "top_pool_count": len(ranked),
                "estimator": "eb",
                "eb_alpha": eb_alpha,
                "eb_mu": scope.eb_mu,
                "search_method": ranked.attrs.get("search_method", ""),
                "evaluated_pool_count": ranked.attrs.get(
                    "evaluated_pool_count", np.nan
                ),
                "best_pool": ", ".join(best_pool),
                "best_pool_score": float(ranked.iloc[0]["pool_score"]),
                "top_10_score_range": float(
                    ranked.iloc[0]["pool_score"] - ranked.iloc[9]["pool_score"]
                ),
                **uncertainty,
            }
        )

    scope_summary = _scope_similarity_columns(pd.DataFrame(summary_rows))
    best_pools = pd.concat(best_pool_frames, ignore_index=True)
    inclusion = pd.concat(inclusion_frames, ignore_index=True)
    exclusion = pd.concat(exclusion_frames, ignore_index=True)
    focus_matchups = add_focus_matchup_stability(
        pd.concat(focus_frames, ignore_index=True)
    )
    disjoint_safe, disjoint_reason = assess_disjoint_bucket_safety(scope_files)
    scope_summary["disjoint_rank_bucket_status"] = (
        "safe" if disjoint_safe else "not_safe"
    )
    scope_summary["disjoint_rank_bucket_reason"] = disjoint_reason

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = ScopeStabilityArtifacts(
        scope_summary=output_dir / "scope_summary.csv",
        best_pools_by_scope=output_dir / "best_pools_by_scope.csv",
        champion_inclusion_by_scope=output_dir / "champion_inclusion_by_scope.csv",
        exclusion_loss_by_scope=output_dir / "exclusion_loss_by_scope.csv",
        sion_pantheon_matchup_stability=(
            output_dir / "sion_pantheon_matchup_stability.csv"
        ),
        report=output_dir / "scope_stability_report.md",
    )
    scope_summary.to_csv(artifacts.scope_summary, index=False)
    best_pools.to_csv(artifacts.best_pools_by_scope, index=False)
    inclusion.to_csv(artifacts.champion_inclusion_by_scope, index=False)
    exclusion.to_csv(artifacts.exclusion_loss_by_scope, index=False)
    focus_matchups.to_csv(
        artifacts.sion_pantheon_matchup_stability,
        index=False,
    )
    write_scope_stability_report(
        artifacts.report,
        scope_summary,
        best_pools,
        inclusion,
        exclusion,
        focus_matchups,
        candidates,
        disjoint_safe,
        disjoint_reason,
    )
    return artifacts
