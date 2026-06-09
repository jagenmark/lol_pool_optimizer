from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import (
    LoadedInputs,
    build_matchup_lookup,
    load_clean_matchup_data,
    load_clean_summary_data,
    load_patch_data,
    merge_enemy_frequencies_into_matchups,
)
from matchup_estimator import apply_matchup_estimator
from optimizer import rank_top_pools
from scoring import pool_score
from selection_bias import build_source_stability
from scope_stability import discover_scope_files
from uncertainty import build_matchup_posteriors, simulate_pool_scores
from utils import canonicalize_champion_name


FOCUS_CHAMPIONS = ("Sion", "Pantheon")
DEFAULT_ALPHA_VALUES = (0.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0)
DEFAULT_OFFMETA_LAMBDAS = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1)
NORMAL_5TH_Z = 1.6448536269514722


@dataclass(frozen=True)
class MethodSweepArtifacts:
    report: Path
    summary: Path
    fixed_policy_simulation_summary: Path
    alpha_sensitivity: Path
    enemy_frequency_sensitivity: Path
    scope_stability: Path
    offmeta_penalty_sensitivity: Path
    contribution_concentration: Path
    residual_model_summary: Path
    robust_objective_comparison: Path


@dataclass(frozen=True)
class AggregateScope:
    scope_id: str
    patch: str
    rank: str
    source_name: str
    summary_path: Path
    matchup_path: Path
    loaded: LoadedInputs


@dataclass(frozen=True)
class PoolMatrix:
    pools: list[tuple[str, ...]]
    pool_labels: list[str]
    pool_indices: np.ndarray
    champion_names: list[str]
    enemy_names: list[str]
    enemy_frequencies: np.ndarray
    pool_values: np.ndarray
    usable_weight_sums: np.ndarray


def _pool_label(pool: tuple[str, ...]) -> str:
    return ", ".join(pool)


def _contains_focus(pool: object, champion: str) -> bool:
    if isinstance(pool, tuple):
        return champion in pool
    return champion in [part.strip() for part in str(pool).split(",")]


def _rate(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if (numeric > 1).any():
        numeric = numeric / 100.0
    return numeric


def _frequency_from_matchups(matchup_df: pd.DataFrame) -> pd.DataFrame:
    real = matchup_df[matchup_df["champion_i"] != matchup_df["champion_j"]]
    frequency = (
        real.groupby("champion_j", as_index=False)["games_ij"]
        .sum()
        .rename(columns={"games_ij": "count_j"})
    )
    frequency["freq_j"] = frequency["count_j"] / frequency["count_j"].sum()
    return frequency.sort_values(["freq_j", "champion_j"], ascending=[False, True]).reset_index(drop=True)


def _loaded_from_paths(
    patch: str,
    rank: str,
    summary_path: Path,
    matchup_path: Path,
    estimator: str,
    eb_alpha: float,
    eb_mu: float | None,
) -> LoadedInputs:
    matchup_df = load_clean_matchup_data(matchup_path)
    frequency_df = _frequency_from_matchups(matchup_df)
    summary_df = load_clean_summary_data(summary_path)
    matchup_df = merge_enemy_frequencies_into_matchups(matchup_df, frequency_df)
    matchup_df, resolved_mu = apply_matchup_estimator(
        matchup_df,
        estimator=estimator,  # type: ignore[arg-type]
        eb_alpha=eb_alpha,
        eb_mu=eb_mu,
    )
    return LoadedInputs(
        patch_label=patch,
        matchup_df=matchup_df,
        frequency_df=frequency_df,
        summary_df=summary_df,
        matchup_lookup=build_matchup_lookup(matchup_df),
        champion_count=int(matchup_df["champion_i"].nunique()),
        matchup_row_count=len(matchup_df),
        frequency_status="derived_from_matchup_games",
        estimator=estimator,  # type: ignore[arg-type]
        eb_alpha=eb_alpha,
        eb_mu=resolved_mu,
    )


def discover_local_opgg_scopes(
    data_dir: Path,
    extra_data_dir: Path | None,
    estimator: str,
    eb_alpha: float,
    eb_mu: float | None,
) -> list[AggregateScope]:
    """Find local OP.GG aggregate scope pairs without making network calls."""
    aggregate_dir = extra_data_dir or data_dir / "__no_aggregate_scope_dir__"
    scopes: list[AggregateScope] = []
    for discovered in discover_scope_files(data_dir, aggregate_dir):
        source = {
            "prepared_patch_directory": "opgg_patch_folder",
            "aggregate_patch_rank_pair": "opgg_local_extract",
            "dated_raw_patch_rank_pair": "opgg_raw_extract",
        }[discovered.source_format]
        loaded = (
            load_patch_data(
                discovered.patch,
                data_dir,
                estimator=estimator,
                eb_alpha=eb_alpha,
                eb_mu=eb_mu,
            )
            if source == "opgg_patch_folder"
            else _loaded_from_paths(
                discovered.patch,
                discovered.rank_scope,
                discovered.summary_path,
                discovered.matchup_path,
                estimator,
                eb_alpha,
                eb_mu,
            )
        )
        scopes.append(
            AggregateScope(
                scope_id=f"opgg_{discovered.rank_scope}_{discovered.patch}",
                patch=discovered.patch,
                rank=discovered.rank_scope,
                source_name=source,
                summary_path=discovered.summary_path,
                matchup_path=discovered.matchup_path,
                loaded=loaded,
            )
        )
    return scopes


def build_pool_matrix(
    candidates: Sequence[str],
    pool_size: int,
    enemy_frequencies: pd.DataFrame,
    matchup_lookup: dict[tuple[str, str], float],
    pools: Sequence[tuple[str, ...]] | None = None,
) -> PoolMatrix:
    champion_names = list(candidates)
    enemy_names = [str(enemy) for enemy in enemy_frequencies["champion_j"].tolist()]
    champion_index = {champion: index for index, champion in enumerate(champion_names)}
    enemy_frequencies_array = enemy_frequencies["freq_j"].to_numpy(dtype=float)
    champion_values = np.full((len(champion_names), len(enemy_names)), np.nan, dtype=float)
    for c_index, champion in enumerate(champion_names):
        for e_index, enemy in enumerate(enemy_names):
            if champion == enemy:
                continue
            value = matchup_lookup.get((champion, enemy))
            if value is not None:
                champion_values[c_index, e_index] = value

    pool_list = list(pools) if pools is not None else list(combinations(champion_names, pool_size))
    pool_indices = np.array(
        [[champion_index[champion] for champion in pool] for pool in pool_list],
        dtype=int,
    )
    pool_values = np.empty((len(pool_list), len(enemy_names)), dtype=float)
    usable_weight_sums = np.empty(len(pool_list), dtype=float)
    for index, indices in enumerate(pool_indices):
        selected = champion_values[indices, :]
        usable = np.any(~np.isnan(selected), axis=0)
        values = np.max(np.where(np.isnan(selected), -np.inf, selected), axis=0)
        values[~usable] = np.nan
        pool_values[index] = values
        usable_weight_sums[index] = float(enemy_frequencies_array[usable].sum())
    return PoolMatrix(
        pools=pool_list,
        pool_labels=[_pool_label(pool) for pool in pool_list],
        pool_indices=pool_indices,
        champion_names=champion_names,
        enemy_names=enemy_names,
        enemy_frequencies=enemy_frequencies_array,
        pool_values=pool_values,
        usable_weight_sums=usable_weight_sums,
    )


def score_pool_matrix(pool_matrix: PoolMatrix, enemy_frequencies: np.ndarray | None = None) -> np.ndarray:
    frequencies = pool_matrix.enemy_frequencies if enemy_frequencies is None else enemy_frequencies
    usable = ~np.isnan(pool_matrix.pool_values)
    weighted_values = np.nan_to_num(pool_matrix.pool_values, nan=0.0) @ frequencies
    usable_weights = usable.astype(float) @ frequencies
    return weighted_values / usable_weights


def top_pool_frame_from_scores(
    pool_matrix: PoolMatrix,
    scores: np.ndarray,
    score_column: str,
    top_n: int,
) -> pd.DataFrame:
    order = np.lexsort((np.array(pool_matrix.pool_labels), -scores))[:top_n]
    rows = [
        {
            "rank": rank,
            "pool": pool_matrix.pools[index],
            "pool_label": pool_matrix.pool_labels[index],
            score_column: float(scores[index]),
        }
        for rank, index in enumerate(order, start=1)
    ]
    return pd.DataFrame(rows)


def champion_inclusion_from_ranked(
    ranked: pd.DataFrame,
    candidates: Sequence[str],
    top_n: int,
) -> pd.DataFrame:
    top = ranked.head(top_n)
    rows = []
    for champion in candidates:
        ranks = [
            int(row.rank)
            for row in top.itertuples(index=False)
            if _contains_focus(row.pool, champion)
        ]
        rows.append(
            {
                "champion": champion,
                "top_pool_appearances": len(ranks),
                "top_pool_share": len(ranks) / len(top) if len(top) else 0.0,
                "best_rank": min(ranks) if ranks else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_fixed_policy_simulations(
    ranked_pools: pd.DataFrame,
    loaded: LoadedInputs,
    prior_strength: float,
    sample_count: int,
    seed: int,
    top_n: int,
) -> pd.DataFrame:
    posterior = build_matchup_posteriors(
        loaded.matchup_df,
        prior_strength=prior_strength,
        prior_mean=loaded.eb_mu,
    )
    pools = [tuple(pool) for pool in ranked_pools.head(top_n)["pool"].tolist()]
    summaries = []
    for mode in ("fixed-policy", "oracle"):
        _, summary = simulate_pool_scores(
            pools=pools,
            enemy_frequencies=loaded.frequency_df,
            posterior_df=posterior,
            sample_count=sample_count,
            seed=seed,
            simulation_mode=mode,  # type: ignore[arg-type]
        )
        summary["point_estimate_rank_source"] = f"top_{top_n}_deterministic_{loaded.estimator}"
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True).sort_values(
        ["simulation_mode", "mean_score", "pool"], ascending=[True, False, True]
    ).reset_index(drop=True)


def run_alpha_sensitivity(
    loaded: LoadedInputs,
    candidates: list[str],
    pool_size: int,
    top_n: int,
    alpha_values: Sequence[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for alpha in alpha_values:
        estimated, mu = apply_matchup_estimator(
            loaded.matchup_df,
            estimator="eb",
            eb_alpha=float(alpha),
            eb_mu=loaded.eb_mu,
        )
        lookup = build_matchup_lookup(estimated)
        pool_matrix = build_pool_matrix(candidates, pool_size, loaded.frequency_df, lookup)
        scores = score_pool_matrix(pool_matrix)
        ranked = top_pool_frame_from_scores(pool_matrix, scores, "pool_score", top_n)
        best_pool = tuple(ranked.iloc[0]["pool"])
        for row in champion_inclusion_from_ranked(ranked, candidates, top_n).itertuples(index=False):
            rows.append(
                {
                    "alpha": float(alpha),
                    "prior_mean": mu,
                    "champion": row.champion,
                    "best_pool": _pool_label(best_pool),
                    "best_pool_score": float(ranked.iloc[0]["pool_score"]),
                    "best_pool_member": row.champion in best_pool,
                    "top_pool_appearances": int(row.top_pool_appearances),
                    "top_pool_share": float(row.top_pool_share),
                    "best_rank": row.best_rank,
                }
            )
    return pd.DataFrame(rows)


def run_enemy_frequency_sensitivity(
    loaded: LoadedInputs,
    candidates: list[str],
    pool_size: int,
    top_n: int,
    sample_count: int,
    effective_sample_size: float,
    seed: int,
) -> pd.DataFrame:
    pool_matrix = build_pool_matrix(candidates, pool_size, loaded.frequency_df, loaded.matchup_lookup)
    base_scores = score_pool_matrix(pool_matrix)
    base_best_index = int(np.argmax(base_scores))
    base_best_pool = pool_matrix.pools[base_best_index]
    rng = np.random.default_rng(seed)
    alpha = np.maximum(pool_matrix.enemy_frequencies * effective_sample_size, 1e-6)
    champion_wins = {champion: 0 for champion in candidates}
    champion_top_n = {champion: 0 for champion in candidates}
    best_pool_counts: dict[str, int] = {}
    base_best_ranks = []
    base_best_scores = []

    for _ in range(sample_count):
        sampled_frequency = rng.dirichlet(alpha)
        scores = score_pool_matrix(pool_matrix, sampled_frequency)
        best_index = int(np.argmax(scores))
        best_pool = pool_matrix.pools[best_index]
        label = pool_matrix.pool_labels[best_index]
        best_pool_counts[label] = best_pool_counts.get(label, 0) + 1
        for champion in best_pool:
            champion_wins[champion] += 1

        top_indices = np.argpartition(-scores, min(top_n, len(scores) - 1))[:top_n]
        for index in top_indices:
            for champion in pool_matrix.pools[int(index)]:
                champion_top_n[champion] += 1
        base_score = float(scores[base_best_index])
        base_best_scores.append(base_score)
        base_best_ranks.append(int(1 + np.sum(scores > base_score)))

    rows = []
    for champion in candidates:
        rows.append(
            {
                "record_type": "champion",
                "champion": champion,
                "pool": "",
                "base_best_pool": _pool_label(base_best_pool),
                "base_best_score": float(base_scores[base_best_index]),
                "winner_inclusion_rate": champion_wins[champion] / sample_count,
                "top_pool_slot_share": champion_top_n[champion] / (sample_count * top_n),
                "base_best_mean_rank": float(np.mean(base_best_ranks)),
                "base_best_p95_rank": float(np.quantile(base_best_ranks, 0.95)),
                "base_best_mean_score": float(np.mean(base_best_scores)),
                "sample_count": sample_count,
                "effective_sample_size": effective_sample_size,
            }
        )
    for pool, count in sorted(best_pool_counts.items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            {
                "record_type": "winning_pool",
                "champion": "",
                "pool": pool,
                "base_best_pool": _pool_label(base_best_pool),
                "base_best_score": float(base_scores[base_best_index]),
                "winner_inclusion_rate": count / sample_count,
                "top_pool_slot_share": np.nan,
                "base_best_mean_rank": float(np.mean(base_best_ranks)),
                "base_best_p95_rank": float(np.quantile(base_best_ranks, 0.95)),
                "base_best_mean_score": float(np.mean(base_best_scores)),
                "sample_count": sample_count,
                "effective_sample_size": effective_sample_size,
            }
        )
    return pd.DataFrame(rows)


def run_scope_stability(
    scopes: Sequence[AggregateScope],
    candidates: list[str],
    pool_size: int,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    common_keys = {
        canonicalize_champion_name(champion) for champion in candidates
    }
    for scope in scopes:
        common_keys &= {
            canonicalize_champion_name(champion)
            for champion in scope.loaded.matchup_df["champion_i"].unique()
        }
    for scope in scopes:
        local_by_key = {
            canonicalize_champion_name(champion): champion
            for champion in scope.loaded.matchup_df["champion_i"].unique()
        }
        available = [
            local_by_key[canonicalize_champion_name(champion)]
            for champion in candidates
            if canonicalize_champion_name(champion) in common_keys
        ]
        if len(available) < pool_size:
            continue
        pool_matrix = build_pool_matrix(available, pool_size, scope.loaded.frequency_df, scope.loaded.matchup_lookup)
        scores = score_pool_matrix(pool_matrix)
        ranked = top_pool_frame_from_scores(pool_matrix, scores, "pool_score", top_n)
        best_pool = tuple(ranked.iloc[0]["pool"])
        focus_indices = {
            available.index(champion)
            for champion in FOCUS_CHAMPIONS
            if champion in available
        }
        allowed = np.array(
            [
                not any(index in focus_indices for index in pool)
                for pool in pool_matrix.pool_indices
            ],
            dtype=bool,
        )
        pair_best = float(np.nanmax(scores[allowed])) if allowed.any() else np.nan
        pair_drop = (
            float(ranked.iloc[0]["pool_score"]) - pair_best
            if np.isfinite(pair_best)
            else np.nan
        )
        scope_metadata = scope.loaded.summary_df.iloc[0]
        for row in champion_inclusion_from_ranked(ranked, available, top_n).itertuples(index=False):
            if row.champion not in FOCUS_CHAMPIONS:
                continue
            rows.append(
                {
                    "scope_id": scope.scope_id,
                    "patch": scope.patch,
                    "rank": scope.rank,
                    "source_name": scope.source_name,
                    "champion": row.champion,
                    "best_pool": _pool_label(best_pool),
                    "best_pool_score": float(ranked.iloc[0]["pool_score"]),
                    "best_pool_member": row.champion in best_pool,
                    "top_pool_share": float(row.top_pool_share),
                    "best_rank": row.best_rank,
                    "focus_pair_exclusion_drop": pair_drop,
                    "candidate_count": len(available),
                    "role": scope_metadata.get("lane", "mid"),
                    "retrieval_date": scope_metadata.get("extraction_date", ""),
                    "representative_source_url": scope_metadata.get("source_url", ""),
                    "summary_path": str(scope.summary_path),
                    "matchup_path": str(scope.matchup_path),
                }
            )
    return pd.DataFrame(rows)


def build_offmeta_penalties(
    candidates: list[str],
    summary_df: pd.DataFrame,
    source_stability_df: pd.DataFrame,
    baseline_inclusion_df: pd.DataFrame,
    pool_size: int,
) -> pd.DataFrame:
    summary = summary_df.rename(columns={"champion_name": "champion"}).copy()
    summary = summary[summary["champion"].isin(candidates)]
    summary["pickrate"] = _rate(summary["pickrate"])
    frame = pd.DataFrame({"champion": candidates}).merge(
        summary[["champion", "pickrate"]],
        on="champion",
        how="left",
    )
    frame = frame.merge(
        baseline_inclusion_df[["champion", "top_pool_share"]],
        on="champion",
        how="left",
    )
    frame = frame.merge(
        source_stability_df[
            [
                column
                for column in [
                    "champion",
                    "lolalytics_available",
                    "lolalytics_breadth",
                    "lolalytics_depth",
                    "lolalytics_classification",
                    "specialist_heuristic_flag",
                    "lolalytics_source_url",
                    "lolalytics_extraction_date",
                    "lolalytics_pickrate_scope",
                    "lolalytics_depth_scope",
                ]
                if column in source_stability_df.columns
            ]
        ],
        on="champion",
        how="left",
    )
    slot_share = frame["top_pool_share"].fillna(0) / max(1, pool_size)
    pickrate_share = frame["pickrate"] / frame["pickrate"].sum()
    ratio = slot_share / pickrate_share.replace(0, np.nan)
    max_ratio = float(np.nanmax(np.log1p(ratio))) if np.isfinite(np.nanmax(np.log1p(ratio))) else 1.0
    frame["low_pickrate_penalty"] = np.clip((0.02 - frame["pickrate"].fillna(0)) / 0.02, 0, 1)
    frame["importance_ratio_penalty"] = np.log1p(ratio.fillna(0)) / max(max_ratio, 1e-9)
    depth = pd.to_numeric(frame.get("lolalytics_depth", pd.Series(np.nan, index=frame.index)), errors="coerce")
    classification = frame.get("lolalytics_classification", pd.Series("", index=frame.index)).fillna("")
    frame["lolalytics_penalty"] = np.maximum(
        np.clip((depth.fillna(1.0) - 1.0) / 0.5, 0, 1),
        classification.isin(["niche"]).astype(float),
    )
    frame["offmeta_penalty"] = frame[
        ["low_pickrate_penalty", "importance_ratio_penalty", "lolalytics_penalty"]
    ].mean(axis=1)
    return frame


def run_offmeta_penalty_sensitivity(
    loaded: LoadedInputs,
    candidates: list[str],
    pool_size: int,
    top_n: int,
    lambda_values: Sequence[float],
    source_stability_df: pd.DataFrame,
) -> pd.DataFrame:
    pool_matrix = build_pool_matrix(candidates, pool_size, loaded.frequency_df, loaded.matchup_lookup)
    base_scores = score_pool_matrix(pool_matrix)
    baseline_ranked = top_pool_frame_from_scores(pool_matrix, base_scores, "pool_score", top_n)
    baseline_inclusion = champion_inclusion_from_ranked(baseline_ranked, candidates, top_n)
    penalties = build_offmeta_penalties(
        candidates,
        loaded.summary_df,
        source_stability_df,
        baseline_inclusion,
        pool_size,
    )
    penalty_lookup = dict(zip(penalties["champion"], penalties["offmeta_penalty"]))
    champion_penalties = np.array([penalty_lookup[champion] for champion in candidates], dtype=float)
    pool_penalties = champion_penalties[pool_matrix.pool_indices].mean(axis=1)
    rows: list[dict[str, object]] = []
    for lambda_value in lambda_values:
        objective = base_scores - float(lambda_value) * pool_penalties
        ranked = top_pool_frame_from_scores(pool_matrix, objective, "objective_score", top_n)
        score_lookup = {label: score for label, score in zip(pool_matrix.pool_labels, base_scores)}
        best_pool = tuple(ranked.iloc[0]["pool"])
        for row in champion_inclusion_from_ranked(ranked, candidates, top_n).itertuples(index=False):
            penalty_row = penalties[penalties["champion"] == row.champion].iloc[0]
            output_row = {
                "lambda": float(lambda_value),
                "champion": row.champion,
                "best_pool": _pool_label(best_pool),
                "best_pool_point_score": float(score_lookup[_pool_label(best_pool)]),
                "best_pool_objective_score": float(ranked.iloc[0]["objective_score"]),
                "best_pool_member": row.champion in best_pool,
                "top_pool_appearances": int(row.top_pool_appearances),
                "top_pool_share": float(row.top_pool_share),
                "mid_pickrate": float(penalty_row.pickrate),
                "offmeta_penalty": float(penalty_row.offmeta_penalty),
                "low_pickrate_penalty": float(penalty_row.low_pickrate_penalty),
                "importance_ratio_penalty": float(penalty_row.importance_ratio_penalty),
                "lolalytics_penalty": float(penalty_row.lolalytics_penalty),
            }
            for column in [
                "lolalytics_available",
                "lolalytics_breadth",
                "lolalytics_depth",
                "lolalytics_classification",
                "specialist_heuristic_flag",
                "lolalytics_source_url",
                "lolalytics_extraction_date",
                "lolalytics_pickrate_scope",
                "lolalytics_depth_scope",
            ]:
                output_row[column] = penalty_row.get(column, np.nan)
            rows.append(output_row)
    return pd.DataFrame(rows)


def run_contribution_concentration(
    ranked_pools: pd.DataFrame,
    loaded: LoadedInputs,
    top_n: int,
) -> pd.DataFrame:
    posterior = build_matchup_posteriors(loaded.matchup_df, prior_strength=loaded.eb_alpha, prior_mean=loaded.eb_mu)
    posterior_lookup = {
        (row.champion, row.enemy_champion): row
        for row in posterior.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    for pool_row in ranked_pools.head(top_n).itertuples(index=False):
        pool = tuple(pool_row.pool)
        champion_components: dict[str, list[dict[str, float]]] = {champion: [] for champion in pool}
        selected = []
        for enemy_row in loaded.frequency_df.itertuples(index=False):
            enemy = str(enemy_row.champion_j)
            values = sorted(
                [
                    (champion, loaded.matchup_lookup[(champion, enemy)])
                    for champion in pool
                    if champion != enemy and (champion, enemy) in loaded.matchup_lookup
                ],
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )
            if values:
                selected.append((enemy, float(enemy_row.freq_j), values))
        usable_weight = sum(weight for _, weight, _ in selected)
        for enemy, weight, values in selected:
            normalized_weight = weight / usable_weight
            champion, best = values[0]
            second = values[1][1] if len(values) > 1 else 0.5
            posterior_row = posterior_lookup[(champion, enemy)]
            champion_components[champion].append(
                {
                    "enemy_frequency": normalized_weight,
                    "value": float(best),
                    "marginal_lift": normalized_weight * max(0.0, float(best) - float(second)),
                    "contribution": normalized_weight * float(best),
                    "excess_over_50": normalized_weight * (float(best) - 0.5),
                    "games": float(posterior_row.games),
                }
            )
        for champion, components in champion_components.items():
            if not components:
                rows.append(
                    {
                        "pool_rank": int(pool_row.rank) if hasattr(pool_row, "rank") else np.nan,
                        "pool": _pool_label(pool),
                        "pool_score": float(pool_row.pool_score),
                        "champion": champion,
                        "covered_enemy_count": 0,
                        "enemy_frequency_mass_covered": 0.0,
                        "total_contribution": 0.0,
                        "excess_contribution_over_50": 0.0,
                        "top5_marginal_lift_share": np.nan,
                        "effective_matchups": np.nan,
                        "median_matchup_games": np.nan,
                        "min_matchup_games": np.nan,
                    }
                )
                continue
            frame = pd.DataFrame(components)
            lift_total = float(frame["marginal_lift"].sum())
            shares = frame["marginal_lift"] / lift_total if lift_total > 0 else pd.Series(dtype=float)
            hhi = float(np.square(shares).sum()) if len(shares) else np.nan
            rows.append(
                {
                    "pool_rank": int(getattr(pool_row, "rank", np.nan)),
                    "pool": _pool_label(pool),
                    "pool_score": float(pool_row.pool_score),
                    "champion": champion,
                    "covered_enemy_count": int(len(frame)),
                    "enemy_frequency_mass_covered": float(frame["enemy_frequency"].sum()),
                    "total_contribution": float(frame["contribution"].sum()),
                    "excess_contribution_over_50": float(frame["excess_over_50"].sum()),
                    "total_marginal_lift": lift_total,
                    "top5_marginal_lift_share": float(frame["marginal_lift"].nlargest(5).sum() / lift_total) if lift_total > 0 else np.nan,
                    "effective_matchups": float(1.0 / hhi) if pd.notna(hhi) and hhi > 0 else np.nan,
                    "median_matchup_games": float(frame["games"].median()),
                    "min_matchup_games": float(frame["games"].min()),
                }
            )
    return pd.DataFrame(rows)


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-4, 1 - 1e-4)
    return np.log(clipped / (1 - clipped))


def _inv_logit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def fit_residual_model(
    loaded: LoadedInputs,
    candidates: list[str],
    pool_size: int,
    top_n: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str], float]]:
    df = loaded.matchup_df[loaded.matchup_df["champion_i"] != loaded.matchup_df["champion_j"]].copy()
    df["y"] = _logit(df["winrate_ij"].to_numpy(dtype=float))
    df["weight"] = np.maximum(df["games_ij"].to_numpy(dtype=float), 1.0)
    global_mean = float(np.average(df["y"], weights=df["weight"]))
    champions = sorted(df["champion_i"].unique())
    enemies = sorted(df["champion_j"].unique())
    champion_effect = pd.Series(0.0, index=champions)
    enemy_effect = pd.Series(0.0, index=enemies)

    for _ in range(30):
        residual_for_champion = df["y"] - global_mean - df["champion_j"].map(enemy_effect)
        champion_effect = (
            pd.DataFrame({"champion": df["champion_i"], "value": residual_for_champion, "weight": df["weight"]})
            .groupby("champion")
            .apply(lambda group: np.average(group["value"], weights=group["weight"]), include_groups=False)
        )
        champion_effect -= np.average(
            champion_effect,
            weights=df.groupby("champion_i")["weight"].sum().reindex(champion_effect.index).fillna(1.0),
        )
        residual_for_enemy = df["y"] - global_mean - df["champion_i"].map(champion_effect)
        enemy_effect = (
            pd.DataFrame({"enemy": df["champion_j"], "value": residual_for_enemy, "weight": df["weight"]})
            .groupby("enemy")
            .apply(lambda group: np.average(group["value"], weights=group["weight"]), include_groups=False)
        )
        enemy_effect -= np.average(
            enemy_effect,
            weights=df.groupby("champion_j")["weight"].sum().reindex(enemy_effect.index).fillna(1.0),
        )

    df["champion_effect_logit"] = df["champion_i"].map(champion_effect)
    df["enemy_effect_logit"] = df["champion_j"].map(enemy_effect)
    df["fitted_logit"] = global_mean + df["champion_effect_logit"] + df["enemy_effect_logit"]
    df["residual_logit"] = df["y"] - df["fitted_logit"]
    df["residual_adjusted_logit"] = global_mean + df["enemy_effect_logit"] + df["residual_logit"]
    df["residual_adjusted_winrate"] = _inv_logit(df["residual_adjusted_logit"].to_numpy(dtype=float))
    residual_lookup = {
        (row.champion_i, row.champion_j): float(row.residual_adjusted_winrate)
        for row in df.itertuples(index=False)
    }
    # Add neutral self rows where needed so scoring can skip or use 50%.
    for champion in candidates:
        residual_lookup.setdefault((champion, champion), 0.5)

    ranked = rank_top_pools(
        candidates,
        pool_size,
        loaded.frequency_df,
        residual_lookup,
        top_n=top_n,
    )
    rows: list[dict[str, object]] = []
    for champion, effect in champion_effect.sort_values(ascending=False).items():
        if champion in candidates:
            rows.append(
                {
                    "record_type": "champion_effect",
                    "champion": champion,
                    "enemy_champion": "",
                    "pool": "",
                    "rank": np.nan,
                    "score": np.nan,
                    "global_logit_mean": global_mean,
                    "champion_effect_logit": float(effect),
                    "champion_effect_odds_ratio": float(np.exp(effect)),
                    "enemy_effect_logit": np.nan,
                    "residual_logit": np.nan,
                    "observed_winrate": np.nan,
                    "residual_adjusted_winrate": np.nan,
                    "games": np.nan,
                }
            )
    for rank, row in enumerate(ranked.itertuples(index=False), start=1):
        rows.append(
            {
                "record_type": "residual_adjusted_pool",
                "champion": "",
                "enemy_champion": "",
                "pool": row.pool_label,
                "rank": rank,
                "score": float(row.pool_score),
                "global_logit_mean": global_mean,
                "champion_effect_logit": np.nan,
                "champion_effect_odds_ratio": np.nan,
                "enemy_effect_logit": np.nan,
                "residual_logit": np.nan,
                "observed_winrate": np.nan,
                "residual_adjusted_winrate": np.nan,
                "games": np.nan,
            }
        )
    extremes = pd.concat(
        [
            df.sort_values("residual_logit", ascending=False).head(50),
            df.sort_values("residual_logit", ascending=True).head(50),
        ],
        ignore_index=True,
    )
    for row in extremes.itertuples(index=False):
        rows.append(
            {
                "record_type": "matchup_residual",
                "champion": row.champion_i,
                "enemy_champion": row.champion_j,
                "pool": "",
                "rank": np.nan,
                "score": np.nan,
                "global_logit_mean": global_mean,
                "champion_effect_logit": float(row.champion_effect_logit),
                "champion_effect_odds_ratio": float(np.exp(row.champion_effect_logit)),
                "enemy_effect_logit": float(row.enemy_effect_logit),
                "residual_logit": float(row.residual_logit),
                "observed_winrate": float(row.winrate_ij),
                "residual_adjusted_winrate": float(row.residual_adjusted_winrate),
                "games": float(row.games_ij),
            }
        )
    return pd.DataFrame(rows), residual_lookup


def fixed_policy_analytic_pool_stats(
    pools: Sequence[tuple[str, ...]],
    enemy_frequencies: pd.DataFrame,
    posterior_df: pd.DataFrame,
) -> pd.DataFrame:
    posterior_lookup = {
        (row.champion, row.enemy_champion): row
        for row in posterior_df.itertuples(index=False)
    }
    rows = []
    for pool in pools:
        means = []
        variances = []
        weights = []
        for enemy_row in enemy_frequencies.itertuples(index=False):
            enemy = str(enemy_row.champion_j)
            candidates = [
                posterior_lookup[(champion, enemy)]
                for champion in pool
                if champion != enemy and (champion, enemy) in posterior_lookup
            ]
            if not candidates:
                continue
            selected = max(candidates, key=lambda row: (float(row.posterior_mean), row.champion))
            total = float(selected.posterior_alpha + selected.posterior_beta)
            variance = float(
                selected.posterior_alpha
                * selected.posterior_beta
                / (total**2 * (total + 1.0))
            )
            means.append(float(selected.posterior_mean))
            variances.append(variance)
            weights.append(float(enemy_row.freq_j))
        weights_array = np.array(weights, dtype=float)
        normalized = weights_array / weights_array.sum()
        mean = float(np.sum(normalized * np.array(means, dtype=float)))
        sd = float(np.sqrt(np.sum((normalized**2) * np.array(variances, dtype=float))))
        rows.append(
            {
                "pool": pool,
                "pool_label": _pool_label(pool),
                "fixed_policy_mean": mean,
                "fixed_policy_sd": sd,
                "fixed_policy_lower_5_normal": mean - NORMAL_5TH_Z * sd,
                "fixed_policy_mean_minus_1sd": mean - sd,
            }
        )
    return pd.DataFrame(rows)


def run_robust_objective_comparison(
    loaded: LoadedInputs,
    scopes: Sequence[AggregateScope],
    candidates: list[str],
    pool_size: int,
    top_n: int,
) -> pd.DataFrame:
    pools = list(combinations(candidates, pool_size))
    posterior = build_matchup_posteriors(loaded.matchup_df, prior_strength=loaded.eb_alpha, prior_mean=loaded.eb_mu)
    fixed_stats = fixed_policy_analytic_pool_stats(pools, loaded.frequency_df, posterior)
    base_lookup = {_pool_label(pool): score for pool, score in zip(pools, [pool_score(pool, loaded.frequency_df, loaded.matchup_lookup) for pool in pools])}
    rows = []

    objectives = [
        ("deterministic_eb", "deterministic EB point estimate", pd.Series(base_lookup)),
        ("fixed_policy_mean", "fixed-policy posterior mean", fixed_stats.set_index("pool_label")["fixed_policy_mean"]),
        ("fixed_policy_lower_5", "fixed-policy normal-approx lower 5%", fixed_stats.set_index("pool_label")["fixed_policy_lower_5_normal"]),
        ("fixed_policy_mean_minus_1sd", "fixed-policy mean - 1 sd", fixed_stats.set_index("pool_label")["fixed_policy_mean_minus_1sd"]),
    ]
    for objective, definition, scores in objectives:
        best_pool = str(scores.sort_values(ascending=False).index[0])
        rows.append(
            {
                "objective": objective,
                "score_definition": definition,
                "best_pool": best_pool,
                "score": float(scores.loc[best_pool]),
                "sion_in_pool": "Sion" in best_pool.split(", "),
                "pantheon_in_pool": "Pantheon" in best_pool.split(", "),
                "scope_count": 1,
                "notes": "Primary patch 16.07 scope.",
            }
        )

    common_candidates = set(candidates)
    for scope in scopes:
        common_candidates &= set(scope.loaded.matchup_df["champion_i"].unique())
    common_candidates_list = [champion for champion in candidates if champion in common_candidates]
    if len(common_candidates_list) >= pool_size and scopes:
        common_pools = list(combinations(common_candidates_list, pool_size))
        worst_scores = np.full(len(common_pools), np.inf)
        for scope in scopes:
            matrix = build_pool_matrix(common_candidates_list, pool_size, scope.loaded.frequency_df, scope.loaded.matchup_lookup, pools=common_pools)
            scope_scores = score_pool_matrix(matrix)
            worst_scores = np.minimum(worst_scores, scope_scores)
        best_index = int(np.argmax(worst_scores))
        best_pool = _pool_label(common_pools[best_index])
        rows.append(
            {
                "objective": "worst_scope_score",
                "score_definition": "maximize minimum deterministic EB score across local OP.GG scopes",
                "best_pool": best_pool,
                "score": float(worst_scores[best_index]),
                "sion_in_pool": "Sion" in common_pools[best_index],
                "pantheon_in_pool": "Pantheon" in common_pools[best_index],
                "scope_count": len(scopes),
                "notes": "Restricted to champions present in every included scope.",
            }
        )
    return pd.DataFrame(rows)


def build_method_summary(
    fixed_policy_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    freq_df: pd.DataFrame,
    scope_df: pd.DataFrame,
    offmeta_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    robust_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for mode in ("fixed-policy", "oracle"):
        subset = fixed_policy_df[fixed_policy_df["simulation_mode"] == mode]
        if not subset.empty:
            best = subset.sort_values(["mean_score", "pool"], ascending=[False, True]).iloc[0]
            rows.append(
                {
                    "method": f"posterior_{mode}",
                    "score_definition": "posterior simulation mean",
                    "best_pool": best.pool,
                    "score": float(best.mean_score),
                    "sion_in_pool": "Sion" in str(best.pool),
                    "pantheon_in_pool": "Pantheon" in str(best.pool),
                    "notes": "Fixed-policy is practical; oracle is an uncertainty upper bound.",
                }
            )
    for alpha in sorted(alpha_df["alpha"].drop_duplicates()):
        subset = alpha_df[alpha_df["alpha"] == alpha]
        best = subset.iloc[0]
        rows.append(
            {
                "method": f"alpha_{alpha:g}",
                "score_definition": "deterministic EB point estimate",
                "best_pool": best.best_pool,
                "score": float(best.best_pool_score),
                "sion_in_pool": "Sion" in str(best.best_pool),
                "pantheon_in_pool": "Pantheon" in str(best.best_pool),
                "notes": "Shrinkage sensitivity.",
            }
        )
    winner_pools = freq_df[freq_df["record_type"] == "winning_pool"].sort_values("winner_inclusion_rate", ascending=False)
    if not winner_pools.empty:
        best = winner_pools.iloc[0]
        rows.append(
            {
                "method": "enemy_frequency_perturbation",
                "score_definition": "winner rate under Dirichlet-perturbed enemy frequencies",
                "best_pool": best.pool,
                "score": float(best.winner_inclusion_rate),
                "sion_in_pool": "Sion" in str(best.pool),
                "pantheon_in_pool": "Pantheon" in str(best.pool),
                "notes": "Score is frequency of being the winning pool.",
            }
        )
    scope_best = scope_df.drop_duplicates(["scope_id", "best_pool"]) if not scope_df.empty else pd.DataFrame()
    if not scope_best.empty:
        most_common = scope_best["best_pool"].value_counts().sort_values(ascending=False)
        best_pool = most_common.index[0]
        rows.append(
            {
                "method": "scope_stability",
                "score_definition": "count of local OP.GG scopes where pool is best",
                "best_pool": best_pool,
                "score": float(most_common.iloc[0]),
                "sion_in_pool": "Sion" in best_pool,
                "pantheon_in_pool": "Pantheon" in best_pool,
                "notes": "Across local aggregate patch/rank scopes.",
            }
        )
    for lambda_value in sorted(offmeta_df["lambda"].drop_duplicates()):
        subset = offmeta_df[offmeta_df["lambda"] == lambda_value]
        best = subset.iloc[0]
        rows.append(
            {
                "method": f"offmeta_lambda_{lambda_value:g}",
                "score_definition": "poolscore - lambda * mean offmeta penalty",
                "best_pool": best.best_pool,
                "score": float(best.best_pool_objective_score),
                "sion_in_pool": "Sion" in str(best.best_pool),
                "pantheon_in_pool": "Pantheon" in str(best.best_pool),
                "notes": "Aggregate generalizability stress test.",
            }
        )
    residual_pools = residual_df[residual_df["record_type"] == "residual_adjusted_pool"]
    if not residual_pools.empty:
        best = residual_pools.sort_values("rank").iloc[0]
        rows.append(
            {
                "method": "residual_adjusted",
                "score_definition": "logit residual model with champion main effect removed",
                "best_pool": best.pool,
                "score": float(best.score),
                "sion_in_pool": "Sion" in str(best.pool),
                "pantheon_in_pool": "Pantheon" in str(best.pool),
                "notes": "Diagnostic only; not a causal correction.",
            }
        )
    for row in robust_df.itertuples(index=False):
        rows.append(
            {
                "method": row.objective,
                "score_definition": row.score_definition,
                "best_pool": row.best_pool,
                "score": float(row.score),
                "sion_in_pool": bool(row.sion_in_pool),
                "pantheon_in_pool": bool(row.pantheon_in_pool),
                "notes": row.notes,
            }
        )
    return pd.DataFrame(rows)


def _focus_text(summary: pd.DataFrame, champion: str) -> str:
    included = int(summary["sion_in_pool"].sum()) if champion == "Sion" else int(summary["pantheon_in_pool"].sum())
    total = len(summary)
    return f"{champion} appears in {included}/{total} method-summary best pools."


def write_method_sweep_report(
    path: Path,
    summary: pd.DataFrame,
    fixed_policy: pd.DataFrame,
    alpha: pd.DataFrame,
    frequency: pd.DataFrame,
    scope: pd.DataFrame,
    offmeta: pd.DataFrame,
    concentration: pd.DataFrame,
    residual: pd.DataFrame,
    robust: pd.DataFrame,
    scopes: Sequence[AggregateScope],
) -> None:
    fixed_best = fixed_policy[fixed_policy["simulation_mode"] == "fixed-policy"].sort_values("mean_score", ascending=False).iloc[0]
    oracle_best = fixed_policy[fixed_policy["simulation_mode"] == "oracle"].sort_values("mean_score", ascending=False).iloc[0]
    fixed_lower_best = fixed_policy[fixed_policy["simulation_mode"] == "fixed-policy"].sort_values(
        ["lower_5_score", "pool"], ascending=[False, True]
    ).iloc[0]
    residual_best = residual[residual["record_type"] == "residual_adjusted_pool"].sort_values("rank").iloc[0]
    worst_scope = robust[robust["objective"] == "worst_scope_score"]
    worst_text = (
        f"{worst_scope.iloc[0].best_pool} ({worst_scope.iloc[0].score:.2%})"
        if not worst_scope.empty
        else "not available"
    )
    focus_scope = scope[scope["champion"].isin(FOCUS_CHAMPIONS)]
    alpha_levels = sorted(alpha["alpha"].drop_duplicates())
    alpha_findings = {}
    penalty_findings = {}
    frequency_findings = {}
    for champion in FOCUS_CHAMPIONS:
        alpha_subset = alpha[alpha["champion"] == champion]
        alpha_findings[champion] = int(alpha_subset["best_pool_member"].sum())
        penalty_subset = offmeta[offmeta["champion"] == champion].sort_values("lambda")
        excluded = penalty_subset[~penalty_subset["best_pool_member"].astype(bool)]
        penalty_findings[champion] = (
            float(excluded.iloc[0]["lambda"]) if not excluded.empty else None
        )
        frequency_subset = frequency[
            (frequency["record_type"] == "champion")
            & (frequency["champion"] == champion)
        ]
        frequency_findings[champion] = (
            float(frequency_subset.iloc[0]["winner_inclusion_rate"])
            if not frequency_subset.empty
            else np.nan
        )
    residual_pool_members = {
        part.strip() for part in str(residual_best.pool).split(",")
    }
    residual_effects = residual[residual["record_type"] == "champion_effect"].set_index(
        "champion"
    )
    practical_pool = str(fixed_lower_best.pool)
    practical_score = float(fixed_lower_best.lower_5_score)
    mild_penalty_retained = all(
        offmeta[
            (offmeta["champion"] == champion) & (offmeta["lambda"] <= 0.01)
        ]["best_pool_member"].astype(bool).all()
        for champion in FOCUS_CHAMPIONS
    )
    lines = [
        "# Aggregate Method Sweep Report",
        "",
        "## Executive Summary",
        "",
        f"- Fixed-policy posterior mean favors **{fixed_best.pool}** at **{fixed_best.mean_score:.2%}**.",
        f"- Oracle posterior mean favors **{oracle_best.pool}** at **{oracle_best.mean_score:.2%}** and should be read as an upper-bound diagnostic.",
        f"- Residual-adjusted scoring favors **{residual_best.pool}** at **{float(residual_best.score):.2%}**.",
        f"- Worst-scope robust scoring favors **{worst_text}**.",
        f"- {_focus_text(summary, 'Sion')} {_focus_text(summary, 'Pantheon')}",
        f"- The most defensible single-scope uncertainty recommendation is **{practical_pool}**, with a fixed-policy simulated lower-5 score of **{practical_score:.2%}**.",
        "",
        "## Score Definitions",
        "",
        "- `deterministic_eb`: `sum_j f_j max_i EB(W_ij)`.",
        "- `fixed-policy`: choose `argmax_i posterior_mean_ij` once for each enemy, then simulate that locked policy.",
        "- `oracle`: resample every matchup and then take the max in each draw; this is optimistic and not a practical policy.",
        "- `offmeta_penalty`: deterministic score minus a transparent aggregate penalty from low pickrate, importance/pickrate ratio, and LoLalytics breadth/depth flags.",
        "- `residual_adjusted`: two-way logit diagnostic with champion main effect removed, preserving enemy and matchup residual terms.",
        "",
        "## Data Sources",
        "",
        "- No live data was fetched. Every source below was already present locally.",
    ]
    for aggregate_scope in scopes:
        metadata = aggregate_scope.loaded.summary_df.iloc[0]
        source_url = str(metadata.get("source_url", ""))
        source_link = f"[representative URL]({source_url})" if source_url else "URL unavailable"
        lines.append(
            f"- `{aggregate_scope.scope_id}`: {aggregate_scope.source_name}, patch {aggregate_scope.patch}, "
            f"rank {aggregate_scope.rank}, role {metadata.get('lane', 'mid')}, retrieved "
            f"{metadata.get('extraction_date', 'unknown')}; {source_link}. "
            f"Local matchup file: `{aggregate_scope.matchup_path}`."
        )
    lolalytics_rows = offmeta[
        offmeta.get("lolalytics_available", pd.Series(False, index=offmeta.index)).astype(bool)
    ]
    if not lolalytics_rows.empty:
        lolalytics_row = lolalytics_rows.iloc[0]
        lines.append(
            f"- LoLalytics pickrate/breadth/depth extract: patch scope "
            f"`{lolalytics_row.get('lolalytics_pickrate_scope', 'unknown')}`, depth scope "
            f"`{lolalytics_row.get('lolalytics_depth_scope', 'unknown')}`, retrieved "
            f"{lolalytics_row.get('lolalytics_extraction_date', 'unknown')}; "
            f"[representative URL]({lolalytics_row.get('lolalytics_source_url', 'https://lolalytics.com')})."
        )
    lines.extend(
        [
        "",
        "## Main Findings",
        "",
        f"- Sion is in the best pool for {alpha_findings['Sion']}/{len(alpha_levels)} EB alpha settings; Pantheon is in {alpha_findings['Pantheon']}/{len(alpha_levels)}.",
        f"- Under enemy-frequency perturbation, Sion is in the winning pool {frequency_findings['Sion']:.1%} of draws and Pantheon {frequency_findings['Pantheon']:.1%}.",
        f"- The residual-adjusted best pool is `{residual_best.pool}`. Sion is {'retained' if 'Sion' in residual_pool_members else 'removed'} and Pantheon is {'retained' if 'Pantheon' in residual_pool_members else 'removed'}.",
        "- Enemy-frequency perturbation is mostly a meta-weight stress test; it cannot reveal player-selection bias in `W_ij`.",
        f"- The scope sweep used {len(scopes)} local OP.GG aggregate scopes and no Riot API calls.",
        (
            "- Neither focus champion leaves the best pool at tested penalty strengths up to `lambda=0.01`; larger values are deliberately stronger stress tests."
            if mild_penalty_retained
            else "- At least one focus champion leaves the best pool at a tested penalty strength no greater than `lambda=0.01`."
        ),
        f"- First tested offmeta lambda removing Sion from the best pool: `{penalty_findings['Sion'] if penalty_findings['Sion'] is not None else 'not reached'}`. "
        f"For Pantheon: `{penalty_findings['Pantheon'] if penalty_findings['Pantheon'] is not None else 'not reached'}`.",
        "",
        "## Focus Champion Scope Stability",
        "",
        ]
    )
    if not focus_scope.empty:
        for champion in FOCUS_CHAMPIONS:
            subset = focus_scope[focus_scope["champion"] == champion]
            lines.append(
                f"- **{champion}:** best-pool member in {int(subset['best_pool_member'].sum())}/{len(subset)} scope rows; "
                f"median top-pool share {subset['top_pool_share'].median():.2%}."
            )
    if not residual_effects.empty:
        lines.extend(["", "## Residual Model"])
        for champion in FOCUS_CHAMPIONS:
            if champion in residual_effects.index:
                effect = residual_effects.loc[champion]
                lines.append(
                    f"- **{champion}:** aggregate champion main effect "
                    f"{float(effect.champion_effect_logit):+.3f} log-odds "
                    f"(odds ratio {float(effect.champion_effect_odds_ratio):.3f}). "
                    "This term is removed in residual-adjusted scoring."
                )
    lines.extend(
        [
            "",
            "## Contribution Concentration",
            "",
        ]
    )
    focus_concentration = concentration[
        concentration["champion"].isin(FOCUS_CHAMPIONS)
    ].sort_values(["pool_rank", "champion"]).head(8)
    for row in focus_concentration.itertuples(index=False):
        lines.append(
            f"- Pool rank {row.pool_rank}, **{row.champion}** in `{row.pool}`: "
            f"{row.covered_enemy_count} enemies, {row.enemy_frequency_mass_covered:.2%} enemy mass, "
            f"effective matchups {row.effective_matchups:.1f}, top-5 lift share {row.top5_marginal_lift_share:.2%}."
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- No Riot Match-V5 or live Riot API data was collected.",
            "- These methods do not observe player identity, champion familiarity, pick order, team composition, or repeated-player effects.",
            "- LoLalytics breadth/depth is a heuristic with its own recorded population scope; it is not a causal selection-bias correction.",
            "- The residual model is an aggregate decomposition, not a replacement for the optimizer and not a causal adjustment.",
            "- The enemy-frequency perturbation uses a Dirichlet model and treats the configured effective sample size as a sensitivity parameter, not a known sampling design.",
            "- Fixed-policy lower-5 analytic objectives use an independence and normal approximation; the CSV also includes direct posterior simulation results.",
            "",
            "## Next Non-Riot-API Step",
            "",
            "Fetch or locally archive a second aggregate matchup source with W_ij by patch/rank, then rerun this sweep with true cross-source matchup agreement rather than only cross-source pickrate/depth heuristics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_method_sweep(
    loaded: LoadedInputs,
    data_dir: Path,
    candidates: list[str],
    pool_size: int,
    ranked_pools: pd.DataFrame,
    output_dir: Path,
    extra_data_dir: Path | None,
    top_n: int,
    posterior_samples: int,
    posterior_seed: int,
    frequency_samples: int,
    frequency_effective_sample_size: float,
    alpha_values: Sequence[float] = DEFAULT_ALPHA_VALUES,
    offmeta_lambdas: Sequence[float] = DEFAULT_OFFMETA_LAMBDAS,
) -> MethodSweepArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_policy = run_fixed_policy_simulations(
        ranked_pools=ranked_pools,
        loaded=loaded,
        prior_strength=loaded.eb_alpha,
        sample_count=posterior_samples,
        seed=posterior_seed,
        top_n=top_n,
    )
    alpha = run_alpha_sensitivity(loaded, candidates, pool_size, top_n, alpha_values)
    frequency = run_enemy_frequency_sensitivity(
        loaded,
        candidates,
        pool_size,
        top_n,
        sample_count=frequency_samples,
        effective_sample_size=frequency_effective_sample_size,
        seed=posterior_seed,
    )
    scopes = discover_local_opgg_scopes(
        data_dir=data_dir,
        extra_data_dir=extra_data_dir,
        estimator=loaded.estimator,
        eb_alpha=loaded.eb_alpha,
        eb_mu=loaded.eb_mu,
    )
    scope = run_scope_stability(scopes, candidates, pool_size, top_n)
    source_stability = build_source_stability(loaded.summary_df, loaded.patch_label, extra_data_dir)
    offmeta = run_offmeta_penalty_sensitivity(
        loaded,
        candidates,
        pool_size,
        top_n,
        offmeta_lambdas,
        source_stability,
    )
    ranked_for_contribution = ranked_pools.head(top_n).copy()
    if "rank" not in ranked_for_contribution.columns:
        ranked_for_contribution.insert(0, "rank", range(1, len(ranked_for_contribution) + 1))
    concentration = run_contribution_concentration(ranked_for_contribution, loaded, top_n)
    residual, _ = fit_residual_model(loaded, candidates, pool_size, top_n)
    robust = run_robust_objective_comparison(loaded, scopes, candidates, pool_size, top_n)
    summary = build_method_summary(
        fixed_policy,
        alpha,
        frequency,
        scope,
        offmeta,
        residual,
        robust,
    )

    artifacts = MethodSweepArtifacts(
        report=output_dir / "method_sweep_report.md",
        summary=output_dir / "method_sweep_summary.csv",
        fixed_policy_simulation_summary=output_dir / "fixed_policy_simulation_summary.csv",
        alpha_sensitivity=output_dir / "alpha_sensitivity.csv",
        enemy_frequency_sensitivity=output_dir / "enemy_frequency_sensitivity.csv",
        scope_stability=output_dir / "scope_stability.csv",
        offmeta_penalty_sensitivity=output_dir / "offmeta_penalty_sensitivity.csv",
        contribution_concentration=output_dir / "contribution_concentration.csv",
        residual_model_summary=output_dir / "residual_model_summary.csv",
        robust_objective_comparison=output_dir / "robust_objective_comparison.csv",
    )
    summary.to_csv(artifacts.summary, index=False)
    fixed_policy.to_csv(artifacts.fixed_policy_simulation_summary, index=False)
    alpha.to_csv(artifacts.alpha_sensitivity, index=False)
    frequency.to_csv(artifacts.enemy_frequency_sensitivity, index=False)
    scope.to_csv(artifacts.scope_stability, index=False)
    offmeta.to_csv(artifacts.offmeta_penalty_sensitivity, index=False)
    concentration.to_csv(artifacts.contribution_concentration, index=False)
    residual.to_csv(artifacts.residual_model_summary, index=False)
    robust.to_csv(artifacts.robust_objective_comparison, index=False)
    write_method_sweep_report(
        artifacts.report,
        summary,
        fixed_policy,
        alpha,
        frequency,
        scope,
        offmeta,
        concentration,
        residual,
        robust,
        scopes,
    )
    return artifacts
