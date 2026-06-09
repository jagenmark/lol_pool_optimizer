from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data_loader import (
    LoadedInputs,
    build_matchup_lookup,
    load_clean_matchup_data,
    load_clean_summary_data,
    load_patch_data,
)
from matchup_estimator import apply_matchup_estimator
from optimizer import rank_top_pools
from scoring import compute_blind_scores
from uncertainty import build_matchup_posteriors
from utils import canonicalize_champion_name


FOCUS_CHAMPIONS = ("Sion", "Pantheon")


@dataclass(frozen=True)
class SelectionBiasArtifacts:
    champion_summary: Path
    matchup_enrichment: Path
    favorable_selection: Path
    pool_dependency: Path
    source_stability: Path
    patch_rank_stability: Path
    sources: Path
    report: Path


def _normalize_patch_label(value: object) -> str:
    parts = str(value).strip().split(".")
    if len(parts) != 2:
        return str(value).strip()
    return f"{int(parts[0])}.{int(parts[1]):02d}"


def _champion_name_lookup(champions: Iterable[str]) -> dict[str, str]:
    return {canonicalize_champion_name(champion): champion for champion in champions}


def _resolve_named_champions(
    requested: Iterable[str],
    available: Iterable[str],
) -> list[str]:
    available_lookup = _champion_name_lookup(available)
    return [
        available_lookup[key]
        for key in (canonicalize_champion_name(value) for value in requested)
        if key in available_lookup
    ]


def compute_matchup_enrichment(
    matchup_df: pd.DataFrame,
    enemy_frequencies: pd.DataFrame,
    posterior_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare each champion's observed opponent mix with the general enemy mix.

    Both distributions are renormalized over the champion's available,
    non-self matchup rows. This avoids treating missing matchup rows as zero.
    """
    frequency_lookup = {
        str(row.champion_j): float(row.freq_j)
        for row in enemy_frequencies.itertuples(index=False)
    }
    posterior_lookup = {
        (str(row.champion), str(row.enemy_champion)): row
        for row in posterior_df.itertuples(index=False)
    }
    summary_lookup = {
        canonicalize_champion_name(str(row.champion_name)): row
        for row in summary_df.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []

    for champion, group in matchup_df.groupby("champion_i", sort=True):
        available = group[
            (group["champion_j"] != champion)
            & group["champion_j"].isin(frequency_lookup)
        ].copy()
        if available.empty:
            continue

        total_games = float(available["games_ij"].sum())
        general_total = float(
            sum(frequency_lookup[str(enemy)] for enemy in available["champion_j"])
        )
        if total_games <= 0 or general_total <= 0:
            continue

        posterior_values = np.array(
            [
                float(posterior_lookup[(str(champion), str(enemy))].posterior_mean)
                for enemy in available["champion_j"]
            ],
            dtype=float,
        )
        games = available["games_ij"].to_numpy(dtype=float)
        champion_baseline = float(np.average(posterior_values, weights=games))
        summary_row = summary_lookup.get(canonicalize_champion_name(str(champion)))
        summary_total_games = (
            float(summary_row.total_games)
            if summary_row is not None
            and hasattr(summary_row, "total_games")
            and pd.notna(summary_row.total_games)
            else np.nan
        )

        for matchup_row in available.itertuples(index=False):
            enemy = str(matchup_row.champion_j)
            matchup_games = float(matchup_row.games_ij)
            conditional_frequency = matchup_games / total_games
            general_frequency = frequency_lookup[enemy] / general_total
            frequency_delta = conditional_frequency - general_frequency
            expected_games = total_games * general_frequency
            enrichment_ratio = (
                conditional_frequency / general_frequency
                if general_frequency > 0
                else np.nan
            )
            standard_error = np.sqrt(
                total_games * general_frequency * (1.0 - general_frequency)
            )
            enrichment_z = (
                (matchup_games - expected_games) / standard_error
                if standard_error > 0
                else np.nan
            )
            posterior = posterior_lookup[(str(champion), enemy)]
            advantage = float(posterior.posterior_mean) - champion_baseline

            rows.append(
                {
                    "champion": champion,
                    "champion_key": canonicalize_champion_name(str(champion)),
                    "enemy_champion": enemy,
                    "enemy_key": canonicalize_champion_name(enemy),
                    "matchup_games": matchup_games,
                    "champion_recorded_matchup_games": total_games,
                    "champion_summary_total_games": summary_total_games,
                    "recorded_matchup_coverage": (
                        total_games / summary_total_games
                        if summary_total_games > 0
                        else np.nan
                    ),
                    "conditional_enemy_frequency": conditional_frequency,
                    "general_enemy_frequency": general_frequency,
                    "frequency_delta": frequency_delta,
                    "expected_matchup_games": expected_games,
                    "enrichment_ratio": enrichment_ratio,
                    "enrichment_z": enrichment_z,
                    "raw_winrate": float(posterior.raw_winrate),
                    "posterior_mean": float(posterior.posterior_mean),
                    "posterior_lower_5": float(posterior.posterior_lower_5),
                    "posterior_upper_95": float(posterior.posterior_upper_95),
                    "posterior_interval_width": float(
                        posterior.posterior_upper_95 - posterior.posterior_lower_5
                    ),
                    "champion_matchup_baseline": champion_baseline,
                    "matchup_advantage": advantage,
                    "selection_component": frequency_delta * advantage,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["champion", "selection_component", "enemy_champion"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_favorable_selection_summary(
    enrichment_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for champion, group in enrichment_df.groupby("champion", sort=True):
        observed_advantage = float(
            (group["conditional_enemy_frequency"] * group["matchup_advantage"]).sum()
        )
        general_advantage = float(
            (group["general_enemy_frequency"] * group["matchup_advantage"]).sum()
        )
        positive = group.sort_values(
            ["selection_component", "enemy_champion"],
            ascending=[False, True],
        ).head(5)
        negative = group.sort_values(
            ["selection_component", "enemy_champion"],
            ascending=[True, True],
        ).head(5)
        rows.append(
            {
                "champion": champion,
                "champion_key": canonicalize_champion_name(str(champion)),
                "selection_advantage": observed_advantage - general_advantage,
                "observed_mix_weighted_advantage": observed_advantage,
                "general_mix_weighted_advantage": general_advantage,
                "positive_selection_component": float(
                    group.loc[group["selection_component"] > 0, "selection_component"].sum()
                ),
                "negative_selection_component": float(
                    group.loc[group["selection_component"] < 0, "selection_component"].sum()
                ),
                "mean_absolute_frequency_delta": float(
                    group["frequency_delta"].abs().mean()
                ),
                "max_enrichment_ratio": float(group["enrichment_ratio"].max()),
                "recorded_matchup_games": float(
                    group["champion_recorded_matchup_games"].iloc[0]
                ),
                "recorded_matchup_coverage": float(
                    group["recorded_matchup_coverage"].iloc[0]
                ),
                "top_positive_matchups": "; ".join(
                    f"{row.enemy_champion}:{row.selection_component:.4f}"
                    for row in positive.itertuples(index=False)
                ),
                "top_negative_matchups": "; ".join(
                    f"{row.enemy_champion}:{row.selection_component:.4f}"
                    for row in negative.itertuples(index=False)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selection_advantage", "champion"], ascending=[False, True]
    ).reset_index(drop=True)


def build_best_pool_matchup_detail(
    best_pool: tuple[str, ...],
    enemy_frequencies: pd.DataFrame,
    value_lookup: dict[tuple[str, str], float],
    posterior_df: pd.DataFrame,
) -> pd.DataFrame:
    posterior_lookup = {
        (str(row.champion), str(row.enemy_champion)): row
        for row in posterior_df.itertuples(index=False)
    }
    selected: list[tuple[object, list[tuple[str, float]]]] = []
    for enemy_row in enemy_frequencies.itertuples(index=False):
        enemy = str(enemy_row.champion_j)
        values = sorted(
            [
                (champion, float(value_lookup[(champion, enemy)]))
                for champion in best_pool
                if champion != enemy and (champion, enemy) in value_lookup
            ],
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        if values:
            selected.append((enemy_row, values))

    usable_frequency = sum(float(enemy.freq_j) for enemy, _ in selected)
    rows: list[dict[str, object]] = []
    for enemy_row, values in selected:
        enemy = str(enemy_row.champion_j)
        champion, best_value = values[0]
        second_value = values[1][1] if len(values) > 1 else 0.5
        frequency = float(enemy_row.freq_j) / usable_frequency
        posterior = posterior_lookup[(champion, enemy)]
        rows.append(
            {
                "champion": champion,
                "enemy_champion": enemy,
                "best_pool_selected": True,
                "best_pool_enemy_frequency": frequency,
                "best_pool_value": best_value,
                "best_alternative_value": second_value,
                "best_pool_marginal_lift": frequency
                * max(0.0, best_value - second_value),
                "best_pool_weighted_value": frequency * best_value,
                "best_pool_matchup_games": float(posterior.games),
                "best_pool_posterior_mean": float(posterior.posterior_mean),
                "best_pool_posterior_lower_5": float(posterior.posterior_lower_5),
                "best_pool_posterior_upper_95": float(posterior.posterior_upper_95),
            }
        )
    return pd.DataFrame(rows)


def build_pool_dependency(
    candidates: list[str],
    pool_size: int,
    enemy_frequencies: pd.DataFrame,
    matchup_lookup: dict[tuple[str, str], float],
    ranked_pools: pd.DataFrame,
    specialist_or_low_presence: Iterable[str] = (),
) -> pd.DataFrame:
    baseline_pool = tuple(ranked_pools.iloc[0]["pool"])
    baseline_score = float(ranked_pools.iloc[0]["pool_score"])
    top_pool_count = len(ranked_pools)
    appearances = {
        champion: int(
            sum(champion in tuple(pool) for pool in ranked_pools["pool"].tolist())
        )
        for champion in candidates
    }

    def exclusion_row(
        scenario: str,
        excluded: tuple[str, ...],
        champion: str | None = None,
    ) -> dict[str, object]:
        excluded_set = set(excluded)
        if excluded_set.isdisjoint(baseline_pool):
            best_pool = baseline_pool
            best_score = baseline_score
        else:
            remaining = [
                candidate for candidate in candidates if candidate not in excluded_set
            ]
            if len(remaining) < pool_size:
                best_pool = ()
                best_score = np.nan
            else:
                reranked = rank_top_pools(
                    remaining,
                    pool_size,
                    enemy_frequencies,
                    matchup_lookup,
                    top_n=1,
                )
                best_pool = tuple(reranked.iloc[0]["pool"])
                best_score = float(reranked.iloc[0]["pool_score"])
        score_drop = baseline_score - best_score if pd.notna(best_score) else np.nan
        return {
            "scenario": scenario,
            "champion": champion or "",
            "excluded_champions": ", ".join(excluded),
            "top_pool_appearances": appearances.get(champion, np.nan),
            "top_pool_share": (
                appearances.get(champion, 0) / top_pool_count
                if champion is not None and top_pool_count
                else np.nan
            ),
            "baseline_best_pool": ", ".join(baseline_pool),
            "baseline_best_score": baseline_score,
            "best_pool_after_exclusion": ", ".join(best_pool),
            "best_score_after_exclusion": best_score,
            "score_drop": score_drop,
            "relative_score_drop": (
                score_drop / baseline_score
                if pd.notna(score_drop) and baseline_score
                else np.nan
            ),
        }

    rows = [
        exclusion_row("single_champion", (champion,), champion=champion)
        for champion in candidates
    ]
    focus = tuple(_resolve_named_champions(FOCUS_CHAMPIONS, candidates))
    if focus:
        rows.append(exclusion_row("focus_pair", focus))

    heuristic_exclusions = tuple(
        sorted(set(specialist_or_low_presence).intersection(candidates))
    )
    if heuristic_exclusions:
        rows.append(
            exclusion_row(
                "heuristic_specialist_or_low_presence",
                heuristic_exclusions,
            )
        )
    return pd.DataFrame(rows).sort_values(
        ["scenario", "score_drop", "champion"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_source_stability(
    summary_df: pd.DataFrame,
    patch: str,
    extra_data_dir: Path | None,
) -> pd.DataFrame:
    opgg = summary_df.copy()
    opgg["champion_key"] = opgg["champion_name"].map(canonicalize_champion_name)
    opgg_columns = [
        "champion_name",
        "champion_key",
        "pickrate",
        "winrate",
        "banrate",
        "total_games",
    ]
    for column in ("source_url", "extraction_date"):
        if column in summary_df.columns:
            opgg_columns.append(column)
    opgg = opgg[opgg_columns].rename(
        columns={
            "champion_name": "champion",
            "pickrate": "opgg_pickrate",
            "winrate": "opgg_winrate",
            "banrate": "opgg_banrate",
            "total_games": "opgg_total_games",
            "source_url": "opgg_source_url",
            "extraction_date": "opgg_extraction_date",
        }
    )

    lolalytics_path: Path | None = None
    if extra_data_dir is not None and extra_data_dir.exists():
        patch_short = f"{int(patch.split('.')[0])}.{int(patch.split('.')[1])}"
        candidates = sorted(
            extra_data_dir.glob(
                f"lolalytics_mid_pickrate_mainrate_{patch_short}_plat_plus_full.csv"
            )
        )
        lolalytics_path = candidates[-1] if candidates else None

    if lolalytics_path is None:
        source = opgg.copy()
        source["lolalytics_available"] = False
        source["specialist_heuristic_flag"] = False
        source["low_mid_presence_flag"] = source["opgg_pickrate"] < 0.01
        source["matchup_cross_source_comparable"] = False
        source["matchup_cross_source_note"] = (
            "LoLalytics matchup estimates were not available."
        )
        return source

    lolalytics = pd.read_csv(lolalytics_path)
    lolalytics["champion_key"] = lolalytics["champion_name"].map(
        canonicalize_champion_name
    )
    lolalytics["lolalytics_pickrate"] = pd.to_numeric(
        lolalytics["pick_rate"], errors="coerce"
    )
    if (lolalytics["lolalytics_pickrate"] > 1).any():
        lolalytics["lolalytics_pickrate"] /= 100.0
    lolalytics = lolalytics.rename(
        columns={
            "breadth": "lolalytics_breadth",
            "depth": "lolalytics_depth",
            "classification": "lolalytics_classification",
            "unique_players": "lolalytics_unique_players",
            "total_ranked_games": "lolalytics_total_ranked_games",
            "source_url": "lolalytics_source_url",
            "extraction_date": "lolalytics_extraction_date",
            "population_scope_pickrate": "lolalytics_pickrate_scope",
            "population_scope_depth": "lolalytics_depth_scope",
        }
    )
    columns = [
        "champion_key",
        "lolalytics_pickrate",
        "lolalytics_breadth",
        "lolalytics_depth",
        "lolalytics_classification",
        "lolalytics_unique_players",
        "lolalytics_total_ranked_games",
        "lolalytics_source_url",
        "lolalytics_extraction_date",
        "lolalytics_pickrate_scope",
        "lolalytics_depth_scope",
    ]
    source = opgg.merge(lolalytics[columns], on="champion_key", how="left")
    source["lolalytics_available"] = source["lolalytics_pickrate"].notna()
    source["pickrate_absolute_difference"] = (
        source["opgg_pickrate"] - source["lolalytics_pickrate"]
    ).abs()
    denominator = source[["opgg_pickrate", "lolalytics_pickrate"]].mean(axis=1)
    source["pickrate_relative_difference"] = (
        source["pickrate_absolute_difference"] / denominator.replace(0, np.nan)
    )
    source["pickrate_sources_agree_within_20pct"] = (
        source["pickrate_relative_difference"] <= 0.20
    )
    source["specialist_heuristic_flag"] = (
        (source["lolalytics_depth"] >= 1.15)
        | source["lolalytics_classification"].isin(["niche"])
    )
    source["low_mid_presence_flag"] = source["opgg_pickrate"] < 0.01
    source["matchup_cross_source_comparable"] = False
    source["matchup_cross_source_note"] = (
        "LoLalytics extract has pick rate and breadth/depth, not matchup W_ij."
    )
    return source.sort_values("champion").reset_index(drop=True)


def _top_pool_membership(
    ranked_pools: pd.DataFrame,
    candidates: list[str],
) -> pd.DataFrame:
    rows = []
    top_count = len(ranked_pools)
    for champion in candidates:
        ranks = [
            index + 1
            for index, pool in enumerate(ranked_pools["pool"].tolist())
            if champion in tuple(pool)
        ]
        rows.append(
            {
                "champion": champion,
                "top_pool_appearances": len(ranks),
                "top_pool_share": len(ranks) / top_count if top_count else 0.0,
                "best_top_pool_rank": min(ranks) if ranks else np.nan,
                "mean_top_pool_rank": float(np.mean(ranks)) if ranks else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_champion_summary(
    loaded: LoadedInputs,
    candidates: list[str],
    ranked_pools: pd.DataFrame,
    favorable_df: pd.DataFrame,
    dependency_df: pd.DataFrame,
    source_df: pd.DataFrame,
    best_pool_detail_df: pd.DataFrame,
) -> pd.DataFrame:
    membership = _top_pool_membership(ranked_pools, candidates)
    blind = compute_blind_scores(
        candidates, loaded.frequency_df, loaded.matchup_lookup
    )
    summary = loaded.summary_df.copy()
    summary["champion_key"] = summary["champion_name"].map(
        canonicalize_champion_name
    )
    summary = summary.rename(columns={"champion_name": "champion"})
    summary = summary[summary["champion"].isin(candidates)]
    result = summary.merge(membership, on="champion", how="left")
    result = result.merge(blind, on="champion", how="left")
    result = result.merge(
        favorable_df.drop(columns=["champion_key"], errors="ignore"),
        on="champion",
        how="left",
    )
    single_dependency = dependency_df[
        dependency_df["scenario"] == "single_champion"
    ][
        [
            "champion",
            "score_drop",
            "relative_score_drop",
            "best_pool_after_exclusion",
        ]
    ].rename(
        columns={
            "score_drop": "exclusion_score_drop",
            "relative_score_drop": "exclusion_relative_score_drop",
        }
    )
    result = result.merge(single_dependency, on="champion", how="left")

    source_columns = [
        "champion",
        "lolalytics_pickrate",
        "lolalytics_breadth",
        "lolalytics_depth",
        "lolalytics_classification",
        "specialist_heuristic_flag",
        "low_mid_presence_flag",
        "pickrate_relative_difference",
        "pickrate_sources_agree_within_20pct",
    ]
    result = result.merge(
        source_df[[column for column in source_columns if column in source_df.columns]],
        on="champion",
        how="left",
    )

    concentration_rows = []
    for champion in candidates:
        detail = best_pool_detail_df[
            best_pool_detail_df["champion"] == champion
        ].copy()
        lifts = detail["best_pool_marginal_lift"].clip(lower=0)
        lift_total = float(lifts.sum())
        shares = lifts / lift_total if lift_total > 0 else pd.Series(dtype=float)
        hhi = float(np.square(shares).sum()) if len(shares) else np.nan
        concentration_rows.append(
            {
                "champion": champion,
                "best_pool_matchups_covered": int(len(detail)),
                "best_pool_enemy_frequency_covered": float(
                    detail["best_pool_enemy_frequency"].sum()
                ),
                "best_pool_marginal_lift": lift_total,
                "best_pool_top5_marginal_lift_share": (
                    float(lifts.nlargest(5).sum() / lift_total)
                    if lift_total > 0
                    else np.nan
                ),
                "best_pool_marginal_lift_hhi": hhi,
                "best_pool_effective_matchups": (
                    1.0 / hhi if pd.notna(hhi) and hhi > 0 else np.nan
                ),
                "best_pool_median_matchup_games": (
                    float(detail["best_pool_matchup_games"].median())
                    if len(detail)
                    else np.nan
                ),
                "best_pool_min_matchup_games": (
                    float(detail["best_pool_matchup_games"].min())
                    if len(detail)
                    else np.nan
                ),
            }
        )
    result = result.merge(pd.DataFrame(concentration_rows), on="champion", how="left")

    slot_share = result["top_pool_appearances"] / (
        len(ranked_pools) * int(len(tuple(ranked_pools.iloc[0]["pool"])))
    )
    candidate_pickrate_share = result["pickrate"] / result["pickrate"].sum()
    result["top_pool_slot_share"] = slot_share
    result["candidate_pickrate_share"] = candidate_pickrate_share
    result["importance_to_pickrate_ratio"] = (
        slot_share / candidate_pickrate_share.replace(0, np.nan)
    )
    best_pool = tuple(ranked_pools.iloc[0]["pool"])
    result["best_pool_member"] = result["champion"].isin(best_pool)
    result["patch"] = loaded.patch_label
    result["estimator"] = loaded.estimator
    return result.sort_values(
        ["top_pool_appearances", "exclusion_score_drop", "champion"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _rank_scope(
    scope_label: str,
    patch: str,
    rank: str,
    matchup_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    frequency_df: pd.DataFrame,
    matchup_lookup: dict[tuple[str, str], float],
    candidates: list[str],
    pool_size: int,
    top_n: int,
) -> pd.DataFrame:
    available = sorted(
        set(candidates).intersection(set(matchup_df["champion_i"].unique()))
    )
    ranked = rank_top_pools(
        available,
        pool_size,
        frequency_df,
        matchup_lookup,
        top_n=top_n,
    )
    membership = _top_pool_membership(ranked, available)
    dependency = build_pool_dependency(
        available,
        pool_size,
        frequency_df,
        matchup_lookup,
        ranked,
    )
    dependency = dependency[dependency["scenario"] == "single_champion"][
        ["champion", "score_drop"]
    ].rename(columns={"score_drop": "exclusion_score_drop"})
    summary = summary_df.copy()
    summary = summary.rename(columns={"champion_name": "champion"})
    summary = summary[summary["champion"].isin(available)]
    frame = membership.merge(summary, on="champion", how="left").merge(
        dependency, on="champion", how="left"
    )
    best_pool = tuple(ranked.iloc[0]["pool"])
    frame["best_pool_member"] = frame["champion"].isin(best_pool)
    frame["best_pool"] = ", ".join(best_pool)
    frame["best_pool_score"] = float(ranked.iloc[0]["pool_score"])
    frame["scope"] = scope_label
    frame["patch"] = patch
    frame["rank"] = rank
    return frame


def build_patch_rank_stability(
    data_dir: Path,
    extra_data_dir: Path | None,
    current_patch: str,
    candidates: list[str],
    pool_size: int,
    top_n: int,
    estimator: str,
    eb_alpha: float,
    eb_mu: float | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    patch_dirs = sorted(
        path.name
        for path in data_dir.iterdir()
        if path.is_dir() and (path / "opgg_mid_matchups_clean.csv").exists()
    )
    for patch in patch_dirs:
        loaded = load_patch_data(
            patch,
            data_dir,
            estimator=estimator,
            eb_alpha=eb_alpha,
            eb_mu=eb_mu,
        )
        frames.append(
            _rank_scope(
                scope_label=f"opgg_plat_plus_patch_{patch}",
                patch=patch,
                rank="plat_plus",
                matchup_df=loaded.matchup_df,
                summary_df=loaded.summary_df,
                frequency_df=loaded.frequency_df,
                matchup_lookup=loaded.matchup_lookup,
                candidates=candidates,
                pool_size=pool_size,
                top_n=top_n,
            )
        )

    if extra_data_dir is not None:
        raw_dir = extra_data_dir / "raw"
        summary_files = sorted(
            raw_dir.glob(
                "opgg_mid_champion_summary__global__emerald_plus__*.csv"
            )
        )
        for summary_path in reversed(summary_files):
            raw_summary = pd.read_csv(summary_path)
            if raw_summary.empty or _normalize_patch_label(
                raw_summary["patch"].iloc[0]
            ) != _normalize_patch_label(current_patch):
                continue
            matchup_path = summary_path.with_name(
                summary_path.name.replace("champion_summary", "matchups")
            )
            if not matchup_path.exists():
                continue
            matchup_df = load_clean_matchup_data(matchup_path)
            summary_df = load_clean_summary_data(summary_path)
            frequency_df = (
                matchup_df[matchup_df["champion_i"] != matchup_df["champion_j"]]
                .groupby("champion_j", as_index=False)["games_ij"]
                .sum()
                .rename(columns={"games_ij": "count_j"})
            )
            frequency_df["freq_j"] = (
                frequency_df["count_j"] / frequency_df["count_j"].sum()
            )
            matchup_df = matchup_df.merge(
                frequency_df[["champion_j", "freq_j"]],
                on="champion_j",
                how="left",
            )
            matchup_df, _ = apply_matchup_estimator(
                matchup_df,
                estimator=estimator,
                eb_alpha=eb_alpha,
                eb_mu=eb_mu,
            )
            frames.append(
                _rank_scope(
                    scope_label=f"opgg_emerald_plus_patch_{current_patch}",
                    patch=current_patch,
                    rank="emerald_plus",
                    matchup_df=matchup_df,
                    summary_df=summary_df,
                    frequency_df=frequency_df,
                    matchup_lookup=build_matchup_lookup(matchup_df),
                    candidates=candidates,
                    pool_size=pool_size,
                    top_n=top_n,
                )
            )
            break

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["scope", "top_pool_share", "champion"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_source_inventory(
    loaded: LoadedInputs,
    source_df: pd.DataFrame,
    stability_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "source_name": "OP.GG champion summary and matchup aggregates",
            "source_type": "aggregated public statistics",
            "patch": loaded.patch_label,
            "rank": (
                str(loaded.summary_df["elo"].iloc[0])
                if "elo" in loaded.summary_df.columns
                else "plat_plus"
            ),
            "role": "mid",
            "retrieval_date": (
                str(loaded.summary_df["extraction_date"].iloc[0])
                if "extraction_date" in loaded.summary_df.columns
                else ""
            ),
            "url": (
                str(loaded.summary_df["source_url"].iloc[0])
                if "source_url" in loaded.summary_df.columns
                else "https://op.gg/lol/champions"
            ),
            "used_for": "W_ij, matchup games, pick rate, win rate, enemy frequency",
            "limitations": "Aggregated; no player familiarity, pick order, or team context.",
        }
    ]
    if "lolalytics_available" in source_df.columns and source_df[
        "lolalytics_available"
    ].any():
        first = source_df[source_df["lolalytics_available"]].iloc[0]
        rows.append(
            {
                "source_name": "LoLalytics breadth/depth and mid pick rate",
                "source_type": "aggregated public statistics",
                "patch": loaded.patch_label,
                "rank": "plat_plus for pick rate; all ranks for depth",
                "role": "mid for pick rate; all roles for depth",
                "retrieval_date": first.get("lolalytics_extraction_date", ""),
                "url": first.get("lolalytics_source_url", "https://lolalytics.com"),
                "used_for": "pick-rate agreement and specialist heuristic",
                "limitations": (
                    "Depth scope is all regions/all ranks/last 7 days and is not "
                    "historically patch-specific."
                ),
            }
        )
    if not stability_df.empty and (stability_df["rank"] == "emerald_plus").any():
        rows.append(
            {
                "source_name": "OP.GG Emerald+ validation extract",
                "source_type": "aggregated public statistics",
                "patch": loaded.patch_label,
                "rank": "emerald_plus",
                "role": "mid",
                "retrieval_date": "2026-04-05",
                "url": "https://op.gg/lol/champions",
                "used_for": "rank stability of optimized pools and champion importance",
                "limitations": "Same provider and aggregation method as primary data.",
            }
        )
    rows.append(
        {
            "source_name": "Riot Match-V5 (proposed next collection)",
            "source_type": "official raw match API",
            "patch": "future sample",
            "rank": "sample-defined",
            "role": "role assignment derived per match",
            "retrieval_date": str(date.today()),
            "url": "https://developer.riotgames.com/apis#match-v5",
            "used_for": "not used in this run; proposed player/match-level validation",
            "limitations": "Requires API key, sampling design, rate-limit handling, and player-history joins.",
        }
    )
    return pd.DataFrame(rows)


def _format_percent(value: object) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.2%}"


def _focus_report_lines(
    champion: str,
    champion_summary: pd.DataFrame,
    stability_df: pd.DataFrame,
) -> list[str]:
    row = champion_summary[champion_summary["champion"] == champion]
    if row.empty:
        return [f"- {champion}: not present in the diagnostic candidate universe."]
    item = row.iloc[0]
    scopes = stability_df[stability_df["champion"] == champion]
    scope_count = int(len(scopes))
    best_count = int(scopes["best_pool_member"].sum()) if scope_count else 0
    specialist = bool(item.get("specialist_heuristic_flag", False))
    return [
        (
            f"- **{champion}:** top-pool share {_format_percent(item.top_pool_share)}, "
            f"slot-share/pick-rate ratio {float(item.importance_to_pickrate_ratio):.2f}x, "
            f"and exclusion loss {_format_percent(item.exclusion_score_drop)}."
        ),
        (
            f"  Favorable-selection score {_format_percent(item.selection_advantage)}; "
            f"best-pool coverage spans {int(item.best_pool_matchups_covered)} enemies "
            f"with median {item.best_pool_median_matchup_games:.0f} games. The top five "
            f"matchups supply {_format_percent(item.best_pool_top5_marginal_lift_share)} "
            f"of its marginal lift, equivalent to about "
            f"{float(item.best_pool_effective_matchups):.1f} equally weighted matchups."
        ),
        (
            f"  LoLalytics specialist heuristic: {'flagged' if specialist else 'not flagged'}; "
            f"best-pool member in {best_count}/{scope_count} patch/rank scopes."
        ),
    ]


def _reliability_lines(
    champion: str,
    best_pool_detail_df: pd.DataFrame,
) -> list[str]:
    detail = best_pool_detail_df[
        best_pool_detail_df["champion"] == champion
    ].sort_values(
        ["best_pool_marginal_lift", "enemy_champion"],
        ascending=[False, True],
    )
    if detail.empty:
        return [f"- **{champion}:** does not cover a matchup in the best pool."]
    top = detail.head(5)
    matchup_text = "; ".join(
        (
            f"{row.enemy_champion} ({row.best_pool_matchup_games:.0f} games, "
            f"posterior {row.best_pool_posterior_mean:.1%}, "
            f"90% interval {row.best_pool_posterior_lower_5:.1%}-"
            f"{row.best_pool_posterior_upper_95:.1%})"
        )
        for row in top.itertuples(index=False)
    )
    low_sample_count = int((detail["best_pool_matchup_games"] < 100).sum())
    return [
        f"- **{champion} largest marginal matchups:** {matchup_text}.",
        (
            f"  {low_sample_count} of {len(detail)} selected matchups have fewer "
            "than 100 games; these should be treated as tail-risk evidence rather "
            "than primary support."
        ),
    ]


def write_selection_bias_report(
    path: Path,
    loaded: LoadedInputs,
    ranked_pools: pd.DataFrame,
    champion_summary: pd.DataFrame,
    dependency_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    source_df: pd.DataFrame,
    best_pool_detail_df: pd.DataFrame,
    top_n: int,
) -> None:
    best_pool = tuple(ranked_pools.iloc[0]["pool"])
    focus_rows = champion_summary[
        champion_summary["champion"].isin(FOCUS_CHAMPIONS)
    ]
    positive_selection = focus_rows[
        focus_rows["selection_advantage"] >= 0.0025
    ]["champion"].tolist()
    specialist_flags = focus_rows[
        focus_rows.get("specialist_heuristic_flag", False) == True  # noqa: E712
    ]["champion"].tolist()
    stable_focus = []
    for champion in FOCUS_CHAMPIONS:
        scopes = stability_df[stability_df["champion"] == champion]
        if len(scopes) and float(scopes["top_pool_share"].median()) >= 0.25:
            stable_focus.append(champion)

    pair_row = dependency_df[dependency_df["scenario"] == "focus_pair"]
    pair_loss = (
        float(pair_row.iloc[0]["score_drop"]) if not pair_row.empty else np.nan
    )
    source_matchups_available = bool(
        source_df.get("matchup_cross_source_comparable", pd.Series(False)).any()
    )
    conclusion = (
        "The evidence is mixed but leans toward a real aggregate signal with "
        "unresolved generalizability risk."
    )
    if specialist_flags or positive_selection:
        conclusion = (
            "The evidence remains ambiguous: the aggregate signal is material, "
            "but selection diagnostics add meaningful generalizability concerns."
        )

    lines = [
        "# Selection Bias Diagnostics for the Champion Pool Optimizer",
        "",
        "## Executive Summary",
        "",
        f"- Patch **{loaded.patch_label}**, estimator **{loaded.estimator}**, "
        f"pool size **{len(best_pool)}**, and top **{top_n}** pools were analyzed.",
        f"- The best pool is **{', '.join(best_pool)}** with score "
        f"**{float(ranked_pools.iloc[0]['pool_score']):.2%}**.",
        (
            f"- That score is the deterministic **{loaded.estimator} point-estimate "
            "pool score**: `sum_j f_j max_i W_ij` using the selected estimator. "
            "It is not a posterior simulation mean, lower-5 score, or "
            "probability-of-being-best."
        ),
        f"- Excluding Sion and Pantheon together changes the best score by "
        f"**{_format_percent(pair_loss)}**.",
        f"- **Conclusion:** {conclusion}",
        "",
        "## Why Posterior Simulation Can Differ",
        "",
        (
            "Posterior simulation samples uncertain matchup values instead of "
            "scoring one fixed matrix. Fixed-policy simulation locks each "
            "enemy's response using posterior means, while oracle simulation "
            "reselects the maximum after every draw and is therefore optimistic. "
            "Those different estimands can change both scores and pool ordering."
        ),
        "",
        "## Why Sion and Pantheon Rank Highly",
        "",
        (
            "The optimizer rewards complementary matchup coverage, not overall "
            "win rate alone. A champion dominates when it is the best available "
            "answer to many high-frequency enemies or supplies large marginal "
            "gains where the rest of the pool is weak."
        ),
        "",
    ]
    for champion in FOCUS_CHAMPIONS:
        lines.extend(_focus_report_lines(champion, champion_summary, stability_df))
    lines.extend(
        [
            "",
            "## Within-Matchup Reliability",
            "",
        ]
    )
    for champion in FOCUS_CHAMPIONS:
        lines.extend(_reliability_lines(champion, best_pool_detail_df))
    lines.extend(
        [
            "",
            "## Evidence Consistent With Selection Bias",
            "",
            (
                "- Low mid pick rate makes both champions less representative of "
                "the ordinary mid-player population, even when matchup game counts "
                "are adequate."
            ),
            (
                "- Positive favorable-selection scores indicate that the recorded "
                "opponent mix is tilted toward matchups where the champion performs "
                "better than its own matchup baseline. This is descriptive and does "
                "not prove the matchup win rates themselves are biased."
            ),
            (
                "- Matchup records cover only the source's recorded mid-opponent "
                "universe. Missing offrole opponents and unobserved draft context "
                "can still affect generalizability."
            ),
            "",
            "## Evidence Consistent With a Robust Aggregate Signal",
            "",
            (
                "- The report checks whether contributions are spread over many "
                "enemies, and reports median/minimum games plus posterior intervals "
                "for every selected matchup."
            ),
            (
                f"- Patch/rank stability is available for {stability_df['scope'].nunique()} "
                "scopes. Repeated top-pool appearance is harder to explain as one "
                "isolated noisy matchup."
            ),
            (
                "- OP.GG and LoLalytics pick rates can be compared at the same patch "
                "and rank. Agreement supports the low-popularity diagnosis, though "
                "it does not validate W_ij."
            ),
            "",
            "## Cross-Source Limitation",
            "",
            (
                "A true matchup-level cross-source stability test was "
                f"{'available' if source_matchups_available else 'not available'} "
                "in this run. The LoLalytics extract contains pick rate and "
                "breadth/depth, not a second W_ij matrix."
            ),
            "",
            "## What Cannot Be Identified From Aggregates",
            "",
            "- Player familiarity, one-trick status, and repeated-player weighting.",
            "- Whether the champion was selected before or after the lane opponent.",
            "- Team composition, bans, autofill, role swaps, and premade context.",
            "- Within-player performance on the same matchup with and without specialization.",
            "",
            "## Best Next Data Collection Step",
            "",
            (
                "Use Riot Match-V5 to build a stratified match-level sample across "
                "regions and ranks. Derive actual role, patch, champion matchup, "
                "pick-order proxy where available, player champion-game history, "
                "and repeated-player identifiers. Then fit a hierarchical model "
                "with matchup effects plus player familiarity and rank controls, "
                "and compare adjusted matchup estimates with the current aggregates."
            ),
            "",
            "## Data Sources",
            "",
            (
                "- OP.GG aggregate champion and matchup files, URLs and retrieval "
                "dates recorded in `selection_bias_sources.csv`."
            ),
            (
                "- LoLalytics pick rate and breadth/depth extract, with its mixed "
                "population scopes explicitly retained."
            ),
            (
                "- Riot Match-V5 documentation: "
                "https://developer.riotgames.com/apis#match-v5 (proposed, not used)."
            ),
            "",
            "## Interpretation Guardrail",
            "",
            (
                "These diagnostics do not causally correct selection bias and do "
                "not replace the optimizer. They identify dependence, concentration, "
                "instability, and generalizability warnings around its inputs."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_selection_bias_diagnostics(
    loaded: LoadedInputs,
    data_dir: Path,
    candidates: list[str],
    pool_size: int,
    ranked_pools: pd.DataFrame,
    output_dir: Path,
    top_n: int,
    prior_strength: float,
    extra_data_dir: Path | None = None,
) -> SelectionBiasArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = ranked_pools.head(top_n).copy()
    posterior_df = build_matchup_posteriors(
        loaded.matchup_df,
        prior_strength=prior_strength,
        prior_mean=loaded.eb_mu,
    )
    enrichment = compute_matchup_enrichment(
        loaded.matchup_df,
        loaded.frequency_df,
        posterior_df,
        loaded.summary_df,
    )
    favorable = build_favorable_selection_summary(enrichment)
    source_stability = build_source_stability(
        loaded.summary_df, loaded.patch_label, extra_data_dir
    )
    specialist_flags = source_stability["specialist_heuristic_flag"].fillna(False)
    low_presence_flags = source_stability["low_mid_presence_flag"].fillna(False)
    heuristic_exclusions = source_stability.loc[
        specialist_flags | low_presence_flags,
        "champion",
    ].tolist()
    dependency = build_pool_dependency(
        candidates,
        pool_size,
        loaded.frequency_df,
        loaded.matchup_lookup,
        ranked,
        specialist_or_low_presence=heuristic_exclusions,
    )
    best_pool = tuple(ranked.iloc[0]["pool"])
    best_pool_detail = build_best_pool_matchup_detail(
        best_pool,
        loaded.frequency_df,
        loaded.matchup_lookup,
        posterior_df,
    )
    enrichment = enrichment.merge(
        best_pool_detail,
        on=["champion", "enemy_champion"],
        how="left",
    )
    enrichment["best_pool_selected"] = enrichment["best_pool_selected"].fillna(False)
    champion_summary = build_champion_summary(
        loaded,
        candidates,
        ranked,
        favorable,
        dependency,
        source_stability,
        best_pool_detail,
    )
    patch_rank = build_patch_rank_stability(
        data_dir=data_dir,
        extra_data_dir=extra_data_dir,
        current_patch=loaded.patch_label,
        candidates=candidates,
        pool_size=pool_size,
        top_n=top_n,
        estimator=loaded.estimator,
        eb_alpha=loaded.eb_alpha,
        eb_mu=loaded.eb_mu,
    )
    sources = build_source_inventory(loaded, source_stability, patch_rank)

    artifacts = SelectionBiasArtifacts(
        champion_summary=output_dir / "selection_bias_champion_summary.csv",
        matchup_enrichment=output_dir / "selection_bias_matchup_enrichment.csv",
        favorable_selection=output_dir / "selection_bias_favorable_selection.csv",
        pool_dependency=output_dir / "selection_bias_pool_dependency.csv",
        source_stability=output_dir / "selection_bias_source_stability.csv",
        patch_rank_stability=output_dir / "selection_bias_patch_rank_stability.csv",
        sources=output_dir / "selection_bias_sources.csv",
        report=output_dir / "selection_bias_report.md",
    )
    champion_summary.to_csv(artifacts.champion_summary, index=False)
    enrichment.to_csv(artifacts.matchup_enrichment, index=False)
    favorable.to_csv(artifacts.favorable_selection, index=False)
    dependency.to_csv(artifacts.pool_dependency, index=False)
    source_stability.to_csv(artifacts.source_stability, index=False)
    patch_rank.to_csv(artifacts.patch_rank_stability, index=False)
    sources.to_csv(artifacts.sources, index=False)
    write_selection_bias_report(
        artifacts.report,
        loaded,
        ranked,
        champion_summary,
        dependency,
        patch_rank,
        source_stability,
        best_pool_detail,
        top_n,
    )
    return artifacts
