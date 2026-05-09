from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiment_config import DEFAULT_SHRINKAGE_C_VALUES, OUTPUT_DIR, PATCH_A, PATCH_B
from experiment_data import intersect_patch_data, load_patch_data
from experiment_models import (
    build_model_grid,
    build_training_frame,
    estimate_matchup_winrates,
    format_model_parameter,
)
from experiment_pool import build_lookup, brute_force_best_pools, pool_score, weighted_error_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the first cross-patch champion pool optimization experiment."
    )
    parser.add_argument("--pool-sizes", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument(
        "--shrinkage-c",
        nargs="+",
        type=float,
        default=list(DEFAULT_SHRINKAGE_C_VALUES),
        help="Grid of c values for simple shrinkage toward champion overall win rate.",
    )
    parser.add_argument(
        "--candidate-champions",
        nargs="*",
        default=None,
        help="Optional champion ids or names. If omitted, all common champions are used.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for exported CSV results.",
    )
    return parser.parse_args()


def _candidate_table(summary_a: pd.DataFrame, summary_b: pd.DataFrame) -> pd.DataFrame:
    merged = summary_a[["champion_id", "champion_name"]].merge(
        summary_b[["champion_id", "champion_name"]],
        on="champion_id",
        how="inner",
        suffixes=("_a", "_b"),
    )
    merged["champion_name"] = merged["champion_name_a"]
    return (
        merged[["champion_id", "champion_name"]]
        .drop_duplicates()
        .sort_values("champion_id")
        .reset_index(drop=True)
    )


def _resolve_candidates(
    candidate_df: pd.DataFrame,
    requested_candidates: list[str] | None,
) -> list[str]:
    if not requested_candidates:
        return candidate_df["champion_id"].tolist()

    by_id = {row.champion_id.lower(): row.champion_id for row in candidate_df.itertuples(index=False)}
    by_name = {row.champion_name.lower(): row.champion_id for row in candidate_df.itertuples(index=False)}
    resolved: list[str] = []
    unknown: list[str] = []
    for raw_value in requested_candidates:
        key = raw_value.strip().lower()
        champion_id = by_id.get(key) or by_name.get(key)
        if champion_id is None:
            unknown.append(raw_value)
        else:
            resolved.append(champion_id)

    if unknown:
        raise ValueError(f"Unknown candidate champions: {', '.join(sorted(unknown))}")

    return sorted(set(resolved))


def _pool_to_names(pool: tuple[str, ...], candidate_df: pd.DataFrame) -> str:
    name_lookup = dict(zip(candidate_df["champion_id"], candidate_df["champion_name"]))
    return ", ".join(name_lookup[champion_id] for champion_id in pool)


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    patch_a_raw = load_patch_data(PATCH_A)
    patch_b_raw = load_patch_data(PATCH_B)
    patch_a, patch_b = intersect_patch_data(patch_a_raw, patch_b_raw)

    candidate_df = _candidate_table(patch_a.summary_df, patch_b.summary_df)
    candidate_ids = _resolve_candidates(candidate_df, args.candidate_champions)
    if not candidate_ids:
        raise ValueError("No candidate champions available after filtering")

    training_frame = build_training_frame(patch_a.summary_df, patch_a.matchup_df)
    observed_b_lookup = build_lookup(patch_b.matchup_df, "matchup_winrate")

    results_rows: list[dict[str, object]] = []
    top_pool_rows: list[dict[str, object]] = []

    for model_spec in build_model_grid(tuple(args.shrinkage_c)):
        estimated_train_df = estimate_matchup_winrates(training_frame, model_spec)
        estimated_lookup = build_lookup(estimated_train_df, "estimated_winrate")
        weighted_mae, weighted_mse, pair_count = weighted_error_metrics(
            estimated_train_df,
            patch_b.matchup_df,
            patch_b.enemy_weights,
        )

        for pool_size in sorted(set(args.pool_sizes)):
            if pool_size <= 0:
                raise ValueError("Pool sizes must be positive")
            if pool_size > len(candidate_ids):
                raise ValueError(
                    f"Pool size {pool_size} exceeds candidate count {len(candidate_ids)}"
                )

            ranked_pools = brute_force_best_pools(
                candidate_ids=candidate_ids,
                pool_size=pool_size,
                weights_df=patch_a.enemy_weights,
                value_lookup=estimated_lookup,
            )
            best_pool = tuple(ranked_pools.iloc[0]["pool"])
            best_train_score = float(ranked_pools.iloc[0]["score"])
            eval_score_b = pool_score(best_pool, patch_b.enemy_weights, observed_b_lookup)

            results_rows.append(
                {
                    "train_patch": patch_a.patch_label,
                    "eval_patch": patch_b.patch_label,
                    "model": model_spec.model_name,
                    "parameter": format_model_parameter(model_spec),
                    "pool_size": pool_size,
                    "candidate_count": len(candidate_ids),
                    "selected_pool_ids": "|".join(best_pool),
                    "selected_pool_names": _pool_to_names(best_pool, candidate_df),
                    "training_score_a": best_train_score,
                    "evaluation_score_b": eval_score_b,
                    "weighted_mae": weighted_mae,
                    "weighted_mse": weighted_mse,
                    "common_pair_count": pair_count,
                }
            )

            for rank, row in ranked_pools.head(10).reset_index(drop=True).iterrows():
                top_pool = tuple(row["pool"])
                top_pool_rows.append(
                    {
                        "train_patch": patch_a.patch_label,
                        "eval_patch": patch_b.patch_label,
                        "model": model_spec.model_name,
                        "parameter": format_model_parameter(model_spec),
                        "pool_size": pool_size,
                        "rank": rank + 1,
                        "pool_ids": "|".join(top_pool),
                        "pool_names": _pool_to_names(top_pool, candidate_df),
                        "training_score_a": float(row["score"]),
                        "evaluation_score_b": pool_score(top_pool, patch_b.enemy_weights, observed_b_lookup),
                    }
                )

    return pd.DataFrame(results_rows), pd.DataFrame(top_pool_rows)


def export_results(results_df: pd.DataFrame, top_pools_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "cross_patch_results.csv", index=False)
    top_pools_df.to_csv(output_dir / "top_pools.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    results_df, top_pools_df = run_experiment(args)
    export_results(results_df, top_pools_df, output_dir)

    print("Experiment pipeline complete.")
    print(f"Patch A: {PATCH_A.patch_label} -> {PATCH_A.summary_path.name} / {PATCH_A.matchup_path.name}")
    print(f"Patch B: {PATCH_B.patch_label} -> {PATCH_B.summary_path.name} / {PATCH_B.matchup_path.name}")
    print(f"Rows exported: {len(results_df)} summary rows, {len(top_pools_df)} ranked-pool rows")
    print(f"Summary CSV: {output_dir / 'cross_patch_results.csv'}")
    print(f"Pool CSV: {output_dir / 'top_pools.csv'}")
    print()
    print(results_df.to_string(index=False))
    print()
    print("Extension point: add a beta-binomial / empirical Bayes model in experiment_models.py.")


if __name__ == "__main__":
    main()
