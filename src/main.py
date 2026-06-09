from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import pandas as pd

from data_loader import collect_missing_matchup_pairs, load_patch_data
from matchup_estimator import DEFAULT_EB_ALPHA, build_shrinkage_comparison
from optimizer import rank_top_pools
from pool_contribution import build_pool_contribution_report
from rank_comparison import (
    build_pool_rank_comparison,
    top_rank_table,
    unique_normalized_pools,
)
from scoring import build_counterpick_table, compute_blind_scores
from utils import (
    canonicalize_champion_name,
    dataframe_for_console,
    describe_frequency_status,
    parse_candidates_from_args,
    resolve_data_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="League of Legends midlane champion pool optimizer."
    )
    parser.add_argument(
        "--patch",
        required=True,
        help="Patch label to load from data/<patch>/, for example 16.05 or 16.06.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(resolve_data_dir(__file__)),
        help="Base data directory containing patch folders under data/<patch>/.",
    )
    parser.add_argument(
        "--candidates",
        nargs="*",
        help="Candidate champions. Use spaces, commas, or both.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=3,
        help="Desired champion pool size n.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top pools to display.",
    )
    parser.add_argument(
        "--lowest-pickrate",
        type=float,
        default=None,
        help=(
            "Optional candidate-only filter in percent. "
            "Example: --lowest-pickrate 1 keeps only champions with pick rate > 1%% "
            "as eligible pool members. Enemy weights and enemy champions are unchanged."
        ),
    )
    parser.add_argument(
        "--estimator",
        choices=("raw", "eb"),
        default="raw",
        help="Matchup estimator used by the optimizer. Default: raw.",
    )
    parser.add_argument(
        "--eb-alpha",
        type=float,
        default=DEFAULT_EB_ALPHA,
        help="Empirical Bayes prior sample size. Default: 100.",
    )
    parser.add_argument(
        "--eb-mu",
        type=float,
        default=None,
        help="Optional EB prior mean in [0, 1]. Defaults to the games-weighted global mean.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for matchup_shrinkage_comparison.csv.",
    )
    parser.add_argument(
        "--uncertainty",
        action="store_true",
        help="Simulate beta-posterior uncertainty for the top point-estimate pools.",
    )
    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=5000,
        help="Number of posterior matchup matrices to simulate. Default: 5000.",
    )
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=None,
        help="Beta prior pseudo-games. Defaults to --eb-alpha.",
    )
    parser.add_argument(
        "--posterior-seed",
        type=int,
        default=42,
        help="Random seed for posterior simulation. Default: 42.",
    )
    parser.add_argument(
        "--simulation-mode",
        choices=("fixed-policy", "oracle"),
        default="fixed-policy",
        help=(
            "Posterior pool simulation policy. fixed-policy locks each enemy's "
            "best response by posterior mean; oracle reselects after every draw."
        ),
    )
    parser.add_argument(
        "--simulate-top-pools",
        type=int,
        default=100,
        help="Simulate only the top N point-estimate pools. Default: 100.",
    )
    parser.add_argument(
        "--output-posterior-matchups",
        default=None,
        help="Posterior matchup CSV. Defaults under --output-dir.",
    )
    parser.add_argument(
        "--output-pool-simulation",
        default=None,
        help="Detailed pool simulation CSV. Defaults under --output-dir.",
    )
    parser.add_argument(
        "--compare-ranks",
        action="store_true",
        help="Compare raw, EB, and posterior simulation pool rankings.",
    )
    parser.add_argument(
        "--compare-top-n",
        type=int,
        default=10,
        help="Number of leaders from each ranking included in the comparison. Default: 10.",
    )
    parser.add_argument(
        "--comparison-output",
        default=None,
        help="Rank comparison CSV. Defaults to pool_rank_comparison.csv under --output-dir.",
    )
    parser.add_argument(
        "--pool-contribution-output",
        default=None,
        help="Top-pool matchup contribution CSV. Defaults under --output-dir.",
    )
    parser.add_argument(
        "--exclude-champions",
        nargs="*",
        default=None,
        help="Champions to remove from the optimizer candidate set.",
    )
    parser.add_argument(
        "--selection-bias-diagnostics",
        action="store_true",
        help="Generate selection-bias and generalizability diagnostic outputs.",
    )
    parser.add_argument(
        "--selection-bias-output-dir",
        default=None,
        help="Diagnostic output directory. Defaults to --output-dir.",
    )
    parser.add_argument(
        "--selection-bias-top-pools",
        type=int,
        default=100,
        help="Number of leading pools used for dependency diagnostics. Default: 100.",
    )
    parser.add_argument(
        "--selection-bias-extra-data-dir",
        default=None,
        help=(
            "Optional directory containing LoLalytics and alternate-rank OP.GG "
            "extracts. The workspace-level data directory is auto-detected when present."
        ),
    )
    parser.add_argument(
        "--method-sweep",
        action="store_true",
        help="Run aggregate-data robustness diagnostics without Riot API data.",
    )
    parser.add_argument(
        "--method-sweep-output-dir",
        default=None,
        help="Method-sweep output directory. Defaults under --output-dir.",
    )
    parser.add_argument(
        "--method-sweep-extra-data-dir",
        default=None,
        help="Optional directory containing local OP.GG and LoLalytics extracts.",
    )
    parser.add_argument(
        "--method-sweep-top-pools",
        type=int,
        default=100,
        help="Number of leading pools retained by each sweep method. Default: 100.",
    )
    parser.add_argument(
        "--method-sweep-posterior-samples",
        type=int,
        default=5000,
        help="Posterior draws for fixed-policy and oracle comparisons. Default: 5000.",
    )
    parser.add_argument(
        "--method-sweep-frequency-samples",
        type=int,
        default=250,
        help="Dirichlet enemy-frequency perturbations. Default: 250.",
    )
    parser.add_argument(
        "--method-sweep-frequency-effective-sample-size",
        type=float,
        default=5000.0,
        help="Dirichlet concentration for enemy-frequency perturbations. Default: 5000.",
    )
    parser.add_argument(
        "--method-sweep-alpha-values",
        nargs="+",
        type=float,
        default=[0.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0],
        help="EB prior-strength grid.",
    )
    parser.add_argument(
        "--method-sweep-offmeta-lambdas",
        nargs="+",
        type=float,
        default=[0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1],
        help="Offmeta penalty-strength grid in score-probability units.",
    )
    return parser


def filter_candidates_by_pickrate(
    candidates: list[str],
    summary_df: pd.DataFrame,
    threshold_percent: float,
) -> tuple[list[str], int]:
    threshold = threshold_percent / 100.0
    summary_lookup = {
        row.champion_key: float(row.pickrate)
        for row in summary_df.itertuples(index=False)
    }

    unmatched_candidates = [
        champion
        for champion in candidates
        if canonicalize_champion_name(champion) not in summary_lookup
    ]
    if unmatched_candidates:
        raise ValueError(
            "Could not match some candidate champions to opgg_mid_champion_summary.csv: "
            + ", ".join(unmatched_candidates)
        )

    filtered_candidates = [
        champion
        for champion in candidates
        if summary_lookup[canonicalize_champion_name(champion)] >= threshold
    ]
    removed_count = len(candidates) - len(filtered_candidates)
    return filtered_candidates, removed_count


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    loaded = load_patch_data(
        args.patch,
        data_dir,
        estimator=args.estimator,
        eb_alpha=args.eb_alpha,
        eb_mu=args.eb_mu,
    )
    matchup_df = loaded.matchup_df
    enemy_frequencies = loaded.frequency_df
    summary_df = loaded.summary_df
    matchup_lookup = loaded.matchup_lookup

    available_candidates = sorted(matchup_df["champion_i"].unique())
    requested_candidates = parse_candidates_from_args(args.candidates)
    candidates = requested_candidates or available_candidates
    candidate_count_before_filtering = len(candidates)
    removed_by_pickrate = 0

    unknown_candidates = sorted(set(candidates) - set(available_candidates))
    if unknown_candidates:
        raise ValueError(
            "Some requested candidates are not present in the matchup dataset: "
            + ", ".join(unknown_candidates)
        )

    excluded_requested = parse_candidates_from_args(args.exclude_champions)
    if excluded_requested:
        available_by_key = {
            canonicalize_champion_name(champion): champion
            for champion in available_candidates
        }
        unknown_exclusions = [
            champion
            for champion in excluded_requested
            if canonicalize_champion_name(champion) not in available_by_key
        ]
        if unknown_exclusions:
            raise ValueError(
                "Some excluded champions are not present in the matchup dataset: "
                + ", ".join(unknown_exclusions)
            )
        excluded = {
            available_by_key[canonicalize_champion_name(champion)]
            for champion in excluded_requested
        }
        candidates = [champion for champion in candidates if champion not in excluded]

    if args.lowest_pickrate is not None:
        if args.lowest_pickrate < 0:
            raise ValueError("--lowest-pickrate must be non-negative")
        candidates, removed_by_pickrate = filter_candidates_by_pickrate(
            candidates=candidates,
            summary_df=summary_df,
            threshold_percent=args.lowest_pickrate,
        )
        if not candidates:
            raise ValueError(
                "Pickrate filter removed all candidate champions. "
                "Lower the threshold or pass a broader candidate list."
            )

    if args.pool_size <= 0:
        raise ValueError("pool size must be positive")

    if args.pool_size > len(candidates):
        raise ValueError(
            f"pool size {args.pool_size} is larger than the number of candidates ({len(candidates)})"
        )
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.posterior_samples <= 0:
        raise ValueError("--posterior-samples must be positive")
    if args.simulate_top_pools <= 0:
        raise ValueError("--simulate-top-pools must be positive")
    if args.compare_top_n <= 0:
        raise ValueError("--compare-top-n must be positive")
    if args.selection_bias_top_pools <= 0:
        raise ValueError("--selection-bias-top-pools must be positive")
    if args.method_sweep_top_pools <= 0:
        raise ValueError("--method-sweep-top-pools must be positive")
    if args.method_sweep_posterior_samples <= 0:
        raise ValueError("--method-sweep-posterior-samples must be positive")
    if args.method_sweep_frequency_samples <= 0:
        raise ValueError("--method-sweep-frequency-samples must be positive")
    if args.method_sweep_frequency_effective_sample_size <= 0:
        raise ValueError(
            "--method-sweep-frequency-effective-sample-size must be positive"
        )
    if any(value < 0 for value in args.method_sweep_alpha_values):
        raise ValueError("--method-sweep-alpha-values must be non-negative")
    if any(value < 0 for value in args.method_sweep_offmeta_lambdas):
        raise ValueError("--method-sweep-offmeta-lambdas must be non-negative")

    prior_strength = (
        args.eb_alpha if args.prior_strength is None else args.prior_strength
    )
    if prior_strength <= 0:
        raise ValueError("--prior-strength must be positive")

    missing_pairs = collect_missing_matchup_pairs(
        candidates=candidates,
        enemy_champions=enemy_frequencies["champion_j"].tolist(),
        matchup_lookup=matchup_lookup,
    )

    blind_scores = compute_blind_scores(candidates, enemy_frequencies, matchup_lookup)
    run_uncertainty = args.uncertainty or args.compare_ranks
    retained_pool_count = max(
        args.top_k,
        args.simulate_top_pools if run_uncertainty else 0,
        args.compare_top_n if args.compare_ranks else 0,
        args.selection_bias_top_pools if args.selection_bias_diagnostics else 0,
        args.method_sweep_top_pools if args.method_sweep else 0,
    )
    ranked_pools = rank_top_pools(
        candidates,
        args.pool_size,
        enemy_frequencies,
        matchup_lookup,
        top_n=retained_pool_count,
    )

    raw_ranked_pools = None
    eb_ranked_pools = None
    if args.compare_ranks:
        raw_lookup = {
            (row.champion_i, row.champion_j): float(row.raw_winrate)
            for row in matchup_df.itertuples(index=False)
        }
        eb_lookup = {
            (row.champion_i, row.champion_j): float(row.shrinked_winrate)
            for row in matchup_df.itertuples(index=False)
        }
        raw_ranked_pools = rank_top_pools(
            candidates,
            args.pool_size,
            enemy_frequencies,
            raw_lookup,
            top_n=retained_pool_count,
        )
        eb_ranked_pools = rank_top_pools(
            candidates,
            args.pool_size,
            enemy_frequencies,
            eb_lookup,
            top_n=retained_pool_count,
        )

    best_pool = tuple(ranked_pools.iloc[0]["pool"])
    best_blind_pick = blind_scores.iloc[0]["champion"]
    best_blind_score = float(blind_scores.iloc[0]["blind_score"])
    top_pools = ranked_pools.head(args.top_k)
    counterpick_table = build_counterpick_table(best_pool, enemy_frequencies, matchup_lookup)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shrinkage_report_path = output_dir / "matchup_shrinkage_comparison.csv"
    build_shrinkage_comparison(matchup_df).to_csv(shrinkage_report_path, index=False)

    posterior_df = None
    simulation_summary_df = None
    posterior_output_path = Path(
        args.output_posterior_matchups
        or output_dir / "posterior_matchups.csv"
    )
    simulation_output_path = Path(
        args.output_pool_simulation
        or output_dir / "pool_score_simulation.csv"
    )
    comparison_output_path = Path(
        args.comparison_output
        or output_dir / "pool_rank_comparison.csv"
    )
    contribution_output_path = Path(
        args.pool_contribution_output
        or output_dir / "pool_contributions.csv"
    )
    summary_output_path = simulation_output_path.with_name(
        f"{simulation_output_path.stem}_summary{simulation_output_path.suffix}"
    )
    comparison_df = None
    contribution_df = None
    selection_bias_artifacts = None
    method_sweep_artifacts = None
    if run_uncertainty:
        from uncertainty import (
            build_matchup_posteriors,
            simulate_pool_scores,
            simulation_summary_path,
        )

        summary_output_path = simulation_summary_path(simulation_output_path)
        posterior_df = build_matchup_posteriors(
            matchup_df,
            prior_strength=prior_strength,
            prior_mean=loaded.eb_mu,
        )
        posterior_output_path.parent.mkdir(parents=True, exist_ok=True)
        posterior_df.to_csv(posterior_output_path, index=False)

        simulation_pools = ranked_pools.head(args.simulate_top_pools)["pool"].tolist()
        if raw_ranked_pools is not None and eb_ranked_pools is not None:
            simulation_pools = unique_normalized_pools(
                raw_ranked_pools.head(args.simulate_top_pools)["pool"].tolist()
                + eb_ranked_pools.head(args.simulate_top_pools)["pool"].tolist()
            )

        simulation_detail_df, simulation_summary_df = simulate_pool_scores(
            pools=simulation_pools,
            enemy_frequencies=enemy_frequencies,
            posterior_df=posterior_df,
            sample_count=args.posterior_samples,
            seed=args.posterior_seed,
            simulation_mode=args.simulation_mode,
        )
        simulation_output_path.parent.mkdir(parents=True, exist_ok=True)
        simulation_detail_df.to_csv(simulation_output_path, index=False)
        simulation_summary_df.to_csv(summary_output_path, index=False)

        if raw_ranked_pools is not None and eb_ranked_pools is not None:
            comparison_df = build_pool_rank_comparison(
                raw_ranked_df=raw_ranked_pools,
                eb_ranked_df=eb_ranked_pools,
                simulation_summary_df=simulation_summary_df,
                top_n=args.compare_top_n,
            )
            comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(comparison_output_path, index=False)
            contribution_df = build_pool_contribution_report(
                comparison_df=comparison_df,
                enemy_frequencies=enemy_frequencies,
                posterior_df=posterior_df,
            )
            contribution_output_path.parent.mkdir(parents=True, exist_ok=True)
            contribution_df.to_csv(contribution_output_path, index=False)

    if args.selection_bias_diagnostics:
        from selection_bias import run_selection_bias_diagnostics

        diagnostic_output_dir = Path(
            args.selection_bias_output_dir or args.output_dir
        )
        if args.selection_bias_extra_data_dir:
            extra_data_dir = Path(args.selection_bias_extra_data_dir)
        else:
            auto_extra_data_dir = data_dir.resolve().parent.parent / "data"
            extra_data_dir = (
                auto_extra_data_dir if auto_extra_data_dir.exists() else None
            )
        selection_bias_artifacts = run_selection_bias_diagnostics(
            loaded=loaded,
            data_dir=data_dir,
            candidates=candidates,
            pool_size=args.pool_size,
            ranked_pools=ranked_pools,
            output_dir=diagnostic_output_dir,
            top_n=args.selection_bias_top_pools,
            prior_strength=prior_strength,
            extra_data_dir=extra_data_dir,
        )

    if args.method_sweep:
        from method_sweep import run_method_sweep

        method_sweep_output_dir = Path(
            args.method_sweep_output_dir
            or output_dir / f"method_sweep_{loaded.patch_label.replace('.', '_')}"
        )
        if args.method_sweep_extra_data_dir:
            method_sweep_extra_data_dir = Path(args.method_sweep_extra_data_dir)
        else:
            auto_extra_data_dir = data_dir.resolve().parent.parent / "data"
            method_sweep_extra_data_dir = (
                auto_extra_data_dir if auto_extra_data_dir.exists() else None
            )
        method_sweep_artifacts = run_method_sweep(
            loaded=loaded,
            data_dir=data_dir,
            candidates=candidates,
            pool_size=args.pool_size,
            ranked_pools=ranked_pools,
            output_dir=method_sweep_output_dir,
            extra_data_dir=method_sweep_extra_data_dir,
            top_n=args.method_sweep_top_pools,
            posterior_samples=args.method_sweep_posterior_samples,
            posterior_seed=args.posterior_seed,
            frequency_samples=args.method_sweep_frequency_samples,
            frequency_effective_sample_size=(
                args.method_sweep_frequency_effective_sample_size
            ),
            alpha_values=args.method_sweep_alpha_values,
            offmeta_lambdas=args.method_sweep_offmeta_lambdas,
        )

    print("=== Inputs ===")
    print(f"Patch: {loaded.patch_label}")
    print(f"Candidates: {', '.join(candidates)}")
    print(f"Pool size (n): {args.pool_size}")
    print(f"Champions loaded: {loaded.champion_count}")
    print(f"Matchup rows loaded: {loaded.matchup_row_count}")
    print(f"Estimator: {loaded.estimator}")
    print(f"EB alpha: {loaded.eb_alpha:g}")
    print(f"EB prior mean (mu): {loaded.eb_mu:.6f}")
    print(f"Shrinkage comparison: {shrinkage_report_path.resolve()}")
    if run_uncertainty:
        print(f"Posterior prior strength: {prior_strength:g}")
        print(f"Posterior samples: {args.posterior_samples}")
        print(f"Posterior seed: {args.posterior_seed}")
        print(f"Simulation mode: {args.simulation_mode}")
    print("Enemy frequencies: " + describe_frequency_status(loaded.frequency_status))
    if args.lowest_pickrate is not None:
        print(f"Lowest pickrate threshold: > {args.lowest_pickrate:.2f}%")
        print(f"Candidate count before filtering: {candidate_count_before_filtering}")
        print(f"Candidates removed by pickrate filter: {removed_by_pickrate}")
        print(f"Candidate count after filtering: {len(candidates)}")
        print("Pickrate filter scope: candidate pool only; enemy distribution is unchanged")
    if excluded_requested:
        print(f"Excluded champions: {', '.join(sorted(excluded))}")
    if missing_pairs:
        print(
            "Scoring note: singular impossible or missing matchups are skipped and the "
            "remaining enemy frequencies are renormalized per champion/pool."
        )
    print()

    print("=== Normalized Enemy Frequencies ===")
    print(dataframe_for_console(enemy_frequencies, percentage_columns=["freq_j"]))
    print()

    print("=== Blind Scores ===")
    print(dataframe_for_console(blind_scores, percentage_columns=["blind_score"]))
    print()

    print("=== Pool Search Summary ===")
    search_method = ranked_pools.attrs.get("search_method", "unknown")
    evaluated_pool_count = ranked_pools.attrs.get(
        "evaluated_pool_count", len(ranked_pools)
    )
    total_combination_count = ranked_pools.attrs.get(
        "total_combination_count", comb(len(candidates), args.pool_size)
    )
    print(f"Pool search method: {search_method}")
    if search_method == "exact_brute_force":
        print(f"Evaluated all {total_combination_count} possible pools.")
    else:
        scored_candidate_count = ranked_pools.attrs.get(
            "scored_candidate_count", evaluated_pool_count
        )
        print(
            f"Evaluated {evaluated_pool_count} final beam candidates and "
            f"{scored_candidate_count} candidate states while searching "
            f"{total_combination_count} possible pools."
        )
    print(f"Retained the top {len(ranked_pools)} pools.")
    print(f"Best pool: {', '.join(best_pool)}")
    print(f"Best pool score: {ranked_pools.iloc[0]['pool_score']:.2%}")
    print(f"Best blind pick: {best_blind_pick} ({best_blind_score:.2%})")
    print()

    print("=== Top Pools ===")
    print(
        dataframe_for_console(
            top_pools[["pool_label", "pool_score"]],
            percentage_columns=["pool_score"],
        )
    )
    print()

    if simulation_summary_df is not None:
        print("=== Posterior Pool Simulation ===")
        print(
            dataframe_for_console(
                simulation_summary_df.head(args.top_k),
                percentage_columns=[
                    "mean_score",
                    "median_score",
                    "sd_score",
                    "lower_5_score",
                    "upper_95_score",
                    "probability_of_being_best",
                ],
            )
        )
        print()
        print(f"Posterior matchups CSV: {posterior_output_path.resolve()}")
        print(f"Pool simulation CSV: {simulation_output_path.resolve()}")
        print(f"Pool simulation summary CSV: {summary_output_path.resolve()}")
        print()

    if comparison_df is not None:
        ranking_specs = [
            ("Raw Score", "raw_rank", "raw_score"),
            ("EB Score", "eb_rank", "eb_score"),
            ("Posterior Mean Score", "posterior_mean_rank", "mean_score"),
            ("Posterior Lower 5 Score", "lower_5_rank", "lower_5_score"),
            (
                "Probability Of Being Best",
                "prob_best_rank",
                "probability_of_being_best",
            ),
        ]
        for title, rank_column, score_column in ranking_specs:
            print(f"=== Top {args.compare_top_n}: {title} ===")
            print(
                dataframe_for_console(
                    top_rank_table(
                        comparison_df,
                        rank_column,
                        score_column,
                        args.compare_top_n,
                    ),
                    percentage_columns=[score_column],
                )
            )
            print()

        posterior_changes = comparison_df.dropna(
            subset=["rank_change_raw_to_posterior_mean"]
        )
        risers = posterior_changes[
            posterior_changes["rank_change_raw_to_posterior_mean"] > 0
        ]
        fallers = posterior_changes[
            posterior_changes["rank_change_raw_to_posterior_mean"] < 0
        ]
        print("=== Biggest Rank Risers: Raw To Posterior Mean ===")
        print(
            risers.sort_values(
                ["rank_change_raw_to_posterior_mean", "pool"],
                ascending=[False, True],
            )
            .head(args.compare_top_n)[
                [
                    "pool",
                    "raw_rank",
                    "posterior_mean_rank",
                    "rank_change_raw_to_posterior_mean",
                ]
            ]
            .to_string(index=False)
        )
        print()
        print("=== Biggest Rank Fallers: Raw To Posterior Mean ===")
        print(
            fallers.sort_values(
                ["rank_change_raw_to_posterior_mean", "pool"],
                ascending=[True, True],
            )
            .head(args.compare_top_n)[
                [
                    "pool",
                    "raw_rank",
                    "posterior_mean_rank",
                    "rank_change_raw_to_posterior_mean",
                ]
            ]
            .to_string(index=False)
        )
        print()
        print(f"Pool rank comparison CSV: {comparison_output_path.resolve()}")
        print(f"Pool contribution CSV: {contribution_output_path.resolve()}")
        print()

    if selection_bias_artifacts is not None:
        print("=== Selection Bias Diagnostics ===")
        for artifact_path in selection_bias_artifacts.__dict__.values():
            print(Path(artifact_path).resolve())
        print()

    if method_sweep_artifacts is not None:
        print("=== Aggregate Method Sweep ===")
        for artifact_path in method_sweep_artifacts.__dict__.values():
            print(Path(artifact_path).resolve())
        print()

    print("=== Counterpick Table For Best Pool ===")
    print(
        dataframe_for_console(
            counterpick_table,
            percentage_columns=["matchup_value", "enemy_frequency"],
        )
    )
    print()
    print("Extension point: add train-patch vs eval-patch workflows on top of load_patch_data().")


if __name__ == "__main__":
    main()
