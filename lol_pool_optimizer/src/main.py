from __future__ import annotations

import argparse
from pathlib import Path

from data_loader import (
    collect_missing_matchup_pairs,
    load_inputs,
)
from optimizer import rank_pools
from scoring import build_counterpick_table, compute_blind_scores
from utils import (
    dataframe_for_console,
    describe_frequency_status,
    parse_candidates_from_args,
    resolve_data_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="League of Legends midlane champion pool optimizer (v1 prototype)."
    )
    parser.add_argument(
        "--data-dir",
        default=str(resolve_data_dir(__file__)),
        help="Project data directory. By default this loads data/clean/*.csv.",
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
        "--dataset",
        choices=["clean", "synthetic"],
        default="clean",
        help="Use cleaned real data by default, or the synthetic fallback dataset for testing.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    loaded = load_inputs(data_dir, dataset=args.dataset)
    matchup_df = loaded.matchup_df
    enemy_frequencies = loaded.frequency_df
    matchup_lookup = loaded.matchup_lookup

    available_candidates = sorted(matchup_df["champion_i"].unique())
    requested_candidates = parse_candidates_from_args(args.candidates)
    candidates = requested_candidates or available_candidates

    unknown_candidates = sorted(set(candidates) - set(available_candidates))
    if unknown_candidates:
        raise ValueError(
            "Some requested candidates are not present in the matchup dataset: "
            + ", ".join(unknown_candidates)
        )

    if args.pool_size <= 0:
        raise ValueError("pool size must be positive")

    if args.pool_size > len(candidates):
        raise ValueError(
            f"pool size {args.pool_size} is larger than the number of candidates ({len(candidates)})"
        )

    missing_pairs = collect_missing_matchup_pairs(
        candidates=candidates,
        enemy_champions=enemy_frequencies["champion_j"].tolist(),
        matchup_lookup=matchup_lookup,
    )

    blind_scores = compute_blind_scores(candidates, enemy_frequencies, matchup_lookup)
    ranked_pools = rank_pools(candidates, args.pool_size, enemy_frequencies, matchup_lookup)

    best_pool = tuple(ranked_pools.iloc[0]["pool"])
    best_blind_pick = blind_scores.iloc[0]["champion"]
    best_blind_score = float(blind_scores.iloc[0]["blind_score"])
    top_pools = ranked_pools.head(args.top_k)
    counterpick_table = build_counterpick_table(best_pool, enemy_frequencies, matchup_lookup)

    print("=== Inputs ===")
    print(f"Dataset mode: {args.dataset}")
    print(f"Candidates: {', '.join(candidates)}")
    print(f"Pool size (n): {args.pool_size}")
    print(f"Champions loaded: {loaded.champion_count}")
    print(f"Matchup rows loaded: {loaded.matchup_row_count}")
    print(
        "Enemy frequencies: "
        + describe_frequency_status(loaded.frequency_status)
    )
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
    print(f"Generated {len(ranked_pools)} pools via brute force.")
    print(f"Best pool: {', '.join(best_pool)}")
    print(f"Best pool score: {ranked_pools.iloc[0]['pool_score']:.2%}")
    print(f"Best blind pick: {best_blind_pick} ({best_blind_score:.2%})")
    print()

    print("=== Top Pools ===")
    print(dataframe_for_console(top_pools[["pool_label", "pool_score"]], percentage_columns=["pool_score"]))
    print()

    print("=== Counterpick Table For Best Pool ===")
    print(
        dataframe_for_console(
            counterpick_table,
            percentage_columns=["matchup_value", "enemy_frequency"],
        )
    )
    print()
    print("TODO: Add shrinkage / Empirical Bayes preprocessing as a separate module.")
    print("TODO: Add optional banrate adjustment module.")
    print("TODO: Add a Streamlit UI once the CLI workflow is stable.")


if __name__ == "__main__":
    main()
