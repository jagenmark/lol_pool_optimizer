from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_loader import collect_missing_matchup_pairs, load_patch_data
from optimizer import rank_pools
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
    loaded = load_patch_data(args.patch, data_dir)
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
    print(f"Patch: {loaded.patch_label}")
    print(f"Candidates: {', '.join(candidates)}")
    print(f"Pool size (n): {args.pool_size}")
    print(f"Champions loaded: {loaded.champion_count}")
    print(f"Matchup rows loaded: {loaded.matchup_row_count}")
    print("Enemy frequencies: " + describe_frequency_status(loaded.frequency_status))
    if args.lowest_pickrate is not None:
        print(f"Lowest pickrate threshold: > {args.lowest_pickrate:.2f}%")
        print(f"Candidate count before filtering: {candidate_count_before_filtering}")
        print(f"Candidates removed by pickrate filter: {removed_by_pickrate}")
        print(f"Candidate count after filtering: {len(candidates)}")
        print("Pickrate filter scope: candidate pool only; enemy distribution is unchanged")
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
    print(
        dataframe_for_console(
            top_pools[["pool_label", "pool_score"]],
            percentage_columns=["pool_score"],
        )
    )
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
