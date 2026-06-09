from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scope_stability import (
    UnsafeRankBucketError,
    build_inclusion_frequency,
    discover_scope_files,
    jaccard_similarity,
    pairwise_mean_jaccard,
    subtract_cumulative_rank_bucket,
)


def _write_scope_pair(root: Path, patch: str, rank: str) -> None:
    matchup = pd.DataFrame(
        {
            "champion_i": ["A"],
            "champion_j": ["B"],
            "matchup_games": [100],
            "matchup_winrate_i_vs_j": [55.0],
        }
    )
    summary = pd.DataFrame(
        {
            "champion_name": ["A"],
            "pickrate": [10.0],
            "winrate": [51.0],
        }
    )
    matchup.to_csv(root / f"opgg_mid_matchups__{rank}__{patch}.csv", index=False)
    summary.to_csv(
        root / f"opgg_mid_champion_summary__{rank}__{patch}.csv",
        index=False,
    )


def test_scope_discovery_pairs_files_and_prefers_explicit_aggregate(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared"
    aggregate = tmp_path / "aggregate"
    patch_dir = prepared / "16.10"
    patch_dir.mkdir(parents=True)
    aggregate.mkdir()
    for name in (
        "opgg_mid_matchups_clean.csv",
        "opgg_mid_champion_summary.csv",
        "enemy_freq_df.csv",
    ):
        (patch_dir / name).write_text("placeholder\n", encoding="utf-8")
    _write_scope_pair(aggregate, "16.10", "plat_plus")
    _write_scope_pair(aggregate, "16.10", "emerald_plus")
    (aggregate / "opgg_mid_matchups__diamond_plus__16.10.csv").write_text(
        "unpaired\n",
        encoding="utf-8",
    )

    scopes = discover_scope_files(prepared, aggregate)

    assert [scope.scope_id for scope in scopes] == [
        "16.10__plat_plus__cumulative",
        "16.10__emerald_plus__cumulative",
    ]
    assert scopes[0].source_format == "aggregate_patch_rank_pair"


def test_scope_discovery_includes_unique_dated_raw_scope(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    aggregate = tmp_path / "aggregate"
    raw = aggregate / "raw"
    prepared.mkdir()
    raw.mkdir(parents=True)
    matchup = pd.DataFrame(
        {
            "champion_i": ["A"],
            "champion_j": ["B"],
            "matchup_games": [100],
            "matchup_winrate_i_vs_j": [55.0],
            "patch": ["16.07"],
            "elo": ["emerald_plus"],
            "lane": ["mid"],
        }
    )
    summary = pd.DataFrame(
        {
            "champion_name": ["A"],
            "pickrate": [10.0],
            "winrate": [51.0],
            "patch": ["16.07"],
            "elo": ["emerald_plus"],
            "lane": ["mid"],
        }
    )
    matchup.to_csv(
        raw / "opgg_mid_matchups__global__emerald_plus__2026-04-05.csv",
        index=False,
    )
    summary.to_csv(
        raw / "opgg_mid_champion_summary__global__emerald_plus__2026-04-05.csv",
        index=False,
    )

    scopes = discover_scope_files(prepared, aggregate)

    assert [scope.scope_id for scope in scopes] == [
        "16.07__emerald_plus__cumulative"
    ]
    assert scopes[0].source_format == "dated_raw_patch_rank_pair"


def test_rank_bucket_subtraction_uses_exact_additive_counts() -> None:
    lower = pd.DataFrame(
        {
            "champion_i": ["A", "A"],
            "champion_j": ["X", "Y"],
            "games_ij": [100.0, 80.0],
            "wins_i": [60.0, 36.0],
        }
    )
    higher = pd.DataFrame(
        {
            "champion_i": ["A", "A"],
            "champion_j": ["X", "Y"],
            "games_ij": [40.0, 30.0],
            "wins_i": [22.0, 12.0],
        }
    )

    bucket = subtract_cumulative_rank_bucket(lower, higher)

    assert bucket["games_ij"].tolist() == [60.0, 50.0]
    assert bucket["wins_i"].tolist() == [38.0, 24.0]
    assert np.allclose(bucket["winrate_ij"], [38.0 / 60.0, 24.0 / 50.0])


def test_rank_bucket_subtraction_rejects_rounded_rate_inputs() -> None:
    lower = pd.DataFrame(
        {
            "champion_i": ["A"],
            "champion_j": ["X"],
            "games_ij": [100.0],
            "winrate_ij": [0.55],
        }
    )
    higher = lower.copy()

    with pytest.raises(UnsafeRankBucketError, match="exact additive columns"):
        subtract_cumulative_rank_bucket(lower, higher)


def test_rank_bucket_subtraction_rejects_non_nested_keys() -> None:
    lower = pd.DataFrame(
        {
            "champion_i": ["A"],
            "champion_j": ["X"],
            "games_ij": [100.0],
            "wins_i": [55.0],
        }
    )
    higher = pd.DataFrame(
        {
            "champion_i": ["A"],
            "champion_j": ["Y"],
            "games_ij": [20.0],
            "wins_i": [10.0],
        }
    )

    with pytest.raises(UnsafeRankBucketError, match="keys absent"):
        subtract_cumulative_rank_bucket(lower, higher)


def test_stability_and_inclusion_calculations() -> None:
    ranked = pd.DataFrame(
        {
            "pool": [("A", "B"), ("A", "C"), ("B", "C")],
            "pool_label": ["A, B", "A, C", "B, C"],
            "pool_score": [0.60, 0.59, 0.58],
        }
    )

    inclusion = build_inclusion_frequency("scope", ranked, ["A", "B", "C"])
    a_row = inclusion[inclusion["champion"] == "A"].iloc[0]

    assert a_row["top_100_appearances"] == 2
    assert np.isclose(a_row["top_100_inclusion_frequency"], 2 / 3)
    assert a_row["best_pool_member"]
    assert np.isclose(jaccard_similarity({"A", "B"}, {"B", "C"}), 1 / 3)
    assert np.isclose(
        pairwise_mean_jaccard([{"A", "B"}, {"A", "C"}, {"A", "B"}]),
        (1 / 3 + 1.0 + 1 / 3) / 3,
    )
