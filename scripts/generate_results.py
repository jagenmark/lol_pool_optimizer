from __future__ import annotations

import argparse
import sys
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import LoadedInputs, load_patch_data
from main import filter_candidates_by_pickrate
from optimizer import rank_pools
from scoring import build_counterpick_table, compute_blind_scores, pool_score
from utils import canonicalize_champion_name, parse_candidates_from_args, resolve_data_dir


MODEL_OPTIMIZED = "optimized_pool_score"
MODEL_WINRATE = "baseline_highest_winrate"
MODEL_BLINDSCORE = "baseline_highest_blindscore"
MODEL_PICKRATE = "baseline_highest_pickrate"
MODEL_OPTIMIZED_FORCED = "optimized_model_forced"
MODEL_WINRATE_FORCED = "baseline_winrate_forced"
MODEL_BLINDSCORE_FORCED = "baseline_blindscore_forced"
MODEL_PICKRATE_FORCED = "baseline_pickrate_forced"
METHOD_LABELS_SV = {
    MODEL_OPTIMIZED: "Optimerad modell",
    MODEL_WINRATE: "Högst winrate",
    MODEL_BLINDSCORE: "Högst blindscore",
    MODEL_PICKRATE: "Högst pickrate",
    MODEL_OPTIMIZED_FORCED: "Optimerad modell",
    MODEL_WINRATE_FORCED: "Högst winrate",
    MODEL_BLINDSCORE_FORCED: "Högst blindscore",
    MODEL_PICKRATE_FORCED: "Högst pickrate",
}


def method_label(method: str) -> str:
    return METHOD_LABELS_SV.get(method, method)


def transition_label(train_patch: str, test_patch: str) -> str:
    return f"{train_patch} → {test_patch}"


def style_report_axes(ax, use_y_grid: bool = True) -> None:
    if use_y_grid:
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def apply_score_ylim(
    ax,
    values: Iterable[float],
    y_min: float | None = None,
    y_max: float | None = None,
    auto_zoom: bool = False,
) -> None:
    numeric_values = [float(value) for value in values if pd.notna(value)]
    if not numeric_values:
        return
    if auto_zoom:
        observed_min = min(numeric_values)
        observed_max = max(numeric_values)
        observed_range = observed_max - observed_min
        padding = observed_range * 0.08 if observed_range > 0 else 0.01
        y_min = observed_min - padding if y_min is None else y_min
        y_max = observed_max + padding if y_max is None else y_max
    if y_min is not None or y_max is not None:
        current_min, current_max = ax.get_ylim()
        ax.set_ylim(
            current_min if y_min is None else y_min,
            current_max if y_max is None else y_max,
        )


def comparison_plot_paths(path: Path) -> tuple[Path, Path]:
    fullscale_path = path.with_name(f"{path.stem}_fullscale{path.suffix}")
    zoomed_path = path.with_name(f"{path.stem}_zoomed{path.suffix}")
    return fullscale_path, zoomed_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate CSV and PNG outputs for the champion pool optimizer results section."
    )
    parser.add_argument("--patch", default=None, help="Primary patch for single-patch results.")
    parser.add_argument(
        "--patches",
        nargs="*",
        default=None,
        help="Patch sequence for validation, for example: --patches 16.05 16.06 16.07.",
    )
    parser.add_argument("--pool-size", type=int, default=3, help="Recommended pool size k.")
    parser.add_argument("--max-pool-size", type=int, default=8, help="Maximum k for pool-size sweeps.")
    parser.add_argument("--candidates", nargs="*", help="Optional candidate champions. Use spaces, commas, or both.")
    parser.add_argument(
        "--candidates-file",
        default=None,
        help="Optional text file with one candidate champion name per line.",
    )
    parser.add_argument(
        "--force-champion",
        nargs="*",
        default=None,
        help="Force one or more champions into optimized and forced-baseline pools.",
    )
    parser.add_argument(
        "--force-champions",
        default=None,
        help="Comma-separated forced champion list. Example: --force-champions Ahri,Syndra.",
    )
    parser.add_argument(
        "--force-champion-batch",
        default=None,
        help='Batch forced champion test. Use "top_pickrate:10" or a comma-separated list like "Ahri,Syndra,Vex".',
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for CSV and PNG files. Defaults to results/<patch>.",
    )
    parser.add_argument(
        "--lowest-pickrate",
        type=float,
        default=None,
        help="Optional candidate-only filter in percent. Enemy weights remain unchanged.",
    )
    parser.add_argument(
        "--top-enemies",
        type=int,
        default=15,
        help="Number of most common enemies for matchup coverage output.",
    )
    parser.add_argument(
        "--max-exact-combinations",
        type=int,
        default=500_000,
        help="Use exact brute force only when n choose k is at or below this limit; otherwise use greedy selection.",
    )
    return parser


def available_patch_labels(data_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in data_dir.iterdir()
        if path.is_dir()
        and (path / "opgg_mid_matchups_clean.csv").exists()
        and (path / "enemy_freq_df.csv").exists()
        and (path / "opgg_mid_champion_summary.csv").exists()
    )


def normalize_patch_label(data_dir: Path, patch: str) -> str:
    available = set(available_patch_labels(data_dir))
    if patch in available:
        return patch
    if patch.startswith("26."):
        suffix = patch.split(".", 1)[1]
        candidate = f"16.{int(suffix):02d}"
        if candidate in available:
            return candidate
    return patch


def resolve_run_patches(args: argparse.Namespace, data_dir: Path) -> tuple[str, list[str]]:
    patches = [normalize_patch_label(data_dir, patch) for patch in (args.patches or available_patch_labels(data_dir))]
    if not patches:
        raise ValueError(f"No patch folders found under {data_dir}")

    primary_patch = normalize_patch_label(data_dir, args.patch) if args.patch else patches[0]
    if primary_patch not in patches:
        patches = [primary_patch, *[patch for patch in patches if patch != primary_patch]]
    return primary_patch, patches


def parse_forced_champions(args: argparse.Namespace) -> list[str]:
    raw_values: list[str] = []
    if args.force_champion:
        raw_values.extend(args.force_champion)
    if args.force_champions:
        raw_values.append(args.force_champions)
    return parse_candidates_from_args(raw_values)


def resolve_forced_champions(raw_forced: list[str], candidates: list[str]) -> list[str]:
    if not raw_forced:
        return []
    matched, missing = match_candidate_names(raw_forced, candidates)
    if missing:
        raise ValueError(
            "Forced champion(s) are not in the eligible candidate set: "
            + ", ".join(missing)
        )
    return matched


def resolve_force_champion_batch(
    batch_spec: str | None,
    candidates: list[str],
    summary_df: pd.DataFrame,
) -> list[str]:
    if not batch_spec:
        return []
    if batch_spec.startswith("top_pickrate:"):
        raw_count = batch_spec.split(":", 1)[1]
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError("--force-champion-batch top_pickrate:N requires an integer N") from exc
        if count <= 0:
            raise ValueError("--force-champion-batch top_pickrate:N requires N > 0")
        return list(top_by_summary_value(candidates, summary_df, "pickrate", count))

    requested = parse_candidates_from_args([batch_spec])
    matched, missing = match_candidate_names(requested, candidates)
    if missing:
        raise ValueError(
            "Batch forced champion(s) are not in the eligible candidate set: "
            + ", ".join(missing)
        )
    return matched


def create_run_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_candidates_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def match_candidate_names(
    requested_candidates: list[str],
    available_candidates: list[str],
) -> tuple[list[str], list[str]]:
    available_by_key = {
        canonicalize_champion_name(champion): champion
        for champion in available_candidates
    }
    matched: list[str] = []
    missing: list[str] = []
    seen = set()
    for requested in requested_candidates:
        key = canonicalize_champion_name(requested)
        champion = available_by_key.get(key)
        if champion is None:
            missing.append(requested)
            continue
        if champion not in seen:
            matched.append(champion)
            seen.add(champion)
    return matched, missing


def select_candidates(
    loaded: LoadedInputs,
    raw_candidates: list[str] | None,
    candidates_file: str | None,
    lowest_pickrate: float | None,
) -> tuple[list[str], dict[str, int | float | None]]:
    available_candidates = sorted(loaded.matchup_df["champion_i"].unique())
    requested_candidates = parse_candidates_from_args(raw_candidates)
    file_candidates = read_candidates_file(Path(candidates_file)) if candidates_file else []
    combined_requested = [*requested_candidates, *file_candidates]
    candidates, missing_candidates = (
        match_candidate_names(combined_requested, available_candidates)
        if combined_requested
        else (available_candidates, [])
    )
    if combined_requested and not candidates:
        raise ValueError("No requested candidate champions matched the selected patch data")

    before_count = len(candidates)
    removed_count = 0
    if lowest_pickrate is not None:
        if lowest_pickrate < 0:
            raise ValueError("--lowest-pickrate must be non-negative")
        candidates, removed_count = filter_candidates_by_pickrate(
            candidates,
            loaded.summary_df,
            lowest_pickrate,
        )
        if not candidates:
            raise ValueError(
                f"Pickrate filter removed all candidates for patch {loaded.patch_label}"
            )

    return candidates, {
        "lowest_pickrate": lowest_pickrate,
        "candidate_file_requested_count": len(file_candidates),
        "candidate_cli_requested_count": len(requested_candidates),
        "candidate_requested_count": len(combined_requested),
        "candidate_matched_count": len(candidates),
        "candidate_missing_count": len(missing_candidates),
        "candidate_count_before_filtering": before_count,
        "candidate_count_after_filtering": len(candidates),
        "candidates_removed_by_pickrate": removed_count,
        "missing_candidates": "; ".join(missing_candidates),
    }


def summary_lookup(summary_df: pd.DataFrame, value_column: str) -> dict[str, float]:
    return {
        canonicalize_champion_name(row.champion_name): float(getattr(row, value_column))
        for row in summary_df.itertuples(index=False)
    }


def top_by_summary_value(candidates: Iterable[str], summary_df: pd.DataFrame, value_column: str, k: int) -> tuple[str, ...]:
    values = summary_lookup(summary_df, value_column)
    unmatched = [
        candidate for candidate in candidates
        if canonicalize_champion_name(candidate) not in values
    ]
    if unmatched:
        raise ValueError(
            "Could not match candidates to champion summary for baseline ranking: "
            + ", ".join(unmatched)
        )
    return tuple(
        sorted(
            candidates,
            key=lambda champion: (
                values[canonicalize_champion_name(champion)],
                champion,
            ),
            reverse=True,
        )[:k]
    )


def greedy_optimized_pool(
    loaded: LoadedInputs,
    candidates: list[str],
    k: int,
    forced_champions: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], float]:
    selected = list(forced_champions)
    remaining = [candidate for candidate in candidates if candidate not in selected]
    for _ in range(k - len(selected)):
        best_candidate = max(
            remaining,
            key=lambda champion: (
                score_named_pool(loaded, tuple([*selected, champion])),
                champion,
            ),
        )
        selected.append(best_candidate)
        remaining.remove(best_candidate)
    pool = tuple(selected)
    return pool, score_named_pool(loaded, pool)


def best_optimized_pool(
    loaded: LoadedInputs,
    candidates: list[str],
    k: int,
    max_exact_combinations: int,
    forced_champions: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], float, str]:
    if len(forced_champions) > k:
        raise ValueError(
            f"Pool size {k} is smaller than forced champion count {len(forced_champions)}"
        )
    missing_forced = [champion for champion in forced_champions if champion not in candidates]
    if missing_forced:
        raise ValueError(
            "Forced champion(s) are not in the eligible candidate set: "
            + ", ".join(missing_forced)
        )

    remaining_count = len(candidates) - len(forced_champions)
    choose_count = k - len(forced_champions)
    combination_count = comb(remaining_count, choose_count)
    if combination_count <= max_exact_combinations:
        pool, score = exact_optimized_pool(loaded, candidates, k, forced_champions)
        return pool, score, "exact_bruteforce"

    pool, score = greedy_optimized_pool(loaded, candidates, k, forced_champions)
    return pool, score, "greedy_forward"


def exact_optimized_pool(
    loaded: LoadedInputs,
    candidates: list[str],
    k: int,
    forced_champions: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], float]:
    if not forced_champions:
        ranked = rank_pools(candidates, k, loaded.frequency_df, loaded.matchup_lookup)
        best_pool = tuple(ranked.iloc[0]["pool"])
        return best_pool, float(ranked.iloc[0]["pool_score"])

    remaining_candidates = [candidate for candidate in candidates if candidate not in forced_champions]
    best_pool: tuple[str, ...] | None = None
    best_score = float("-inf")
    for complement in combinations(remaining_candidates, k - len(forced_champions)):
        pool = tuple([*forced_champions, *complement])
        score = score_named_pool(loaded, pool)
        if score > best_score:
            best_pool = pool
            best_score = score
    if best_pool is None:
        raise ValueError("No valid forced pool could be generated")
    return best_pool, best_score


def pool_from_ranked_order(
    forced_champions: tuple[str, ...],
    ranked_champions: Iterable[str],
    k: int,
) -> tuple[str, ...]:
    if len(forced_champions) > k:
        raise ValueError(
            f"Pool size {k} is smaller than forced champion count {len(forced_champions)}"
    )
    selected = list(forced_champions)
    if len(selected) == k:
        return tuple(selected)
    for champion in ranked_champions:
        if champion not in selected:
            selected.append(champion)
        if len(selected) == k:
            break
    if len(selected) < k:
        raise ValueError("Not enough candidate champions to fill the pool")
    return tuple(selected)


def score_named_pool(loaded: LoadedInputs, pool: tuple[str, ...]) -> float:
    return pool_score(pool, loaded.frequency_df, loaded.matchup_lookup)


def generate_recommended_pool(
    loaded: LoadedInputs,
    candidates: list[str],
    pool_size: int,
    max_exact_combinations: int,
    forced_champions: tuple[str, ...] = (),
) -> pd.DataFrame:
    pool, score, strategy = best_optimized_pool(
        loaded,
        candidates,
        pool_size,
        max_exact_combinations,
        forced_champions,
    )
    return pd.DataFrame(
        [
            {
                "patch": loaded.patch_label,
                "pool_size": pool_size,
                "method": MODEL_OPTIMIZED,
                "selected_pool": ", ".join(pool),
                "score": score,
            }
        ]
    )


def generate_baseline_comparison(
    loaded: LoadedInputs,
    candidates: list[str],
    max_pool_size: int,
    max_exact_combinations: int,
    forced_champions: tuple[str, ...] = (),
) -> pd.DataFrame:
    blind_scores = compute_blind_scores(candidates, loaded.frequency_df, loaded.matchup_lookup)
    blind_order = blind_scores["champion"].tolist()
    pickrate_order = list(top_by_summary_value(candidates, loaded.summary_df, "pickrate", len(candidates)))
    max_k = min(max_pool_size, len(candidates))
    rows: list[dict[str, object]] = []

    for k in range(1, max_k + 1):
        if k < len(forced_champions):
            continue
        optimized_pool, optimized_score, strategy = best_optimized_pool(
            loaded,
            candidates,
            k,
            max_exact_combinations,
            forced_champions,
        )
        winrate_pool = pool_from_ranked_order(
            forced_champions,
            top_by_summary_value(candidates, loaded.summary_df, "winrate", len(candidates)),
            k,
        )
        blindscore_pool = pool_from_ranked_order(forced_champions, blind_order, k)
        pickrate_pool = pool_from_ranked_order(forced_champions, pickrate_order, k)

        rows.extend(
            [
                {
                    "patch": loaded.patch_label,
                    "pool_size": k,
                    "method": MODEL_OPTIMIZED,
                    "selected_pool": ", ".join(optimized_pool),
                    "score": optimized_score,
                },
                {
                    "patch": loaded.patch_label,
                    "pool_size": k,
                    "method": MODEL_WINRATE,
                    "selected_pool": ", ".join(winrate_pool),
                    "score": score_named_pool(loaded, winrate_pool),
                },
                {
                    "patch": loaded.patch_label,
                    "pool_size": k,
                    "method": MODEL_BLINDSCORE,
                    "selected_pool": ", ".join(blindscore_pool),
                    "score": score_named_pool(loaded, blindscore_pool),
                },
                {
                    "patch": loaded.patch_label,
                    "pool_size": k,
                    "method": MODEL_PICKRATE,
                    "selected_pool": ", ".join(pickrate_pool),
                    "score": score_named_pool(loaded, pickrate_pool),
                },
            ]
        )

    return pd.DataFrame(rows)


def plot_baseline_comparison(
    df: pd.DataFrame,
    path: Path,
    y_min: float | None = None,
    y_max: float | None = None,
    auto_zoom: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, model_df in df.groupby("method"):
        ax.plot(model_df["pool_size"], model_df["score"], marker="o", label=method_label(method))
    ax.set_xlabel("Poolstorlek k")
    ax.set_ylabel("Poolscore")
    ax.set_title("Poolscore vid olika poolstorlekar")
    ax.legend(title="Metod")
    style_report_axes(ax)
    apply_score_ylim(ax, df["score"], y_min=y_min, y_max=y_max, auto_zoom=auto_zoom)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_baseline_comparison_versions(df: pd.DataFrame, path: Path) -> None:
    plot_baseline_comparison(df, path)
    fullscale_path, zoomed_path = comparison_plot_paths(path)
    plot_baseline_comparison(df, fullscale_path)
    plot_baseline_comparison(df, zoomed_path, auto_zoom=True)


def generate_marginal_utility(baseline_df: pd.DataFrame) -> pd.DataFrame:
    optimized = (
        baseline_df[baseline_df["method"] == MODEL_OPTIMIZED]
        .sort_values("pool_size")
        .reset_index(drop=True)
    )
    rows = []
    previous_score = None
    for row in optimized.itertuples(index=False):
        delta = pd.NA if previous_score is None else float(row.score) - previous_score
        rows.append(
            {
                "patch": row.patch,
                "pool_size": row.pool_size,
                "optimized_pool": row.selected_pool,
                "score": float(row.score),
                "marginal_gain": delta,
            }
        )
        previous_score = float(row.score)
    return pd.DataFrame(rows)


def plot_marginal_utility(df: pd.DataFrame, path: Path) -> None:
    plot_df = df.dropna(subset=["marginal_gain"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(plot_df["pool_size"], plot_df["marginal_gain"])
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xlabel("Poolstorlek k")
    ax.set_ylabel("Ökning i poolscore")
    ax.set_title("Marginalnytta av att lägga till champions")
    style_report_axes(ax)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def generate_patch_validation(
    data_dir: Path,
    patches: list[str],
    raw_candidates: list[str] | None,
    candidates_file: str | None,
    lowest_pickrate: float | None,
    pool_size: int,
    max_exact_combinations: int,
    raw_forced_champions: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for train_patch, test_patch in zip(patches, patches[1:]):
        train = load_patch_data(train_patch, data_dir)
        test = load_patch_data(test_patch, data_dir)
        train_candidates, _ = select_candidates(
            train,
            raw_candidates,
            candidates_file,
            lowest_pickrate,
        )

        test_available = set(test.matchup_df["champion_i"].unique())
        candidates = [candidate for candidate in train_candidates if candidate in test_available]
        if pool_size > len(candidates):
            raise ValueError(
                f"Pool size {pool_size} exceeds common candidate count for {train_patch}->{test_patch}"
            )
        forced_champions = tuple(resolve_forced_champions(raw_forced_champions, candidates))
        if len(forced_champions) > pool_size:
            raise ValueError(
                f"Pool size {pool_size} is smaller than forced champion count {len(forced_champions)}"
            )

        blind_scores = compute_blind_scores(candidates, train.frequency_df, train.matchup_lookup)
        blind_order = blind_scores["champion"].tolist()
        pickrate_order = top_by_summary_value(candidates, train.summary_df, "pickrate", len(candidates))

        method_pools = {
            MODEL_OPTIMIZED: best_optimized_pool(
                train,
                candidates,
                pool_size,
                max_exact_combinations,
                forced_champions,
            )[0],
            MODEL_WINRATE: pool_from_ranked_order(
                forced_champions,
                top_by_summary_value(candidates, train.summary_df, "winrate", len(candidates)),
                pool_size,
            ),
            MODEL_BLINDSCORE: pool_from_ranked_order(forced_champions, blind_order, pool_size),
            MODEL_PICKRATE: pool_from_ranked_order(forced_champions, pickrate_order, pool_size),
        }

        for model_name, selected_pool in method_pools.items():
            rows.append(
                {
                    "train_patch": train_patch,
                    "test_patch": test_patch,
                    "pool_size": pool_size,
                    "forced_champions": ", ".join(forced_champions),
                    "method": model_name,
                    "selected_pool": ", ".join(selected_pool),
                    "train_score": score_named_pool(train, selected_pool),
                    "test_score": score_named_pool(test, selected_pool),
                }
            )

    return pd.DataFrame(rows)


def add_bar_labels(ax, fmt: str = "{:.3f}") -> None:
    for container in ax.containers:
        labels = [fmt.format(value) if pd.notna(value) else "" for value in container.datavalues]
        ax.bar_label(container, labels=labels, fontsize=8, padding=3)


def patch_transition_label(df: pd.DataFrame) -> pd.Series:
    return df["train_patch"] + " → " + df["test_patch"]


def plot_patch_validation_absolute(
    df: pd.DataFrame,
    path: Path,
    y_min: float | None = None,
    y_max: float | None = None,
    auto_zoom: bool = False,
) -> None:
    df = df.copy()
    df["transition"] = patch_transition_label(df)
    df["method_label"] = df["method"].map(method_label)
    pivot = df.pivot(index="transition", columns="method_label", values="test_score")
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_xlabel("Patchövergång")
    ax.set_ylabel("Testscore")
    ax.set_title("Poolscore på efterföljande patch")
    ax.legend(title="Metod")
    add_bar_labels(ax)
    style_report_axes(ax)
    apply_score_ylim(ax, df["test_score"], y_min=y_min, y_max=y_max, auto_zoom=auto_zoom)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_patch_validation_absolute_versions(df: pd.DataFrame, path: Path) -> None:
    plot_patch_validation_absolute(df, path)
    fullscale_path, zoomed_path = comparison_plot_paths(path)
    plot_patch_validation_absolute(df, fullscale_path)
    plot_patch_validation_absolute(df, zoomed_path, auto_zoom=True)


def generate_patch_validation_delta(validation_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (train_patch, test_patch), group in validation_df.groupby(["train_patch", "test_patch"]):
        baseline_rows = group[group["method"] == MODEL_WINRATE]
        if baseline_rows.empty:
            continue
        baseline_score = float(baseline_rows.iloc[0]["test_score"])
        for row in group.itertuples(index=False):
            if row.method == MODEL_WINRATE:
                continue
            rows.append(
                {
                    "train_patch": train_patch,
                    "test_patch": test_patch,
                    "transition": transition_label(train_patch, test_patch),
                    "pool_size": row.pool_size,
                    "method": row.method,
                    "selected_pool": row.selected_pool,
                    "test_score": float(row.test_score),
                    "baseline_highest_winrate_test_score": baseline_score,
                    "delta_vs_baseline_highest_winrate": float(row.test_score) - baseline_score,
                }
            )
    return pd.DataFrame(rows)


def plot_patch_validation_delta(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    pivot = df.pivot(
        index="transition",
        columns="method",
        values="delta_vs_baseline_highest_winrate",
    )
    ordered_columns = [column for column in [MODEL_OPTIMIZED, MODEL_BLINDSCORE, MODEL_PICKRATE] if column in pivot.columns]
    pivot = pivot[ordered_columns]
    pivot = pivot.rename(columns=method_label)
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Patchövergång")
    ax.set_ylabel("Skillnad i testscore mot winrate-baseline")
    ax.set_title("Skillnad mot winrate-baseline på efterföljande patch")
    ax.legend(title="Metod")
    add_bar_labels(ax, fmt="{:.4f}")
    style_report_axes(ax)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def forced_method_pools(
    loaded: LoadedInputs,
    candidates: list[str],
    forced_champions: tuple[str, ...],
    pool_size: int,
    max_exact_combinations: int,
) -> dict[str, tuple[str, ...]]:
    blind_scores = compute_blind_scores(candidates, loaded.frequency_df, loaded.matchup_lookup)
    blind_order = blind_scores["champion"].tolist()
    winrate_order = top_by_summary_value(candidates, loaded.summary_df, "winrate", len(candidates))
    pickrate_order = top_by_summary_value(candidates, loaded.summary_df, "pickrate", len(candidates))
    return {
        MODEL_OPTIMIZED_FORCED: best_optimized_pool(
            loaded,
            candidates,
            pool_size,
            max_exact_combinations,
            forced_champions,
        )[0],
        MODEL_WINRATE_FORCED: pool_from_ranked_order(forced_champions, winrate_order, pool_size),
        MODEL_BLINDSCORE_FORCED: pool_from_ranked_order(forced_champions, blind_order, pool_size),
        MODEL_PICKRATE_FORCED: pool_from_ranked_order(forced_champions, pickrate_order, pool_size),
    }


def generate_forced_comparison(
    loaded: LoadedInputs,
    candidates: list[str],
    forced_champions: tuple[str, ...],
    pool_size: int,
    max_exact_combinations: int,
) -> pd.DataFrame:
    rows = []
    for model_name, selected_pool in forced_method_pools(
        loaded,
        candidates,
        forced_champions,
        pool_size,
        max_exact_combinations,
    ).items():
        rows.append(
            {
                "patch": loaded.patch_label,
                "pool_size": pool_size,
                "forced_champions": ", ".join(forced_champions),
                "model_name": model_name,
                "selected_pool": ", ".join(selected_pool),
                "score": score_named_pool(loaded, selected_pool),
            }
        )
    return pd.DataFrame(rows)


def generate_forced_by_pool_size(
    loaded: LoadedInputs,
    candidates: list[str],
    forced_champions: tuple[str, ...],
    max_pool_size: int,
    max_exact_combinations: int,
) -> pd.DataFrame:
    rows = []
    for pool_size in range(1, min(max_pool_size, len(candidates)) + 1):
        if pool_size < len(forced_champions):
            continue
        comparison_df = generate_forced_comparison(
            loaded,
            candidates,
            forced_champions,
            pool_size,
            max_exact_combinations,
        )
        rows.extend(comparison_df.to_dict(orient="records"))
    return pd.DataFrame(rows)


def plot_forced_comparison(
    df: pd.DataFrame,
    path: Path,
    y_min: float | None = None,
    y_max: float | None = None,
    auto_zoom: bool = False,
) -> None:
    plot_df = df.copy()
    plot_df["method_label"] = plot_df["model_name"].map(method_label)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(plot_df["method_label"], plot_df["score"])
    ax.set_xlabel("model_name")
    ax.set_ylabel("Poolscore")
    ax.set_title("Forced champion complement comparison")
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    style_report_axes(ax)
    apply_score_ylim(ax, plot_df["score"], y_min=y_min, y_max=y_max, auto_zoom=auto_zoom)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_forced_comparison_versions(df: pd.DataFrame, path: Path) -> None:
    plot_forced_comparison(df, path)
    fullscale_path, zoomed_path = comparison_plot_paths(path)
    plot_forced_comparison(df, fullscale_path)
    plot_forced_comparison(df, zoomed_path, auto_zoom=True)


def plot_forced_by_pool_size(
    df: pd.DataFrame,
    path: Path,
    y_min: float | None = None,
    y_max: float | None = None,
    auto_zoom: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, model_df in df.groupby("model_name"):
        ax.plot(
            model_df["pool_size"],
            model_df["score"],
            marker="o",
            label=method_label(model_name),
        )
    ax.set_xlabel("Poolstorlek k")
    ax.set_ylabel("Poolscore")
    ax.set_title("Forced champion complement comparison by pool size")
    ax.legend(title="Metod")
    style_report_axes(ax)
    apply_score_ylim(ax, df["score"], y_min=y_min, y_max=y_max, auto_zoom=auto_zoom)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_forced_by_pool_size_versions(df: pd.DataFrame, path: Path) -> None:
    plot_forced_by_pool_size(df, path)
    fullscale_path, zoomed_path = comparison_plot_paths(path)
    plot_forced_by_pool_size(df, fullscale_path)
    plot_forced_by_pool_size(df, zoomed_path, auto_zoom=True)


def generate_forced_champion_batch(
    loaded: LoadedInputs,
    candidates: list[str],
    forced_champions: list[str],
    pool_size: int,
    max_exact_combinations: int,
) -> pd.DataFrame:
    rows = []
    if pool_size < 1:
        raise ValueError("Pool size must be positive for forced champion batch evaluation")
    for forced_champion in forced_champions:
        comparison_df = generate_forced_comparison(
            loaded,
            candidates,
            (forced_champion,),
            pool_size,
            max_exact_combinations,
        )
        comparison_df = comparison_df.rename(columns={"forced_champions": "forced_champion"})
        rows.extend(comparison_df.to_dict(orient="records"))
    return pd.DataFrame(rows)


def summarize_forced_champion_batch(batch_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    average_score_df = (
        batch_df.groupby("model_name", as_index=False)["score"]
        .mean()
        .rename(columns={"score": "average_score"})
        .sort_values("average_score", ascending=False)
    )

    baseline_df = batch_df[batch_df["model_name"] == MODEL_WINRATE_FORCED][
        ["forced_champion", "score"]
    ].rename(columns={"score": "winrate_baseline_score"})
    delta_df = batch_df.merge(baseline_df, on="forced_champion", how="left")
    delta_df["delta_vs_winrate_baseline"] = delta_df["score"] - delta_df["winrate_baseline_score"]

    average_delta_df = (
        delta_df.groupby("model_name", as_index=False)["delta_vs_winrate_baseline"]
        .mean()
        .rename(columns={"delta_vs_winrate_baseline": "average_delta_vs_winrate_baseline"})
        .sort_values("average_delta_vs_winrate_baseline", ascending=False)
    )

    optimized_scores = batch_df[batch_df["model_name"] == MODEL_OPTIMIZED_FORCED][
        ["forced_champion", "score"]
    ].rename(columns={"score": "optimized_score"})
    beat_rows = []
    for baseline_model in [MODEL_WINRATE_FORCED, MODEL_BLINDSCORE_FORCED, MODEL_PICKRATE_FORCED]:
        comparison = batch_df[batch_df["model_name"] == baseline_model][
            ["forced_champion", "score"]
        ].merge(optimized_scores, on="forced_champion", how="left")
        beat_rows.append(
            {
                "baseline_model_name": baseline_model,
                "optimized_win_count": int((comparison["optimized_score"] > comparison["score"]).sum()),
                "comparison_count": int(len(comparison)),
            }
        )
    beat_count_df = pd.DataFrame(beat_rows)
    return average_score_df, average_delta_df, beat_count_df, delta_df


def plot_forced_batch_average_score(df: pd.DataFrame, path: Path) -> None:
    plot_df = df.copy()
    plot_df["method_label"] = plot_df["model_name"].map(method_label)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(plot_df["method_label"], plot_df["average_score"])
    ax.set_xlabel("Metod")
    ax.set_ylabel("Genomsnittlig poolscore")
    ax.set_title("Genomsnittlig score per forced-champion-metod")
    add_bar_labels(ax)
    style_report_axes(ax)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_forced_batch_average_delta(df: pd.DataFrame, path: Path) -> None:
    plot_df = df.copy()
    plot_df["method_label"] = plot_df["model_name"].map(method_label)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(plot_df["method_label"], plot_df["average_delta_vs_winrate_baseline"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Metod")
    ax.set_ylabel("Genomsnittlig skillnad mot winrate-baseline")
    ax.set_title("Genomsnittlig skillnad mot winrate-baseline")
    add_bar_labels(ax, fmt="{:.4f}")
    style_report_axes(ax)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_forced_batch_delta_by_champion(delta_df: pd.DataFrame, path: Path) -> None:
    plot_df = delta_df[delta_df["model_name"] == MODEL_OPTIMIZED_FORCED].copy()
    plot_df = plot_df.sort_values("delta_vs_winrate_baseline", ascending=False)
    fig, ax = plt.subplots(figsize=(max(9, len(plot_df) * 0.6), 5))
    bars = ax.bar(plot_df["forced_champion"], plot_df["delta_vs_winrate_baseline"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Forced champion")
    ax.set_ylabel("Skillnad mot winrate-baseline")
    ax.set_title("Optimerad modell: delta per forced champion")
    add_bar_labels(ax, fmt="{:.4f}")
    style_report_axes(ax)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def generate_matchup_coverage(
    loaded: LoadedInputs,
    candidates: list[str],
    pool_size: int,
    top_enemies: int,
    max_exact_combinations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    optimized_pool, _, _ = best_optimized_pool(
        loaded,
        candidates,
        pool_size,
        max_exact_combinations,
    )
    coverage_df = build_counterpick_table(optimized_pool, loaded.frequency_df, loaded.matchup_lookup)
    coverage_df = coverage_df.head(top_enemies).rename(
        columns={
            "recommended_pick": "best_pool_champion_against_enemy",
            "matchup_value": "best_matchup_value",
        }
    )
    coverage_df.insert(0, "pool_size", pool_size)
    coverage_df.insert(0, "patch", loaded.patch_label)
    coverage_df = coverage_df[
        [
            "patch",
            "pool_size",
            "enemy_champion",
            "enemy_frequency",
            "best_pool_champion_against_enemy",
            "best_matchup_value",
        ]
    ]

    heatmap_rows = []
    for champion in optimized_pool:
        row = {"pool_champion": champion}
        for enemy in coverage_df["enemy_champion"]:
            row[enemy] = loaded.matchup_lookup.get((champion, enemy), pd.NA)
        heatmap_rows.append(row)
    heatmap_df = pd.DataFrame(heatmap_rows)
    return coverage_df, heatmap_df


def plot_matchup_heatmap(heatmap_df: pd.DataFrame, path: Path) -> None:
    plot_df = heatmap_df.iloc[:, :13]
    values = plot_df.drop(columns=["pool_champion"]).astype(float)
    fig, ax = plt.subplots(figsize=(max(8, values.shape[1] * 0.6), max(3, values.shape[0] * 0.65)))
    image = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=0.6)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Matchup-winrate")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["pool_champion"])
    ax.set_xticks(range(len(values.columns)))
    ax.set_xticklabels(values.columns, rotation=45, ha="right")
    ax.set_title("Matchup-täckning för optimerad pool")
    style_report_axes(ax, use_y_grid=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def generate_matchup_games_histogram(
    loaded: LoadedInputs,
    path: Path,
    log_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    games_df = loaded.matchup_df[["champion_i", "champion_j", "games_ij"]].copy()
    real_games_df = games_df[games_df["champion_i"] != games_df["champion_j"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts, bin_edges, _ = ax.hist(real_games_df["games_ij"], bins=30)
    ax.set_xlabel("Antal matcher bakom matchup")
    ax.set_ylabel("Antal matchup-rader")
    ax.set_title("Fördelning av datamängd per matchup")
    style_report_axes(ax)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(real_games_df["games_ij"], bins=30)
    ax.set_xscale("log")
    ax.set_xlabel("Antal matcher bakom matchup")
    ax.set_ylabel("Antal matchup-rader")
    ax.set_title("Fördelning av datamängd per matchup")
    style_report_axes(ax)
    plt.tight_layout()
    plt.savefig(log_path, dpi=300)
    plt.close()

    histogram_df = pd.DataFrame(
        {
            "bin_start": bin_edges[:-1],
            "bin_end": bin_edges[1:],
            "count": counts.astype(int),
        }
    )
    return real_games_df, histogram_df


def write_run_metadata(path: Path, args: argparse.Namespace, candidate_info: dict[str, int | float | None]) -> None:
    lines = [
        "# Results Run Metadata",
        "",
        f"patch: {args.patch}",
        f"patches: {', '.join(args.patches or []) if args.patches else 'auto-discovered'}",
        f"pool_size: {args.pool_size}",
        f"max_pool_size: {args.max_pool_size}",
        f"max_exact_combinations: {args.max_exact_combinations}",
        f"lowest_pickrate: {args.lowest_pickrate}",
        f"candidates_file: {args.candidates_file}",
        f"force_champion: {', '.join(parse_forced_champions(args)) if parse_forced_champions(args) else ''}",
        f"force_champion_batch: {args.force_champion_batch}",
        f"candidate_file_requested_count: {candidate_info['candidate_file_requested_count']}",
        f"candidate_cli_requested_count: {candidate_info['candidate_cli_requested_count']}",
        f"candidate_requested_count: {candidate_info['candidate_requested_count']}",
        f"candidate_matched_count: {candidate_info['candidate_matched_count']}",
        f"candidate_missing_count: {candidate_info['candidate_missing_count']}",
        f"missing_candidates: {candidate_info['missing_candidates']}",
        f"candidate_count_before_filtering: {candidate_info['candidate_count_before_filtering']}",
        f"candidate_count_after_filtering: {candidate_info['candidate_count_after_filtering']}",
        f"candidates_removed_by_pickrate: {candidate_info['candidates_removed_by_pickrate']}",
        "",
        "TODO: If shrinkage is added later, extend the diagnostic with shrinkage adjustments.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_warnings(path: Path, warnings: list[str]) -> None:
    if not warnings:
        return
    path.write_text("\n".join(warnings), encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = resolve_data_dir(str(SRC_DIR / "main.py"))
    primary_patch, patches = resolve_run_patches(args, data_dir)
    args.patch = primary_patch
    args.patches = patches
    raw_forced_champions = parse_forced_champions(args)

    loaded = load_patch_data(primary_patch, data_dir)
    candidates, candidate_info = select_candidates(
        loaded,
        args.candidates,
        args.candidates_file,
        args.lowest_pickrate,
    )
    forced_champions = tuple(resolve_forced_champions(raw_forced_champions, candidates))
    batch_forced_champions = resolve_force_champion_batch(
        args.force_champion_batch,
        candidates,
        loaded.summary_df,
    )
    if args.pool_size > len(candidates):
        raise ValueError(f"--pool-size {args.pool_size} exceeds candidate count {len(candidates)}")
    if len(forced_champions) > args.pool_size:
        raise ValueError(
            f"--pool-size {args.pool_size} is smaller than forced champion count {len(forced_champions)}"
        )

    output_dir = Path(args.output_dir) if args.output_dir else ROOT_DIR / "results" / primary_patch
    run_dir = create_run_dir(output_dir)
    warnings: list[str] = []

    # Recommended pool: the main model output for the chosen k.
    recommended_df = generate_recommended_pool(
        loaded,
        candidates,
        args.pool_size,
        args.max_exact_combinations,
        forced_champions,
    )
    recommended_df.to_csv(run_dir / "recommended_pool.csv", index=False)

    # Baseline comparison: shows whether optimization beats simple ranking heuristics.
    baseline_df = generate_baseline_comparison(
        loaded,
        candidates,
        args.max_pool_size,
        args.max_exact_combinations,
        forced_champions,
    )
    baseline_df.to_csv(run_dir / "baseline_comparison.csv", index=False)
    plot_baseline_comparison_versions(baseline_df, run_dir / "baseline_comparison.png")

    # Marginal utility: shows diminishing returns as the pool grows.
    marginal_df = generate_marginal_utility(baseline_df)
    marginal_df.to_csv(run_dir / "marginal_utility.csv", index=False)
    plot_marginal_utility(marginal_df, run_dir / "marginal_utility.png")

    # Patch validation: select on patch A, evaluate the same pool on patch B.
    try:
        validation_df = generate_patch_validation(
            data_dir,
            patches,
            args.candidates,
            args.candidates_file,
            args.lowest_pickrate,
            args.pool_size,
            args.max_exact_combinations,
            raw_forced_champions,
        )
        validation_df.to_csv(run_dir / "patch_validation.csv", index=False)
        if not validation_df.empty:
            plot_patch_validation_absolute_versions(validation_df, run_dir / "patch_validation_absolute.png")
            validation_delta_df = generate_patch_validation_delta(validation_df)
            validation_delta_df.to_csv(run_dir / "patch_validation_delta.csv", index=False)
            plot_patch_validation_delta(validation_delta_df, run_dir / "patch_validation_delta.png")
        else:
            pd.DataFrame().to_csv(run_dir / "patch_validation_delta.csv", index=False)
    except Exception as exc:
        warning = f"Patch validation skipped: {exc}"
        warnings.append(warning)
        print(f"WARNING: {warning}")
        pd.DataFrame().to_csv(run_dir / "patch_validation.csv", index=False)
        pd.DataFrame().to_csv(run_dir / "patch_validation_delta.csv", index=False)

    if forced_champions:
        # Forced comparison: how well each method fills the remaining slots around the required champions.
        forced_comparison_df = generate_forced_comparison(
            loaded,
            candidates,
            forced_champions,
            args.pool_size,
            args.max_exact_combinations,
        )
        forced_comparison_df.to_csv(run_dir / "forced_champion_comparison.csv", index=False)
        plot_forced_comparison_versions(forced_comparison_df, run_dir / "forced_champion_comparison.png")

        forced_by_k_df = generate_forced_by_pool_size(
            loaded,
            candidates,
            forced_champions,
            args.max_pool_size,
            args.max_exact_combinations,
        )
        forced_by_k_df.to_csv(run_dir / "forced_champion_by_pool_size.csv", index=False)
        plot_forced_by_pool_size_versions(forced_by_k_df, run_dir / "forced_champion_by_pool_size.png")

    if batch_forced_champions:
        # Batch forced testing: repeat forced-complement evaluation for many anchor champions.
        forced_batch_df = generate_forced_champion_batch(
            loaded,
            candidates,
            batch_forced_champions,
            args.pool_size,
            args.max_exact_combinations,
        )
        forced_batch_df.to_csv(run_dir / "forced_champion_batch.csv", index=False)

        average_score_df, average_delta_df, beat_count_df, delta_df = summarize_forced_champion_batch(
            forced_batch_df
        )
        average_score_df.to_csv(run_dir / "forced_champion_batch_average_score.csv", index=False)
        average_delta_df.to_csv(run_dir / "forced_champion_batch_average_delta_vs_winrate.csv", index=False)
        beat_count_df.to_csv(run_dir / "forced_champion_batch_optimized_beat_counts.csv", index=False)
        delta_df.to_csv(run_dir / "forced_champion_batch_delta_by_champion.csv", index=False)

        plot_forced_batch_average_score(
            average_score_df,
            run_dir / "forced_champion_batch_average_score.png",
        )
        plot_forced_batch_average_delta(
            average_delta_df,
            run_dir / "forced_champion_batch_average_delta_vs_winrate.png",
        )
        plot_forced_batch_delta_by_champion(
            delta_df,
            run_dir / "forced_champion_batch_delta_by_champion.png",
        )

    # Matchup coverage: explains which pool member handles common enemy champions.
    coverage_df, heatmap_df = generate_matchup_coverage(
        loaded,
        candidates,
        args.pool_size,
        args.top_enemies,
        args.max_exact_combinations,
    )
    coverage_df.to_csv(run_dir / "matchup_coverage.csv", index=False)
    heatmap_df.to_csv(run_dir / "matchup_heatmap_values.csv", index=False)
    plot_matchup_heatmap(heatmap_df, run_dir / "matchup_heatmap.png")

    # Uncertainty diagnostic placeholder: sample sizes matter even before shrinkage exists.
    games_df, histogram_df = generate_matchup_games_histogram(
        loaded,
        run_dir / "matchup_games_histogram.png",
        run_dir / "matchup_games_histogram_log.png",
    )
    games_df.to_csv(run_dir / "matchup_games.csv", index=False)
    histogram_df.to_csv(run_dir / "matchup_games_histogram.csv", index=False)

    write_run_metadata(run_dir / "run_metadata.md", args, candidate_info)
    write_warnings(run_dir / "warnings.txt", warnings)

    print("Results workflow complete.")
    print(f"Output directory: {run_dir}")
    print(f"Candidates requested: {candidate_info['candidate_requested_count']}")
    print(f"Candidates matched: {candidate_info['candidate_matched_count']}")
    print(f"Candidates missing: {candidate_info['missing_candidates'] or 'none'}")
    print(f"Forced champions: {', '.join(forced_champions) if forced_champions else 'none'}")
    print(f"Forced champion batch: {', '.join(batch_forced_champions) if batch_forced_champions else 'none'}")
    print("Created files:")
    for path in sorted(run_dir.iterdir()):
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
