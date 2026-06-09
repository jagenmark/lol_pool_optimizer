from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matchup_estimator import DEFAULT_EB_ALPHA
from scope_stability import run_scope_stability_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run cumulative patch/rank EB robustness and stability diagnostics."
        )
    )
    parser.add_argument(
        "--prepared-data-dir",
        default=str(PROJECT_ROOT / "data"),
        help="Prepared patch directory root.",
    )
    parser.add_argument(
        "--aggregate-data-dir",
        default=str(PROJECT_ROOT / "data" / "external"),
        help="Aggregate patch/rank file root.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "scope_stability"),
        help="Output directory for requested stability artifacts.",
    )
    parser.add_argument("--pool-size", type=int, default=3)
    parser.add_argument("--top-pools", type=int, default=100)
    parser.add_argument("--eb-alpha", type=float, default=DEFAULT_EB_ALPHA)
    parser.add_argument("--eb-mu", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.pool_size <= 0:
        raise ValueError("--pool-size must be positive")
    if args.top_pools < 100:
        raise ValueError("--top-pools must be at least 100")
    if args.eb_alpha < 0:
        raise ValueError("--eb-alpha must be non-negative")

    artifacts = run_scope_stability_workflow(
        prepared_data_dir=Path(args.prepared_data_dir),
        aggregate_data_dir=Path(args.aggregate_data_dir),
        output_dir=Path(args.output_dir),
        pool_size=args.pool_size,
        top_pool_count=args.top_pools,
        eb_alpha=args.eb_alpha,
        eb_mu=args.eb_mu,
    )
    print("Scope stability outputs:")
    for path in artifacts.__dict__.values():
        print(Path(path).resolve())


if __name__ == "__main__":
    main()
