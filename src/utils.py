from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


def resolve_data_dir(current_file: str) -> Path:
    return Path(current_file).resolve().parent.parent / "data"


def describe_frequency_status(status: str) -> str:
    descriptions = {
        "present_in_prepared_enemy_frequency_file": "loaded directly from the prepared enemy frequency file",
        "present_in_synthetic_frequency_file": "present in the synthetic fallback frequency file",
    }
    return descriptions.get(status, status)


def canonicalize_champion_name(name: str) -> str:
    return "".join(ch for ch in name.strip().lower() if ch.isalnum())


def normalize_candidate_names(raw_candidates: Iterable[str]) -> List[str]:
    cleaned = []
    seen = set()
    for name in raw_candidates:
        candidate = name.strip()
        if not candidate:
            continue
        if candidate not in seen:
            seen.add(candidate)
            cleaned.append(candidate)
    return cleaned


def parse_candidates_from_args(raw_candidates: list[str] | None) -> List[str]:
    if not raw_candidates:
        return []

    expanded = []
    for item in raw_candidates:
        expanded.extend(part.strip() for part in item.split(","))
    return normalize_candidate_names(expanded)


def format_percentage(value: float) -> str:
    return f"{value:.2%}"


def dataframe_for_console(df: pd.DataFrame, percentage_columns: Iterable[str] | None = None) -> str:
    display_df = df.copy()
    for column in percentage_columns or []:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(format_percentage)
    return display_df.to_string(index=False)
