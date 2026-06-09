from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT / "site"
PATCHES = ("16.05", "16.06", "16.07")


def normalized_icon_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def normalize_rate(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return numeric
    return numeric / 100.0 if numeric.dropna().max() > 1 else numeric


def optional_number(value: object) -> float | int | None:
    if pd.isna(value):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def build_icon_lookup() -> dict[str, str]:
    icon_dir = ROOT / "assets" / "champion_icons"
    return {
        normalized_icon_key(path.stem): path.name
        for path in icon_dir.glob("*.png")
    }


def export_patch(patch: str, icon_lookup: dict[str, str]) -> set[str]:
    patch_dir = ROOT / "data" / patch
    summary = pd.read_csv(patch_dir / "opgg_mid_champion_summary.csv")
    frequencies = pd.read_csv(patch_dir / "enemy_freq_df.csv")
    matchups = pd.read_csv(patch_dir / "opgg_mid_matchups_clean.csv")

    for column in ("winrate", "pickrate", "banrate"):
        if column in summary:
            summary[column] = normalize_rate(summary[column])
    matchups["matchup_winrate_i_vs_j"] = normalize_rate(
        matchups["matchup_winrate_i_vs_j"]
    )

    champions = []
    for row in summary.sort_values("champion_name").itertuples(index=False):
        name = str(row.champion_name)
        champions.append(
            {
                "name": name,
                "icon": icon_lookup.get(normalized_icon_key(name)),
                "winrate": optional_number(getattr(row, "winrate", None)),
                "pickrate": optional_number(getattr(row, "pickrate", None)),
                "banrate": optional_number(getattr(row, "banrate", None)),
                "total_games": optional_number(getattr(row, "total_games", None)),
            }
        )

    payload = {
        "patch": patch,
        "champions": champions,
        "frequencies": [
            {
                "enemy": str(row.champion_j),
                "frequency": float(row.f_j),
                "games": optional_number(getattr(row, "enemy_total_games", None)),
            }
            for row in frequencies.sort_values(
                ["f_j", "champion_j"], ascending=[False, True]
            ).itertuples(index=False)
        ],
        "matchups": [
            {
                "champion": str(row.champion_i),
                "enemy": str(row.champion_j),
                "winrate": float(row.matchup_winrate_i_vs_j),
                "games": optional_number(getattr(row, "matchup_games", None)),
            }
            for row in matchups.itertuples(index=False)
            if pd.notna(row.matchup_winrate_i_vs_j)
        ],
    }

    output = SITE_DIR / "data" / f"{patch}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return {
        champion["icon"]
        for champion in champions
        if champion["icon"] is not None
    }


def main() -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    icon_lookup = build_icon_lookup()
    used_icons: set[str] = set()
    for patch in PATCHES:
        used_icons.update(export_patch(patch, icon_lookup))

    source_icons = ROOT / "assets" / "champion_icons"
    target_icons = SITE_DIR / "assets" / "champion_icons"
    if target_icons.exists():
        shutil.rmtree(target_icons)
    target_icons.mkdir(parents=True)
    for filename in sorted(used_icons):
        shutil.copy2(source_icons / filename, target_icons / filename)
    print(f"Exported {len(PATCHES)} patches to {SITE_DIR}")


if __name__ == "__main__":
    main()
