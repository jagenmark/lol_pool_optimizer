from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ColumnMapping:
    champion: str
    champion_normalized: str
    enemy: str | None
    enemy_normalized: str | None
    overall_winrate: str | None
    pickrate: str | None
    banrate: str | None
    total_games: str | None
    matchup_winrate: str | None
    matchup_games: str | None


@dataclass(frozen=True)
class PatchFiles:
    patch_label: str
    summary_path: Path
    matchup_path: Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SUMMARY_MAPPING = ColumnMapping(
    champion="champion_name",
    champion_normalized="champion_name_normalized",
    enemy=None,
    enemy_normalized=None,
    overall_winrate="winrate",
    pickrate="pickrate",
    banrate="banrate",
    total_games="total_games",
    matchup_winrate=None,
    matchup_games=None,
)

MATCHUP_MAPPING = ColumnMapping(
    champion="champion_i",
    champion_normalized="champion_i_normalized",
    enemy="champion_j",
    enemy_normalized="champion_j_normalized",
    overall_winrate=None,
    pickrate=None,
    banrate=None,
    total_games=None,
    matchup_winrate="matchup_winrate_i_vs_j",
    matchup_games="matchup_games",
)

PATCH_A = PatchFiles(
    patch_label="16.05",
    summary_path=REPO_ROOT / "data" / "raw" / "opgg_mid_champion_summary__baseline_2026-03-13__plat_plus.csv",
    matchup_path=REPO_ROOT / "data" / "raw" / "opgg_mid_matchups__baseline_2026-03-13__plat_plus.csv",
)

PATCH_B = PatchFiles(
    patch_label="16.06",
    summary_path=REPO_ROOT / "data" / "opgg_mid_champion_summary__plat_plus__16.06.csv",
    matchup_path=REPO_ROOT / "data" / "opgg_mid_matchups__plat_plus__16.06.csv",
)

DEFAULT_SHRINKAGE_C_VALUES = (50.0, 100.0, 250.0, 500.0, 1000.0)
