from __future__ import annotations

import csv
import random
import re
import statistics
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://op.gg"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "external"
LANE = "mid"
REGION = "global"
TIERS = (
    ("emerald_plus", "emerald_plus", "Emerald +"),
    ("diamond_plus", "diamond_plus", "Diamond +"),
    ("master_plus", "master_plus", "Master +"),
)
PATCHES = ("16.10", "16.11")
REQUEST_TIMEOUT = 30
REQUEST_DELAY_SECONDS = 0.8
REQUEST_DELAY_JITTER_SECONDS = 0.4
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class ChampionIndexEntry:
    name: str
    normalized_name: str
    slug: str
    build_url: str
    counters_url: str
    winrate: float | None
    pickrate: float | None
    banrate: float | None


def create_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


def polite_sleep() -> None:
    time.sleep(REQUEST_DELAY_SECONDS + random.random() * REQUEST_DELAY_JITTER_SECONDS)


def fetch_soup(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    polite_sleep()
    return BeautifulSoup(response.text, "html.parser")


def normalize_champion_name(name: str) -> str:
    ascii_name = (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    ascii_name = ascii_name.replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "", ascii_name)


def parse_percent(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    return float(match.group(1)) if match else None


def parse_int(text: str) -> int | None:
    match = re.search(r"(\d[\d,]*)", text)
    return int(match.group(1).replace(",", "")) if match else None


def slug_from_href(href: str) -> str:
    path = urlparse(href).path.rstrip("/")
    parts = path.split("/")
    try:
        champion_index = parts.index("champions") + 1
    except ValueError as exc:
        raise ValueError(f"Unexpected champion URL: {href}") from exc
    return parts[champion_index]


def verify_scope(
    soup: BeautifulSoup,
    patch: str,
    tier_display: str,
    context: str,
) -> None:
    text = soup.get_text(" ", strip=True)
    selected_filters = {
        label.get_text(" ", strip=True) for label in soup.select("label.select-label")
    }
    checks = {
        "Middle": "Middle" in text,
        "Global selected": "Global" in selected_filters,
        f"{tier_display} selected": tier_display in selected_filters,
        f"Patch {patch} selected": f"Ver: {patch}" in selected_filters,
    }
    failed = [label for label, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            f"Scope verification failed for {context} ({patch}): missing {', '.join(failed)}"
        )


def get_summary_index(
    session: requests.Session,
    patch: str,
    tier_query: str,
    tier_display: str,
) -> list[ChampionIndexEntry]:
    summary_url = (
        f"{BASE_URL}/lol/champions?region={REGION}"
        f"&tier={tier_query}&position={LANE}&patch={patch}"
    )
    soup = fetch_soup(session, summary_url)
    verify_scope(soup, patch, tier_display, "summary index")

    table = soup.find("table")
    if table is None:
        raise RuntimeError(f"Could not find the summary table for patch {patch}.")

    champions: list[ChampionIndexEntry] = []
    for row in table.find_all("tr")[1:]:
        build_link = row.find("a", href=re.compile(r"/lol/champions/.+/build/mid"))
        if build_link is None:
            continue

        cells = row.find_all("td")
        if len(cells) < 7:
            continue

        champion_name = build_link.get_text(" ", strip=True)
        slug = slug_from_href(build_link["href"])
        champions.append(
            ChampionIndexEntry(
                name=champion_name,
                normalized_name=normalize_champion_name(champion_name),
                slug=slug,
                build_url=(
                    f"{BASE_URL}/lol/champions/{slug}/build/{LANE}"
                    f"?region={REGION}&tier={tier_query}&patch={patch}"
                ),
                counters_url=(
                    f"{BASE_URL}/lol/champions/{slug}/counters/{LANE}"
                    f"?region={REGION}&tier={tier_query}&type=ranked&patch={patch}"
                ),
                winrate=parse_percent(cells[4].get_text(" ", strip=True)),
                pickrate=parse_percent(cells[5].get_text(" ", strip=True)),
                banrate=parse_percent(cells[6].get_text(" ", strip=True)),
            )
        )

    unique_names = {champion.normalized_name for champion in champions}
    if len(unique_names) != len(champions):
        raise RuntimeError(f"Duplicate champion rows detected for patch {patch}.")

    return champions


def infer_total_games(soup: BeautifulSoup) -> int | None:
    candidates: list[tuple[int, int]] = []

    for table in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
        if "Pick rate" not in headers:
            continue

        for row in table.find_all("tr")[1:]:
            row_text = row.get_text(" ", strip=True)
            if "Games" not in row_text:
                continue

            numbers = re.findall(r"\d[\d,]*\.?\d*", row_text)
            if len(numbers) < 2:
                continue

            pick_rate = parse_percent(numbers[0])
            games = parse_int(numbers[1])
            if not pick_rate or not games or pick_rate <= 0:
                continue

            total_games = round(games / (pick_rate / 100.0))
            candidates.append((games, total_games))

    if not candidates:
        return None

    top_candidates = sorted(candidates, reverse=True)[:5]
    inferred_values = [value for _, value in top_candidates]
    return round(statistics.median(inferred_values))


def get_summary_rows(
    session: requests.Session,
    champions: list[ChampionIndexEntry],
    patch: str,
    elo_label: str,
    tier_display: str,
    extraction_date: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for champion in champions:
        soup = fetch_soup(session, champion.build_url)
        verify_scope(soup, patch, tier_display, f"{champion.name} build page")
        rows.append(
            {
                "champion_name": champion.name,
                "champion_name_normalized": champion.normalized_name,
                "lane": LANE,
                "patch": patch,
                "elo": elo_label,
                "winrate": champion.winrate,
                "pickrate": champion.pickrate,
                "banrate": champion.banrate,
                "total_games": infer_total_games(soup),
                "source_url": champion.build_url,
                "extraction_date": extraction_date,
            }
        )
    return rows


def extract_matchup_name(list_item: Tag) -> str:
    for span in list_item.find_all("span"):
        text = span.get_text(" ", strip=True)
        if text and text != "%" and text.lower() != "games":
            return text
    raise RuntimeError("Could not determine the opponent name from a matchup row.")


def extract_matchup_metrics(list_item: Tag) -> tuple[float | None, int | None]:
    winrate_tag = list_item.find("strong")
    games_tag = list_item.find("span", string=re.compile(r"\d"))
    winrate = parse_percent(winrate_tag.get_text(" ", strip=True)) if winrate_tag else None
    games = parse_int(games_tag.get_text(" ", strip=True)) if games_tag else None
    return winrate, games


def get_matchup_rows(
    session: requests.Session,
    champions: list[ChampionIndexEntry],
    patch: str,
    tier_query: str,
    elo_label: str,
    tier_display: str,
    extraction_date: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for champion in champions:
        soup = fetch_soup(session, champion.counters_url)
        verify_scope(soup, patch, tier_display, f"{champion.name} counters page")

        search_box = soup.find("input", {"id": "championSearchAndFilter"})
        if search_box is None:
            raise RuntimeError(f"Could not locate matchup list for {champion.name}.")

        aside = search_box.find_parent("aside")
        if aside is None:
            raise RuntimeError(f"Could not locate matchup sidebar for {champion.name}.")

        list_items = aside.find("ul").find_all("li")
        opponent_options = soup.select("select#select-champion option")
        if len(list_items) != len(opponent_options):
            raise RuntimeError(
                f"Mismatch between matchup rows and options for {champion.name} ({patch})."
            )

        for list_item, option in zip(list_items, opponent_options):
            opponent_name = extract_matchup_name(list_item)
            opponent_slug = option.get("value", "").strip() or normalize_champion_name(
                opponent_name
            )
            opponent_normalized = normalize_champion_name(opponent_name)
            winrate, games = extract_matchup_metrics(list_item)
            rows.append(
                {
                    "champion_i": champion.name,
                    "champion_i_normalized": champion.normalized_name,
                    "champion_j": opponent_name,
                    "champion_j_normalized": opponent_normalized,
                    "lane": LANE,
                    "patch": patch,
                    "elo": elo_label,
                    "matchup_winrate_i_vs_j": winrate,
                    "matchup_games": games,
                    "matchup_occurrence_count": games,
                    "matchup_occurrence_rate": None,
                    "source_url": (
                        f"{BASE_URL}/lol/champions/{champion.slug}/counters/{LANE}"
                        f"?region={REGION}&tier={tier_query}&type=ranked"
                        f"&patch={patch}&target_champion={opponent_slug}"
                    ),
                    "extraction_date": extraction_date,
                }
            )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {path}")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_rows(summary_rows: list[dict[str, object]], matchup_rows: list[dict[str, object]]) -> list[str]:
    warnings: list[str] = []

    normalized_names = [row["champion_name_normalized"] for row in summary_rows]
    if len(set(normalized_names)) != len(normalized_names):
        warnings.append("Duplicate champion rows detected.")

    missing_total_games = sum(1 for row in summary_rows if row["total_games"] in (None, ""))
    if missing_total_games:
        warnings.append(f"{missing_total_games} summary rows are missing total_games.")

    missing_matchup_games = sum(1 for row in matchup_rows if row["matchup_games"] in (None, ""))
    if missing_matchup_games:
        warnings.append(f"{missing_matchup_games} matchup rows are missing matchup_games.")

    return warnings


def main() -> None:
    session = create_session()
    extraction_date = datetime.now(timezone.utc).date().isoformat()

    summary_fieldnames = [
        "champion_name",
        "champion_name_normalized",
        "lane",
        "patch",
        "elo",
        "winrate",
        "pickrate",
        "banrate",
        "total_games",
        "source_url",
        "extraction_date",
    ]
    matchup_fieldnames = [
        "champion_i",
        "champion_i_normalized",
        "champion_j",
        "champion_j_normalized",
        "lane",
        "patch",
        "elo",
        "matchup_winrate_i_vs_j",
        "matchup_games",
        "matchup_occurrence_count",
        "matchup_occurrence_rate",
        "source_url",
        "extraction_date",
    ]

    for tier_query, elo_label, tier_display in TIERS:
        for patch in PATCHES:
            print(
                f"Collecting OP.GG validation data for patch {patch}, "
                f"tier {tier_display}..."
            )
            try:
                champions = get_summary_index(
                    session,
                    patch,
                    tier_query,
                    tier_display,
                )
            except RuntimeError as exc:
                print(f"- Skipped patch {patch}, tier {tier_display}: {exc}")
                continue

            print(f"- Found {len(champions)} champions.")

            summary_rows = get_summary_rows(
                session,
                champions,
                patch,
                elo_label,
                tier_display,
                extraction_date,
            )
            matchup_rows = get_matchup_rows(
                session,
                champions,
                patch,
                tier_query,
                elo_label,
                tier_display,
                extraction_date,
            )

            summary_path = (
                OUTPUT_DIR
                / f"opgg_mid_champion_summary__{elo_label}__{patch}.csv"
            )
            matchup_path = (
                OUTPUT_DIR / f"opgg_mid_matchups__{elo_label}__{patch}.csv"
            )

            write_csv(summary_path, summary_rows, summary_fieldnames)
            write_csv(matchup_path, matchup_rows, matchup_fieldnames)

            print(
                f"- Wrote {len(summary_rows)} summary rows to "
                f"{summary_path.as_posix()}"
            )
            print(
                f"- Wrote {len(matchup_rows)} matchup rows to "
                f"{matchup_path.as_posix()}"
            )

            warnings = validate_rows(summary_rows, matchup_rows)
            if warnings:
                print("- Warnings:")
                for warning in warnings:
                    print(f"  - {warning}")
            else:
                print("- Validation checks passed.")


if __name__ == "__main__":
    main()
