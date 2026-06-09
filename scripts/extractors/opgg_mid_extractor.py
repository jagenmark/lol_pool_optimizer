from __future__ import annotations

import argparse
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
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://op.gg"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUEST_TIMEOUT = 30
REQUEST_DELAY_SECONDS = 0.8
REQUEST_DELAY_JITTER_SECONDS = 0.4
OUTPUT_DIR = PROJECT_ROOT / "data" / "external"
SUMMARY_OUTPUT = OUTPUT_DIR / "opgg_mid_champion_summary.csv"
MATCHUPS_OUTPUT = OUTPUT_DIR / "opgg_mid_matchups.csv"
RAW_OUTPUT_DIR = OUTPUT_DIR / "raw"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)
ELO_LABELS = {
    "platinum_plus": "plat_plus",
    "emerald_plus": "emerald_plus",
}


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


@dataclass(frozen=True)
class ExtractConfig:
    lane: str
    region: str
    tier_query: str
    elo_label: str

    @property
    def summary_url(self) -> str:
        return (
            f"{BASE_URL}/lol/champions?region={self.region}"
            f"&tier={self.tier_query}&position={self.lane}"
        )


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
    return BeautifulSoup(response.text, "lxml")


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


def extract_patch(text: str) -> str | None:
    match = re.search(r"Patch\s+(\d+\.\d+)", text)
    if match:
        return match.group(1)

    match = re.search(r"Version:\s*(\d+\.\d+)", text)
    if match:
        return match.group(1)

    return None


def slug_from_href(href: str) -> str:
    path = urlparse(href).path.rstrip("/")
    parts = path.split("/")
    try:
        champion_index = parts.index("champions") + 1
    except ValueError as exc:
        raise ValueError(f"Unexpected champion URL: {href}") from exc
    return parts[champion_index]


def get_champion_list(
    session: requests.Session, config: ExtractConfig
) -> tuple[list[ChampionIndexEntry], str]:
    soup = fetch_soup(session, config.summary_url)
    page_text = soup.get_text(" ", strip=True)
    patch = extract_patch(page_text)
    if not patch:
        raise RuntimeError("Could not determine the current patch from the summary page.")

    table = soup.find("table")
    if table is None:
        raise RuntimeError("Could not find the champion summary table on OP.GG.")

    champions: list[ChampionIndexEntry] = []
    for row in table.find_all("tr")[1:]:
        build_link = row.find(
            "a", href=re.compile(rf"/lol/champions/.+/build/{re.escape(config.lane)}")
        )
        if build_link is None:
            continue

        cells = row.find_all("td")
        if len(cells) < 7:
            continue

        champion_name = build_link.get_text(" ", strip=True)
        build_url = urljoin(BASE_URL, build_link["href"])
        slug = slug_from_href(build_link["href"])
        counters_url = (
            f"{BASE_URL}/lol/champions/{slug}/counters/{config.lane}"
            f"?region={config.region}&tier={config.tier_query}&type=ranked"
        )

        champions.append(
            ChampionIndexEntry(
                name=champion_name,
                normalized_name=normalize_champion_name(champion_name),
                slug=slug,
                build_url=build_url,
                counters_url=counters_url,
                winrate=parse_percent(cells[4].get_text(" ", strip=True)),
                pickrate=parse_percent(cells[5].get_text(" ", strip=True)),
                banrate=parse_percent(cells[6].get_text(" ", strip=True)),
            )
        )

    unique_names = {champion.normalized_name for champion in champions}
    if len(unique_names) != len(champions):
        raise RuntimeError("Duplicate champion rows detected while parsing the summary page.")

    return champions, patch


def get_summary_stats(
    session: requests.Session,
    champion: ChampionIndexEntry,
    config: ExtractConfig,
    patch: str,
    extraction_date: str,
) -> dict[str, object]:
    soup = fetch_soup(session, champion.build_url)
    total_games = infer_total_games(soup)

    return {
        "champion_name": champion.name,
        "champion_name_normalized": champion.normalized_name,
        "lane": config.lane,
        "patch": patch,
        "elo": config.elo_label,
        "winrate": champion.winrate,
        "pickrate": champion.pickrate,
        "banrate": champion.banrate,
        "total_games": total_games,
        "source_url": champion.build_url,
        "extraction_date": extraction_date,
    }


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

    # Favor the most common high-sample estimate across the page.
    top_candidates = sorted(candidates, reverse=True)[:5]
    inferred_values = [value for _, value in top_candidates]
    return round(statistics.median(inferred_values))


def get_matchup_stats_for_champion(
    session: requests.Session,
    champion: ChampionIndexEntry,
    config: ExtractConfig,
    patch: str,
    extraction_date: str,
) -> list[dict[str, object]]:
    soup = fetch_soup(session, champion.counters_url)
    search_box = soup.find("input", {"id": "championSearchAndFilter"})
    if search_box is None:
        raise RuntimeError(f"Could not locate the matchup list for {champion.name}.")

    aside = search_box.find_parent("aside")
    if aside is None:
        raise RuntimeError(f"Could not locate the matchup sidebar for {champion.name}.")

    list_items = aside.find("ul").find_all("li")
    opponent_options = soup.select("select#select-champion option")

    if len(list_items) != len(opponent_options):
        raise RuntimeError(
            f"Mismatch between visible matchup rows and opponent options for {champion.name}."
        )

    matchup_rows: list[dict[str, object]] = []
    for list_item, option in zip(list_items, opponent_options):
        opponent_name = extract_matchup_name(list_item)
        opponent_slug = option.get("value", "").strip() or normalize_champion_name(
            opponent_name
        )
        opponent_normalized = normalize_champion_name(opponent_name)
        winrate, games = extract_matchup_metrics(list_item)
        source_url = (
            f"{BASE_URL}/lol/champions/{champion.slug}/counters/{config.lane}"
            f"?region={config.region}&tier={config.tier_query}"
            f"&type=ranked&target_champion={opponent_slug}"
        )

        matchup_rows.append(
            {
                "champion_i": champion.name,
                "champion_i_normalized": champion.normalized_name,
                "champion_j": opponent_name,
                "champion_j_normalized": opponent_normalized,
                "lane": config.lane,
                "patch": patch,
                "elo": config.elo_label,
                "matchup_winrate_i_vs_j": winrate,
                "matchup_games": games,
                "matchup_occurrence_count": games,
                "matchup_occurrence_rate": None,
                "source_url": source_url,
                "extraction_date": extraction_date,
            }
        )

    return matchup_rows


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


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def clean_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: dict[str, dict[str, object]] = {}
    ordered_rows: list[dict[str, object]] = []
    for row in rows:
        key = row["champion_name_normalized"]
        if key not in deduped:
            ordered_rows.append(row)
        deduped[key] = row
    return [deduped[row["champion_name_normalized"]] for row in ordered_rows]


def clean_matchup_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: dict[tuple[object, object], dict[str, object]] = {}
    ordered_keys: list[tuple[object, object]] = []
    for row in rows:
        key = (row["champion_i_normalized"], row["champion_j_normalized"])
        if key not in deduped:
            ordered_keys.append(key)
        deduped[key] = row
    return [deduped[key] for key in ordered_keys]


def validate_summary_rows(rows: list[dict[str, object]]) -> list[str]:
    warnings: list[str] = []
    normalized_names = [row["champion_name_normalized"] for row in rows]
    if len(set(normalized_names)) != len(normalized_names):
        warnings.append("Duplicate champion rows were detected in the summary output.")

    missing_total_games = sum(1 for row in rows if row["total_games"] in (None, ""))
    if missing_total_games:
        warnings.append(
            f"Could not determine total_games for {missing_total_games} champion rows."
        )

    return warnings


def validate_matchup_rows(rows: list[dict[str, object]]) -> list[str]:
    warnings: list[str] = []
    incomplete_rows = [
        row
        for row in rows
        if not row["champion_i"] or not row["champion_j"]
    ]
    if incomplete_rows:
        warnings.append(
            f"{len(incomplete_rows)} matchup rows were missing champion_i or champion_j."
        )

    missing_games = sum(1 for row in rows if row["matchup_games"] in (None, ""))
    if missing_games:
        warnings.append(
            f"Could not determine matchup_games for {missing_games} matchup rows."
        )

    missing_occurrence = sum(
        1
        for row in rows
        if row["matchup_occurrence_count"] in (None, "")
        and row["matchup_occurrence_rate"] in (None, "")
    )
    if missing_occurrence:
        warnings.append(
            f"{missing_occurrence} matchup rows were missing both occurrence fields."
        )

    return warnings


def build_raw_path(base_filename: str, extraction_date: str, config: ExtractConfig) -> Path:
    stem = Path(base_filename).stem
    suffix = Path(base_filename).suffix
    return (
        RAW_OUTPUT_DIR
        / f"{stem}__{config.region}__{config.tier_query}__{extraction_date}{suffix}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MID matchup data from OP.GG.")
    parser.add_argument("--lane", default="mid")
    parser.add_argument("--region", default="global")
    parser.add_argument("--tier", default="emerald_plus")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExtractConfig(
        lane=args.lane,
        region=args.region,
        tier_query=args.tier,
        elo_label=ELO_LABELS.get(args.tier, args.tier),
    )
    session = create_session()
    extraction_date = datetime.now(timezone.utc).date().isoformat()

    print("Loading champion index from OP.GG...")
    champions, patch = get_champion_list(session, config)
    print(
        f"Found {len(champions)} {config.lane}lane champions for "
        f"{config.elo_label} on patch {patch}."
    )

    raw_summary_rows: list[dict[str, object]] = []
    raw_matchup_rows: list[dict[str, object]] = []

    for index, champion in enumerate(champions, start=1):
        print(f"[{index}/{len(champions)}] Collecting {champion.name}...")
        raw_summary_rows.append(
            get_summary_stats(session, champion, config, patch, extraction_date)
        )
        raw_matchup_rows.extend(
            get_matchup_stats_for_champion(
                session, champion, config, patch, extraction_date
            )
        )

    clean_summary = clean_summary_rows(raw_summary_rows)
    clean_matchups = clean_matchup_rows(raw_matchup_rows)

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

    raw_summary_output = build_raw_path(SUMMARY_OUTPUT.name, extraction_date, config)
    raw_matchups_output = build_raw_path(MATCHUPS_OUTPUT.name, extraction_date, config)

    write_csv(raw_summary_output, raw_summary_rows, summary_fieldnames)
    write_csv(raw_matchups_output, raw_matchup_rows, matchup_fieldnames)
    write_csv(SUMMARY_OUTPUT, clean_summary, summary_fieldnames)
    write_csv(MATCHUPS_OUTPUT, clean_matchups, matchup_fieldnames)

    print(
        f"Wrote {len(raw_summary_rows)} raw champion summary rows to "
        f"{raw_summary_output.as_posix()}."
    )
    print(
        f"Wrote {len(raw_matchup_rows)} raw matchup rows to "
        f"{raw_matchups_output.as_posix()}."
    )
    print(
        f"Wrote {len(clean_summary)} cleaned champion summary rows to "
        f"{SUMMARY_OUTPUT.as_posix()}."
    )
    print(
        f"Wrote {len(clean_matchups)} cleaned matchup rows to "
        f"{MATCHUPS_OUTPUT.as_posix()}."
    )

    warnings = validate_summary_rows(clean_summary) + validate_matchup_rows(clean_matchups)
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("Validation checks passed without warnings.")


if __name__ == "__main__":
    main()
