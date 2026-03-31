from __future__ import annotations

import csv
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from playwright.sync_api import Browser, Page, sync_playwright


BASE_URL = "https://lolalytics.com"
TIERLIST_URL = "https://lolalytics.com/lol/tierlist/?lane=middle&tier=platinum_plus"
LANE = "mid"
LANE_QUERY = "middle"
RANK = "plat_plus"
RANK_QUERY = "platinum_plus"
OUTPUT_PATH = Path("data") / "lolalytics_mid_pickrate_mainrate.csv"
REQUEST_DELAY_SECONDS = 1.0
REQUEST_DELAY_JITTER_SECONDS = 0.5
ROW_STABLE_POLLS = 2
MAX_SCROLL_PASSES = 8
EDGE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
)
POPULATION_SCOPE_PICKRATE = "global_platinum_plus_ranked_solo_duo_mid"
POPULATION_SCOPE_DEPTH = "all_regions_all_ranks_last_7_days"


@dataclass(frozen=True)
class ChampionRow:
    champion_name: str
    champion_name_normalized: str
    pick_rate: float
    source_url: str


def normalize_champion_name(name: str) -> str:
    ascii_name = (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    ascii_name = ascii_name.replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "", ascii_name)


def parse_float(text: str) -> float:
    cleaned = (
        text.replace("%", "")
        .replace(",", "")
        .replace("\xa0", "")
        .strip()
    )
    return float(cleaned)


def parse_int(text: str) -> int:
    cleaned = text.replace(",", "").replace("\xa0", "").strip()
    return int(cleaned)


def polite_sleep(multiplier: float = 1.0) -> None:
    delay = REQUEST_DELAY_SECONDS + random.random() * REQUEST_DELAY_JITTER_SECONDS
    time.sleep(delay * multiplier)


def find_browser_executable() -> str | None:
    for candidate in EDGE_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


def build_canonical_champion_url(href: str) -> str:
    url = urljoin(BASE_URL, href)
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query["lane"] = [LANE_QUERY]
    query["tier"] = [RANK_QUERY]
    canonical_query = urlencode(sorted(query.items()), doseq=True)
    return urlunparse(parsed._replace(query=canonical_query))


def create_browser() -> Browser:
    executable_path = find_browser_executable()
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(
        headless=True,
        executable_path=executable_path,
    )
    browser._codex_playwright = playwright  # type: ignore[attr-defined]
    return browser


def close_browser(browser: Browser) -> None:
    playwright = getattr(browser, "_codex_playwright", None)
    browser.close()
    if playwright is not None:
        playwright.stop()


def dismiss_consent_banner(page: Page) -> None:
    for label in ("Accept", "I Agree"):
        button = page.get_by_role("button", name=label)
        if button.count():
            try:
                button.first.click(timeout=1_500)
                page.wait_for_timeout(500)
            except Exception:
                pass


def extract_patch(page: Page) -> str:
    body_text = page.locator("body").inner_text()
    match = re.search(r"Patch\s+(\d+\.\d+)", body_text)
    if not match:
        raise RuntimeError("Could not determine the current patch from the tier list page.")
    return match.group(1)


def scroll_tierlist_until_stable(page: Page) -> None:
    previous_count = -1
    stable_polls = 0

    for _ in range(MAX_SCROLL_PASSES):
        page.wait_for_timeout(1_500)
        current_count = len(extract_tierlist_rows(page))
        if current_count == previous_count:
            stable_polls += 1
        else:
            stable_polls = 0
        if stable_polls >= ROW_STABLE_POLLS:
            return
        previous_count = current_count
        page.mouse.wheel(0, 15_000)
        polite_sleep(0.5)


def extract_tierlist_rows(page: Page) -> list[dict[str, object]]:
    return page.evaluate(
        """
() => Array.from(document.querySelectorAll('div'))
  .filter((el) => {
    const cls = typeof el.className === 'string' ? el.className : '';
    return cls.includes('h-[52px]')
      && cls.includes('justify-between')
      && cls.includes('text-[13px]')
      && !!el.querySelector('a[href*="/build/"]');
  })
  .map((row) => ({
    lines: row.innerText.split('\\n').map((line) => line.trim()).filter(Boolean),
    hrefs: Array.from(row.querySelectorAll('a[href]')).map((a) => a.getAttribute('href')),
  }))
        """
    )


def get_champion_rows(page: Page) -> tuple[list[ChampionRow], str]:
    page.goto(TIERLIST_URL, wait_until="domcontentloaded", timeout=60_000)
    dismiss_consent_banner(page)
    page.wait_for_timeout(4_000)
    patch = extract_patch(page)
    scroll_tierlist_until_stable(page)

    raw_rows = extract_tierlist_rows(page)
    champions: list[ChampionRow] = []
    seen_names: set[str] = set()

    for raw_row in raw_rows:
        lines = raw_row["lines"]
        if len(lines) < 7:
            continue

        champion_name = str(lines[1]).strip()
        normalized_name = normalize_champion_name(champion_name)
        pick_rate = parse_float(str(lines[6]))

        href = next(
            (
                value
                for value in raw_row["hrefs"]
                if value and "/build/" in value
            ),
            None,
        )
        if not href:
            continue

        if normalized_name in seen_names:
            continue
        seen_names.add(normalized_name)

        champions.append(
            ChampionRow(
                champion_name=champion_name,
                champion_name_normalized=normalized_name,
                pick_rate=pick_rate,
                source_url=build_canonical_champion_url(str(href)),
            )
        )

    if not champions:
        raise RuntimeError("No champion rows were extracted from the LoLalytics tier list.")

    return champions, patch


def extract_depth_metrics(page: Page, url: str) -> dict[str, object]:
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    dismiss_consent_banner(page)
    page.locator("text=Best on").first.scroll_into_view_if_needed()
    page.wait_for_timeout(2_500)

    chart_svg = page.locator("svg").filter(
        has_text="Normalised Champion Ranked Player Base"
    ).first
    if chart_svg.count() == 0:
        raise RuntimeError(f"Could not locate the depth chart on {url}")

    circle = chart_svg.locator("circle").first
    if circle.count() == 0:
        raise RuntimeError(f"Could not locate the depth chart point on {url}")

    circle.hover()
    page.wait_for_timeout(500)
    body_text = page.locator("body").inner_text()
    match = re.search(
        (
            r"Classification:\s*(?P<classification>\w+)\s+"
            r"Breadth:\s*(?P<breadth>\d+(?:\.\d+)?)\s+"
            r"Depth:\s*(?P<depth>\d+(?:\.\d+)?)\s+"
            r"Unique Players:\s*(?P<players>[\d \xa0,]+)\s+"
            r"Total Ranked Games:\s*(?P<games>[\d \xa0,]+)\s+"
            r"(?P<scope>7 Days, All Ranks, All Regions)"
        ),
        body_text,
    )
    if not match:
        raise RuntimeError(f"Could not parse the depth tooltip on {url}")

    return {
        "breadth": parse_float(match.group("breadth")),
        "depth": parse_float(match.group("depth")),
        "classification": match.group("classification").lower(),
        "unique_players": parse_int(match.group("players")),
        "total_ranked_games": parse_int(match.group("games")),
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    extraction_date = datetime.now(timezone.utc).date().isoformat()
    browser = create_browser()
    list_page = browser.new_page(viewport={"width": 1600, "height": 2400})
    detail_page = browser.new_page(viewport={"width": 1400, "height": 3000})

    try:
        print("Loading LoLalytics midlane tier list...")
        champions, patch = get_champion_rows(list_page)
        print(f"Found {len(champions)} midlane champions on patch {patch}.")

        rows: list[dict[str, object]] = []
        for index, champion in enumerate(champions, start=1):
            print(
                f"[{index}/{len(champions)}] Extracting {champion.champion_name} "
                f"from {champion.source_url}"
            )
            depth_metrics = extract_depth_metrics(detail_page, champion.source_url)
            polite_sleep()
            rows.append(
                {
                    "champion_name": champion.champion_name,
                    "champion_name_normalized": champion.champion_name_normalized,
                    "lane": LANE,
                    "patch": patch,
                    "rank": RANK,
                    "population_scope_pickrate": POPULATION_SCOPE_PICKRATE,
                    "population_scope_depth": POPULATION_SCOPE_DEPTH,
                    "pick_rate": champion.pick_rate,
                    "breadth": depth_metrics["breadth"],
                    "depth": depth_metrics["depth"],
                    "classification": depth_metrics["classification"],
                    "unique_players": depth_metrics["unique_players"],
                    "total_ranked_games": depth_metrics["total_ranked_games"],
                    "source_url": champion.source_url,
                    "extraction_date": extraction_date,
                }
            )

        unique_names = {row["champion_name_normalized"] for row in rows}
        if len(unique_names) != len(rows):
            raise RuntimeError("Duplicate champion rows were detected in the final output.")

        fieldnames = [
            "champion_name",
            "champion_name_normalized",
            "lane",
            "patch",
            "rank",
            "population_scope_pickrate",
            "population_scope_depth",
            "pick_rate",
            "breadth",
            "depth",
            "classification",
            "unique_players",
            "total_ranked_games",
            "source_url",
            "extraction_date",
        ]
        write_csv(OUTPUT_PATH, rows, fieldnames)
        print(f"Wrote {len(rows)} rows to {OUTPUT_PATH.as_posix()}.")
        print(f"Validation: found {len(champions)} champions, extracted {len(rows)} rows.")
    finally:
        close_browser(browser)


if __name__ == "__main__":
    main()
