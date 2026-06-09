from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://lolalytics.com"
LANE = "mid"
LANE_QUERY = "middle"
REQUEST_TIMEOUT = 30
REQUEST_DELAY_SECONDS = 0.8
REQUEST_DELAY_JITTER_SECONDS = 0.4
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)
RANK_LABELS = {
    "platinum_plus": "plat_plus",
    "emerald_plus": "emerald_plus",
}
POPULATION_SCOPE_DEPTH = "all_regions_all_ranks_last_7_days"
REFERENCE_CHAMPION_SLUG = "ahri"


@dataclass(frozen=True)
class ChampionEntry:
    slug: str
    source_url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", required=True, help="Historical patch like 16.5")
    parser.add_argument("--tier", default="platinum_plus", help="LoLalytics tier query")
    parser.add_argument("--output", required=True, help="Output CSV path")
    return parser.parse_args()


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


def polite_sleep(multiplier: float = 1.0) -> None:
    delay = REQUEST_DELAY_SECONDS + random.random() * REQUEST_DELAY_JITTER_SECONDS
    time.sleep(delay * multiplier)


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
    cleaned = text.replace("%", "").replace(",", "").replace("\xa0", "").strip()
    return float(cleaned)


def build_champion_url(slug: str, patch: str, tier: str) -> str:
    return (
        f"{BASE_URL}/lol/{slug}/build/?lane={LANE_QUERY}"
        f"&patch={patch}&tier={tier}"
    )


def population_scope_pickrate(tier: str) -> str:
    if tier == "platinum_plus":
        return "global_platinum_plus_ranked_solo_duo_mid"
    if tier == "emerald_plus":
        return "global_emerald_plus_ranked_solo_duo_mid"
    return f"global_{tier}_ranked_solo_duo_mid"


def fetch_html(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    polite_sleep()
    return response.text


def extract_qwik_objects(html: str) -> list[Any]:
    match = re.search(r'<script type="qwik/json">(.*?)</script>', html, re.S)
    if not match:
        raise RuntimeError("Could not locate LoLalytics Qwik serialized state.")
    return json.loads(match.group(1))["objs"]


def maybe_resolve_token(objs: list[Any], value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if not re.fullmatch(r"[0-9a-z]+", value):
        return value
    index = int(value, 36)
    if index < 0 or index >= len(objs):
        return value
    return objs[index]


def extract_patch_from_text(page_text: str) -> str:
    match = re.search(r"Patch\s+(\d+\.\d+)", page_text)
    if not match:
        raise RuntimeError("Could not determine the patch from the LoLalytics page.")
    return match.group(1)


def extract_champion_name(soup: BeautifulSoup) -> str:
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    match = re.match(r"(.+?)\s+Build,", title)
    if match:
        return match.group(1).strip()
    heading = soup.find("h1")
    if heading:
        heading_text = heading.get_text(" ", strip=True).strip()
        if heading_text:
            return heading_text
    raise RuntimeError("Could not determine champion name from the LoLalytics page.")


def extract_pick_rate(page_text: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*Pick Rate", page_text)
    if not match:
        raise RuntimeError("Could not parse Pick Rate from the LoLalytics page text.")
    return parse_float(match.group(1))


def extract_depth_payload(objs: list[Any]) -> dict[str, object]:
    payload_dict = next(
        (
            obj
            for obj in objs
            if isinstance(obj, dict)
            and {"topList", "depth", "topStats", "stats", "time", "objective"} <= set(obj)
        ),
        None,
    )
    if payload_dict is None:
        raise RuntimeError("Could not locate the breadth/depth payload in serialized state.")

    depth_list = maybe_resolve_token(objs, payload_dict["depth"])
    if not isinstance(depth_list, list) or len(depth_list) < 5:
        raise RuntimeError("Could not resolve the breadth/depth payload list.")

    resolved = [maybe_resolve_token(objs, item) for item in depth_list[:5]]
    total_ranked_games, unique_players, breadth, depth, classification = resolved
    return {
        "total_ranked_games": int(total_ranked_games),
        "unique_players": int(unique_players),
        "breadth": parse_float(str(breadth)),
        "depth": parse_float(str(depth)),
        "classification": str(classification).lower(),
    }


def get_champion_universe(session: requests.Session, patch: str, tier: str) -> list[ChampionEntry]:
    reference_url = build_champion_url(REFERENCE_CHAMPION_SLUG, patch, tier)
    html = fetch_html(session, reference_url)
    objs = extract_qwik_objects(html)

    slug_map = next(
        (
            obj
            for obj in objs
            if isinstance(obj, dict)
            and "aatrox" in obj
            and "ahri" in obj
            and "zed" in obj
            and "yuumi" in obj
        ),
        None,
    )
    if slug_map is None:
        raise RuntimeError("Could not locate the full LoLalytics champion slug map.")

    return [
        ChampionEntry(
            slug=slug,
            source_url=build_champion_url(slug, patch, tier),
        )
        for slug in sorted(slug_map)
    ]


def collect_rows(session: requests.Session, patch: str, tier: str) -> list[dict[str, object]]:
    extraction_date = datetime.now(timezone.utc).date().isoformat()
    rank_label = RANK_LABELS.get(tier, tier)
    champions = get_champion_universe(session, patch, tier)
    print(
        f"Found {len(champions)} champions in the LoLalytics serialized champion universe "
        f"for patch {patch}."
    )

    rows: list[dict[str, object]] = []
    seen_names: set[str] = set()

    for index, champion in enumerate(champions, start=1):
        print(f"[{index}/{len(champions)}] Collecting {champion.slug}...")
        html = fetch_html(session, champion.source_url)
        soup = BeautifulSoup(html, "lxml")
        page_text = soup.get_text(" ", strip=True)
        visible_patch = extract_patch_from_text(page_text)
        objs = extract_qwik_objects(html)

        champion_name = extract_champion_name(soup)
        champion_name_normalized = normalize_champion_name(champion_name)
        if champion_name_normalized in seen_names:
            continue
        seen_names.add(champion_name_normalized)

        pick_rate = extract_pick_rate(page_text)
        depth_payload = extract_depth_payload(objs)

        rows.append(
            {
                "champion_name": champion_name,
                "champion_name_normalized": champion_name_normalized,
                "lane": LANE,
                "patch": visible_patch,
                "rank": rank_label,
                "population_scope_pickrate": population_scope_pickrate(tier),
                "population_scope_depth": POPULATION_SCOPE_DEPTH,
                "pick_rate": pick_rate,
                "breadth": depth_payload["breadth"],
                "depth": depth_payload["depth"],
                "classification": depth_payload["classification"],
                "unique_players": depth_payload["unique_players"],
                "total_ranked_games": depth_payload["total_ranked_games"],
                "source_url": champion.source_url,
                "extraction_date": extraction_date,
            }
        )

    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    session = create_session()
    rows = collect_rows(session, args.patch, args.tier)

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
    output_path = Path(args.output)
    write_csv(output_path, rows, fieldnames)
    print(f"Wrote {len(rows)} rows to {output_path.as_posix()}.")
    print(f"Validation: extracted {len(rows)} unique champion rows.")


if __name__ == "__main__":
    main()
