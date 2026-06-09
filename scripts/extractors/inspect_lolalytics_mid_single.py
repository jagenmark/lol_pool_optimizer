from __future__ import annotations

import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import Page, sync_playwright


CHAMPION_SLUG = "ahri"
LANE_QUERY = "middle"
RANK_QUERY = "platinum_plus"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)
EDGE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
)


def build_url(slug: str) -> str:
    return f"https://lolalytics.com/lol/{slug}/build/?lane={LANE_QUERY}&tier={RANK_QUERY}"


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


def find_browser_executable() -> str | None:
    for candidate in EDGE_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


def inspect_html(session: requests.Session, url: str) -> None:
    print("== HTML inspection ==")
    response = session.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")

    text = soup.get_text(" ", strip=True)
    pick_rate_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*Pick Rate", text)
    qwik_json_present = '<script type="qwik/json">' in response.text
    depth_label_present = "Depth (Games per player)" in response.text

    print(f"URL: {response.url}")
    print(f"HTTP status: {response.status_code}")
    print(f"Pick Rate found directly in HTML text: {bool(pick_rate_match)}")
    if pick_rate_match:
        print(f"Pick Rate value in HTML: {pick_rate_match.group(1)}")
    print(f"Embedded Qwik JSON present: {qwik_json_present}")
    print(f"Depth label present in serialized HTML: {depth_label_present}")
    print("Conclusion: Pick Rate is server-rendered; depth/main-style data is hydrated.")


def inspect_rendered_depth(page: Page, url: str) -> None:
    print("\n== Rendered depth inspection ==")
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    page.locator("text=Best on").first.scroll_into_view_if_needed()
    page.wait_for_timeout(2_500)

    chart_svg = page.locator("svg").filter(
        has_text="Normalised Champion Ranked Player Base"
    ).first
    chart_svg.locator("circle").first.hover()
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
        raise RuntimeError("Could not locate the rendered depth tooltip text.")

    print(f"Classification: {match.group('classification')}")
    print(f"Breadth: {match.group('breadth')}")
    print(f"Depth: {match.group('depth')}")
    print(f"Unique Players: {match.group('players')}")
    print(f"Total Ranked Games: {match.group('games')}")
    print(f"Tooltip scope text: {match.group('scope')}")
    print("Conclusion: depth is the closest LoLalytics specialization metric to use.")


def main() -> None:
    url = build_url(CHAMPION_SLUG)
    session = create_session()
    inspect_html(session, url)

    executable_path = find_browser_executable()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=executable_path,
        )
        page = browser.new_page(viewport={"width": 1400, "height": 3000})
        inspect_rendered_depth(page, url)
        browser.close()


if __name__ == "__main__":
    main()
