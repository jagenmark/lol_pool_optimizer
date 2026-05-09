from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DDRAGON_BASE_URL = "https://ddragon.leagueoflegends.com"
VERSIONS_URL = f"{DDRAGON_BASE_URL}/api/versions.json"
USER_AGENT = "lol-pool-optimizer-icon-downloader/1.0"


def fetch_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        destination.write_bytes(response.read())


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download League of Legends champion icons from Riot Data Dragon."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload icons even when the local file already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()
    icon_dir = root / "assets" / "champion_icons"
    manifest_path = icon_dir / "champion_icon_manifest.csv"
    icon_dir.mkdir(parents=True, exist_ok=True)

    try:
        versions = fetch_json(VERSIONS_URL)
        version = versions[0]
        champion_data_url = f"{DDRAGON_BASE_URL}/cdn/{version}/data/en_US/champion.json"
        champion_payload = fetch_json(champion_data_url)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, IndexError) as exc:
        print(f"Failed to fetch Data Dragon metadata: {exc}", file=sys.stderr)
        return 1

    champions = champion_payload.get("data", {})
    manifest_rows: list[dict[str, str]] = []
    downloaded = 0
    skipped = 0
    failed = 0

    for champion in sorted(champions.values(), key=lambda item: item.get("name", "")):
        champion_name = str(champion.get("name", ""))
        champion_id = str(champion.get("id", ""))
        champion_key = str(champion.get("key", ""))
        image = champion.get("image", {})
        icon_filename = str(image.get("full", ""))

        if not icon_filename:
            print(f"Missing image.full for {champion_name or champion_id}", file=sys.stderr)
            failed += 1
            continue

        source_url = f"{DDRAGON_BASE_URL}/cdn/{version}/img/champion/{icon_filename}"
        destination = icon_dir / icon_filename
        local_path = Path("assets") / "champion_icons" / icon_filename

        manifest_rows.append(
            {
                "champion_name": champion_name,
                "champion_id": champion_id,
                "champion_key": champion_key,
                "icon_filename": icon_filename,
                "local_path": local_path.as_posix(),
                "ddragon_version": version,
                "source_url": source_url,
            }
        )

        if destination.exists() and not args.force:
            skipped += 1
            continue

        try:
            download_file(source_url, destination)
            downloaded += 1
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            print(f"Failed to download {champion_name} from {source_url}: {exc}", file=sys.stderr)
            failed += 1

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        fieldnames = [
            "champion_name",
            "champion_id",
            "champion_key",
            "icon_filename",
            "local_path",
            "ddragon_version",
            "source_url",
        ]
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(
        f"Downloaded {downloaded} icons, skipped {skipped} existing icons, failed {failed}. "
        f"Manifest: {manifest_path}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
