from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import resolve_extra_data_dir


def test_resolve_extra_data_dir_prefers_bundled_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "project" / "data"
    bundled = data_dir / "external"
    legacy = tmp_path / "data"
    bundled.mkdir(parents=True)
    legacy.mkdir()

    assert resolve_extra_data_dir(data_dir) == bundled.resolve()


def test_resolve_extra_data_dir_honors_explicit_path(tmp_path: Path) -> None:
    configured = tmp_path / "custom"

    assert resolve_extra_data_dir(tmp_path / "data", configured) == configured
