from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from matchup_estimator import empirical_bayes_winrate, raw_winrate


class MatchupEstimatorTests(unittest.TestCase):
    def test_high_games_stays_close_to_raw_winrate(self) -> None:
        raw = raw_winrate(wins=6000, games=10_000)
        shrunk = empirical_bayes_winrate(
            wins=6000,
            games=10_000,
            alpha=100,
            mu=0.5,
        )
        self.assertLess(abs(shrunk - raw), 0.002)

    def test_low_games_moves_strongly_toward_mu(self) -> None:
        raw = raw_winrate(wins=8, games=10)
        shrunk = empirical_bayes_winrate(wins=8, games=10, alpha=100, mu=0.5)
        self.assertLess(abs(shrunk - 0.5), abs(raw - 0.5) / 5)

    def test_zero_games_does_not_crash(self) -> None:
        self.assertEqual(raw_winrate(wins=0, games=0, fallback=0.52), 0.52)
        self.assertEqual(
            empirical_bayes_winrate(wins=0, games=0, alpha=100, mu=0.52),
            0.52,
        )

    def test_zero_alpha_reproduces_raw_winrate_with_games(self) -> None:
        raw = raw_winrate(wins=7, games=13)
        shrunk = empirical_bayes_winrate(wins=7, games=13, alpha=0, mu=0.5)
        self.assertAlmostEqual(shrunk, raw)


if __name__ == "__main__":
    unittest.main()
