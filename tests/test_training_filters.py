import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import chess.pgn
import pandas as pd

from maia2.main import evaluate_MAIA1_data, game_filter, process_per_chunk
from maia2.utils import create_elo_dict


def _game_with_titles(white_title=None, black_title=None, event="Rated Rapid game"):
    title_headers = []
    if white_title is not None:
        title_headers.append(f'[WhiteTitle "{white_title}"]')
    if black_title is not None:
        title_headers.append(f'[BlackTitle "{black_title}"]')

    pgn = "\n".join(
        [
            f'[Event "{event}"]',
            '[Site "https://lichess.org/testgame"]',
            '[Date "2026.07.14"]',
            '[Round "-"]',
            '[White "WhitePlayer"]',
            '[Black "BlackPlayer"]',
            '[Result "1/2-1/2"]',
            '[WhiteElo "1500"]',
            '[BlackElo "1500"]',
            '[TimeControl "600+0"]',
            *title_headers,
            "",
            "1. e4 { [%clk 0:10:00] } e5 { [%clk 0:10:00] } 1/2-1/2",
        ]
    )
    return chess.pgn.read_game(io.StringIO(pgn))


class TrainingGameFilterTest(unittest.TestCase):
    def test_keeps_human_games(self):
        self.assertIsNotNone(game_filter(_game_with_titles()))
        self.assertIsNotNone(game_filter(_game_with_titles("GM", "IM")))
        self.assertIsNotNone(
            game_filter(_game_with_titles(event="Rated Blitz game"), "blitz")
        )

    def test_drops_rapid_events_without_a_rated_marker(self):
        game = _game_with_titles()
        game.headers["Event"] = "≤2000 Rapid Arena"
        self.assertIsNone(game_filter(game))

    def test_drops_an_explicitly_casual_rapid_export(self):
        game = _game_with_titles()
        game.headers["Event"] = "Casual Rapid game"
        self.assertIsNone(game_filter(game))

    def test_drops_rated_games_without_a_rapid_marker(self):
        game = _game_with_titles()
        game.headers["Event"] = "Rated Blitz game"
        self.assertIsNone(game_filter(game))

    def test_blitz_filter_remains_strictly_rated_and_speed_specific(self):
        for event in (
            "≤2000 Blitz Arena",
            "Casual Blitz game",
            "Rated Rapid game",
        ):
            with self.subTest(event=event):
                self.assertIsNone(
                    game_filter(_game_with_titles(event=event), game_type="blitz")
                )

    def test_event_markers_remain_case_sensitive(self):
        for event in ("rated Rapid game", "Rated rapid game"):
            with self.subTest(event=event):
                self.assertIsNone(game_filter(_game_with_titles(event=event)))

    def test_rejects_unknown_game_types(self):
        for game_type in ("bullet", "", None, 1):
            with self.subTest(game_type=game_type):
                with self.assertRaisesRegex(ValueError, "rapid.*blitz"):
                    game_filter(_game_with_titles(), game_type=game_type)

    def test_chunk_preprocessing_uses_the_configured_game_type(self):
        games = "\n\n".join(
            [
                str(_game_with_titles(event="Rated Rapid game")),
                str(_game_with_titles(event="Rated Blitz game")),
            ]
        )
        base_cfg = dict(
            first_n_moves=0,
            max_ply=300,
            clock_threshold=0,
            max_games_per_elo_range=20,
        )

        with tempfile.TemporaryDirectory() as directory:
            pgn_path = Path(directory, "mixed.pgn")
            pgn_path.write_text(games, encoding="utf-8")
            args = (0, pgn_path.stat().st_size, pgn_path, create_elo_dict())

            for game_type in ("rapid", "blitz"):
                with self.subTest(game_type=game_type):
                    data, game_count, frequency = process_per_chunk(
                        (*args, SimpleNamespace(**base_cfg, game_type=game_type))
                    )
                    self.assertEqual(len(data), 2)
                    self.assertEqual(game_count, 1)
                    self.assertEqual(sum(frequency.values()), 1)

    def test_blitz_maia1_evaluation_skips_empty_speed_slices(self):
        rapid_only = pd.DataFrame(
            [
                {
                    "type": "Rapid",
                    "board": "unused",
                    "move": "e2e4",
                    "active_elo": 1500,
                    "opponent_elo": 1500,
                    "white_active": True,
                }
            ]
        )
        cfg = SimpleNamespace(
            game_type="blitz",
            maia1_test_root="/unused",
            batch_size=1,
            num_workers=0,
            verbose=0,
        )

        with (
            mock.patch("maia2.main.pd.read_csv", return_value=rapid_only),
            self.assertWarnsRegex(RuntimeWarning, "no Blitz rows"),
        ):
            evaluate_MAIA1_data(None, None, None, cfg, tiny=True)

    def test_drops_games_with_a_labeled_bot(self):
        bot_title_pairs = [
            ("BOT", None),
            (None, "BOT"),
            ("BOT", "BOT"),
            ("bot", None),
        ]

        for white_title, black_title in bot_title_pairs:
            with self.subTest(white_title=white_title, black_title=black_title):
                self.assertIsNone(
                    game_filter(_game_with_titles(white_title, black_title))
                )

    def test_drops_games_with_malformed_elo_headers(self):
        for header in ("WhiteElo", "BlackElo"):
            with self.subTest(header=header):
                game = _game_with_titles()
                game.headers[header] = "not-an-elo"
                self.assertIsNone(game_filter(game))


if __name__ == "__main__":
    unittest.main()
