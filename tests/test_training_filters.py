import io
import unittest

import chess.pgn

from maia2.main import game_filter


def _game_with_titles(white_title=None, black_title=None):
    title_headers = []
    if white_title is not None:
        title_headers.append(f'[WhiteTitle "{white_title}"]')
    if black_title is not None:
        title_headers.append(f'[BlackTitle "{black_title}"]')

    pgn = "\n".join(
        [
            '[Event "Rated Rapid game"]',
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

    def test_keeps_rated_database_games_with_arena_event_names(self):
        game = _game_with_titles()
        game.headers["Event"] = "≤2000 Rapid Arena"
        self.assertIsNotNone(game_filter(game))

    def test_drops_an_explicitly_casual_rapid_export(self):
        game = _game_with_titles()
        game.headers["Event"] = "Casual Rapid game"
        self.assertIsNone(game_filter(game))

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
