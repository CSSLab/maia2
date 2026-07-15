import math
import unittest
from types import SimpleNamespace
from unittest import mock

import chess
import torch
import torch.nn as nn

from maia2.main import MAIA2Dataset, MAIA2Model, train_chunks
from maia2.utils import create_elo_dict, get_all_possible_moves, get_side_info


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class TrainingDatasetMemoryRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all_moves = get_all_possible_moves()
        cls.all_moves_dict = {move: index for index, move in enumerate(cls.all_moves)}
        cls.data = [(START_FEN, "e2e4", 5, 5, 1)]

    def test_enabled_side_info_returns_one_combined_auxiliary_target(self):
        cfg = self._tiny_config(side_info=True)

        with mock.patch(
            "maia2.main.get_side_info", wraps=get_side_info
        ) as mocked_get_side_info:
            sample = MAIA2Dataset(self.data, self.all_moves_dict, cfg)[0]

        self.assertEqual(len(sample), 6)
        mocked_get_side_info.assert_called_once()

        side_info = sample[4]
        legal_moves, expected_side_info = get_side_info(
            chess.Board(START_FEN), "e2e4", self.all_moves_dict
        )
        torch.testing.assert_close(side_info, expected_side_info)
        torch.testing.assert_close(side_info[-len(self.all_moves_dict) :], legal_moves)

    def test_disabled_side_info_returns_no_auxiliary_target(self):
        cfg = self._tiny_config(side_info=False)

        with mock.patch(
            "maia2.main.get_side_info",
            side_effect=AssertionError("side target should not be generated"),
        ) as mocked_get_side_info:
            sample = MAIA2Dataset(self.data, self.all_moves_dict, cfg)[0]

        self.assertEqual(len(sample), 5)
        mocked_get_side_info.assert_not_called()

    def test_real_tiny_cpu_train_step_accepts_both_batch_shapes(self):
        for side_info_enabled in (True, False):
            with self.subTest(side_info=side_info_enabled):
                cfg = self._tiny_config(side_info=side_info_enabled)
                model = MAIA2Model(len(self.all_moves), create_elo_dict(), cfg).to(
                    "cpu"
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                before = model.fc_1.weight.detach().clone()

                losses = train_chunks(
                    cfg,
                    self.data,
                    model,
                    optimizer,
                    self.all_moves_dict,
                    nn.CrossEntropyLoss(),
                    nn.BCEWithLogitsLoss(),
                    nn.MSELoss(),
                )

                self.assertTrue(all(math.isfinite(loss) for loss in losses))
                self.assertFalse(torch.equal(before, model.fc_1.weight.detach()))

    @staticmethod
    def _tiny_config(**overrides):
        values = dict(
            batch_size=1,
            num_workers=0,
            verbose=0,
            side_info=True,
            side_info_coefficient=1.0,
            value=True,
            value_coefficient=1.0,
            input_channels=18,
            dim_cnn=8,
            num_blocks_cnn=1,
            vit_length=2,
            dim_vit=32,
            num_blocks_vit=1,
            elo_dim=8,
        )
        values.update(overrides)
        return SimpleNamespace(**values)


if __name__ == "__main__":
    unittest.main()
