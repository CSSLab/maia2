import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from maia2.main import MAIA2Model, train_chunks
from maia2 import train
from maia2.train import load_model_state_dict, resolve_device
from maia2.utils import create_elo_dict, get_all_possible_moves


class TrainingDeviceTest(unittest.TestCase):
    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.backends.mps.is_available", return_value=True),
        ):
            self.assertEqual(resolve_device("auto"), torch.device("cuda"))

        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=True),
        ):
            self.assertEqual(resolve_device("auto"), torch.device("mps"))

        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=False),
        ):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))

    def test_unavailable_explicit_accelerators_have_clear_errors(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                resolve_device("cuda")

        with mock.patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "MPS was requested"):
                resolve_device("mps")

    def test_loads_data_parallel_checkpoint_into_unwrapped_model(self):
        source = nn.Linear(3, 2)
        checkpoint = {
            f"module.{key}": value.clone()
            for key, value in source.state_dict().items()
        }
        target = nn.Linear(3, 2)

        load_model_state_dict(target, checkpoint)

        for key, value in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], value)

    def test_training_keeps_at_least_one_preprocessing_process(self):
        with mock.patch("maia2.train.cpu_count", return_value=8):
            self.assertEqual(train.get_num_processes(num_cpu_left=16), 1)

    def test_cpu_training_step_updates_the_maia2_model(self):
        cfg = self._tiny_config(
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
        all_moves = get_all_possible_moves()
        all_moves_dict = {move: index for index, move in enumerate(all_moves)}
        model = MAIA2Model(len(all_moves), create_elo_dict(), cfg).to("cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.fc_1.weight.detach().clone()

        losses = train_chunks(
            cfg,
            [
                (
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "e2e4",
                    5,
                    5,
                    1,
                )
            ],
            model,
            optimizer,
            all_moves_dict,
            nn.CrossEntropyLoss(),
            nn.BCEWithLogitsLoss(),
            nn.MSELoss(),
        )

        self.assertTrue(all(math.isfinite(loss) for loss in losses))
        self.assertFalse(torch.equal(before, model.fc_1.weight.detach()))

    @staticmethod
    def _tiny_config(**overrides):
        values = dict(
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
