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

    def test_gpu_remains_an_alias_for_cuda(self):
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device("gpu"), torch.device("cuda"))

    def test_indexed_cuda_is_validated_and_never_uses_data_parallel(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.device_count", return_value=2),
        ):
            self.assertEqual(resolve_device("cuda:1"), torch.device("cuda:1"))
            with self.assertRaisesRegex(RuntimeError, "only 2 CUDA device"):
                resolve_device("cuda:2")
            self.assertFalse(train._should_use_data_parallel(torch.device("cuda:1")))
            self.assertTrue(train._should_use_data_parallel(torch.device("cuda")))

    def test_optimizer_resume_state_has_cpu_step_and_parameter_local_moments(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter])
        parameter.grad = torch.tensor(1.0)
        optimizer.step()

        train._normalize_optimizer_state_devices(optimizer)

        state = optimizer.state[parameter]
        self.assertEqual(state["step"].device, torch.device("cpu"))
        self.assertEqual(state["exp_avg"].device, parameter.device)
        self.assertEqual(state["exp_avg_sq"].device, parameter.device)

    def test_loads_data_parallel_checkpoint_into_unwrapped_model(self):
        source = nn.Linear(3, 2)
        checkpoint = {
            f"module.{key}": value.clone() for key, value in source.state_dict().items()
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

    def test_empty_training_chunk_is_skipped_safely(self):
        model = nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        with self.assertWarnsRegex(RuntimeWarning, "no positions"):
            losses = train_chunks(
                self._tiny_config(),
                [],
                model,
                optimizer,
                {},
                criterion,
                criterion,
                criterion,
            )

        self.assertEqual(losses, (0.0, 0.0, 0.0, 0.0))

    def test_nonfinite_training_loss_fails_before_optimizer_step(self):
        cfg = self._tiny_config(
            batch_size=1,
            num_workers=0,
            verbose=0,
            side_info=False,
            value=False,
        )
        all_moves = get_all_possible_moves()
        all_moves_dict = {move: index for index, move in enumerate(all_moves)}
        model = MAIA2Model(len(all_moves), create_elo_dict(), cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.fc_1.weight.detach().clone()

        def nonfinite_criterion(logits, _labels):
            return logits.sum() * float("nan")

        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            train_chunks(
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
                nonfinite_criterion,
                nn.BCEWithLogitsLoss(),
                nn.MSELoss(),
            )

        self.assertTrue(torch.equal(before, model.fc_1.weight.detach()))

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
