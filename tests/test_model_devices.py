import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from maia2 import model
from maia2.main import MAIA2Model
from maia2.utils import create_elo_dict, get_all_possible_moves


class PretrainedModelDeviceTest(unittest.TestCase):
    def test_pretrained_loader_uses_shared_device_resolution_and_old_checkpoint(self):
        cfg = SimpleNamespace(
            input_channels=18,
            dim_cnn=8,
            num_blocks_cnn=1,
            vit_length=2,
            dim_vit=32,
            num_blocks_vit=1,
            elo_dim=8,
        )
        all_moves = get_all_possible_moves()
        source = MAIA2Model(len(all_moves), create_elo_dict(), cfg)
        checkpoint = {
            "model_state_dict": {
                f"module.{key}": value.detach().clone()
                for key, value in source.state_dict().items()
            }
        }

        with tempfile.TemporaryDirectory() as save_root:
            Path(save_root, "rapid_model.pt").touch()
            Path(save_root, "config.yaml").touch()
            with (
                mock.patch("maia2.model.parse_args", return_value=cfg),
                mock.patch("maia2.model.torch.load", return_value=checkpoint),
                mock.patch(
                    "maia2.model.resolve_device",
                    return_value=torch.device("cpu"),
                ) as resolve,
            ):
                loaded = model.from_pretrained(
                    type="rapid",
                    device="mps",
                    save_root=save_root,
                )

        resolve.assert_called_once_with("mps")
        self.assertEqual(next(loaded.parameters()).device, torch.device("cpu"))
        for key, value in source.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value)


if __name__ == "__main__":
    unittest.main()
