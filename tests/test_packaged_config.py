import hashlib
from importlib.resources import files
import unittest

from maia2.main import MAIA2Model
from maia2.utils import Config, create_elo_dict, get_all_possible_moves
import yaml


RELEASED_CONFIG_SHA256 = (
    "4b06a5e6917dba8a55defaf3947ce97a73edca3ae2c9d225779a620353c1371b"
)


class PackagedConfigTest(unittest.TestCase):
    def test_released_config_hash_and_model_shapes(self):
        config_root = files("maia2.configs")
        config_resource = config_root.joinpath("maia2-training.yaml")
        config_bytes = config_resource.read_bytes()
        self.assertEqual(
            hashlib.sha256(config_bytes).hexdigest(), RELEASED_CONFIG_SHA256
        )

        cfg = Config(yaml.safe_load(config_bytes))
        all_moves = get_all_possible_moves()
        model = MAIA2Model(len(all_moves), create_elo_dict(), cfg)

        self.assertEqual(len(all_moves), 1880)
        self.assertEqual(tuple(model.chess_cnn.conv1.weight.shape), (256, 18, 3, 3))
        self.assertEqual(tuple(model.pos_embedding.shape), (1, 8, 1024))
        self.assertEqual(tuple(model.fc_1.weight.shape), (1880, 1024))
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()), 23_297_486
        )

    def test_speed_presets_match_the_released_training_configuration(self):
        config_root = files("maia2.configs")
        released = yaml.safe_load(
            config_root.joinpath("maia2-training.yaml").read_bytes()
        )
        self.assertEqual(released.pop("data_root"), "/datadrive2/lichess_data")
        presets = {}

        for game_type in ("rapid", "blitz"):
            name = f"maia2-training-{game_type}.yaml"
            with self.subTest(name=name):
                preset = yaml.safe_load(config_root.joinpath(name).read_bytes())
                self.assertEqual(preset.pop("game_type"), game_type)
                save_root = preset.pop("save_root")
                self.assertEqual(save_root, f"../saves/{game_type}")
                self.assertEqual(preset.pop("data_root"), "./lichess_data")
                self.assertEqual(preset, released)
                presets[game_type] = save_root

        self.assertNotEqual(presets["rapid"], presets["blitz"])


if __name__ == "__main__":
    unittest.main()
