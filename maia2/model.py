import os
from importlib.resources import as_file, files

import torch

from .main import MAIA2Model
from .train import load_model_state_dict, resolve_device
from .utils import create_elo_dict, download_google_drive_file
from .utils import get_all_possible_moves, parse_args


_MODEL_ASSETS = {
    "blitz": {
        "url": "https://drive.google.com/uc?id=1X-Z4J3PX3MQFJoa8gRt3aL8CIH0PWoyt",
        "filename": "blitz_model.pt",
        "sha256": "5090d5d0d49dc29787c08d13febbed5b2da81a18c2ddcd8a90070cd3b43c44b2",
    },
    "rapid": {
        "url": "https://drive.google.com/uc?id=1gbC1-c7c0EQOPPAVpGWubezeEW8grVwc",
        "filename": "rapid_model.pt",
        "sha256": "65aae8465eed5e65df66a24ea7370715579f9e5435098d06fe18bdb1e267e997",
    },
}


def from_pretrained(type, device="auto", save_root="./maia2_models"):
    device = resolve_device(device)

    if type not in _MODEL_ASSETS:
        raise ValueError("Invalid model type. Choose between 'blitz' and 'rapid'.")

    os.makedirs(save_root, exist_ok=True)
    model_asset = _MODEL_ASSETS[type]
    output_path = os.path.join(save_root, model_asset["filename"])
    print(f"Downloading or validating the model for {type} games.")
    download_google_drive_file(
        model_asset["url"],
        output_path,
        sha256=model_asset["sha256"],
        quiet=False,
    )

    # Model architecture is immutable package data. Older releases downloaded
    # it into ``save_root/config.yaml``, which users may have edited for their
    # own training paths; do not overwrite or checksum-reject that legacy file.
    config_resource = files("maia2.configs").joinpath("maia2-training.yaml")
    with as_file(config_resource) as cfg_path:
        cfg = parse_args(cfg_path)

    all_moves = get_all_possible_moves()
    elo_dict = create_elo_dict()

    model = MAIA2Model(len(all_moves), elo_dict, cfg)

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    load_model_state_dict(model, checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Model for {type} games loaded to {device}.")

    return model
