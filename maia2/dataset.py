import os

import pandas as pd

from .utils import download_google_drive_file


_EXAMPLE_TEST_ASSET = {
    "url": "https://drive.google.com/uc?id=1fSu4Yp8uYj7xocbHAbjBP6DthsgiJy9X",
    "filename": "example_test_dataset.csv",
    "sha256": "cd4defb7213f052eb0c3e78c1af32f40ee67b8cb85b859e890862db31dcb7bd9",
}
_EXAMPLE_TRAIN_ASSET = {
    "url": "https://drive.google.com/uc?id=1XBeuhB17z50mFK4tDvPG9rQRbxLSzNqB",
    "filename": "example_train_dataset.csv",
    "sha256": "40907a6356fa02c644d307b7090b093b6e0138f8b197b4eae79a932907c3a106",
}


def load_example_test_dataset(save_root="./maia2_data"):
    os.makedirs(save_root, exist_ok=True)
    output_path = os.path.join(save_root, _EXAMPLE_TEST_ASSET["filename"])
    download_google_drive_file(
        _EXAMPLE_TEST_ASSET["url"],
        output_path,
        sha256=_EXAMPLE_TEST_ASSET["sha256"],
        quiet=False,
    )

    data = pd.read_csv(output_path)
    data = data[data.move_ply > 10][
        ["board", "move", "active_elo", "opponent_elo"]
    ].copy()

    return data


def load_example_train_dataset(save_root="./maia2_data"):
    os.makedirs(save_root, exist_ok=True)
    output_path = os.path.join(save_root, _EXAMPLE_TRAIN_ASSET["filename"])
    download_google_drive_file(
        _EXAMPLE_TRAIN_ASSET["url"],
        output_path,
        sha256=_EXAMPLE_TRAIN_ASSET["sha256"],
        quiet=False,
    )

    return output_path
