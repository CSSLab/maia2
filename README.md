# Maia-2: A Unified Model for Human-AI Alignment in Chess

[![CI](https://github.com/CSSLab/maia2/actions/workflows/ci.yml/badge.svg)](https://github.com/CSSLab/maia2/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/maia2.svg)](https://pypi.org/project/maia2/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/CSSLab/maia2/blob/main/pyproject.toml)
[![License](https://img.shields.io/github/license/CSSLab/maia2.svg)](https://github.com/CSSLab/maia2/blob/main/LICENSE)

> [!IMPORTANT]
> [Maia-3](https://github.com/CSSLab/maia3) is recommended for new projects.
> See its [pre-trained models](https://huggingface.co/collections/UofTCSSLab/maia3),
> [paper](https://arxiv.org/abs/2605.19091), and
> [website](https://www.maiachess.com/).

Maia-2 is a unified, skill-aware chess model for predicting human moves and
outcomes across Elo levels. This repository is the official implementation of
the NeurIPS 2024 paper
[Maia-2: A Unified Model for Human-AI Alignment in Chess](https://arxiv.org/abs/2409.20553),
led by [CSSLab](https://csslab.cs.toronto.edu/) at the University of Toronto.

## Installation

Maia-2 supports Python 3.10–3.12 and runs on CUDA, Apple MPS, or CPU.

```sh
pip install maia2
```

For development or the validated dependency baseline, use a fresh Python 3.12
environment:

```sh
conda create -n maia2 python=3.12 -y
conda activate maia2
git clone https://github.com/CSSLab/maia2.git
cd maia2
python -m pip install -r maia2/requirements.txt
python -m pip install -e . --no-deps
```

Install contributor tools with:

```sh
python -m pip install -e ".[dev]"
```

The `main` branch may contain changes not yet published on PyPI. Use a source
checkout when validating an exact commit.

## Inference

### Batch inference

Load a released Rapid or Blitz model and run it on the example dataset:

```python
from maia2 import dataset, inference, model

maia2_model = model.from_pretrained(type="rapid", device="auto")
data = dataset.load_example_test_dataset()

data, accuracy = inference.inference_batch(
    data,
    maia2_model,
    verbose=1,
    batch_size=1024,
    num_workers=0,
)
print(accuracy)
```

`"auto"` selects CUDA first, then MPS, then CPU. Set `device` to `"cuda"`,
`"mps"`, or `"cpu"` to choose explicitly. Adjust `batch_size` and
`num_workers` for the available memory.

### Position-wise inference

Prepare the move and Elo mappings once, then reuse them:

```python
prepared = inference.prepare()
columns = ["board", "move", "active_elo", "opponent_elo"]

for fen, move, elo_self, elo_oppo in data.loc[:, columns].head(10).itertuples(
    index=False, name=None
):
    move_probs, white_expected_score = inference.inference_each(
        maia2_model,
        prepared,
        fen,
        elo_self,
        elo_oppo,
    )
    predicted_move = max(move_probs, key=move_probs.get)
    print(
        f"Move: {move}; predicted: {predicted_move}; "
        f"White expected score: {white_expected_score}"
    )
```

The second return value is a White-perspective expected score, not a calibrated
win probability or the active player's score. Values `0`, `0.5`, and `1`
represent a White loss, draw, and win.

## Training

### Data

Maia-2 trains on monthly `.pgn.zst` archives from the
[Lichess standard-rated database](https://database.lichess.org/). A game's PGN
`Event` must contain the exact, case-sensitive `Rated` marker and the selected
`Rapid` or `Blitz` marker. Arena, tournament, and casual Event names without
`Rated` are excluded. Games involving a player titled `BOT` or missing
per-move clock annotations are also excluded.

Download one month for a local test:

```sh
./maia2/fetch_data.sh /path/to/lichess_data 2023-01 2023-01
```

Download the released training range:

```sh
./maia2/fetch_data.sh /path/to/lichess_data 2018-05 2023-11
```

December 2019 is skipped to match the original training pipeline.

### Configurations

Choose one of the maintained presets:

- [`maia2-training-rapid.yaml`](https://github.com/CSSLab/maia2/blob/main/maia2/configs/maia2-training-rapid.yaml)
  accepts Events containing both `Rated` and `Rapid`.
- [`maia2-training-blitz.yaml`](https://github.com/CSSLab/maia2/blob/main/maia2/configs/maia2-training-blitz.yaml)
  accepts Events containing both `Rated` and `Blitz`.

The presets use separate checkpoint roots. Keep Rapid and Blitz outputs
separate when overriding `save_root`. Older configurations without
`game_type` default to Rapid.

[`maia2-training.yaml`](https://github.com/CSSLab/maia2/blob/main/maia2/configs/maia2-training.yaml)
is the preserved legacy configuration matching the released checkpoint
architecture. Use an explicit Rapid or Blitz preset for new training runs.

Run training from a Python script:

```python
from importlib.resources import as_file, files

from maia2 import train, utils


def main():
    game_type = "rapid"  # Use "blitz" for rated Blitz training.
    config_resource = files("maia2.configs").joinpath(
        f"maia2-training-{game_type}.yaml"
    )
    with as_file(config_resource) as config_path:
        cfg = utils.parse_args(config_path)

    cfg.data_root = "/path/to/lichess_data"
    cfg.save_root = f"/path/to/checkpoints/{game_type}"
    train.run(cfg, device="auto")


if __name__ == "__main__":
    main()
```

Keep the `__main__` guard: training uses spawned preprocessing workers.
Checkpoints are written below
`<save_root>/<lr>_<batch_size>_<weight_decay>/`.

The default configuration covers May 2018 through November 2023, trains for
three epochs, and uses a batch size of 8192. Reduce the date range,
`batch_size`, `num_workers`, and `chunk_size` for a short or laptop run.
Unindexed `"cuda"` uses all visible CUDA devices through `DataParallel`;
`"cuda:N"` selects one device.

The packaged configuration matches the released architecture and training
settings, but does not guarantee bit-for-bit weight reproduction. Data
filtering, dependency versions, hardware, and parallel execution can affect
the result.

### Resume and reproducibility

`train.run` starts a new model unless checkpoint restoration is enabled; a
model returned by `model.from_pretrained` is not used automatically. To resume,
set `from_checkpoint`, `checkpoint_epoch`, `checkpoint_year`, and
`checkpoint_month`, while keeping the original full date range in the
configuration.

Maia-2 records data provenance and rejects incompatible checkpoints. Use a
separate `save_root` for different configurations. Legacy checkpoints without
training metadata can only be resumed with Rapid configurations.

## Interpretability

The [Maia-2 skill-adaptation repository](https://github.com/CSSLab/maia2-skill-adaptation)
contains tools for extracting intermediate activations and training
Elo-conditioned probes over 172 chess concepts. It extends the paper's concept
analysis rather than exactly reproducing every result in Figure 4.

## Citation

```bibtex
@inproceedings{
tang2024maia,
title={Maia-2: A Unified Model for Human-{AI} Alignment in Chess},
author={Zhenwei Tang and Difan Jiao and Reid McIlroy-Young and Jon Kleinberg and Siddhartha Sen and Ashton Anderson},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=XWlkhRn14K}
}
```

```bibtex
@inproceedings{monroe2026chessformer,
title={Chessformer: A Unified Architecture for Chess Modeling},
author={Daniel Monroe and George Eilender and Philip Chalmers and Zhenwei Tang and Ashton Anderson},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=2ltBRzEHyd}
}
```

Please cite the relevant paper(s) and consider starring both
[Maia-2](https://github.com/CSSLab/maia2) and
[Maia-3](https://github.com/CSSLab/maia3).

## Contributing and contact

Contributions are welcome; see
[CONTRIBUTING.md](https://github.com/CSSLab/maia2/blob/main/CONTRIBUTING.md).
For questions, email josephtang@cs.toronto.edu or open a GitHub issue.

Report security vulnerabilities privately as described in
[SECURITY.md](https://github.com/CSSLab/maia2/blob/main/SECURITY.md), not in a
public issue.

## License

Maia-2 is released under the
[MIT License](https://github.com/CSSLab/maia2/blob/main/LICENSE).
