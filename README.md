# Maia2: A Unified Model for Human-AI Alignment in Chess

[![CI](https://github.com/CSSLab/maia2/actions/workflows/ci.yml/badge.svg)](https://github.com/CSSLab/maia2/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/maia2.svg)](https://pypi.org/project/maia2/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/CSSLab/maia2/blob/main/pyproject.toml)
[![License](https://img.shields.io/github/license/CSSLab/maia2.svg)](https://github.com/CSSLab/maia2/blob/main/LICENSE)

> [!IMPORTANT]
> **[Maia-3](https://github.com/CSSLab/maia3) is now available and is
> recommended for new projects.** It is the latest generation of our human
> chess modeling work, built on the Chessformer architecture. See the
> [code](https://github.com/CSSLab/maia3),
> [pre-trained models](https://huggingface.co/collections/UofTCSSLab/maia3),
> [paper](https://arxiv.org/abs/2605.19091), and
> [website](https://www.maiachess.com/).

This is the official implementation of the NeurIPS 2024 paper
**Maia-2: A Unified Model for Human-AI Alignment in Chess**
[[paper](https://arxiv.org/abs/2409.20553)]. This work was led by
[CSSLab](https://csslab.cs.toronto.edu/) at the University of Toronto.

## Abstract

There are an increasing number of domains in which artificial intelligence
(AI) systems both surpass human ability and accurately model human behavior.
This introduces the possibility of algorithmically-informed teaching in these
domains through more relatable AI partners and deeper insights into human
decision-making. Critical to achieving this goal, however, is coherently
modeling human behavior at various skill levels. Chess is an ideal model
system for conducting research into this kind of human-AI alignment, with its
rich history as a pivotal testbed for AI research, mature superhuman AI
systems like AlphaZero, and precise measurements of skill via chess rating
systems. Previous work in modeling human decision-making in chess uses
completely independent models to capture human style at different skill
levels, meaning they lack coherence in their ability to adapt to the full
spectrum of human improvement and are ultimately limited in their
effectiveness as AI partners and teaching tools. In this work, we propose a
unified modeling approach for human-AI alignment in chess that coherently
captures human style across different skill levels and directly captures how
people improve. Recognizing the complex, non-linear nature of human learning,
we introduce a skill-aware attention mechanism to dynamically integrate
players' strengths with encoded chess positions, enabling our model to be
sensitive to evolving player skill. Our experimental results demonstrate that
this unified framework significantly enhances the alignment between AI and
human players across a diverse range of expertise levels, paving the way for
deeper insights into human decision-making and AI-guided teaching tools.

## Requirements

Maia-2 supports Python 3.10, 3.11, and 3.12. The release-validation baseline
uses PyTorch 2.8, and the device interface supports:

- NVIDIA CUDA GPUs;
- Apple Silicon through MPS; and
- CPU-only training and inference.

The package metadata uses compatible dependency ranges so Maia-2 can coexist
with other Python packages. To match the direct dependency versions used in
release validation, use the pinned
[release-validation requirements](https://github.com/CSSLab/maia2/blob/main/maia2/requirements.txt),
which also include `pyyaml` for loading training and model configurations.

## Installation

Install the latest release from PyPI:

```sh
pip install maia2
```

The `main` branch can contain maintenance work for the next release.
Clone and install the repository as shown below when validating an exact
commit rather than the latest published PyPI artifact.

For development or matching the release-validation dependency baseline, we
recommend a fresh Python 3.12 environment:

```sh
conda create -n maia2 python=3.12 -y
conda activate maia2
git clone https://github.com/CSSLab/maia2.git
cd maia2
python -m pip install -r maia2/requirements.txt
python -m pip install -e . --no-deps
```

Contributors can install the test, formatting, and build tools with:

```sh
python -m pip install -e ".[dev]"
```

## Current release candidate

The source version on `main` is the 0.11.0 release candidate. It provides a
consistent `"auto"`, `"cuda"`, `"mps"`, and `"cpu"` device interface for
training and inference, preserves compatibility with older DataParallel
checkpoints, and supports strict rated Rapid or rated Blitz training through
packaged configurations.

Version 0.11.0 also adds cryptographic training-data provenance, safe chunk
caches and checkpoint resumption, legal-move masking fixes, guarded downloads,
BOT filtering, the released-model configuration, and expanded tests and
release metadata. Until 0.11.0 appears on PyPI, install an exact source commit
to validate these changes.

## Quick Start: Batch Inference

```python
from maia2 import dataset, inference, model
```

Load a model for `"rapid"` or `"blitz"` games. The default `"auto"` setting
selects CUDA first, then MPS, and finally CPU.

```python
maia2_model = model.from_pretrained(type="rapid", device="auto")
```

Set `device` explicitly to `"cuda"`, `"mps"`, or `"cpu"` when needed. The
older `"gpu"` value remains supported as an alias for `"cuda"`.
For training on a multi-GPU host, unindexed `"cuda"` uses all visible CUDA
devices through `DataParallel`; `"cuda:N"` selects only device `N`. `"auto"`
selects CUDA first, then MPS, and finally CPU.

Load the example test dataset:

```python
data = dataset.load_example_test_dataset()
```

Run batch inference:

```python
data, acc = inference.inference_batch(
    data,
    maia2_model,
    verbose=1,
    batch_size=1024,
    num_workers=4,
)
print(acc)
```

The returned `data` contains the inference results. Tune `batch_size` and
`num_workers` for the available device and memory.

## Position-wise Inference

Prepare the move and Elo mappings once, then reuse them across positions:

```python
prepared = inference.prepare()

columns = ["board", "move", "active_elo", "opponent_elo"]
for fen, move, elo_self, elo_oppo in data.loc[:, columns].head(10).itertuples(
    index=False, name=None
):
    move_probs, win_prob = inference.inference_each(
        maia2_model,
        prepared,
        fen,
        elo_self,
        elo_oppo,
    )
    predicted_move = max(move_probs, key=move_probs.get)
    print(f"Move: {move}; predicted: {predicted_move}; White expected score: {win_prob}")
```

Despite its legacy name, `win_prob` is a White-perspective expected score in
the original FEN orientation, not a calibrated probability of winning and not
the active player's score. The value head is mapped from `[-1, 1]` to `[0, 1]`,
where `0` represents a White loss, `0.5` a draw, and `1` a White win. When
Black is to move, Maia-2 mirrors the position internally and converts the
score back to White's perspective.

Try varying the active player's skill level (`elo_self`) and the opponent's
skill level (`elo_oppo`) to inspect how the predictions change.

## Training

### Download Lichess data

Maia-2 trains on the monthly standard-rated archives from the
[Lichess database](https://database.lichess.org/). Keep the downloads in
`.pgn.zst` format; Maia-2 decompresses each archive while training.
For compatibility with the released training pipeline, a game's PGN `Event`
must contain the exact, case-sensitive marker `Rated` and the selected speed
marker, either `Rapid` or `Blitz`. Tournament, arena, or casual Event names
without `Rated` are intentionally excluded even when the surrounding archive
is standard-rated.

Download one month for a local training check:

```sh
./maia2/fetch_data.sh /path/to/lichess_data 2023-01 2023-01
```

Download the full date range used by the released training configuration:

```sh
./maia2/fetch_data.sh /path/to/lichess_data 2018-05 2023-11
```

December 2019 is intentionally skipped to match the original Maia-2 training
pipeline.

### Choose a packaged Rapid or Blitz configuration

Two maintained training presets use the released architecture and training
parameters while making the data selection explicit:

- [`maia2-training-rapid.yaml`](https://github.com/CSSLab/maia2/blob/main/maia2/configs/maia2-training-rapid.yaml)
  selects Events containing both `Rated` and `Rapid`;
- [`maia2-training-blitz.yaml`](https://github.com/CSSLab/maia2/blob/main/maia2/configs/maia2-training-blitz.yaml)
  selects Events containing both `Rated` and `Blitz`.

The presets default to separate Rapid and Blitz checkpoint roots. Keep them
separate when overriding `save_root`: a run manifest prevents checkpoints from
different game types from being mixed or resumed together. If `game_type` is
absent from an older configuration, Maia-2 defaults to `rapid` for backward
compatibility. Other values are rejected before any archive or output path is
touched.

[`maia2-training.yaml`](https://github.com/CSSLab/maia2/blob/main/maia2/configs/maia2-training.yaml)
remains an immutable, byte-for-byte copy of the configuration downloaded with
the released Rapid and Blitz models. It is the historical Rapid-compatible
alias and is included alongside both maintained presets in source and wheel
distributions. Its SHA-256 digest is:

```text
4b06a5e6917dba8a55defaf3947ce97a73edca3ae2c9d225779a620353c1371b
```

`model.from_pretrained` loads the immutable bundled configuration because the
published Rapid and Blitz checkpoints share the same architecture. Downloads
of both checkpoints are SHA-256 verified before they are loaded.

The model architecture in all three files (256 CNN channels,
1024-dimensional transformer, five CNN blocks, two transformer blocks, and
128-dimensional Elo embeddings) matches the released checkpoint tensor
shapes. Select a maintained preset, then override machine-specific paths after
loading it:

```python
from importlib.resources import as_file, files

from maia2 import train, utils


def main():
    game_type = "rapid"  # Change to "blitz" for strict rated Blitz training.
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

Keep the `__main__` guard when running training from a Python script: Maia-2
uses spawned preprocessing workers so CUDA and MPS initialization remains
safe.

Checkpoints are written below
`<save_root>/<lr>_<batch_size>_<weight_decay>/`, not directly into
`save_root`.

The released configuration covers May 2018 through November 2023, trains for
three epochs, and uses a batch size of 8192. The original run used two A100
GPUs and 16 preprocessing CPUs and took roughly one week per epoch. For a
short validation run, narrow `start_year`/`start_month` and
`end_year`/`end_month`. For a laptop, also reduce `batch_size`, `num_workers`,
and `chunk_size` to fit the available memory.

`last_n_moves` is retained because it is present in the distributed legacy
configuration, but the current training implementation does not consume it.

This file is the best available architecture-compatible reference, not a
guarantee of bit-for-bit weight reproduction: the checkpoint does not embed a
configuration hash, source revision, or training-data manifest. The maintained
code also filters players explicitly labeled `BOT`, and floating-point results
can vary across hardware, drivers, dependencies, and parallel execution.

`train.run` creates a new model unless checkpoint restoration is enabled. A
model returned by `model.from_pretrained` is intended for inference and is not
used automatically by `train.run`. To resume training, set `from_checkpoint`,
`checkpoint_epoch`, `checkpoint_year`, and `checkpoint_month` in the
configuration. Keep the original full `start_year`/`start_month` through
`end_year`/`end_month` range: Maia-2 skips completed months in the checkpoint's
epoch and returns to the full range for each later epoch. New checkpoints
include a configuration snapshot, Maia-2 and PyTorch versions, the current
month's compressed and decompressed source names, sizes, and SHA-256 digests,
optimizer-step counts, and CPU plus available accelerator RNG state.

### Reproducibility and artifact safety

For one configured month, `source_sha256` may be the archive's 64-character
SHA-256 digest:

```python
cfg.source_sha256 = "3b522ebe20bd745b763298efcb0043abfa80a8e3b55a1c9bf4a4f4f8236289e8"
```

For multiple months, provide a complete mapping keyed by `YYYY-MM`, PGN file
name, or archive file name:

```python
cfg.source_sha256 = {
    "2018-05": "3b522ebe20bd745b763298efcb0043abfa80a8e3b55a1c9bf4a4f4f8236289e8",
    "2018-06": "<sha256-of-the-2018-06-archive>",
}
```

If the option is omitted, Maia-2 still calculates and records each digest,
but it cannot compare the archive against a digest supplied independently.
Setting `reuse_decompressed = true` reuses a PGN only when its provenance
sidecar and freshly calculated hash both match the current archive. Chunk
caches are JSON, cover the complete PGN, and are bound to its SHA-256 rather
than only its size.

Each run directory has a `run_manifest.json` that locks architecture, data
range and chunking, source hashes, the selected Rapid or Blitz filter policy,
losses, optimizer settings, and seed. Incompatible settings require a new
`save_root`. Existing checkpoints are never overwritten by default;
`overwrite_checkpoints = true` is required for intentional replacement within
an already compatible run. Checkpoints and manifests are installed atomically.

Legacy checkpoints remain loadable with a warning, but their configuration and
data provenance cannot always be verified. Because historical Maia-2 training
defaulted to Rapid, a legacy checkpoint without training metadata may only be
resumed as Rapid; it is rejected for Blitz. Maia-2 also refuses a resume when
the available source, configuration, epoch, or optimizer hyperparameters
conflict.

## Interpretability and Concept Probing

For follow-up work on interpreting Maia-2's skill-aware representations, see
[maia2-skill-adaptation](https://github.com/CSSLab/maia2-skill-adaptation).
It includes code for extracting intermediate activations and training
Elo-conditioned linear probes over 172 formally defined chess concepts,
including bishop-pair and queen-capture concepts. This is an extension of the
concept analysis in the Maia-2 paper rather than an exact reproduction of
every measurement in the paper's chess-concept figure.

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

If your work uses Maia-2, please cite the Maia-2 paper. If it also uses Maia-3
or Chessformer, we would appreciate citing both relevant papers. If you find
the projects useful, please also consider starring the
[Maia-2](https://github.com/CSSLab/maia2) and
[Maia-3](https://github.com/CSSLab/maia3) repositories.

## Contributing and Security

Contributions are welcome; see
[CONTRIBUTING.md](https://github.com/CSSLab/maia2/blob/main/CONTRIBUTING.md).
Please report potential vulnerabilities privately as described in
[SECURITY.md](https://github.com/CSSLab/maia2/blob/main/SECURITY.md), rather
than opening a public issue.

## Contact

For questions or suggestions, contact josephtang@cs.toronto.edu or open a
GitHub issue.

## License

This project is licensed under the
[MIT License](https://github.com/CSSLab/maia2/blob/main/LICENSE).
