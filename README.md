# Maia2: A Unified Model for Human-AI Alignment in Chess

> [!IMPORTANT]
> **[Maia-3](https://github.com/CSSLab/maia3) is now available and is recommended for new projects.** It is the latest generation of our human chess modeling work, built on the Chessformer architecture. See the [code](https://github.com/CSSLab/maia3), [pre-trained models](https://huggingface.co/collections/UofTCSSLab/maia3), [paper](https://arxiv.org/abs/2605.19091), and [website](https://maiachess.com/).

The official implementation of the NeurIPS 2024 paper **Maia-2** [[paper](https://arxiv.org/abs/2409.20553)]. This work was led by [CSSLab](https://csslab.cs.toronto.edu/) at the University of Toronto.

## Abstract
There are an increasing number of domains in which artificial intelligence (AI) systems both surpass human ability and accurately model human behavior. This introduces the possibility of algorithmically-informed teaching in these domains through more relatable AI partners and deeper insights into human decision-making. Critical to achieving this goal, however, is coherently modeling human behavior at various skill levels. Chess is an ideal model system for conducting research into this kind of human-AI alignment, with its rich history as a pivotal testbed for AI research, mature superhuman AI systems like AlphaZero, and precise measurements of skill via chess rating systems. Previous work in modeling human decision-making in chess uses completely independent models to capture human style at different skill levels, meaning they lack coherence in their ability to adapt to the full spectrum of human improvement and are ultimately limited in their effectiveness as AI partners and teaching tools. In this work, we propose a unified modeling approach for human-AI alignment in chess that coherently captures human style across different skill levels and directly captures how people improve. Recognizing the complex, non-linear nature of human learning, we introduce a skill-aware attention mechanism to dynamically integrate players’ strengths with encoded chess positions, enabling our model to be sensitive to evolving player skill. Our experimental results demonstrate that this unified framework significantly enhances the alignment between AI and human players across a diverse range of expertise levels, paving the way for deeper insights into human decision-making and AI-guided teaching tools.

## Requirements

```sh
chess==1.10.0
einops==0.8.0
gdown==5.2.0
numpy==2.1.3
pandas==2.2.3
pyzstd==0.15.9
Requests==2.32.3
torch==2.4.0
tqdm==4.65.0
```

The version requirements may not be very strict, but the above configuration should work.

## Installation

```sh
pip install maia2
```

## Quick Start: Batch Inference

```python
from maia2 import model, dataset, inference
```

You can load a model for `"rapid"` or `"blitz"` games with either CPU or GPU.

```python
maia2_model = model.from_pretrained(type="rapid", device="gpu")
```

Load a pre-defined example test dataset for demonstration.

```python
data = dataset.load_example_test_dataset()
```

Batch Inference
- `batch_size=1024`: Set the batch size for inference.
- `num_workers=4`: Use multiple worker threads for data loading and processing.
- `verbose=1`: Show the progress bar during the inference process.

```python
data, acc = inference.inference_batch(data, maia2_model, verbose=1, batch_size=1024, num_workers=4)
print(acc)
```

`data` will be updated in-place to include inference results.


## Position-wise Inference

We use the same example test dataset for demonstration.
```python
prepared = inference.prepare()
```

Once the prepapration is done, you can easily run inference position by position:
```python
for fen, move, elo_self, elo_oppo, _, _ in data.values[:10]:
    move_probs, win_prob = inference.inference_each(maia2_model, prepared, fen, elo_self, elo_oppo)
    print(f"Move: {move}, Predicted: {move_probs}, Win Prob: {win_prob}")
    print(f"Correct: {max(move_probs, key=move_probs.get) == move}")
```

Try to tweak the skill level (ELO) of the activce player `elo_self` and opponent play `elo_oppo`! You may find it insightful for some positions.

## Interpretability and Concept Probing

For follow-up work on interpreting Maia-2's skill-aware representations, see
[maia2-skill-adaptation](https://github.com/CSSLab/maia2-skill-adaptation).
It includes code for extracting intermediate activations and training
Elo-conditioned linear probes over 172 formally defined chess concepts,
including bishop-pair and queen-capture concepts. This is an extension of the
concept analysis in the Maia-2 paper rather than an exact reproduction of every
measurement in Figure 5.


## Training

### Download data from [Lichess Database](https://database.lichess.org/)

Please download the game data of the time period you would like to train on in `.pgn.zst` format. Data decompressing is handled by `maia2`, so you don't need to decompress these files before training.

### Training with our default settings

Please modify `data_root` in the config file to indicate where you stored the downloaded lichess data. It will take around 1 week to finish training 1 epoch with 2\*A100 and 16\*CPUs.

```python
from maia2 import train, utils
cfg = utils.parse_args(cfg_file_path="./maia2_models/config.yaml")
train.run(cfg, device="auto")
```

The training device can be set to `"cuda"`, `"mps"` (Apple Silicon), or
`"cpu"`. The default `"auto"` setting selects CUDA first, then MPS, and
finally CPU. CPU training is supported but is likely to be much slower; reduce
the configured batch size and number of workers when training on a laptop.

`train.run` creates a new model unless checkpoint restoration is enabled in the
configuration. A model returned by `model.from_pretrained` is intended for
inference and is not used automatically by `train.run`.

If you would like to restore training from a checkpoint, please modify the `from_checkpoint`, `checkpoint_year`, and `checkpoint_month` to indicate the initialization you need.


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

If you find these projects helpful, please consider citing both papers and starring both repos.

## Contact

If you have any questions or suggestions, please feel free to contact us via email: josephtang@cs.toronto.edu.

## License

This project is licensed under the [MIT License](LICENSE).
