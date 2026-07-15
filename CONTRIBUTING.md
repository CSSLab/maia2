# Contributing to Maia-2

Thank you for helping improve Maia-2. Bug reports, documentation fixes, tests,
and focused code changes are welcome.

## Before opening an issue

- Search the existing issues and pull requests for related work.
- Use the bug-report template for reproducible defects.
- Do not include credentials, private data, or unpublished game data.
- Report security vulnerabilities privately according to
  [SECURITY.md](SECURITY.md).

Questions about new research projects should also consider
[Maia-3](https://github.com/CSSLab/maia3), which is recommended for new work.

## Development setup

Use Python 3.10, 3.11, or 3.12. Python 3.12 is the primary release-validation
version.

```sh
conda create -n maia2 python=3.12 -y
conda activate maia2
python -m pip install -r maia2/requirements.txt
python -m pip install -e . --no-deps
python -m pip install "pytest>=8,<9" "ruff>=0.14,<1" "build>=1.2,<2"
```

Run the same checks as continuous integration:

```sh
python -m pytest -q
python -m ruff check maia2 tests
python -m ruff format --check maia2 tests
python -m build
python -m pip check
```

To apply formatting locally, run `python -m ruff format maia2 tests`.

## Tests and data

- Put tracked, deterministic unit tests in `tests/`.
- Keep any generated validation assets under the ignored local `test/`
  directory.
- Use the smallest real or synthetic dataset that demonstrates the behavior.
- Never commit Lichess monthly archives, decompressed PGNs, model checkpoints,
  credentials, or ad-hoc machine-specific paths. The bundled training config
  is the deliberate exception: it is preserved byte-for-byte for released
  checkpoint compatibility, so override its `data_root` only in local files.
- Do not make tests depend on datasets outside the repository.

CPU tests should always pass. Changes to device handling or tensor operations
should also be validated on CUDA and Apple Silicon MPS when practical, with
`PYTORCH_ENABLE_MPS_FALLBACK=0` for strict MPS checks.

## Pull requests

Keep each pull request focused. Explain the user-visible behavior, include
regression tests for bug fixes, and report the exact test environments used.
Avoid unrelated formatting or generated files. Maintainers may ask for a
small, reproducible example before reviewing changes that require large data.
