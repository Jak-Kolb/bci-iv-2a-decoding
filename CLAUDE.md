# Motor Imagery Decoder

BCI Competition IV Dataset 2a. Per-subject motor imagery classifier. Baseline (CSP + LDA) ships first; EEGNet comes after the baseline runs end-to-end on real data.

## Bash commands
- `uv run pytest -q` — run tests
- `uv run pytest tests/test_csp.py -q` — run a single test file
- `uv run ruff check . && uv run ruff format .` — lint and format
- `uv run mypy src` — type check

Train/eval entry points (`python -m mi_decoder.train`, `python -m mi_decoder.eval`) are not implemented yet.

## Code style
- Python 3.11+, type hints on every public function signature.
- `pathlib.Path` only. Never `os.path` or string concatenation for paths.
- No em dashes in code, comments, docstrings, or commit messages.
- Frozen dataclasses for config objects, not dicts.
- `numpy` arrays use explicit dtype. Never trust the default float.
- Errors are raised, not logged-and-continued, unless there is a written reason in a comment.

## Project layout
- `src/mi_decoder/data.py` — loading, epoching, train/test splits
- `src/mi_decoder/preprocess.py` — filtering, artifact rejection
- `src/mi_decoder/csp.py` — CSP feature extraction
- `tests/` — pytest, mirrors `src/mi_decoder/`
- `notebooks/` — exploration and analysis notebooks
- `scripts/` — one-off scripts
- `data/`, `results/` — gitignored data and run outputs

Models (LDA baseline, EEGNet) and `train.py` / `eval.py` entry points do not exist yet; add them under `src/mi_decoder/` when needed.

## Workflow rules
- Run tests before saying a change is done. If no tests exist for the code path being changed, add one.
- Fix root causes. Do not wrap a bug in try/except to make tests pass.
- Do not add a dependency without checking `pyproject.toml` first and asking.
- Do not write README marketing copy, top-level docstrings about "the framework," or a config system until the code path it describes runs end-to-end on real data.
- Do not introduce abstractions (base classes, registries, plugin systems) until there are at least two concrete uses.

## EEG and BCI gotchas — IMPORTANT, these cause silent bugs
- BCI Competition IV 2a labels: `1=left hand`, `2=right hand`, `3=feet`, `4=tongue`. Confirm label encoding when loading. Do not assume zero-indexed.
- Sampling rate is 250 Hz. Always pass `sfreq` explicitly to filter and epoch functions. Never let it default.
- Units: keep signals in volts internally. Convert to microvolts only for plotting. Mixing them silently breaks model scale.
- Channel order matters for CSP and EEGNet. Use the dataset's documented montage. If reordering, write a test that asserts the order.
- CSP and any preprocessing with learned parameters must be fit on training folds only. Refitting on test data is the easiest way to get fake 95% accuracy.
- EEGNet expects input shape `(batch, 1, channels, time_samples)`. Document and test this at the data loader boundary.
- Set `numpy`, `torch`, and `random` seeds at the top of every entry point. Log the seed in run output.

## Testing
- Total test suite runs in under 10 seconds. Use a tiny synthetic EEG fixture (4 channels, 2 seconds, 250 Hz). Do not load real GDF files in unit tests.
- Test invariants at boundaries: data loader output shape and dtype, preprocessing determinism given a seed, model forward-pass shapes.
- Real-subject integration tests live in `tests/integration/` and are not run by default.

## Before saying done — run all four
1. `uv run ruff check . && uv run ruff format --check .`
2. `uv run mypy src`
3. `uv run pytest -q`
4. If a model was trained, print final accuracy AND the random seed used.

## Git
- Branch names: `feat/<short>`, `fix/<short>`, `exp/<short>`. `exp/` branches are never merged.
- Commit messages: imperative mood, no em dashes, no Claude attribution lines.
- Never `git push` or `git commit -a` without being asked.
