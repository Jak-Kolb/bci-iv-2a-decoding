### BCI-IV-2A-DECODING
Per-subject motor imagery classifier for BCI Competition IV Dataset 2a. Learning project; CSP + LDA baseline first, EEGNet to follow.

## Layout
- `src/mi_decoder/` — core library
  - `data.py` — MOABB data loading, epoching, train/test splits
  - `preprocess.py` — filtering and artifact rejection
  - `csp.py` — CSP feature extraction
  - `lda.py` — LDA classifier
- `src/scoring/cv_scoring.py` — cross-validation scoring helpers
- `tests/` — pytest suite (mirrors `src/mi_decoder/`)
- `notebooks/` — exploration: data, preprocessing, CSP+LDA
- `scripts/` — one-off scripts (e.g. `iterate_outcomes.py`)
- `data/`, `results/` — gitignored inputs and run outputs
- `CLAUDE.md` — project conventions and EEG gotchas
