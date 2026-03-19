# Changelog

All notable changes to **SATTC** are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.0.0] — 2026-03-19

### Added
- Initial public release accompanying CVPR 2026 submission.
- Leave-one-subject-out (LOSO) training & evaluation pipeline (`run_sattc_loso.py`).
- Label-free test-time calibration head:
  - Subject-adaptive whitening of EEG embeddings.
  - Adaptive CSLS (Cross-domain Similarity Local Scaling) geometric expert.
  - Structural expert from mutual nearest neighbors, bidirectional top-*k* ranks, and class popularity.
  - Product-of-Experts fusion.
- Multi-encoder baseline comparison (`contrast_retrieval.py`).
- LOSO dataset loader for THINGS-EEG (`eegdatasets_leaveone.py`).
- Cross-validation split generator (`split_generator.py`).
- EEG preprocessing pipeline (`EEG-preprocessing/`).
- Transformer / attention subject-layer modules (`subject_layers/`).
- Evaluation metrics: Top-1, Top-5 accuracy, hubness (skewness of N_k), per-class imbalance.
- `requirements.txt`, `setup.sh`, MIT `LICENSE`, `CITATION.cff`, and this `CHANGELOG.md`.

---

<!-- Template for future releases:
## [X.Y.Z] — YYYY-MM-DD

### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security
-->
