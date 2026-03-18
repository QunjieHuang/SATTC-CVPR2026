#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split Generator for Double Zero-Shot Nested Validation (LOSO + dev-pack + val-unseen)

Per outer fold (test subject):
  1) Compute 9×9 subject distance matrix using per-subject centers from training data only
  2) Select dev-pack = [easy, medium, hard] via mean distance µ(s)
  3) Sample val-unseen classes (e.g., 200 from 0..1653) with fixed seed
  4) Write split JSON under splits/

Notes
- Subject center for a subject s is computed from s's training file only (preprocessed_eeg_training.npy / .npz)
- No cross-subject fitting; only per-subject normalization allowed (here we simply average trials)
- Distance metric: Euclidean on flattened center vectors (63×250 → 15750)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np


# -------------------------------
# Utilities
# -------------------------------

def list_subjects(data_path: str) -> List[str]:
    subs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    subs = [s for s in subs if s.startswith('sub-')]
    subs.sort()
    return subs


def _load_training_array(data_path: str, subject: str) -> np.ndarray:
    """Load training EEG array for a subject.

    Expected file: <data_path>/<subject>/preprocessed_eeg_training.npy
    The file may actually be an NPZ with keys: 'preprocessed_eeg_data', 'times', 'ch_names'.
    Return shape approximately (N_trials, C=63, T=250). If extra dims exist, we flatten to (N_trials, C, T).
    """
    file_path = os.path.join(data_path, subject, 'preprocessed_eeg_training.npy')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training file not found for {subject}: {file_path}")

    data = np.load(file_path, allow_pickle=True)

    # Handle .npz-like, dict, pickled-object, or raw array cases
    if isinstance(data, np.lib.npyio.NpzFile):
        arr = data['preprocessed_eeg_data']
    elif isinstance(data, dict):
        if 'preprocessed_eeg_data' in data:
            arr = data['preprocessed_eeg_data']
        else:
            raise ValueError(f"Dict for {subject} missing 'preprocessed_eeg_data' key: keys={list(data.keys())}")
    elif isinstance(data, np.ndarray) and data.shape == () and hasattr(data, 'item'):
        # Pickled dict stored in ndarray scalar
        obj = data.item()
        if isinstance(obj, dict) and 'preprocessed_eeg_data' in obj:
            arr = obj['preprocessed_eeg_data']
        else:
            raise ValueError(f"Pickled object for {subject} missing 'preprocessed_eeg_data' key: keys={list(obj.keys()) if isinstance(obj, dict) else type(obj)}")
    else:
        arr = data

    arr = np.array(arr)
    # Ensure (N, C, T)
    if arr.ndim < 3:
        raise ValueError(f"Unexpected training array shape for {subject}: {arr.shape}")
    C, T = arr.shape[-2], arr.shape[-1]
    arr = arr.reshape(-1, C, T)
    return arr


def compute_subject_center(train_arr: np.ndarray, per_channel_zscore: bool = False) -> np.ndarray:
    """Compute per-subject center vector from training trials.

    - train_arr: (N, C, T)
    - If per_channel_zscore: z-score each channel over N trials (mean/std computed within subject)
    - Return flattened center vector: (C*T,)
    """
    x = train_arr.astype(np.float32)
    if per_channel_zscore:
        # mean/std over trials axis
        mu = x.mean(axis=0, keepdims=True)  # (1, C, T)
        std = x.std(axis=0, keepdims=True) + 1e-6
        x = (x - mu) / std
    center = x.mean(axis=0)  # (C, T)
    return center.reshape(-1)


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix for rows in X.

    X: (n, d)
    Return: (n, n) symmetric, diag=0
    """
    # (x - y)^2 = x^2 + y^2 - 2 x·y
    g = np.sum(X * X, axis=1, keepdims=True)  # (n,1)
    D2 = g + g.T - 2.0 * (X @ X.T)
    D2 = np.maximum(D2, 0.0)
    D = np.sqrt(D2, dtype=np.float32)
    np.fill_diagonal(D, 0.0)
    # Symmetrize just in case
    D = 0.5 * (D + D.T)
    return D


def select_dev_pack(candidates: List[str], D: np.ndarray, eps: float = 1e-6) -> Tuple[str, str, str]:
    """Select (easy, medium, hard) from candidates using 9×9 distance matrix D.

    candidates: list of subject IDs in fixed order (length=9)
    D: 9×9 matrix where D[i,j] is distance between candidates[i] and candidates[j]
    """
    assert len(candidates) == D.shape[0] == D.shape[1], "Candidates and D size mismatch"

    # mean distance µ(s)
    mu = np.array([(np.sum(D[i]) - D[i, i]) / (len(candidates) - 1) for i in range(len(candidates))])

    # easy = argmin µ(s)
    min_val = mu.min()
    easy_idxs = np.where(np.abs(mu - min_val) <= eps)[0]
    # stable tie-break by sorted subject id order
    if len(easy_idxs) > 1:
        ids = [candidates[i] for i in easy_idxs]
        easy = sorted(ids)[0]
        easy_idx = candidates.index(easy)
    else:
        easy_idx = int(easy_idxs[0])
        easy = candidates[easy_idx]

    # hard = argmax µ(s)
    max_val = mu.max()
    hard_idxs = np.where(np.abs(mu - max_val) <= eps)[0]
    if len(hard_idxs) > 1:
        ids = [candidates[i] for i in hard_idxs]
        hard = sorted(ids)[0]
        hard_idx = candidates.index(hard)
    else:
        hard_idx = int(hard_idxs[0])
        hard = candidates[hard_idx]

    # medium: remove easy & hard, pick closest to median(µ)
    remaining = [i for i in range(len(candidates)) if i not in {easy_idx, hard_idx}]
    mu_remaining = mu[remaining]
    med = float(np.median(mu_remaining))
    diffs = np.abs(mu_remaining - med)
    min_diff = diffs.min()
    med_idxs = [remaining[i] for i, v in enumerate(diffs) if np.abs(v - min_diff) <= eps]
    if len(med_idxs) > 1:
        ids = [candidates[i] for i in med_idxs]
        medium = sorted(ids)[0]
    else:
        medium = candidates[med_idxs[0]]

    return easy, medium, hard


def sample_val_unseen(seed: int, num: int, total_train_classes: int = 1654) -> List[int]:
    rng = np.random.default_rng(seed)
    classes = rng.choice(total_train_classes, size=num, replace=False)
    return sorted([int(x) for x in classes])


@dataclass
class SplitFold:
    test_subject: str
    dev_subjects: List[str]
    train_subjects: List[str]
    val_unseen_classes: List[int]
    seeds: Dict[str, int]
    notes: str
    mu_scores: Dict[str, float]


def generate_fold_split(
    data_path: str,
    subjects: List[str],
    test_subject: str,
    val_unseen_seed: int = 20250921,
    val_unseen_num: int = 200,
    per_channel_zscore: bool = False,
) -> SplitFold:
    # Candidates = all subjects except test
    candidates = [s for s in subjects if s != test_subject]
    if len(candidates) != 9:
        raise ValueError(f"Expected 9 candidates, got {len(candidates)}. subjects={subjects}, test={test_subject}")

    # Compute centers for candidates
    centers = []
    for s in candidates:
        arr = _load_training_array(data_path, s)
        c = compute_subject_center(arr, per_channel_zscore)
        centers.append(c)
    X = np.stack(centers, axis=0)  # (9, d)

    # 9×9 distance matrix
    D = pairwise_euclidean(X)

    # Select dev-pack
    easy, medium, hard = select_dev_pack(candidates, D)
    dev_subjects = [easy, medium, hard]
    train_subjects = [s for s in candidates if s not in dev_subjects]

    # Compute µ(s) for reporting
    mu = {}
    for i, s in enumerate(candidates):
        mu[s] = float((np.sum(D[i]) - D[i, i]) / (len(candidates) - 1))

    # Sample val-unseen classes
    val_unseen = sample_val_unseen(val_unseen_seed, val_unseen_num, total_train_classes=1654)

    notes = (
        "Fold-specific 9x9 subject distance computed using per-subject training centers; "
        "dev-pack selected as [easy(min µ), medium(closest to median µ), hard(max µ)]."
    )

    return SplitFold(
        test_subject=test_subject,
        dev_subjects=dev_subjects,
        train_subjects=train_subjects,
        val_unseen_classes=val_unseen,
        seeds={"class": val_unseen_seed},
        notes=notes,
        mu_scores=mu,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate LOSO nested validation splits (dev-pack + val-unseen)")
    parser.add_argument('--data_path', type=str, required=True, help='Path to EEG dataset root (contains sub-XX folders)')
    parser.add_argument('--subjects', nargs='+', default=None, help='Optional explicit subject list (default: scan data_path)')
    parser.add_argument('--val_unseen_seed', type=int, default=20250921, help='Seed for sampling val-unseen classes')
    parser.add_argument('--val_unseen_num', type=int, default=200, help='Number of val-unseen classes (default 200)')
    parser.add_argument('--per_channel_zscore', action='store_true', help='Enable per-subject per-channel z-score before averaging')
    parser.add_argument('--output_dir', type=str, default='splits', help='Directory to save split JSON files')
    parser.add_argument('--fold_subject', type=str, default=None, help='Generate only for this test subject (e.g., sub-06)')

    args = parser.parse_args()

    data_path = args.data_path
    subjects = args.subjects or list_subjects(data_path)
    if len(subjects) != 10:
        print(f"[Warn] Expected 10 subjects, got {len(subjects)}: {subjects}")

    os.makedirs(args.output_dir, exist_ok=True)

    fold_subjects = [args.fold_subject] if args.fold_subject else subjects
    for test_subject in fold_subjects:
        if test_subject not in subjects:
            print(f"[Skip] test_subject {test_subject} not in subjects list")
            continue

        split = generate_fold_split(
            data_path=data_path,
            subjects=subjects,
            test_subject=test_subject,
            val_unseen_seed=args.val_unseen_seed,
            val_unseen_num=args.val_unseen_num,
            per_channel_zscore=args.per_channel_zscore,
        )

        out_path = os.path.join(args.output_dir, f"outer_{test_subject}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(split), f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote split: {out_path}")


if __name__ == '__main__':
    main()
