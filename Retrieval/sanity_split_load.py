#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick sanity check for split-driven tune pipeline:
- Reads split JSON
- Builds train (6×1454 classes) and val (3×200 classes, train-gallery) datasets
- Iterates 1-2 batches and prints shapes and class counts
"""
import argparse
import json
from torch.utils.data import DataLoader
from eegdatasets_leaveone import EEGDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--split_file', required=True)
    ap.add_argument('--batch_size', type=int, default=8)
    args = ap.parse_args()

    with open(args.split_file, 'r', encoding='utf-8') as f:
        split = json.load(f)

    test_subject = split['test_subject']
    dev_subjects = split['dev_subjects']
    train_subjects = split['train_subjects']
    val_unseen = split['val_unseen_classes']

    all_train_classes = list(range(1654))
    train_classes = [c for c in all_train_classes if c not in set(val_unseen)]

    print(f"Fold: test={test_subject}")
    print(f" Train subjects (6): {train_subjects}")
    print(f" Dev-pack subjects (3): {dev_subjects}")
    print(f" Classes: train={len(train_classes)}, val_unseen={len(val_unseen)}")

    # Build datasets
    train_ds = EEGDataset(args.data_path, subjects=train_subjects, train=True, classes=train_classes)
    val_ds = EEGDataset(args.data_path, subjects=dev_subjects, train=True, classes=val_unseen)

    print(f" Samples: train={len(train_ds)}, val={len(val_ds)}")
    print(f" Prototypes: train C={train_ds.img_proto.size(0)}, val C={val_ds.img_proto.size(0)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Take one batch from train
    xb, yb, tb, tfb, ib, ifb = next(iter(train_loader))
    print(f" Train batch: x={xb.shape}, y=[{yb.min().item()}, {yb.max().item()}], tf={tfb.shape}, if={ifb.shape}")

    # Take two samples from val
    xv, yv, tv, tfv, iv, ifv = next(iter(val_loader))
    print(f" Val sample: x={xv.shape}, y={yv.item()}, tf={tfv.shape}, if={ifv.shape}")

    print(" Sanity passed: split-driven datasets can be built and iterated.")


if __name__ == '__main__':
    main()
