#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate per-fold results across subjects.

This script scans the output directory structure produced by run_sattc_loso.py
and summarizes key metrics (Top-1, Top-5, R@1/5/10, etc.) across subjects.

Expected layout (per subject, per run timestamp):
  <output_dir>/<encoder_type>/<sub-XX>/<MM-DD_HH-MM>/final_summary.json
  <output_dir>/<encoder_type>/<sub-XX>/<MM-DD_HH-MM>/tune_summary.json (optional, for auto mode)

Usage examples:
  python fold_aggregate.py \
    --root outputs/iTransformer \
    --stage final \
    --pattern final_summary.json \
    --out_csv outputs/aggregate_final.csv

Notes
- If multiple runs exist per subject (multiple timestamps), the latest timestamp
  is used by default. You can switch to "best" selection by Top-5 if desired.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def _parse_timestamp(name: str) -> Tuple[int, int, int, int]:
    """Parse timestamp folder name 'MM-DD_HH-MM' into sortable tuple.
    Returns (MM, DD, HH, mm) or (0,0,0,0) if parsing fails.
    """
    try:
        mmdd, hhmm = name.split("_")
        mm, dd = mmdd.split("-")
        hh, mi = hhmm.split("-")
        return (int(mm), int(dd), int(hh), int(mi))
    except Exception:
        return (0, 0, 0, 0)


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def pick_latest_run(subject_dir: str) -> Optional[str]:
    if not os.path.isdir(subject_dir):
        return None
    candidates = [d for d in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, d))]
    if not candidates:
        return None
    candidates.sort(key=_parse_timestamp)
    return os.path.join(subject_dir, candidates[-1])


def gather_subjects(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Root not found: {root}")
    subs = [d for d in os.listdir(root) if d.startswith('sub-') and os.path.isdir(os.path.join(root, d))]
    subs.sort()
    return subs


def main():
    ap = argparse.ArgumentParser(description="Aggregate per-fold results across subjects")
    ap.add_argument('--root', required=True, help='Root dir: <output_dir>/<encoder_type> (contains sub-XX)')
    ap.add_argument('--stage', choices=['final', 'tune'], default='final', help='Which stage summaries to aggregate')
    ap.add_argument('--pattern', default=None, help='Override JSON filename to read (default: final_summary.json or tune_summary.json)')
    ap.add_argument('--out_csv', default=None, help='Optional CSV path to write aggregate table')
    args = ap.parse_args()

    json_name = args.pattern or (f"{args.stage}_summary.json")
    subjects = gather_subjects(args.root)

    rows: List[Dict[str, Any]] = []
    for sub in subjects:
        sub_dir = os.path.join(args.root, sub)
        latest = pick_latest_run(sub_dir)
        if latest is None:
            continue
        jpath = os.path.join(latest, json_name)
        data = _load_json(jpath)
        if not data:
            continue
        row = {
            'subject': sub,
            'best_epoch': data.get('best_epoch'),
            'top1': data.get('test_top1') or data.get('test_accuracy') or data.get('R@1'),
            'top5': data.get('test_top5') or data.get('R@5'),
            'R@1': data.get('R@1'),
            'R@5': data.get('R@5'),
            'R@10': data.get('R@10'),
            'MRR': data.get('MRR'),
            'mAP@200': data.get('mAP@200'),
            'MedR': data.get('MedR'),
            'NDCG@5': data.get('NDCG@5'),
            'NDCG@10': data.get('NDCG@10'),
        }
        rows.append(row)

    # Print aggregate
    def _safe(vals: List[Optional[float]]) -> List[float]:
        return [float(v) for v in vals if isinstance(v, (int, float))]

    import math
    def _mean_std(vals: List[float]) -> Tuple[float, float]:
        if not vals:
            return float('nan'), float('nan')
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / len(vals)
        return m, math.sqrt(v)

    top1_vals = _safe([r.get('top1') for r in rows])
    top5_vals = _safe([r.get('top5') for r in rows])
    r1_vals = _safe([r.get('R@1') for r in rows])
    r5_vals = _safe([r.get('R@5') for r in rows])
    r10_vals = _safe([r.get('R@10') for r in rows])

    m1, s1 = _mean_std(top1_vals)
    m5, s5 = _mean_std(top5_vals)
    mr1, sr1 = _mean_std(r1_vals)
    mr5, sr5 = _mean_std(r5_vals)
    mr10, sr10 = _mean_std(r10_vals)

    print(f"Subjects covered: {len(rows)}")
    print(f"Top-1:  mean={m1:.4f}  std={s1:.4f}  (n={len(top1_vals)})")
    print(f"Top-5:  mean={m5:.4f}  std={s5:.4f}  (n={len(top5_vals)})")
    print(f"R@1:    mean={mr1:.4f}  std={sr1:.4f}  (n={len(r1_vals)})")
    print(f"R@5:    mean={mr5:.4f}  std={sr5:.4f}  (n={len(r5_vals)})")
    print(f"R@10:   mean={mr10:.4f} std={sr10:.4f} (n={len(r10_vals)})")

    if args.out_csv:
        fieldnames = list(rows[0].keys()) if rows else ['subject']
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"CSV written: {args.out_csv}")


if __name__ == '__main__':
    main()
