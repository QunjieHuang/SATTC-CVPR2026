"""Evaluation runner for offline scoring variants."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch

from metrics import eval_all_metrics
from scoring import csls_adaptive, csls_fixed


def forward_and_cache(model, dataloader, device, img_features, saw_handler, cache_path: Path) -> Path:
    """Run a forward pass, apply SAW, and cache similarities and labels."""
    model.eval()
    eeg_features = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 7:
                eeg_data, batch_labels, *_ = batch
            else:
                eeg_data, batch_labels = batch[0], batch[1]
            eeg_data = eeg_data.to(device)
            subject_ids = torch.zeros(eeg_data.size(0), dtype=torch.long, device=device)
            enc = model(eeg_data, subject_ids)
            eeg_features.append(enc.cpu())
            labels.append(batch_labels.cpu())
    Q = torch.cat(eeg_features, dim=0)
    labels_all = torch.cat(labels, dim=0)
    if saw_handler is not None:
        Q = saw_handler(Q)
    img_tensor = img_features.cpu()
    similarities = Q @ img_tensor.T
    payload = {
        "similarities": similarities,
        "labels": labels_all,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return cache_path


def eval_variants(cache_path: Path, device: torch.device, tune_cfg: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Evaluate no-CSLS, fixed-k CSLS, and Ada-CSLS variants."""
    blob = torch.load(cache_path)
    similarities = blob["similarities"].to(device)
    labels = blob["labels"].to(device)

    results = {
        "no_csls": eval_all_metrics(similarities, labels),
    }

    k_fixed = int(tune_cfg.get("k_fixed", tune_cfg.get("k0", 10)))
    fixed_scores = csls_fixed(similarities, k=k_fixed)
    results["csls_fixed"] = eval_all_metrics(fixed_scores, labels)

    kmin_val = int(tune_cfg.get("kmin", max(1, k_fixed // 2)))
    kmax_val = int(tune_cfg.get("kmax", max(k_fixed, kmin_val)))
    k_side_raw = tune_cfg.get("k_side")
    k_side_val = None if k_side_raw is None else int(k_side_raw)
    ada_scores, _, _, _ = csls_adaptive(
        similarities,
        k0=int(tune_cfg.get("k0", k_fixed)),
        kmin=kmin_val,
        kmax=kmax_val,
        alpha=float(tune_cfg.get("alpha", 1.0)),
        m=int(tune_cfg.get("m", 10)),
        k_side=k_side_val,
    )
    results["csls_ada"] = eval_all_metrics(ada_scores, labels)

    return results
