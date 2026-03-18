"""Metric computation utilities for retrieval evaluation."""
from __future__ import annotations

from typing import Dict, Iterable

import torch


def compute_topk(similarities: torch.Tensor, labels: torch.Tensor, ks: Iterable[int]) -> Dict[int, Dict[str, float]]:
    """Compute top-k accuracy metrics for the provided similarity scores."""
    if similarities.dim() != 2:
        raise ValueError("Similarities must be a 2D tensor [N, C]")
    if similarities.shape[0] != labels.numel():
        raise ValueError("Number of similarity rows must match number of labels")

    device = similarities.device
    labels = labels.to(device)
    num_samples, num_classes = similarities.shape

    unique_ks = sorted(set(int(k) for k in ks))
    metrics: Dict[int, Dict[str, float]] = {}
    for k in unique_ks:
        if k <= 0:
            raise ValueError("k must be positive")
        k_eff = min(k, num_classes)
        topk_indices = similarities.topk(k_eff, dim=1).indices
        correct_any = (topk_indices == labels.unsqueeze(1)).any(dim=1)
        top1_correct = (topk_indices[:, 0] == labels).float()
        metrics[k] = {
            "top1": top1_correct.mean().item(),
            "topk": correct_any.float().mean().item(),
        }
    return metrics


def eval_all_metrics(similarities: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute canonical metrics and return flattened results."""
    ks = [1, 2, 4, 5, 10, 50, 100, similarities.shape[1]]
    topk_stats = compute_topk(similarities, labels, ks)

    flattened: Dict[str, float] = {}
    for k, stats in topk_stats.items():
        flattened[f"top1@{k}"] = stats["top1"]
        flattened[f"topk@{k}"] = stats["topk"]
    return flattened
