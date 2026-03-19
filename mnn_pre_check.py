"""Mutual Nearest Neighbour (MNN) diagnostic utilities for retrieval training.

This module inspects similarity matrices produced during evaluation to identify
whether correctly retrieved queries (top-k hits) are also reciprocally close to
their ground-truth class when the similarity direction is reversed. The goal is
to flag cases where a query reaches its true class only because of one-sided
hubness without mutual support.

TEST: Modified via Copilot Claude from macOS via SSH to Windows dorm machine!

Modified via Copilot Claude - SSH test OK!`n`nModified via Copilot Claude from macOS SSH session - test successful!


"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch

from soft_mnn import estimate_class_popularity, percentile_from_rank, soft_mnn_bundle

try:  # Optional plotting support; the main stats still work without matplotlib.
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib not always available
    plt = None  # type: ignore


@dataclass
class MNNConfig:
    epoch: Optional[int] = None
    stage: Optional[str] = None
    run_prefix: Optional[str] = None
    top_k: int = 5
    output_dir: Optional[str] = None
    generate_plot: bool = True
    save_details: bool = True
    detail_limit: Optional[int] = None


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _format_epoch(epoch: Optional[int]) -> str:
    if epoch is None or epoch < 0:
        return "unknown"
    return f"{epoch:04d}"


def _sanitize_part(part: Optional[str]) -> Optional[str]:
    if not part:
        return None
    clean = str(part).strip().replace(os.sep, "_")
    return clean or None


def _ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _histogram_plot(values: torch.Tensor, save_path: str, title: str, top_k: int) -> Optional[str]:
    if plt is None or values.numel() == 0:
        return None
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = int(min(max(values.max().item(), top_k), 100))
        bins = max(bins, top_k)
        ax.hist(values.cpu().numpy(), bins=bins, color="#2878B5", alpha=0.85)
        ax.set_xlabel("Reverse rank (class �?query)")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.axvline(top_k + 0.5, color="#E24A33", linestyle="--", linewidth=1.0, label=f"top-{top_k} boundary")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(save_path, dpi=220)
        plt.close(fig)
        return save_path
    except Exception:
        return None


def run_mnn_precheck(
    sim_matrix: torch.Tensor,
    labels: torch.Tensor,
    *,
    config: Optional[MNNConfig] = None,
    stage: Optional[str] = None,
    epoch: Optional[int] = None,
    run_prefix: Optional[str] = None,
    output_dir: Optional[str] = None,
    top_k: int = 5,
    generate_plot: Optional[bool] = None,
    save_details: Optional[bool] = None,
    detail_limit: Optional[int] = None,
    subject_indices: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Analyse mutual nearest neighbour behaviour for correctly retrieved queries.

    Args:
        sim_matrix: Similarity or logit matrix of shape (num_queries, num_classes).
                    Higher scores should represent closer matches.
        labels: Ground-truth class indices for each query (shape: num_queries,).
        config: Optional :class:`MNNConfig` to provide defaults.
        stage: Training/eval stage name (e.g. ``"tune"`` or ``"final"``).
        epoch: Current epoch index (0-based).
        run_prefix: Additional folder prefix (e.g. subject identifier).
        output_dir: Root directory used to persist outputs.
        top_k: Hit definition threshold (default: 5).
        generate_plot: Whether to emit a histogram visualisation per epoch.
        save_details: Whether to persist per-query CSV rows for top-k hits.
        detail_limit: Optional cap on number of rows written to the detail CSV.
        subject_indices: Optional tensor mapping queries to subject IDs for logging.

    Returns:
        Dictionary containing aggregate diagnostics and emitted file paths.
    """

    if config is not None:
        stage = config.stage if stage is None else stage
        epoch = config.epoch if epoch is None else epoch
        run_prefix = config.run_prefix if run_prefix is None else run_prefix
        if output_dir is None:
            output_dir = config.output_dir
        top_k = config.top_k if top_k is None else top_k
        if generate_plot is None:
            generate_plot = config.generate_plot
        if save_details is None:
            save_details = config.save_details
        if detail_limit is None:
            detail_limit = config.detail_limit

    if generate_plot is None:
        generate_plot = True
    if save_details is None:
        save_details = True
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    if not isinstance(sim_matrix, torch.Tensor) or sim_matrix.dim() != 2:
        raise ValueError("sim_matrix must be a rank-2 tensor")
    if not isinstance(labels, torch.Tensor) or labels.dim() != 1:
        raise ValueError("labels must be a rank-1 tensor")
    if sim_matrix.size(0) != labels.size(0):
        raise ValueError("Number of queries in sim_matrix and labels must match")

    sim_cpu = sim_matrix.detach().cpu().to(torch.float32)
    labels_cpu = labels.detach().cpu().to(torch.long)
    num_queries, num_classes = sim_cpu.shape

    if num_queries == 0 or num_classes == 0:
        return {
            "enabled": False,
            "reason": "empty-similarity-matrix",
            "num_queries": int(num_queries),
            "num_classes": int(num_classes),
        }

    subject_cpu: Optional[torch.Tensor]
    if subject_indices is not None:
        subject_cpu = subject_indices.detach().cpu().to(torch.long)
        if subject_cpu.numel() != num_queries:
            subject_cpu = None
    else:
        subject_cpu = None

    # Normalise each row to mitigate per-query scale drift introduced by re-scoring.
    row_norms = torch.norm(sim_cpu, p=2, dim=1, keepdim=True)
    zero_mask = row_norms.squeeze(1) == 0
    row_norms[zero_mask] = 1.0
    sim_cpu = sim_cpu / row_norms

    soft_top_l = None
    soft_gamma = 0.3
    if config is not None:
        soft_top_l = getattr(config, 'mnn_soft_topl', None)
        soft_gamma = getattr(config, 'mnn_soft_gamma', soft_gamma)
    if soft_top_l is None or soft_top_l <= 0:
        soft_top_l = top_k
    try:
        soft_gamma = float(soft_gamma)
    except (TypeError, ValueError):
        soft_gamma = 0.3
    soft_gamma = max(0.0, min(0.95, soft_gamma))
    delta_tol = getattr(config, 'mnn_soft_delta_tol', 0.08) if config is not None else 0.08
    tau_delta = getattr(config, 'mnn_soft_tau_delta', 0.08) if config is not None else 0.08
    tau_f = getattr(config, 'mnn_soft_tau_f', 0.05) if config is not None else 0.05
    tau_r = getattr(config, 'mnn_soft_tau_r', 0.12) if config is not None else 0.12
    try:
        delta_tol = max(0.0, float(delta_tol))
    except (TypeError, ValueError):
        delta_tol = 0.08
    try:
        tau_delta = max(1e-6, float(tau_delta))
    except (TypeError, ValueError):
        tau_delta = 0.08
    try:
        tau_f = max(1e-6, float(tau_f))
    except (TypeError, ValueError):
        tau_f = 0.05
    try:
        tau_r = max(1e-6, float(tau_r))
    except (TypeError, ValueError):
        tau_r = 0.12

    # Compute forward rank (query �?class).
    negative_rows = -sim_cpu
    row_ranks = torch.argsort(torch.argsort(negative_rows, dim=1), dim=1) + 1
    forward_ranks = row_ranks.gather(1, labels_cpu.view(-1, 1)).squeeze(1)

    # Compute reverse rank (class �?query) for each query.
    reverse_ranks = torch.empty_like(forward_ranks)
    unique_labels = torch.unique(labels_cpu)
    for cls in unique_labels.tolist():
        col_scores = sim_cpu[:, cls]
        col_rank = torch.argsort(torch.argsort(-col_scores)) + 1
        mask = labels_cpu == cls
        reverse_ranks[mask] = col_rank[mask]

    forward_percentiles_all = percentile_from_rank(forward_ranks, num_classes)
    reverse_percentiles_all = percentile_from_rank(reverse_ranks, num_queries)
    popularity = estimate_class_popularity(sim_cpu, top_l=soft_top_l)

    if num_queries <= 1:
        soft_percentile_threshold = 0.0
    else:
        effective_topk = min(max(1, top_k), num_queries)
        soft_percentile_threshold = (effective_topk - 1) / float(num_queries - 1)

    top_k = int(top_k)
    topk_hits_mask = forward_ranks <= top_k
    top1_hits_mask = forward_ranks == 1

    topk_hit_indices = torch.nonzero(topk_hits_mask, as_tuple=False).view(-1)
    top1_hit_indices = torch.nonzero(top1_hits_mask, as_tuple=False).view(-1)

    reverse_for_hits = reverse_ranks[topk_hits_mask]
    forward_for_hits = forward_ranks[topk_hits_mask]

    forward_percentiles_hits = forward_percentiles_all[topk_hits_mask]
    reverse_percentiles_hits = reverse_percentiles_all[topk_hits_mask]
    labels_hits = labels_cpu[topk_hits_mask]
    soft_hits_bundle = soft_mnn_bundle(
        forward_percentiles_hits,
        reverse_percentiles_hits,
        labels_hits,
        popularity,
        gamma=soft_gamma,
        tau_f=tau_f,
        tau_r=tau_r,
        delta_tol=delta_tol,
        tau_delta=tau_delta,
    )
    reverse_percentiles_soft_hits = soft_hits_bundle["pi_r_soft"]
    w_forward_hits = soft_hits_bundle["w_forward"]
    w_reverse_hits = soft_hits_bundle["w_reverse"]
    w_tol_hits = soft_hits_bundle["w_tol"]
    w_mnn_hits = soft_hits_bundle["w_mnn"]
    delta_hits = soft_hits_bundle["delta"]
    class_pop_hits = torch.zeros_like(reverse_percentiles_hits)
    if popularity.numel() > 0 and labels_hits.numel() > 0:
        valid_label_mask = (labels_hits >= 0) & (labels_hits < popularity.numel())
        if valid_label_mask.any():
            class_pop_hits[valid_label_mask] = popularity[labels_hits[valid_label_mask]]

    if top1_hits_mask.any():
        reverse_percentiles_top1 = reverse_percentiles_all[top1_hits_mask]
        labels_top1 = labels_cpu[top1_hits_mask]
        soft_top1_bundle = soft_mnn_bundle(
            forward_percentiles_all[top1_hits_mask],
            reverse_percentiles_top1,
            labels_top1,
            popularity,
            gamma=soft_gamma,
            tau_f=tau_f,
            tau_r=tau_r,
            delta_tol=delta_tol,
            tau_delta=tau_delta,
        )
        reverse_percentiles_soft_top1 = soft_top1_bundle["pi_r_soft"]
        w_mnn_top1 = soft_top1_bundle["w_mnn"]
        w_tol_top1 = soft_top1_bundle["w_tol"]
        delta_top1 = soft_top1_bundle["delta"]
    else:
        reverse_percentiles_soft_top1 = torch.empty(0, dtype=torch.float32)
        w_mnn_top1 = torch.empty(0, dtype=torch.float32)
        w_tol_top1 = torch.empty(0, dtype=torch.float32)
        delta_top1 = torch.empty(0, dtype=torch.float32)

    mutual_topk_mask = reverse_for_hits <= top_k
    mutual_top1_mask = reverse_ranks[top1_hits_mask] <= top_k if top1_hit_indices.numel() > 0 else torch.tensor([], dtype=torch.bool)
    strict_mnn_mask = reverse_for_hits == 1
    strict_top1_mask = reverse_ranks[top1_hits_mask] == 1 if top1_hit_indices.numel() > 0 else torch.tensor([], dtype=torch.bool)

    soft_mutual_mask = reverse_percentiles_soft_hits <= soft_percentile_threshold
    if reverse_percentiles_soft_top1.numel() > 0:
        soft_mutual_top1_mask = reverse_percentiles_soft_top1 <= soft_percentile_threshold
    else:
        soft_mutual_top1_mask = torch.empty(0, dtype=torch.bool)

    topk_hits = int(topk_hit_indices.numel())
    top1_hits = int(top1_hit_indices.numel())
    mutual_topk = int(mutual_topk_mask.sum().item())
    mutual_top1 = int(mutual_top1_mask.sum().item()) if mutual_top1_mask.numel() > 0 else 0
    strict_mutual_topk = int(strict_mnn_mask.sum().item())
    strict_mutual_top1 = int(strict_top1_mask.sum().item()) if strict_top1_mask.numel() > 0 else 0
    soft_mutual_topk = int(soft_mutual_mask.sum().item())
    soft_mutual_top1 = int(soft_mutual_top1_mask.sum().item()) if soft_mutual_top1_mask.numel() > 0 else 0

    reverse_hits_float = reverse_for_hits.to(torch.float32)
    forward_hits_float = forward_for_hits.to(torch.float32)

    soft_mean_w_mnn = float(w_mnn_hits.mean().item()) if w_mnn_hits.numel() > 0 else None
    soft_mean_w_tol = float(w_tol_hits.mean().item()) if w_tol_hits.numel() > 0 else None
    soft_mean_delta = float(delta_hits.mean().item()) if delta_hits.numel() > 0 else None
    soft_mean_w_forward = float(w_forward_hits.mean().item()) if w_forward_hits.numel() > 0 else None
    soft_mean_w_reverse = float(w_reverse_hits.mean().item()) if w_reverse_hits.numel() > 0 else None

    def _safe_quantiles(values: torch.Tensor, probs: Iterable[float]) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        if values.numel() == 0:
            return stats
        try:
            for p in probs:
                stats[f"q{int(p*100):02d}"] = float(torch.quantile(values, p).item())
        except Exception:
            # torch.quantile not available; fallback to numpy via tensor sorting.
            sorted_vals = torch.sort(values).values
            for p in probs:
                idx = int((len(sorted_vals) - 1) * p)
                stats[f"q{int(p*100):02d}"] = float(sorted_vals[idx].item())
        return stats

    reverse_stats = {
        "mean": float(reverse_hits_float.mean().item()) if reverse_hits_float.numel() else None,
        "max": float(reverse_hits_float.max().item()) if reverse_hits_float.numel() else None,
        "min": float(reverse_hits_float.min().item()) if reverse_hits_float.numel() else None,
        **_safe_quantiles(reverse_hits_float, [0.25, 0.5, 0.75, 0.9]),
    }
    forward_stats = {
        "mean": float(forward_hits_float.mean().item()) if forward_hits_float.numel() else None,
        "max": float(forward_hits_float.max().item()) if forward_hits_float.numel() else None,
        "min": float(forward_hits_float.min().item()) if forward_hits_float.numel() else None,
    }

    epoch_str = _format_epoch(epoch)
    stage_str = _sanitize_part(stage)
    prefix_str = _sanitize_part(run_prefix)

    root_dir = output_dir or os.path.join("./outputs", "mnn_precheck")
    path_parts = [root_dir]
    if stage_str:
        path_parts.append(stage_str)
    if prefix_str:
        path_parts.append(prefix_str)
    out_dir = os.path.join(*path_parts)
    _ensure_directory(out_dir)

    summary_path = os.path.join(out_dir, "summary.csv")
    detail_path = None
    hist_path = None

    # Append to summary CSV.
    summary_header = [
        "epoch",
        "num_queries",
        "num_classes",
        "topk_hits",
        "top1_hits",
        "mutual_topk",
        "mutual_top1",
        "strict_mutual_topk",
        "strict_mutual_top1",
        "soft_mutual_topk",
        "soft_mutual_top1",
        "mutual_topk_ratio",
        "mutual_top1_ratio",
        "soft_mutual_topk_ratio",
        "soft_mutual_top1_ratio",
        "soft_percentile_threshold",
        "soft_gamma",
        "soft_top_l",
    "soft_mean_w_forward",
    "soft_mean_w_reverse",
        "soft_mean_w_mnn",
        "soft_mean_w_tol",
        "soft_mean_delta",
        "mean_reverse_rank",
        "median_reverse_rank",
        "p90_reverse_rank",
    ]
    median_val = reverse_stats.get("q50")
    p90_val = reverse_stats.get("q90")
    summary_row = [
        epoch_str,
        int(num_queries),
        int(num_classes),
        topk_hits,
        top1_hits,
        mutual_topk,
        mutual_top1,
        strict_mutual_topk,
        strict_mutual_top1,
        soft_mutual_topk,
        soft_mutual_top1,
        float(mutual_topk / max(1, topk_hits)),
        float(mutual_top1 / max(1, top1_hits)) if top1_hits > 0 else 0.0,
        float(soft_mutual_topk / max(1, topk_hits)),
        float(soft_mutual_top1 / max(1, top1_hits)) if top1_hits > 0 else 0.0,
        float(soft_percentile_threshold),
        float(soft_gamma),
        int(soft_top_l),
    soft_mean_w_forward if soft_mean_w_forward is not None else "",
    soft_mean_w_reverse if soft_mean_w_reverse is not None else "",
        soft_mean_w_mnn if soft_mean_w_mnn is not None else "",
        soft_mean_w_tol if soft_mean_w_tol is not None else "",
        soft_mean_delta if soft_mean_delta is not None else "",
        reverse_stats.get("mean"),
        median_val,
        p90_val,
    ]

    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as fp:
        writer = csv.writer(fp)
        if write_header:
            writer.writerow(summary_header)
        writer.writerow(summary_row)

    # Emit per-hit CSV when requested.
    if save_details and topk_hit_indices.numel() > 0:
        detail_path = os.path.join(out_dir, f"epoch_{epoch_str}_hits.csv")
        limit = detail_limit if detail_limit is None else max(int(detail_limit), 0)
        with open(detail_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            header = [
                "query_index",
                "subject_id",
                "label",
                "forward_rank",
                "reverse_rank",
                "topk_hit",
                "top1_hit",
                "mutual_topk",
                "mutual_top1",
                "strict_mutual",
                "pi_f",
                "pi_r",
                "pi_r_soft",
                "delta_pi",
                "class_popularity",
                "w_forward",
                "w_reverse",
                "w_tol",
                "w_mnn",
                "soft_mutual",
                "soft_mutual_top1",
            ]
            writer.writerow(header)
            rows_written = 0
            for (
                idx_tensor,
                f_rank,
                r_rank,
                pi_f_val,
                pi_r_val,
                pi_r_soft_val,
                delta_val,
                pop_val,
                w_forward_val,
                w_reverse_val,
                w_tol_val,
                w_mnn_val,
                soft_flag,
            ) in zip(
                topk_hit_indices,
                forward_for_hits.tolist(),
                reverse_for_hits.tolist(),
                forward_percentiles_hits.tolist(),
                reverse_percentiles_hits.tolist(),
                reverse_percentiles_soft_hits.tolist(),
                delta_hits.tolist(),
                class_pop_hits.tolist(),
                w_forward_hits.tolist(),
                w_reverse_hits.tolist(),
                w_tol_hits.tolist(),
                w_mnn_hits.tolist(),
                soft_mutual_mask.tolist(),
            ):
                idx_int = _safe_int(idx_tensor.item(), default=-1)
                subj_val = None
                if subject_cpu is not None and 0 <= idx_int < subject_cpu.numel():
                    subj_val = _safe_int(subject_cpu[idx_int].item(), default=-1)
                mutual_flag = 1 if r_rank <= top_k else 0
                strict_flag = 1 if r_rank == 1 else 0
                top1_flag = 1 if f_rank == 1 else 0
                reverse_for_top1 = 1 if top1_flag and r_rank <= top_k else 0
                soft_flag_bool = bool(soft_flag)
                soft_flag_int = 1 if soft_flag_bool else 0
                soft_top1_flag = 1 if top1_flag and soft_flag_bool else 0
                writer.writerow([
                    idx_int,
                    subj_val if subj_val is not None else "",
                    _safe_int(labels_cpu[idx_int].item(), default=-1) if idx_int >= 0 else "",
                    int(f_rank),
                    int(r_rank),
                    1,
                    top1_flag,
                    mutual_flag,
                    reverse_for_top1,
                    strict_flag,
                    float(pi_f_val),
                    float(pi_r_val),
                    float(pi_r_soft_val),
                    float(delta_val),
                    float(pop_val),
                    float(w_forward_val),
                    float(w_reverse_val),
                    float(w_tol_val),
                    float(w_mnn_val),
                    soft_flag_int,
                    soft_top1_flag,
                ])
                rows_written += 1
                if limit and rows_written >= limit:
                    break

    # Optional histogram visualisation.
    if generate_plot and reverse_hits_float.numel() > 0:
        hist_name = f"epoch_{epoch_str}_reverse_rank_hist.png"
        hist_path = _histogram_plot(
            reverse_hits_float,
            os.path.join(out_dir, hist_name),
            title=f"Reverse-rank distribution @ epoch {epoch_str}",
            top_k=top_k,
        )

    result = {
        "enabled": True,
        "epoch": epoch,
        "stage": stage,
        "output_dir": out_dir,
        "summary_path": summary_path,
        "detail_path": detail_path,
        "hist_path": hist_path,
        "topk_hits": topk_hits,
        "top1_hits": top1_hits,
        "mutual_topk": mutual_topk,
        "mutual_top1": mutual_top1,
        "strict_mutual_topk": strict_mutual_topk,
        "strict_mutual_top1": strict_mutual_top1,
        "soft_mutual_topk": soft_mutual_topk,
        "soft_mutual_top1": soft_mutual_top1,
        "mutual_topk_ratio": float(mutual_topk / max(1, topk_hits)),
        "mutual_top1_ratio": float(mutual_top1 / max(1, top1_hits)) if top1_hits > 0 else 0.0,
        "soft_mutual_topk_ratio": float(soft_mutual_topk / max(1, topk_hits)),
        "soft_mutual_top1_ratio": float(soft_mutual_top1 / max(1, top1_hits)) if top1_hits > 0 else 0.0,
        "soft_percentile_threshold": float(soft_percentile_threshold),
        "soft_gamma": float(soft_gamma),
        "soft_top_l": int(soft_top_l),
    "soft_mean_w_forward": soft_mean_w_forward,
    "soft_mean_w_reverse": soft_mean_w_reverse,
        "soft_mean_w_mnn": soft_mean_w_mnn,
        "soft_mean_w_tol": soft_mean_w_tol,
        "soft_mean_delta": soft_mean_delta,
        "reverse_rank_stats": reverse_stats,
        "forward_rank_stats": forward_stats,
    }
    if detail_limit is not None:
        result["detail_limit"] = int(detail_limit)

    return result
