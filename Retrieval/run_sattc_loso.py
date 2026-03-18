# =================================================================================
# 2025.10.17 v1.5版本已经完成了严谨的可对比的baseline（可选择cos和dot评估）

# 跑完三个seed：42、3407、20251的结果
# dot评价指标会更好看，但是同时也引入了范数的影响，指标偏向于乐观，不视为逻辑错误但仍有质疑的点
# 优化了脚本的运行速度
# 保证了严谨地对比基线，评估、验证划分的方法都是严谨的
# =================================================================================
# 开启v2版本
# 1、先将subject_ids修改为“真实训练样本所属被试”，提高建模的严谨性


# =================================================================================
# eegdatasets_leaveone里修改了加载本地openclip（.pt）文件进行计算

import os
import gc
import time
import inspect

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

os.environ.setdefault("WANDB_MODE", "offline")
from itertools import combinations

# import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from eegdatasets_leaveone import EEGDataset

from einops.layers.torch import Rearrange, Reduce

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from util import wandb_logger
from utils.cli_args import (
    normalize_csls_args,
    normalize_structural_args,
    register_ada_csls_args,
    register_structural_args,
    register_eeg_encoder_arg,
    get_ada_csls_defaults,
)
from mnn_pre_check import run_mnn_precheck
import csv
from torch import Tensor
import itertools
import copy
import json
import math
import re
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

try:
    from braindecode.models import EEGNetv4, ShallowFBCSPNet  # type: ignore
except (ImportError, ModuleNotFoundError):
    try:
        from braindecode.models.eegnet import EEGNetv4  # type: ignore
        from braindecode.models.shallow_fbcsp import ShallowFBCSPNet  # type: ignore
    except (ImportError, ModuleNotFoundError):
        EEGNetv4 = None  # type: ignore
        ShallowFBCSPNet = None  # type: ignore

try:
    from braindecode.models import EEGConformer  # type: ignore
except (ImportError, ModuleNotFoundError):
    try:
        from braindecode.models.eegconformer import EEGConformer  # type: ignore
    except (ImportError, ModuleNotFoundError):
        EEGConformer = None  # type: ignore

ADA_CSLS_DEFAULTS = get_ada_csls_defaults()


def _ada_required(config, key):
    """Resolve required Ada-CSLS knob, falling back to canonical defaults."""
    value = getattr(config, key, None)
    if value is None:
        value = ADA_CSLS_DEFAULTS[key]
    return value


def _ada_required_int(config, key):
    """Resolve required integer knob with canonical fallback."""
    value = _ada_required(config, key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(ADA_CSLS_DEFAULTS[key])


def _ada_required_float(config, key):
    """Resolve required float knob with canonical fallback."""
    value = _ada_required(config, key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(ADA_CSLS_DEFAULTS[key])


def _ada_optional_int(config, key):
    """Resolve optional integer knob that may be explicitly disabled via None."""
    value = getattr(config, key, ADA_CSLS_DEFAULTS[key])
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(ADA_CSLS_DEFAULTS[key])


def _ada_optional_float(config, key):
    """Resolve optional float knob that may be explicitly disabled via None."""
    value = getattr(config, key, ADA_CSLS_DEFAULTS[key])
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(ADA_CSLS_DEFAULTS[key])


def _get_optional_float(config, key):
    """Safely fetch an optional float attribute without default fallback."""
    value = getattr(config, key, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_optional_int(config, key):
    """Safely fetch an optional int attribute without default fallback."""
    value = getattr(config, key, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_metric_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def format_duration_hm(seconds):
    """Format duration in seconds into a string like 'Xh Ym'."""
    if seconds is None:
        return "0h 0m"
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes}m"


def update_summary_duration(path, duration_str, duration_seconds):
    """Attach total duration information into a JSON summary file when it exists."""
    if not path:
        return
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            data = {}
    except Exception:
        data = {}
    data['total_duration'] = duration_str
    data['total_duration_seconds'] = round(float(duration_seconds), 2)
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


_GLOBAL_STATS_CACHE: Dict[str, Dict[str, Any]] = {}


def _to_tensor(value, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Best-effort conversion of JSON/NumPy/Tensor inputs into torch tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=dtype)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.astype(np.float32)).to(dtype)
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=dtype)
    if isinstance(value, (float, int)):
        return torch.tensor([value], dtype=dtype)
    raise TypeError(f"Unsupported stats payload type: {type(value)!r}")


def _load_stats_mapping(path: str) -> Dict[str, Any]:
    """Load statistics dictionaries with caching to minimise disk overhead."""
    if not path:
        return {}
    cached = _GLOBAL_STATS_CACHE.get(path)
    if cached is not None:
        return cached
    ext = os.path.splitext(path)[1].lower()
    data: Any
    try:
        if ext in {'.pt', '.pth'}:
            data = torch.load(path, map_location='cpu')
        elif ext == '.npz':
            npz_file = np.load(path)
            data = {key: npz_file[key] for key in npz_file.files}
        elif ext in {'.json', '.js'}:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported stats file extension: {ext}")
    except Exception as err:
        raise RuntimeError(f"Failed to load stats from {path}: {err}") from err
    if not isinstance(data, dict):
        raise TypeError(f"Stats file {path} must contain a dict, got {type(data)!r}")
    _GLOBAL_STATS_CACHE[path] = data
    return data


def _resolve_whitening_state(config: Any, kind: str) -> Optional[Dict[str, torch.Tensor]]:
    """Resolve global whitening statistics for image/text branches."""
    cache_attr = f'_cache_whiten_{kind}'
    cached = getattr(config, cache_attr, None)
    if cached is not None:
        return cached
    path_attr = f'global_{kind}_whiten_stats'
    stats_path = getattr(config, path_attr, None)
    if not stats_path:
        setattr(config, cache_attr, None)
        return None
    try:
        stats_dict = _load_stats_mapping(stats_path)
        mean_val = None
        for key in ('mean', 'mu', 'center', 'centroid'):
            if key in stats_dict:
                mean_val = _to_tensor(stats_dict[key])
                break
        if mean_val is None:
            raise KeyError("missing mean/mu/center in whitening stats")
        transform_val = None
        for key in ('transform', 'matrix', 'whitening_matrix', 'inv_sqrt', 'projection'):
            if key in stats_dict:
                transform_val = _to_tensor(stats_dict[key])
                break
        if transform_val is None and 'cov' in stats_dict:
            cov_tensor = _to_tensor(stats_dict['cov']).to(torch.float32)
            transform_val = _eigh_inv_sqrt(cov_tensor)
        if transform_val is None:
            raise KeyError("missing transform/inv_sqrt/matrix in whitening stats")
        mean_flat = mean_val.reshape(-1).to(torch.float32)
        transform_mat = transform_val.to(torch.float32)
        if transform_mat.dim() != 2 or transform_mat.size(0) != transform_mat.size(1):
            raise ValueError("whitening transform must be square")
        if transform_mat.size(0) != mean_flat.numel():
            raise ValueError("whitening mean/transform dimension mismatch")
        state = {
            "mean": mean_flat.clone(),
            "transform": transform_mat.clone(),
            "_cache": {},  # device cache populated on demand
        }
    except Exception as err:
        print(f"[GlobalAlign] Warning: failed to load whitening stats ({kind}): {err}")
        state = None
    setattr(config, cache_attr, state)
    return state


def _whiten_features(features: torch.Tensor, state: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Apply cached whitening transform when available."""
    if state is None:
        return features
    cache = state.setdefault('_cache', {})
    key = (features.device, features.dtype)
    if key not in cache:
        mean = state['mean'].to(device=features.device, dtype=features.dtype)
        transform = state['transform'].to(device=features.device, dtype=features.dtype)
        cache[key] = (mean, transform)
    mean_tensor, transform_tensor = cache[key]
    centered = features - mean_tensor.unsqueeze(0)
    return centered @ transform_tensor.T


def _build_whitening_state_from_features(
    features: torch.Tensor,
    shrink: float = 0.05,
    diag: bool = False,
) -> Optional[Dict[str, torch.Tensor]]:
    """Compute whitening transform (ZCA) directly from feature statistics."""
    if features is None or features.numel() == 0:
        return None
    try:
        feats = features.detach().to(device='cpu', dtype=torch.float32)
        mu, cov = estimate_mu_cov(feats, shrink=shrink, diag=diag)
        inv_sqrt = _eigh_inv_sqrt(cov)
        state = {
            "mean": mu.reshape(-1).clone(),
            "transform": inv_sqrt.clone(),
            "_cache": {},
        }
        return state
    except Exception as err:
        print(f"[GlobalAlign] Warning: failed to build auto whitening state: {err}")
        return None


def _resolve_auto_whitening_state(
    config: Any,
    kind: str,
    features: torch.Tensor,
) -> Optional[Dict[str, torch.Tensor]]:
    """Auto-compute whitening state when no external stats are provided."""
    cache_attr = f'_cache_auto_whiten_{kind}'
    cached = getattr(config, cache_attr, None)
    if cached is not None:
        return cached
    shrink = float(getattr(config, 'global_whiten_auto_shrink', 0.05) or 0.0)
    diag_flag = bool(getattr(config, 'global_whiten_auto_diag', False))
    state = _build_whitening_state_from_features(features, shrink=shrink, diag=diag_flag)
    setattr(config, cache_attr, state)
    return state


def _assess_alignment_strict(summary: Dict[str, float]) -> bool:
    """Check whether logits satisfy strict alignment tolerances."""
    if not summary:
        return False
    required_keys = (
        "global_std",
        "global_mean",
        "row_mean_abs",
        "col_mean_abs",
        "row_mean_std",
        "col_mean_std",
    )
    for key in required_keys:
        value = summary.get(key)
        if value is None or not math.isfinite(value):
            return False
    scale = max(abs(summary["global_std"]), 1.0)
    mean_ok = abs(summary["global_mean"]) <= 0.05 * scale
    row_mean_ok = summary["row_mean_abs"] <= 0.05 * scale
    col_mean_ok = summary["col_mean_abs"] <= 0.05 * scale
    row_disp_ok = summary["row_mean_std"] <= 0.1 * scale
    col_disp_ok = summary["col_mean_std"] <= 0.1 * scale
    std_ok = 0.5 <= abs(summary["global_std"]) <= 2.0
    return mean_ok and row_mean_ok and col_mean_ok and row_disp_ok and col_disp_ok and std_ok


def _summarise_logits_alignment(logits: torch.Tensor) -> Dict[str, float]:
    """Compute diagnostics describing global/row/column alignment of logits."""
    if logits is None or logits.numel() == 0:
        return {}
    with torch.no_grad():
        matrix = logits.detach().to(device='cpu', dtype=torch.float32)
    global_mean = float(matrix.mean().item())
    global_std = float(matrix.std(unbiased=False).item())
    row_means = matrix.mean(dim=1)
    row_stds = matrix.std(dim=1, unbiased=False)
    row_norms = matrix.norm(dim=1)
    col_means = matrix.mean(dim=0)
    col_stds = matrix.std(dim=0, unbiased=False)
    summary = {
        "global_mean": global_mean,
        "global_std": global_std,
        "row_mean_abs": float(row_means.abs().mean().item()),
        "row_mean_std": float(row_means.std(unbiased=False).item()),
        "row_std_mean": float(row_stds.mean().item()),
        "row_std_std": float(row_stds.std(unbiased=False).item()),
        "row_norm_mean": float(row_norms.mean().item()),
        "row_norm_std": float(row_norms.std(unbiased=False).item()),
        "col_mean_abs": float(col_means.abs().mean().item()),
        "col_mean_std": float(col_means.std(unbiased=False).item()),
        "col_std_mean": float(col_stds.mean().item()),
        "col_std_std": float(col_stds.std(unbiased=False).item()),
    }
    summary["aligned_strict"] = _assess_alignment_strict(summary)
    return summary


def _resolve_logits_zscore(config: Any) -> Tuple[Optional[float], Optional[float]]:
    """Resolve global logits mean/std for z-score calibration."""
    cache_attr = '_cache_logits_zscore'
    cached = getattr(config, cache_attr, None)
    if cached is not None:
        return cached
    mean_val = getattr(config, 'logits_zscore_mean', None)
    std_val = getattr(config, 'logits_zscore_std', None)
    stats_path = getattr(config, 'logits_zscore_path', None)
    if stats_path:
        try:
            stats_dict = _load_stats_mapping(stats_path)
            if mean_val is None:
                for key in ('mean', 'mu', 'avg', 'offset'):
                    if key in stats_dict:
                        mean_val = float(np.asarray(stats_dict[key]).reshape(-1)[0])
                        break
            if std_val is None:
                for key in ('std', 'sigma', 'scale', 'stddev'):
                    if key in stats_dict:
                        std_val = float(np.asarray(stats_dict[key]).reshape(-1)[0])
                        break
        except Exception as err:
            print(f"[GlobalAlign] Warning: failed to load logits stats: {err}")

    mean_out = None if mean_val is None else float(mean_val)
    std_out = None if std_val is None else float(std_val)
    if std_out is not None and std_out <= 0:
        print("[GlobalAlign] Warning: logits std <= 0, skip z-score calibration.")
        std_out = None
    result = (mean_out, std_out)
    setattr(config, cache_attr, result)
    return result


def print_stage_summary(stage_name, subject, primary_metrics, auxiliary_metrics=None):
    """Pretty-print stage summary with primary and optional auxiliary metrics."""
    stage_title = stage_name.capitalize()
    subject_tag = subject if isinstance(subject, str) else str(subject)
    print(f"\n📊 {stage_title} Stage Summary ({subject_tag}):")
    for label, value in primary_metrics.items():
        print(f"   - {label}: {_format_metric_value(value)}")
    if auxiliary_metrics:
        print("   - Auxiliary Metrics:")
        for label, value in auxiliary_metrics.items():
            print(f"       • {label}: {_format_metric_value(value)}")


def _spearman_corr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """Lightweight Spearman correlation using rank approximation (ties handled by stable argsort)."""
    x = x.flatten().float()
    y = y.flatten().float()
    if x.numel() != y.numel() or x.numel() < 2:
        return float('nan')
    x_rank = torch.argsort(torch.argsort(x))
    y_rank = torch.argsort(torch.argsort(y))
    x_rank = x_rank.float()
    y_rank = y_rank.float()
    x_rank = x_rank - x_rank.mean()
    y_rank = y_rank - y_rank.mean()
    denom = torch.sqrt((x_rank.square().sum()) * (y_rank.square().sum()))
    if denom <= 0:
        return float('nan')
    return float((x_rank * y_rank).sum() / denom)

from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding
import numpy as np
from loss import ClipLoss
import argparse
from torch import nn
from torch.optim import AdamW


def clear_memory(*objects):
    """Release GPU caches and trigger GC; call cleanup() on provided objects when available."""
    for obj in objects:
        if not obj:
            continue
        try:
            if hasattr(obj, 'cleanup') and callable(obj.cleanup):
                obj.cleanup()
        except Exception:
            pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def force_ram_cleanup():
    """Force Python GC and clear CUDA allocator if available."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def shutdown_dataloader(loader):
    """Explicitly shut down DataLoader worker pools to release RAM."""
    if loader is None:
        return
    try:
        iterator_ref = getattr(loader, '_iterator', None)
        if iterator_ref is not None:
            shutdown = getattr(iterator_ref, '_shutdown_workers', None)
            if callable(shutdown):
                shutdown()
        try:
            setattr(loader, '_iterator', None)
        except Exception:
            pass
        direct_shutdown = getattr(loader, '_shutdown_workers', None)
        if callable(direct_shutdown):
            direct_shutdown()
    except Exception:
        pass
    gc.collect()


def _eigh_inv_sqrt(cov, eps=1e-6):
    """Compute inverse square root of covariance with numerical safety."""
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    return eigenvectors @ torch.diag(eigenvalues.pow(-0.5)) @ eigenvectors.T


def estimate_mu_cov(features, shrink=0.2, diag=False):
    """Estimate mean and Ledoit-Wolf style shrunk covariance."""
    mu = features.mean(dim=0, keepdim=True)
    centered = features - mu
    denom = max(1, features.size(0) - 1)
    cov = (centered.T @ centered) / denom
    if diag:
        cov = torch.diag(torch.diag(cov))
    dim = cov.size(0)
    trace_mean = torch.trace(cov) / dim
    identity = torch.eye(dim, device=cov.device, dtype=cov.dtype)
    cov = (1 - shrink) * cov + shrink * trace_mean * identity
    return mu, cov


def subject_adaptive_whiten(features, shrink=0.2, diag=False):
    """Subject-Adaptive Whitening followed by L2 normalisation."""
    if features.size(0) == 0:
        return features
    mu, cov = estimate_mu_cov(features, shrink=shrink, diag=diag)
    inv_sqrt = _eigh_inv_sqrt(cov)
    whitened = (features - mu) @ inv_sqrt.T
    return F.normalize(whitened, dim=1, eps=1e-12)


def csls_scores(similarities: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Apply CSLS重打分缓解hubness问题, 输入为[N,C]相似度矩阵."""
    if similarities.dim() != 2:
        raise ValueError("CSLS期望[N, C]二维相似度矩阵")
    n, c = similarities.size()
    if n == 0 or c == 0:
        return similarities
    k_eff = max(1, min(int(k), n, c))
    rx = similarities.topk(k_eff, dim=1).values.mean(dim=1, keepdim=True)
    ry = similarities.topk(k_eff, dim=0).values.mean(dim=0, keepdim=True)
    return 2 * similarities - rx - ry


@torch.no_grad()
def csls_adaptive(
    similarities: torch.Tensor,
    k0: int = 10,
    kmin: int = 5,
    kmax: int = 20,
    alpha: float = 1.0,
    m: int = 10,
    k_side: Optional[int] = None,
    col_alpha: Optional[float] = None,
    col_m: Optional[int] = None,
    col_kmin: Optional[int] = None,
    col_kmax: Optional[int] = None,
    *,
    return_details: bool = False,
):
    """Adaptive CSLS that only uses row/column density estimates to pick k."""

    if similarities.dim() != 2:
        raise ValueError("Ada-CSLS期望[N, C]二维相似度矩阵")

    n_q, n_c = similarities.shape
    device = similarities.device
    dtype = similarities.dtype

    if n_q == 0 or n_c == 0:
        empty_row_idx = torch.empty(n_q, dtype=torch.int64, device=device)
        empty_col_idx = torch.empty(n_c, dtype=torch.int64, device=device)
        empty_row_vals = torch.empty(n_q, dtype=dtype, device=device)
        empty_col_vals = torch.empty(n_c, dtype=dtype, device=device)
        details = {
            "rho_row": torch.empty(n_q, dtype=dtype, device=device),
            "rho_col": torch.empty(n_c, dtype=dtype, device=device),
            "scale_row": torch.empty(n_q, dtype=dtype, device=device),
            "scale_col": torch.empty(n_c, dtype=dtype, device=device),
            "k_row": empty_row_idx,
            "k_col": empty_col_idx,
            "rT": empty_row_vals,
            "rS": empty_col_vals,
        }
        if return_details:
            return similarities, details
        return similarities

    kmin = max(1, min(int(kmin), n_c))
    kmax = max(kmin, min(int(kmax), n_c))
    k0 = max(kmin, min(int(k0), kmax))
    m_eff = max(1, min(int(m), n_c))

    rho_row = similarities.topk(m_eff, dim=1).values.mean(dim=1)
    med_row = rho_row.median()
    med_row_safe = med_row + similarities.new_tensor(1e-9)
    scale_row = (rho_row / med_row_safe).clamp_min(1e-6).pow(float(alpha))

    k_row = (k0 * scale_row).round().clamp(kmin, kmax).to(torch.int64)

    rT = torch.empty(n_q, dtype=dtype, device=device)
    for k_val in torch.unique(k_row, sorted=True).tolist():
        k_int = max(1, int(k_val))
        mask = k_row == k_int
        if mask.any():
            topk_vals = similarities[mask].topk(k_int, dim=1).values
            rT[mask] = topk_vals.mean(dim=1)

    try:
        base_k_col = k_side if k_side is not None else k0
        base_k_col = int(base_k_col)
    except (TypeError, ValueError):
        base_k_col = k0
    base_k_col = max(1, min(base_k_col, n_q))

    col_alpha_eff = float(col_alpha) if col_alpha is not None else float(alpha)
    col_m_eff = col_m if col_m is not None else m_eff
    col_m_eff = max(1, min(int(col_m_eff), n_c))

    try:
        _, col_top_idx = similarities.topk(col_m_eff, dim=1, largest=True, sorted=False)
        hits = torch.zeros(n_c, dtype=dtype, device=device)
        ones = torch.ones_like(col_top_idx, dtype=dtype)
        hits.scatter_add_(0, col_top_idx.reshape(-1), ones.reshape(-1))
        rho_col = hits / max(1, n_q)
    except RuntimeError as err:
        print(f"[Ada-CSLS] 列密度统计失败，退化为均匀分布: {err}")
        rho_col = torch.ones(n_c, dtype=dtype, device=device)

    if rho_col.numel() > 0:
        try:
            med_col = torch.quantile(rho_col, 0.5)
        except (RuntimeError, AttributeError):
            med_col = rho_col.median()
    else:
        med_col = torch.tensor(1.0, dtype=dtype, device=device)
    med_col_safe = med_col + similarities.new_tensor(1e-9)
    scale_col = (rho_col / med_col_safe).clamp_min(1e-6).pow(col_alpha_eff)

    if col_kmin is None:
        col_kmin_eff = max(1, min(base_k_col // 2 if base_k_col > 1 else 1, n_q))
    else:
        col_kmin_eff = max(1, min(int(col_kmin), n_q))
    if col_kmax is None:
        col_kmax_eff = max(col_kmin_eff, min(base_k_col * 2, n_q))
    else:
        col_kmax_eff = max(col_kmin_eff, min(int(col_kmax), n_q))

    k_col = (base_k_col * scale_col).round().to(torch.int64)
    k_col = k_col.clamp(col_kmin_eff, col_kmax_eff)

    ks_col_max = int(k_col.max().item()) if k_col.numel() > 0 else 1
    ks_col_max = max(1, min(ks_col_max, n_q))
    topk_cols = similarities.topk(ks_col_max, dim=0).values
    cumsum_cols = topk_cols.cumsum(dim=0)
    gather_idx = (k_col - 1).clamp(0, ks_col_max - 1)
    gathered = cumsum_cols.gather(0, gather_idx.unsqueeze(0)).squeeze(0)
    denom_col = k_col.to(dtype).clamp(min=1)
    rS = gathered / denom_col
    rS_query = rS.unsqueeze(0).expand(n_q, -1)

    csls_sim = 2 * similarities - rT.unsqueeze(1) - rS_query

    details = {
        "rho_row": rho_row,
        "rho_col": rho_col,
        "scale_row": scale_row,
        "scale_col": scale_col,
        "k_row": k_row,
        "k_col": k_col,
        "rT": rT,
        "rS": rS,
    }

    try:
        k_row_min = int(k_row.min().item()) if k_row.numel() > 0 else -1
        k_row_max = int(k_row.max().item()) if k_row.numel() > 0 else -1
        k_row_mean = float(k_row.float().mean().item()) if k_row.numel() > 0 else -1.0
        k_col_min = int(k_col.min().item()) if k_col.numel() > 0 else -1
        k_col_max = int(k_col.max().item()) if k_col.numel() > 0 else -1
        k_col_mean = float(k_col.float().mean().item()) if k_col.numel() > 0 else -1.0
        print(
            f"[Ada-CSLS] k_row[min/mean/max]={k_row_min}/{k_row_mean:.2f}/{k_row_max} | "
            f"k_col[min/mean/max]={k_col_min}/{k_col_mean:.2f}/{k_col_max}"
        )
    except Exception as dbg_err:
        print(f"[Ada-CSLS] Debug print失败: {dbg_err}")

    if return_details:
        return csls_sim, details
    return csls_sim


def build_structural_expert(
    pre_csls_evidence: Dict[str, Any],
    S_geom: torch.Tensor,
    device: Optional[torch.device] = None,
    lambda_pen: Optional[float] = None,
    lambda_bonus: Optional[float] = None,
) -> torch.Tensor:
    """根据 pre-CSLS 证据构造结构 expert 的 logits 矩阵。

    规则：
    - 默认 S_struct=0；
    - penalty_scale∈[0,1] 控制惩罚强度，直接转换成负偏置；
    - 对可信 pair（锁定区、保护区）施加正偏置；
    - 其他区域保持 0。
    """

    if device is None:
        device = S_geom.device

    Q, C = S_geom.shape
    S_struct = torch.zeros_like(S_geom)

    def _to_dev(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        return torch.tensor(value, device=device)

    penalty_scale = _to_dev(pre_csls_evidence.get("penalty_scale"))
    lock_mask = _to_dev(pre_csls_evidence.get("lock_mask"))
    case2_mask = _to_dev(pre_csls_evidence.get("case2_mask"))
    case3_mask = _to_dev(pre_csls_evidence.get("case3_mask"))
    case4_mask = _to_dev(pre_csls_evidence.get("case4_mask"))
    is_mnn1_mask = _to_dev(pre_csls_evidence.get("is_mnn1"))

    # 根据 S_geom 的尺度自动设置 lambda 的默认值，确保 S_struct 的变化幅度与主链路 logits 可控
    # 目标：S_struct 的 std 大小在 S_geom.std() 的 20-50% 范围内；默认取 lambda_pen=0.5*std, lambda_bonus=0.25*std
    geom_std = float(S_geom.std().item()) if S_geom.numel() > 0 else 1.0
    geom_std = max(1e-6, geom_std)
    lambda_pen = float(lambda_pen) if lambda_pen is not None else (0.5 * geom_std)
    lambda_bonus = float(lambda_bonus) if lambda_bonus is not None else (0.25 * geom_std)

    # 负偏置：penalty_scale 直接转成惩罚强度
    try:
        if penalty_scale is not None:
            if lock_mask is not None:
                penalty_scale = penalty_scale.masked_fill(lock_mask, 0.0)
            penalty_scale = penalty_scale.clamp(min=0.0, max=1.0)
            S_struct = S_struct - penalty_scale * lambda_pen
    except Exception:
        pass

    # 正偏置：强保护区 -> 基础奖励；case2/3/4 -> 额外奖励
    try:
        bonus_accumulator = torch.zeros_like(S_struct)

        base_protect = 0.5 * lambda_bonus
        case4_bonus = 0.05 * lambda_bonus
        case3_bonus = 0.15 * lambda_bonus
        case2_bonus = 0.20 * lambda_bonus
        case1_bonus = 0.25 * lambda_bonus

        if lock_mask is not None and base_protect != 0.0:
            bonus_accumulator = bonus_accumulator + lock_mask.to(dtype=S_struct.dtype) * base_protect

        if case4_mask is not None and case4_bonus != 0.0:
            bonus_accumulator = bonus_accumulator + case4_mask.to(dtype=S_struct.dtype) * case4_bonus

        if case3_mask is not None and case3_bonus != 0.0:
            bonus_accumulator = bonus_accumulator + case3_mask.to(dtype=S_struct.dtype) * case3_bonus

        if case2_mask is not None and case2_bonus != 0.0:
            bonus_accumulator = bonus_accumulator + case2_mask.to(dtype=S_struct.dtype) * case2_bonus

        if is_mnn1_mask is not None and case1_bonus != 0.0:
            bonus_accumulator = bonus_accumulator + is_mnn1_mask.to(dtype=S_struct.dtype) * case1_bonus

        S_struct = S_struct + bonus_accumulator
        if lock_mask is not None:
            non_negative = torch.clamp(S_struct, min=0.0)
            S_struct = torch.where(lock_mask, non_negative, S_struct)
    except Exception:
        pass

    return S_struct


def fuse_poe_scores(S_geom: torch.Tensor, S_struct: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """返回 PoE 融合后的 logits：S_poe = S_geom + beta * S_struct。"""

    try:
        return S_geom + float(beta) * S_struct
    except Exception:
        S_struct = S_struct.to(device=S_geom.device)
        return S_geom + float(beta) * S_struct


def _strip_modules_if_exists(module: nn.Module, names) -> nn.Module:
    """Remove named child modules from an nn.Module when present."""
    if not isinstance(names, (list, tuple, set)):
        names = (names,)
    for name in names:
        if hasattr(module, name):
            delattr(module, name)
    return module


def _build_eegnet_backbone(output_dim: int, num_channels: int, seq_len: int) -> nn.Module:
    if EEGNetv4 is None:
        raise ImportError("EEGNetv4 is not available; please install braindecode>=0.7.")
    sig = inspect.signature(EEGNetv4.__init__)
    kwargs = {}
    if 'n_outputs' in sig.parameters:
        kwargs['n_outputs'] = output_dim
    elif 'n_classes' in sig.parameters:
        kwargs['n_classes'] = output_dim
    if 'n_chans' in sig.parameters:
        kwargs['n_chans'] = num_channels
    elif 'in_chans' in sig.parameters:
        kwargs['in_chans'] = num_channels
    if 'n_times' in sig.parameters:
        kwargs['n_times'] = seq_len
    elif 'input_window_samples' in sig.parameters:
        kwargs['input_window_samples'] = seq_len
    if 'final_conv_length' in sig.parameters:
        kwargs['final_conv_length'] = 'auto'
    backbone = EEGNetv4(**kwargs)
    _strip_modules_if_exists(backbone, ('softmax',))
    return backbone


def _build_shallow_backbone(output_dim: int, num_channels: int, seq_len: int) -> nn.Module:
    if ShallowFBCSPNet is None:
        raise ImportError("ShallowFBCSPNet is not available; please install braindecode>=0.7.")
    sig = inspect.signature(ShallowFBCSPNet.__init__)
    kwargs = {}
    if 'n_outputs' in sig.parameters:
        kwargs['n_outputs'] = output_dim
    elif 'n_classes' in sig.parameters:
        kwargs['n_classes'] = output_dim
    if 'n_chans' in sig.parameters:
        kwargs['n_chans'] = num_channels
    elif 'in_chans' in sig.parameters:
        kwargs['in_chans'] = num_channels
    if 'n_times' in sig.parameters:
        kwargs['n_times'] = seq_len
    elif 'input_window_samples' in sig.parameters:
        kwargs['input_window_samples'] = seq_len
    if 'final_conv_length' in sig.parameters:
        kwargs['final_conv_length'] = 'auto'
    backbone = ShallowFBCSPNet(**kwargs)
    _strip_modules_if_exists(backbone, ('softmax',))
    return backbone


def _build_conformer_backbone(output_dim: int, num_channels: int, seq_len: int) -> nn.Module:
    if EEGConformer is None:
        raise ImportError("EEGConformer is not available; please install braindecode>=1.2.")
    sig = inspect.signature(EEGConformer.__init__)
    kwargs = {}
    if 'n_outputs' in sig.parameters:
        kwargs['n_outputs'] = output_dim
    elif 'n_classes' in sig.parameters:
        kwargs['n_classes'] = output_dim
    if 'n_chans' in sig.parameters:
        kwargs['n_chans'] = num_channels
    elif 'in_chans' in sig.parameters:
        kwargs['in_chans'] = num_channels
    if 'n_times' in sig.parameters:
        kwargs['n_times'] = seq_len
    elif 'input_window_samples' in sig.parameters:
        kwargs['input_window_samples'] = seq_len
    kwargs.setdefault('return_features', False)
    return EEGConformer(**kwargs)


class Config:
    def __init__(self):
        self.task_name = 'classification'  # Example task name
        self.seq_len = 250                 # Sequence length
        self.pred_len = 250                # Prediction length
        self.output_attention = False      # Whether to output attention weights
        self.d_model = 250                 # Model dimension
        self.embed = 'timeF'               # Time encoding method
        self.freq = 'h'                    # Time frequency
        self.dropout = 0.25                # Dropout rate
        self.factor = 1                    # Attention scaling factor
        self.n_heads = 4                   # Number of attention heads
        self.e_layers = 1                  # Number of encoder layers
        self.d_ff = 256                    # Feedforward network dimension
        self.activation = 'gelu'           # Activation function
        self.enc_in = 63                   # Encoder input dimension (example value)
        
class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False,  num_subjects=10, use_subject_unk=False):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
            joint_train=False,
            num_subjects=num_subjects,
            use_subject_unk=use_subject_unk,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, :63, :]      
        # print("enc_out", enc_out.shape)
        return enc_out



class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)     
        # print("x", x.shape)   
        x = self.tsconv(x)
        # print("tsconv", x.shape)   
        x = self.projection(x)
        # print("projection", x.shape)  
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        input = input.contiguous().view(input.size(0), -1)
        return input


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

        
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )



class SATTC(nn.Module):
    def __init__(
        self,
        num_channels=63,
        sequence_length=250,
        num_subjects=2,
        num_features=64,
        num_latents=1024,
        num_blocks=1,
        use_subject_unk=False,
        encoder_choice: str = 'sattc',
        proj_dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder_choice = (encoder_choice or 'sattc').lower()
        self.num_channels = int(num_channels)
        self.sequence_length = int(sequence_length)
        self.proj_dim = int(num_latents)
        self.logit_scale = nn.Parameter(torch.zeros([]))
        self.loss_func = ClipLoss()
        self.subject_wise_linear = None

        if self.encoder_choice == 'sattc':
            default_config = Config()
            default_config.use_subject_unk = use_subject_unk
            self.encoder = iTransformer(default_config, num_subjects=num_subjects, use_subject_unk=use_subject_unk)
            self.subject_wise_linear = nn.ModuleList([nn.Linear(default_config.d_model, self.sequence_length) for _ in range(num_subjects)])
            self.enc_eeg = Enc_eeg()
            embedding_dim = 1440
        elif self.encoder_choice == 'eegnet':
            backbone = _build_eegnet_backbone(self.proj_dim, self.num_channels, self.sequence_length)
            self.encoder = backbone
            self.enc_eeg = nn.Identity()
            embedding_dim = self.proj_dim
        elif self.encoder_choice == 'shallow':
            backbone = _build_shallow_backbone(self.proj_dim, self.num_channels, self.sequence_length)
            self.encoder = backbone
            self.enc_eeg = nn.Identity()
            embedding_dim = self.proj_dim
        elif self.encoder_choice == 'conformer':
            backbone = _build_conformer_backbone(self.proj_dim, self.num_channels, self.sequence_length)
            self.encoder = backbone
            self.enc_eeg = nn.Identity()
            embedding_dim = self.proj_dim
        else:
            raise ValueError(f"Unsupported encoder_choice: {self.encoder_choice}")

        if embedding_dim != self.proj_dim:
            self.proj_eeg = Proj_eeg(embedding_dim=embedding_dim, proj_dim=self.proj_dim, drop_proj=proj_dropout)
        else:
            self.proj_eeg = nn.Identity()

    def forward(self, x, subject_ids=None):
        x = x.float()
        if self.encoder_choice == 'sattc':
            encoded = self.encoder(x, None, subject_ids)
            eeg_embedding = self.enc_eeg(encoded)
        else:
            # Ensure shape is [batch, channels, seq_len] for braindecode models
            if x.dim() == 4 and x.size(-1) == 1:
                x = x.squeeze(-1)
            if x.dim() == 3 and x.size(1) != self.num_channels and x.size(2) == self.num_channels:
                x = x.permute(0, 2, 1)
            encoded = self.encoder(x)
            eeg_embedding = self.enc_eeg(encoded)

        out = self.proj_eeg(eeg_embedding)
        return out  


    # Backward-compatibility alias during the naming migration window.
    SATTC_COMPAT = SATTC
    
def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def train_model(sub, eeg_model, dataloader, optimizer, device, text_features_all, img_features_all, config, progress_callback=None):
    eeg_model.train()
    use_cos = getattr(config, 'sim', 'cos') == 'cos'
    text_features_all = text_features_all.to(device).float()
    text_whiten_state = _resolve_whitening_state(config, 'text')
    img_whiten_state = _resolve_whitening_state(config, 'img')
    auto_whiten_enabled = bool(getattr(config, 'global_auto_whiten', True))
    if text_whiten_state is None and auto_whiten_enabled:
        text_whiten_state = _resolve_auto_whitening_state(config, 'text', text_features_all)
    if img_whiten_state is None and auto_whiten_enabled:
        img_whiten_state = _resolve_auto_whitening_state(config, 'img', img_features_all.to(device='cpu'))
    if text_whiten_state is not None:
        text_features_all = _whiten_features(text_features_all, text_whiten_state)
    if use_cos:
        text_features_all = F.normalize(text_features_all, dim=1, eps=1e-12)  # (n_cls, d)
    img_features_all = img_features_all[::10].to(device).float()
    if img_whiten_state is not None:
        img_features_all = _whiten_features(img_features_all, img_whiten_state)
    if use_cos:
        img_features_all = F.normalize(img_features_all, dim=1, eps=1e-12)
    total_loss = 0
    correct_top5 = 0
    total = 0
    alpha=0.99
    features_list = []  # List to store features
    save_features= True
    batch_count = 0
    subject_id_mode = getattr(config, 'train_subject_ids', 'spoof') or 'spoof'
    real_ids_requested = subject_id_mode.lower() == 'real'
    warned_missing_real_ids = False
    subject_dropout_p = float(getattr(config, 'subject_dropout_p', 0.0) or 0.0)
    use_subject_unk = bool(getattr(config, 'use_subject_unk', False))
    unk_idx = getattr(getattr(eeg_model, 'encoder', None), 'enc_embedding', None)
    if unk_idx is not None:
        unk_idx = getattr(eeg_model.encoder.enc_embedding, 'unk_subject_index', None)
    used_real_subject_ids = set(getattr(config, '_train_subjects_seen', []))
    dataset_subject_lookup = getattr(getattr(dataloader, 'dataset', None), 'idx_to_subject', None)
    subject_index_to_real = None
    if real_ids_requested and dataset_subject_lookup:
        subject_index_to_real = {}
        for key, value in dataset_subject_lookup.items():
            try:
                idx_key = int(key)
            except (TypeError, ValueError):
                try:
                    idx_key = int(key.item())  # tensors as keys
                except Exception:
                    continue
            real_id = extract_id_from_string(value) if isinstance(value, str) else None
            subject_index_to_real[idx_key] = int(real_id) if real_id is not None else idx_key
    for batch_idx, batch in enumerate(dataloader):
        if len(batch) == 7:
            eeg_data, labels, text, text_features, img, img_features, _subject_ids = batch
        else:
            eeg_data, labels, text, text_features, img, img_features = batch
            _subject_ids = None
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        if text_whiten_state is not None:
            text_features = _whiten_features(text_features, text_whiten_state)
        if img_whiten_state is not None:
            img_features = _whiten_features(img_features, img_whiten_state)
        if use_cos:
            text_features = F.normalize(text_features, dim=1, eps=1e-12)
            img_features = F.normalize(img_features, dim=1, eps=1e-12)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        batch_size = eeg_data.size(0)  
        if real_ids_requested and _subject_ids is not None:
            subject_ids = _subject_ids.to(device=device, dtype=torch.long)
            if subject_ids.ndim == 0:
                subject_ids = subject_ids.view(1).expand(batch_size)
            if subject_index_to_real:
                mapped_ids = [subject_index_to_real.get(int(idx), int(idx)) for idx in subject_ids.cpu().tolist()]
                subject_ids = torch.tensor(mapped_ids, dtype=torch.long, device=device)
        else:
            if real_ids_requested and _subject_ids is None and not warned_missing_real_ids:
                print("[SubjectIDs] Requested real subject IDs but dataset did not provide them; falling back to spoof mode for this dataloader.")
                warned_missing_real_ids = True
            subject_id = extract_id_from_string(sub)
            if subject_id is None:
                subject_id = 0
            subject_ids = torch.full((batch_size,), int(subject_id), dtype=torch.long, device=device)
        batch_real_ids = subject_ids.detach().cpu().tolist()
        for sid in batch_real_ids:
            if sid is None:
                continue
            if unk_idx is not None and sid == int(unk_idx):
                continue
            used_real_subject_ids.add(int(sid))
        if use_subject_unk and subject_dropout_p > 0 and unk_idx is not None:
            drop_mask = torch.rand_like(subject_ids.float()) < subject_dropout_p
            if drop_mask.any():
                subject_ids = subject_ids.masked_fill(drop_mask.bool(), int(unk_idx))
        # eeg_data = eeg_data.permute(0, 2, 1)
        eeg_features = eeg_model(eeg_data, subject_ids).float()
        if use_cos:
            eeg_features = F.normalize(eeg_features, dim=1, eps=1e-12)


        features_list.append(eeg_features)

        logit_scale = torch.clamp(eeg_model.logit_scale, max=np.log(100.0)).exp()

        img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
        text_loss = eeg_model.loss_func(eeg_features, text_features, logit_scale)
        # loss = img_loss + text_loss
        # print("text_loss", text_loss)
        # print("img_loss", img_loss)
        loss = alpha * img_loss + (1 - alpha) * text_loss
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        
        # Compute the corresponding logits
        logits_img = logit_scale * (eeg_features @ img_features_all.T)
        # logits_text = logit_scale * (eeg_features @ text_features_all.T)
        # logits_single = (logits_text + logits_img) / 2.0        
        # logits_text = logit_scale * (eeg_features @ text_features_all.T)
        logits_single = logits_img
        top_k = min(5, logits_single.size(1))
        topk_indices = torch.topk(logits_single, k=top_k, dim=1).indices
        top5_matches = (topk_indices == labels.unsqueeze(1)).any(dim=1)

        batch_size = top5_matches.shape[0]
        total += batch_size
        correct_top5 += top5_matches.sum().item()
        batch_count += 1
        if progress_callback is not None:
            progress_callback(batch_size)
    average_loss = total_loss / max(1, batch_count)
    accuracy = correct_top5 / max(1, total)
    if used_real_subject_ids:
        existing_seen = set(getattr(config, '_train_subjects_seen', []))
        if unk_idx is not None:
            filtered = {int(x) for x in used_real_subject_ids if int(x) != int(unk_idx)}
        else:
            filtered = {int(x) for x in used_real_subject_ids}
        existing_seen.update(filtered)
        setattr(config, '_train_subjects_seen', sorted(existing_seen))
    return average_loss, accuracy, torch.cat(features_list, dim=0)



def evaluate_model(sub, eeg_model, dataloader, device, text_features_all, img_features_all, k_values, config, progress_callback=None):
    eeg_model.eval()
    try:
        setattr(config, '_last_eval_diag', None)
    except Exception:
        pass

    if isinstance(k_values, int):
        ks = [k_values]
        single_return = True
    else:
        ks = list(dict.fromkeys(k_values))  # preserve order without duplicates
        single_return = False

    use_cos = getattr(config, 'sim', 'cos') == 'cos'
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()

    text_whiten_state = _resolve_whitening_state(config, 'text')
    img_whiten_state = _resolve_whitening_state(config, 'img')
    auto_whiten_enabled = bool(getattr(config, 'global_auto_whiten', True))
    if text_whiten_state is None and auto_whiten_enabled:
        text_whiten_state = _resolve_auto_whitening_state(config, 'text', text_features_all)
    if img_whiten_state is None and auto_whiten_enabled:
        img_whiten_state = _resolve_auto_whitening_state(config, 'img', img_features_all)
    if text_whiten_state is not None:
        text_features_all = _whiten_features(text_features_all, text_whiten_state)
    if img_whiten_state is not None:
        img_features_all = _whiten_features(img_features_all, img_whiten_state)

    if use_cos:
        text_features_all = F.normalize(text_features_all, dim=1, eps=1e-12)
        img_features_all = F.normalize(img_features_all, dim=1, eps=1e-12)
    eval_mode = getattr(config, 'test_subject_ids', 'target') or 'target'
    embedding_module = getattr(getattr(eeg_model, 'encoder', None), 'enc_embedding', None)
    subject_wrapper = None
    if embedding_module is not None:
        subject_wrapper = getattr(embedding_module, 'subject_embedding', None)
    unk_idx = getattr(embedding_module, 'unk_subject_index', None) if embedding_module is not None else None
    if eval_mode == 'unk_avg' and unk_idx is not None and subject_wrapper is not None:
        seen_ids = getattr(config, '_train_subjects_seen', [])
        if seen_ids:
            weight = subject_wrapper.subject_embedding.weight.data
            valid_ids = [idx for idx in seen_ids if 0 <= idx < weight.size(0)]
            if valid_ids:
                avg_vec = weight[valid_ids].mean(dim=0, keepdim=True)
                avg_vec = F.normalize(avg_vec, dim=1, eps=1e-12)
                weight[unk_idx] = avg_vec.squeeze(0)
    total_loss = 0.0
    alpha = 0.99
    batch_count = 0

    feature_batches = []
    labels_batches = []
    subject_index_batches = []

    subject_lookup = getattr(getattr(dataloader, 'dataset', None), 'idx_to_subject', None)
    has_subject_indices = False
    fallback_subject_numeric = extract_id_from_string(sub)
    if fallback_subject_numeric is None:
        fallback_subject_numeric = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 7:
                eeg_data, labels, text, text_features, img, img_features, subject_indices = batch
                has_subject_indices = True
            else:
                eeg_data, labels, text, text_features, img, img_features = batch
                subject_indices = None

            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            if text_whiten_state is not None:
                text_features = _whiten_features(text_features, text_whiten_state)
            if img_whiten_state is not None:
                img_features = _whiten_features(img_features, img_whiten_state)
            if use_cos:
                text_features = F.normalize(text_features, dim=1, eps=1e-12)
                img_features = F.normalize(img_features, dim=1, eps=1e-12)

            batch_size = eeg_data.size(0)
            subject_id = fallback_subject_numeric
            if eval_mode == 'target':
                subject_ids = torch.full((batch_size,), int(subject_id), dtype=torch.long, device=device)
            elif eval_mode == 'fixed1':
                subject_ids = torch.full((batch_size,), 1, dtype=torch.long, device=device)
            elif eval_mode in {'unk_avg', 'unk_learned'} and unk_idx is not None:
                subject_ids = torch.full((batch_size,), int(unk_idx), dtype=torch.long, device=device)
            else:
                subject_ids = torch.full((batch_size,), int(subject_id), dtype=torch.long, device=device)

            eeg_features = eeg_model(eeg_data, subject_ids)
            raw_features = eeg_features.detach().float()
            if use_cos:
                eeg_features = F.normalize(eeg_features, dim=1, eps=1e-12)

            logit_scale = torch.clamp(eeg_model.logit_scale, max=np.log(100.0)).exp()

            img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
            text_loss = eeg_model.loss_func(eeg_features, text_features, logit_scale)
            loss = img_loss * alpha + text_loss * (1 - alpha)
            total_loss += loss.item()

            feature_batches.append(raw_features.cpu())
            del raw_features
            labels_batches.append(labels.detach().cpu())
            if subject_indices is not None:
                subject_index_batches.append(subject_indices.detach().cpu())
            batch_count += 1
            if progress_callback is not None:
                progress_callback(batch_size)

    average_loss = total_loss / max(1, batch_count)

    if not feature_batches:
        empty_results = {k: {"top1": 0.0, "top5": 0.0} for k in ks}
        empty_debug = {
            "aligned_strict": False,
            "num_queries": 0,
            "alignment_metrics": {},
            "evidence_pre_csls": {},
        }
        try:
            setattr(config, '_last_eval_rank_debug', empty_debug)
        except Exception:
            pass
        if single_return:
            only_k = ks[0]
            return average_loss, empty_results[only_k]["top1"], empty_results[only_k]["top5"], empty_debug
        return average_loss, empty_results, empty_debug

    features_all = torch.cat(feature_batches, dim=0)
    labels_all = torch.cat(labels_batches, dim=0)
    subject_indices_all = torch.cat(subject_index_batches, dim=0) if subject_index_batches else None

    features_all_device = features_all.to(device)
    logit_scale_eval = torch.clamp(eeg_model.logit_scale, max=np.log(100.0)).exp()
    saw_enabled = bool(getattr(config, 'use_saw', False))
    saw_shrink = float(getattr(config, 'saw_shrink', 0.2))
    saw_diag = bool(getattr(config, 'saw_diag', False))

    if saw_enabled:
        if subject_indices_all is not None:
            subject_ids_tensor = subject_indices_all.to(device=device, dtype=torch.long)
        else:
            subject_ids_tensor = torch.full(
                (features_all_device.size(0),),
                int(fallback_subject_numeric),
                dtype=torch.long,
                device=device,
            )
        processed = torch.empty_like(features_all_device)
        for subj_id in subject_ids_tensor.unique(sorted=True):
            mask = subject_ids_tensor == subj_id
            subj_feats = features_all_device[mask]
            if subj_feats.numel() == 0:
                continue
            try:
                processed[mask] = subject_adaptive_whiten(subj_feats, shrink=saw_shrink, diag=saw_diag)
            except Exception as err:
                print(f"[SAW] Warning: fallback to L2 normalization for subject {int(subj_id)} due to {err}.")
                processed[mask] = F.normalize(subj_feats, dim=1, eps=1e-12)
    else:
        if use_cos:
            processed = F.normalize(features_all_device, dim=1, eps=1e-12)
        else:
            processed = features_all_device

    similarities = processed @ img_features_all.T
    tau = float(getattr(config, 'temp', 1.0) or 1.0)
    logits_all = logit_scale_eval * similarities
    if abs(tau - 1.0) > 1e-6:
        logits_all = logits_all / max(tau, 1e-6)
    global_scale = float(getattr(config, 'global_temp_scale', 1.0) or 1.0)
    if not math.isfinite(global_scale):
        global_scale = 1.0
    global_bias = float(getattr(config, 'global_temp_bias', 0.0) or 0.0)
    if not math.isfinite(global_bias):
        global_bias = 0.0
    if global_scale != 1.0 or global_bias != 0.0:
        logits_all = logits_all * global_scale + global_bias

    zscore_mean, zscore_std = _resolve_logits_zscore(config)
    if zscore_mean is not None:
        logits_all = logits_all - zscore_mean
    if zscore_std is not None:
        logits_all = logits_all / max(zscore_std, 1e-6)

    auto_center = bool(getattr(config, 'logits_auto_center', True))
    auto_scale = bool(getattr(config, 'logits_auto_scale', True))
    auto_center_applied = False
    auto_scale_applied = False
    alignment_metrics_base = {}
    if (auto_center or auto_scale) and zscore_mean is None and zscore_std is None:
        mean_est = logits_all.mean()
        std_est = logits_all.std(unbiased=False)
        if auto_center:
            logits_all = logits_all - mean_est
            auto_center_applied = True
        if auto_scale:
            std_scalar = float(std_est.detach().cpu().item())
            scale_denom = max(std_scalar, 1e-6)
            logits_all = logits_all / scale_denom
            auto_scale_applied = True
        try:
            setattr(config, '_last_auto_logits_mean', float(mean_est.detach().cpu().item()))
            setattr(config, '_last_auto_logits_std', float(std_est.detach().cpu().item()))
        except Exception:
            pass

    base_logits = logits_all.detach().clone()   # S_new for following CSLS/Ada-CSLS steps
    q_dim, c_dim = base_logits.shape
    lock_mask = torch.zeros((q_dim, c_dim), dtype=torch.bool, device=base_logits.device)
    penalty_scale = torch.zeros((q_dim, c_dim), dtype=torch.float32, device=base_logits.device)
    case_id = torch.zeros((q_dim, c_dim), dtype=torch.int8, device=base_logits.device)
    pre_csls_evidence: Dict[str, Any] = {}
    try:
        row_rank_tensor = torch.argsort(base_logits, dim=1, descending=True).argsort(dim=1) + 1
        col_rank_tensor = torch.argsort(base_logits, dim=0, descending=True).argsort(dim=0) + 1
        k_row_safe_default = int(getattr(config, 'pre_csls_row_safe_k', 5) or 5)
        k_col_safe_default = int(getattr(config, 'pre_csls_col_safe_k', 5) or 5)
        K_row_safe = max(1, min(k_row_safe_default, c_dim))
        K_col_safe = max(1, min(k_col_safe_default, q_dim))
        row_top1_mask_tensor = row_rank_tensor == 1
        row_topk_mask_tensor = row_rank_tensor <= K_row_safe
        col_top1_mask_tensor = col_rank_tensor == 1
        col_topk_mask_tensor = col_rank_tensor <= K_col_safe
        top5_row_limit = max(1, min(5, c_dim))
        top5_col_limit = max(1, min(5, q_dim))
        row_top5_mask_tensor = row_rank_tensor <= top5_row_limit
        col_top5_mask_tensor = col_rank_tensor <= top5_col_limit
        mnn1_mask_tensor = (row_rank_tensor == 1) & (col_rank_tensor == 1)
        anchors_idx_tensor = mnn1_mask_tensor.nonzero(as_tuple=False)

        has_mnn1_col = torch.zeros((c_dim,), dtype=torch.bool, device=base_logits.device)
        big_penalty_alpha = 1.0
        k_row_top = min(max(2, 5), c_dim)
        margin_row_tensor = None
        margin_col_tensor = None
        rho_tensor = None
        hub_level_tensor = None
        hub_quantiles = None
        if k_row_top >= 2 and c_dim >= 2:
            top_vals_row, _ = torch.topk(base_logits, k=k_row_top, dim=1)
            row_top1 = top_vals_row[:, 0]
            row_top2 = top_vals_row[:, 1]
            margin_row_tensor = row_top1 - row_top2
        else:
            margin_row_tensor = torch.zeros((q_dim,), dtype=base_logits.dtype, device=base_logits.device)

        if q_dim >= 2:
            k_col_top = min(max(2, 5), q_dim)
            top_vals_col, _ = torch.topk(base_logits.t(), k=k_col_top, dim=1)
            col_top1 = top_vals_col[:, 0]
            col_top2 = top_vals_col[:, 1]
            margin_col_tensor = col_top1 - col_top2
        else:
            margin_col_tensor = torch.zeros((c_dim,), dtype=base_logits.dtype, device=base_logits.device)

        hub_high_q_default = float(getattr(config, 'pre_csls_case5_hub_high_quantile', 0.95) or 0.95)
        hub_mid_q_default = float(getattr(config, 'pre_csls_case5_hub_mid_quantile', 0.80) or 0.80)
        hub_high_q = max(0.0, min(1.0, hub_high_q_default))
        hub_mid_q = max(0.0, min(hub_high_q, hub_mid_q_default))

        if q_dim > 0 and c_dim > 0:
            L_pop_default = int(getattr(config, 'pre_csls_row_topL', 5) or 5)
            L_pop = max(1, min(L_pop_default, c_dim))
            in_topL_mask = row_rank_tensor <= L_pop
            rho_count_tensor = in_topL_mask.sum(dim=0)
            rho_tensor = rho_count_tensor.to(dtype=torch.float32) / float(max(q_dim, 1))
            hub_level_tensor = torch.zeros((c_dim,), dtype=torch.int8, device=base_logits.device)
            try:
                high_thr_val = torch.quantile(rho_tensor, hub_high_q).clamp(min=0.0)
                if hub_mid_q > 0.0:
                    mid_thr_val = torch.quantile(rho_tensor, hub_mid_q).clamp(min=0.0)
                else:
                    mid_thr_val = rho_tensor.new_tensor(0.0)
            except Exception:
                high_thr_val = rho_tensor.new_tensor(0.0)
                mid_thr_val = rho_tensor.new_tensor(0.0)
            if hub_mid_q > 0.0:
                hub_level_tensor[rho_tensor >= mid_thr_val] = 1
            if hub_high_q > 0.0:
                hub_level_tensor[rho_tensor >= high_thr_val] = 2
            hub_quantiles = {
                "high_threshold": float(high_thr_val.detach().cpu().item()) if high_thr_val is not None else 0.0,
                "mid_threshold": float(mid_thr_val.detach().cpu().item()) if mid_thr_val is not None else 0.0,
                "high_quantile": float(hub_high_q),
                "mid_quantile": float(hub_mid_q),
                "L_pop": int(L_pop),
            }
        else:
            rho_tensor = torch.zeros((c_dim,), dtype=torch.float32, device=base_logits.device)
            hub_level_tensor = torch.zeros((c_dim,), dtype=torch.int8, device=base_logits.device)
            hub_quantiles = {
                "high_threshold": 0.0,
                "mid_threshold": 0.0,
                "high_quantile": float(hub_high_q),
                "mid_quantile": float(hub_mid_q),
                "L_pop": int(getattr(config, 'pre_csls_row_topL', 5) or 5),
            }

        for anchor_pair in anchors_idx_tensor:
            q_star = int(anchor_pair[0].item())
            c_star = int(anchor_pair[1].item())
            has_mnn1_col[c_star] = True
            lock_mask[q_star, c_star] = True
            case_id[q_star, c_star] = 1
            rows = torch.arange(q_dim, device=base_logits.device)
            other_rows = rows[rows != q_star]
            if other_rows.numel() > 0:
                penalty_scale[other_rows, c_star] = big_penalty_alpha
                case_id[other_rows, c_star] = -1

        tau_ratio_default = float(getattr(config, 'pre_csls_case2_tau_ratio', 0.5) or 0.5)
        tau_ratio_tensor = margin_row_tensor.new_tensor(tau_ratio_default)
        if margin_row_tensor.numel() > 0:
            try:
                margin_row_median = margin_row_tensor.median()
            except RuntimeError:
                margin_row_median = margin_row_tensor.new_tensor(0.0)
        else:
            margin_row_median = margin_row_tensor.new_tensor(0.0)
        tau_row_tensor = margin_row_median * tau_ratio_tensor
        margin_row_2d = margin_row_tensor.unsqueeze(1).expand(-1, c_dim)
        case1_mask_tensor = mnn1_mask_tensor
        case2_candidate_mask = row_top1_mask_tensor & col_topk_mask_tensor
        case2_mask_tensor = case2_candidate_mask & (margin_row_2d >= tau_row_tensor)
        case2_mask_tensor = case2_mask_tensor & (~case1_mask_tensor)

        case3_mask_tensor = (
            row_topk_mask_tensor
            & col_top1_mask_tensor
            & (~case1_mask_tensor)
            & (~case2_mask_tensor)
        )

        case2_indices = case2_mask_tensor.nonzero(as_tuple=False)
        if case2_indices.numel() > 0:
            weak_alpha = float(getattr(config, 'pre_csls_case2_col_penalty', 0.5) or 0.5)
            weak_alpha_tensor = penalty_scale.new_tensor(weak_alpha)
            for pair in case2_indices:
                q_idx = int(pair[0].item())
                c_idx = int(pair[1].item())
                if lock_mask[q_idx, c_idx]:
                    continue
                lock_mask[q_idx, c_idx] = True
                case_id[q_idx, c_idx] = 2
                column_rows = torch.arange(q_dim, device=base_logits.device)
                other_rows = column_rows[column_rows != q_idx]
                if other_rows.numel() == 0:
                    continue
                other_mask = ~lock_mask[other_rows, c_idx]
                if other_mask.any():
                    penalised_rows = other_rows[other_mask]
                    existing_penalty = penalty_scale[penalised_rows, c_idx]
                    penalty_scale[penalised_rows, c_idx] = torch.maximum(existing_penalty, weak_alpha_tensor)
                    untouched_mask = case_id[penalised_rows, c_idx] == 0
                    if untouched_mask.any():
                        case_id[penalised_rows[untouched_mask], c_idx] = -2

        if case3_mask_tensor.any():
            lock_mask[case3_mask_tensor] = True
            penalty_scale[case3_mask_tensor] = 0.0
            case_id[case3_mask_tensor] = 3

        case4_mask_tensor = (
            row_top5_mask_tensor
            & col_top5_mask_tensor
            & (~case3_mask_tensor)
            & (~case2_mask_tensor)
            & (~case1_mask_tensor)
        )
        if case4_mask_tensor.any():
            lock_mask[case4_mask_tensor] = True
            penalty_scale[case4_mask_tensor] = 0.0
            case_id[case4_mask_tensor] = 4

        case5_mask_tensor = (
            row_top5_mask_tensor
            & (~col_topk_mask_tensor)
            & (~case1_mask_tensor)
            & (~case2_mask_tensor)
            & (~case3_mask_tensor)
            & (~case4_mask_tensor)
        )
        hub_level_2d = hub_level_tensor.view(1, -1).expand(q_dim, -1)
        if case5_mask_tensor.any():
            overwrite_case5 = case5_mask_tensor & (case_id == 0)
            if overwrite_case5.any():
                case_id[overwrite_case5] = 5

            case5_high_hub_mask = case5_mask_tensor & (hub_level_2d == 2)
            case5_mid_hub_mask = case5_mask_tensor & (hub_level_2d == 1)
            case5_low_hub_mask = case5_mask_tensor & (hub_level_2d <= 0)
            if case5_high_hub_mask.any():
                penalty_scale = penalty_scale.masked_fill(case5_high_hub_mask, 1.0)
            if case5_mid_hub_mask.any():
                penalty_scale = penalty_scale.masked_fill(case5_mid_hub_mask, 0.5)
            if case5_low_hub_mask.any():
                penalty_scale = penalty_scale.masked_fill(case5_low_hub_mask, 0.0)

        case6_mask_tensor = (
            (~row_top5_mask_tensor)
            & col_topk_mask_tensor
            & (~case1_mask_tensor)
            & (~case2_mask_tensor)
            & (~case3_mask_tensor)
            & (~case4_mask_tensor)
        )
        if case6_mask_tensor.any():
            overwrite_case6 = case6_mask_tensor & (case_id == 0)
            if overwrite_case6.any():
                case_id[overwrite_case6] = 6

            case6_high_hub_mask = case6_mask_tensor & (hub_level_2d == 2)
            case6_mid_hub_mask = case6_mask_tensor & (hub_level_2d == 1)
            case6_pen_high = float(getattr(config, 'pre_csls_case6_penalty_high', 0.5) or 0.0)
            case6_pen_mid = float(getattr(config, 'pre_csls_case6_penalty_mid', 0.25) or 0.0)
            case6_pen_high = max(0.0, case6_pen_high)
            case6_pen_mid = max(0.0, case6_pen_mid)
            if case6_high_hub_mask.any():
                penalty_scale = torch.maximum(
                    penalty_scale,
                    penalty_scale.new_tensor(case6_pen_high) * case6_high_hub_mask.to(dtype=penalty_scale.dtype),
                )
            if case6_mid_hub_mask.any():
                penalty_scale = torch.maximum(
                    penalty_scale,
                    penalty_scale.new_tensor(case6_pen_mid) * case6_mid_hub_mask.to(dtype=penalty_scale.dtype),
                )

        strong_protect_mask = (
            mnn1_mask_tensor
            | case2_mask_tensor
            | case3_mask_tensor
            | case4_mask_tensor
        )
        lock_mask = lock_mask | strong_protect_mask
        penalty_scale = penalty_scale.masked_fill(lock_mask, 0.0)

        bg_penalty_default = getattr(config, 'pre_csls_bg_penalty', 1.0)
        try:
            bg_penalty_value = float(bg_penalty_default)
        except (TypeError, ValueError):
            bg_penalty_value = 1.0
        bg_penalty_value = max(0.0, bg_penalty_value)

        bg_mask_tensor = (~row_top5_mask_tensor) & (~col_topk_mask_tensor) & (~strong_protect_mask)
        if bg_mask_tensor.any():
            # 🚨 背景区强惩罚：row>5 且 col>5 且不在任何保护区
            penalty_scale = torch.where(
                bg_mask_tensor,
                penalty_scale.new_tensor(bg_penalty_value),
                penalty_scale,
            )

        penalty_scale[lock_mask] = 0.0

        pre_csls_evidence["row_rank"] = row_rank_tensor.to(device='cpu', dtype=torch.int32)
        pre_csls_evidence["col_rank"] = col_rank_tensor.to(device='cpu', dtype=torch.int32)
        pre_csls_evidence["mnn1_mask"] = mnn1_mask_tensor.to(device='cpu', dtype=torch.bool)
        pre_csls_evidence["case1_mask"] = mnn1_mask_tensor.to(device='cpu', dtype=torch.bool)
        pre_csls_evidence["mnn1_indices"] = anchors_idx_tensor.to(device='cpu', dtype=torch.int64)
        pre_csls_evidence["has_mnn1_col"] = has_mnn1_col.to(device='cpu', dtype=torch.bool)
        pre_csls_evidence["row_margin_top1_top2"] = margin_row_tensor.to(device='cpu', dtype=torch.float32)
        pre_csls_evidence["col_margin_top1_top2"] = margin_col_tensor.to(device='cpu', dtype=torch.float32)
        pre_csls_evidence["rho_popularity"] = rho_tensor.to(device='cpu', dtype=torch.float32)
        pre_csls_evidence["hub_level"] = hub_level_tensor.to(device='cpu', dtype=torch.int8)
        pre_csls_evidence["hub_quantiles"] = hub_quantiles
        pre_csls_evidence["row_top1_mask"] = row_top1_mask_tensor.to(device='cpu', dtype=torch.bool)
        pre_csls_evidence["row_topk_mask"] = row_topk_mask_tensor.to(device='cpu', dtype=torch.bool)
        pre_csls_evidence["col_top1_mask"] = col_top1_mask_tensor.to(device='cpu', dtype=torch.bool)
        pre_csls_evidence["col_topk_mask"] = col_topk_mask_tensor.to(device='cpu', dtype=torch.bool)
        pre_csls_evidence["row_top5_mask"] = row_top5_mask_tensor.to(device='cpu', dtype=torch.bool)
        pre_csls_evidence["col_top5_mask"] = col_top5_mask_tensor.to(device='cpu', dtype=torch.bool)
        pre_csls_evidence["row_safe_k"] = int(K_row_safe)
        pre_csls_evidence["col_safe_k"] = int(K_col_safe)
        pre_csls_evidence["case2_mask"] = case2_mask_tensor.to(device='cpu', dtype=torch.bool)
        pre_csls_evidence["case2_tau_row"] = float(tau_row_tensor.detach().cpu().item())
        pre_csls_evidence["case3_mask"] = case3_mask_tensor.to(device='cpu', dtype=torch.bool)
        pre_csls_evidence["case4_mask"] = case4_mask_tensor.to(device='cpu', dtype=torch.bool)
        pre_csls_evidence["case5_mask"] = case5_mask_tensor.to(device='cpu', dtype=torch.bool)
        pre_csls_evidence["case6_mask"] = case6_mask_tensor.to(device='cpu', dtype=torch.bool)
        pre_csls_evidence["strong_protect_mask"] = strong_protect_mask.to(device='cpu', dtype=torch.bool)
        pre_csls_evidence["background_mask"] = bg_mask_tensor.to(device='cpu', dtype=torch.bool)
        pre_csls_evidence["lock_mask"] = lock_mask.to(device='cpu', dtype=torch.bool)
        pre_csls_evidence["penalty_scale"] = penalty_scale.to(device='cpu', dtype=torch.float32)
        pre_csls_evidence["case_id_matrix"] = case_id.to(device='cpu', dtype=torch.int8)
        pre_csls_evidence["is_mnn1"] = pre_csls_evidence["mnn1_mask"].clone()
        pre_csls_evidence["background_penalty"] = float(bg_penalty_value)
        pre_csls_evidence["case6_penalty_high"] = float(case6_pen_high)
        pre_csls_evidence["case6_penalty_mid"] = float(case6_pen_mid)
        case_counts = {
            "case1": int(mnn1_mask_tensor.sum().item()),
            "case2": int(case2_mask_tensor.sum().item()),
            "case3": int(case3_mask_tensor.sum().item()),
            "case4": int(case4_mask_tensor.sum().item()),
            "case5": int(case5_mask_tensor.sum().item()),
            "case6": int(case6_mask_tensor.sum().item()),
            "background": int(bg_mask_tensor.sum().item()),
        }
        penalty_cpu = penalty_scale.detach().to(device='cpu', dtype=torch.float32)
        penalty_stats = {
            "mean": float(penalty_cpu.mean().item()) if penalty_cpu.numel() else 0.0,
            "max": float(penalty_cpu.max().item()) if penalty_cpu.numel() else 0.0,
            "nonzero_fraction": float((penalty_cpu > 0).float().mean().item()) if penalty_cpu.numel() else 0.0,
        }
        pre_csls_evidence["case_counts"] = case_counts
        pre_csls_evidence["penalty_stats"] = penalty_stats
        required_evidence_keys = (
            "is_mnn1",
            "case1_mask",
            "case2_mask",
            "case3_mask",
            "case4_mask",
            "case5_mask",
            "case6_mask",
            "rho_popularity",
            "hub_level",
            "lock_mask",
            "penalty_scale",
        )
        missing_keys = [key for key in required_evidence_keys if key not in pre_csls_evidence]
        if missing_keys:
            print(f"[PreCSLS][Diag] Missing required evidence keys: {missing_keys}")
        else:
            tensor_device_mismatch = [
                key
                for key in required_evidence_keys
                if isinstance(pre_csls_evidence.get(key), torch.Tensor)
                and pre_csls_evidence[key].device.type != 'cpu'
            ]
            if tensor_device_mismatch:
                print(f"[PreCSLS][Diag] Evidence tensors not on CPU: {tensor_device_mismatch}")
        try:
            print(f"[PreCSLS][Diag] cases={case_counts} penalty_stats={penalty_stats}")
        except Exception:
            pass
    except RuntimeError as rank_err:
        print(f"[GlobalAlign][Evidence] Failed to compute rank evidence: {rank_err}")
    try:
        setattr(config, '_last_pre_csls_evidence', pre_csls_evidence)
    except Exception:
        pass

    alignment_metrics_base = _summarise_logits_alignment(base_logits)
    if alignment_metrics_base:
        print(
            "[GlobalAlign] S_new stats | mean={:.4f} | std={:.4f} | row_mean_std={:.4f} | col_mean_std={:.4f} | aligned_strict={}".format(
                alignment_metrics_base.get('global_mean', float('nan')),
                alignment_metrics_base.get('global_std', float('nan')),
                alignment_metrics_base.get('row_mean_std', float('nan')),
                alignment_metrics_base.get('col_mean_std', float('nan')),
                bool(alignment_metrics_base.get('aligned_strict', False)),
            )
        )
        try:
            setattr(config, '_last_alignment_metrics', alignment_metrics_base)
        except Exception:
            pass

    ada_k_col_values = None
    base_top5_hits = None
    base_topk_indices_cpu = None
    base_logits_csls_cpu = None
    k_row_values = None
    rT_values = None
    rS_values = None
    base_rank_tensor = None
    ada_metrics = {}
    csls_aux = None
    if getattr(config, 'use_csls', False):
        csls_k = _ada_required_int(config, 'csls_k')
        use_ada = bool(getattr(config, 'use_ada_csls', False))
        if use_ada:
            csls_kmin = _ada_required_int(config, 'csls_kmin')
            csls_kmax = _ada_required_int(config, 'csls_kmax')
            max_col = logits_all.size(1)
            csls_kmin = max(1, min(csls_kmin, max_col))
            csls_kmax = max(csls_kmin, min(csls_kmax, max_col))
            csls_k = max(csls_kmin, min(csls_k, csls_kmax))
            csls_alpha = _ada_required_float(config, 'csls_alpha')
            csls_m = _ada_required_int(config, 'csls_m')
            csls_m = max(1, min(csls_m, max_col))
            raw_k_side = getattr(config, 'csls_k_side', ADA_CSLS_DEFAULTS['csls_k_side'])
            csls_k_side = None
            if raw_k_side is not None:
                try:
                    csls_k_side = max(1, min(int(raw_k_side), max_col))
                except (TypeError, ValueError):
                    csls_k_side = max(1, min(int(ADA_CSLS_DEFAULTS['csls_k_side']), max_col))
            col_alpha = _ada_optional_float(config, 'csls_col_alpha')
            col_m = _ada_optional_int(config, 'csls_col_m')
            if col_m is not None:
                col_m = max(1, min(col_m, max_col))
            col_kmin = _ada_optional_int(config, 'csls_col_kmin')
            if col_kmin is not None:
                col_kmin = max(1, min(col_kmin, max_col))
            col_kmax = _ada_optional_int(config, 'csls_col_kmax')
            if col_kmax is not None:
                lower_bound = col_kmin if col_kmin is not None else 1
                col_kmax = max(lower_bound, min(col_kmax, max_col))
            k_col_values = None
            try:
                csls_result = csls_adaptive(
                    logits_all,
                    k0=csls_k,
                    kmin=csls_kmin,
                    kmax=csls_kmax,
                    alpha=csls_alpha,
                    m=csls_m,
                    k_side=csls_k_side,
                    col_alpha=col_alpha,
                    col_m=col_m,
                    col_kmin=col_kmin,
                    col_kmax=col_kmax,
                    return_details=True,
                )
                if isinstance(csls_result, tuple) and len(csls_result) == 2:
                    logits_all, csls_aux = csls_result
                else:
                    logits_all = csls_result
                    csls_aux = None
                    k_row_values = None
                    rT_values = None
                    rS_values = None
                    k_col_values = None

                if isinstance(csls_aux, dict):
                    k_row_values = csls_aux.get("k_row")
                    rT_values = csls_aux.get("rT")
                    rS_values = csls_aux.get("rS")
                    k_col_values = csls_aux.get("k_col")
                else:
                    k_col_values = None

                if isinstance(k_col_values, torch.Tensor) and k_col_values.numel() > 0:
                    ada_k_col_values = k_col_values.detach().to(device='cpu', dtype=torch.float32)
                if isinstance(csls_aux, dict):
                    for key_name, metric_prefix in (
                        ("rho_row", "rho_row"),
                        ("rho_col", "rho_col"),
                        ("scale_row", "scale_row"),
                        ("scale_col", "scale_col"),
                    ):
                        tensor_val = csls_aux.get(key_name) if csls_aux is not None else None
                        if isinstance(tensor_val, torch.Tensor) and tensor_val.numel() > 0:
                            try:
                                ada_metrics[f"{metric_prefix}_mean"] = float(tensor_val.mean().item())
                                ada_metrics[f"{metric_prefix}_median"] = float(tensor_val.median().item())
                            except Exception:
                                pass
                if isinstance(k_row_values, torch.Tensor) and k_row_values.numel() > 0:
                    ada_metrics["k_row_min"] = float(k_row_values.min().item())
                    ada_metrics["k_row_mean"] = float(k_row_values.float().mean().item())
                    ada_metrics["k_row_max"] = float(k_row_values.max().item())
                if isinstance(k_col_values, torch.Tensor) and k_col_values.numel() > 0:
                    ada_metrics["k_col_min"] = float(k_col_values.min().item())
                    ada_metrics["k_col_mean"] = float(k_col_values.float().mean().item())
                    ada_metrics["k_col_max"] = float(k_col_values.max().item())
            except Exception as csls_err:
                print(f"[Ada-CSLS] Warning: fallback至固定k CSLS，原因: {csls_err}")
                try:
                    logits_all = csls_scores(logits_all, k=csls_k)
                except Exception as fallback_err:
                    print(f"[CSLS] Warning: fallback至原始相似度，原因: {fallback_err}")
        else:
            try:
                logits_all = csls_scores(logits_all, k=csls_k)
            except Exception as csls_err:
                print(f"[CSLS] Warning: fallback至原始相似度，原因: {csls_err}")
    try:
        baseline_logits = base_logits.clone()
        if getattr(config, 'use_csls', False):
            try:
                baseline_logits = csls_scores(baseline_logits, k=csls_k)
            except Exception as base_csls_err:
                print(f"[Ada-CSLS][Hub] baseline CSLS fallback: {base_csls_err}")
        topk_hits = torch.zeros(baseline_logits.size(1), device=baseline_logits.device, dtype=torch.float32)
        topk_idx = torch.topk(baseline_logits, k=min(5, baseline_logits.size(1)), dim=1).indices
        flat_idx = topk_idx.reshape(-1)
        topk_hits.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        base_top5_hits = topk_hits.cpu()
        base_topk_indices_cpu = topk_idx.detach().cpu()
        base_logits_csls_cpu = baseline_logits.detach().cpu()
    except Exception as base_hit_err:
        print(f"[Ada-CSLS][Hub] baseline hit computation failed: {base_hit_err}")

    if base_logits_csls_cpu is None:
        base_logits_csls_cpu = base_logits.detach().cpu()

    logits_all_cpu = logits_all.detach().cpu()
    csls_sim_reference = logits_all.detach().cpu()
    logits_all = logits_all_cpu
    base_logits_cpu = base_logits.detach().cpu().to(torch.float32)
    del processed, features_all_device, similarities

    # ----- PoE 融合（可选）: 把结构 expert 从 pre_csls_evidence 构造到 device 并融合回 logits -----
    try:
        if getattr(config, 'enable_poe', False):
            try:
                if not pre_csls_evidence:
                    print("[PoE] pre_csls_evidence 空，跳过 PoE")
                else:
                    # S_geom 基于 CSLS 结果（当前在 CPU），搬到评估 device 上以便构造 S_struct
                    S_geom = logits_all_cpu.to(device)
                    lambda_pen_cfg = getattr(config, 'poe_lambda_pen', None)
                    lambda_bonus_cfg = getattr(config, 'poe_lambda_bonus', None)
                    S_struct = build_structural_expert(pre_csls_evidence, S_geom, device=device, lambda_pen=lambda_pen_cfg, lambda_bonus=lambda_bonus_cfg)
                    beta = float(getattr(config, 'poe_beta', 1.0) or 1.0)
                    geom_std_val = float(S_geom.std(unbiased=False).item()) if S_geom.numel() else 0.0
                    lambda_pen_eff = float(lambda_pen_cfg) if lambda_pen_cfg is not None else float(0.5 * max(geom_std_val, 1e-6))
                    lambda_bonus_eff = float(lambda_bonus_cfg) if lambda_bonus_cfg is not None else float(0.25 * max(geom_std_val, 1e-6))
                    struct_mean = float(S_struct.mean().item()) if S_struct.numel() else 0.0
                    struct_std = float(S_struct.std(unbiased=False).item()) if S_struct.numel() else 0.0
                    try:
                        print(
                            "[PoE] enabled | beta={:.3f} | lambda_pen={:.4f} | lambda_bonus={:.4f} | S_struct mean={:.4f} std={:.4f} | geom_std={:.4f}".format(
                                beta,
                                lambda_pen_eff,
                                lambda_bonus_eff,
                                struct_mean,
                                struct_std,
                                geom_std_val,
                            )
                        )
                    except Exception:
                        pass
                    S_poe = fuse_poe_scores(S_geom, S_struct, beta=beta)
                    # 把融合后的 logits 回搬到 CPU 供后续 evaluation 使用
                    logits_all_cpu = S_poe.detach().cpu()
                    logits_all = logits_all_cpu
                    csls_sim_reference = logits_all_cpu.clone()
                    # 记录 debug 信息
                    try:
                        eval_debug_info.setdefault('poe', {})
                        eval_debug_info['poe']['enabled'] = True
                        eval_debug_info['poe']['beta'] = float(beta)
                        eval_debug_info['poe']['lambda_pen'] = float(lambda_pen_eff)
                        eval_debug_info['poe']['lambda_bonus'] = float(lambda_bonus_eff)
                        eval_debug_info['poe']['S_struct_shape'] = list(S_struct.shape)
                        eval_debug_info['poe']['S_struct_dtype'] = str(S_struct.dtype)
                    except Exception:
                        pass
            except Exception as poe_err:
                print(f"[PoE] 构造/融合失败，跳过 PoE: {poe_err}")
    except Exception:
        # 保守：任何上层错误都不会中断评估
        pass

    labels_cpu_all = labels_all.detach().cpu().to(torch.int64)
    mnn_result = None
    if getattr(config, 'enable_mnn_precheck', True):
        try:
            stage_name = getattr(config, 'stage', None)
            epoch_index = getattr(config, '_current_epoch', None)
            exp_id = getattr(config, 'exp_id', None)
            run_parts = []
            if exp_id:
                run_parts.append(str(exp_id))
            run_parts.append(str(sub))
            run_prefix = "_".join(part for part in run_parts if part)
            detail_limit = None
            detail_limit_raw = getattr(config, 'mnn_precheck_detail_limit', None)
            if detail_limit_raw is not None:
                try:
                    detail_limit = int(detail_limit_raw)
                except (TypeError, ValueError):
                    detail_limit = None
            top_k_mnn = getattr(config, 'mnn_precheck_topk', 5) or 5
            try:
                top_k_mnn = int(top_k_mnn)
            except (TypeError, ValueError):
                top_k_mnn = 5
            mnn_source = str(getattr(config, 'mnn_precheck_source', 'post_csls') or 'post_csls').lower()
            if mnn_source not in {'post_csls', 'pre_csls'}:
                mnn_source = 'post_csls'
            sim_for_mnn = logits_all_cpu if mnn_source == 'post_csls' else base_logits_cpu
            mnn_result = run_mnn_precheck(
                sim_matrix=sim_for_mnn,
                labels=labels_cpu_all,
                stage=stage_name,
                epoch=epoch_index,
                run_prefix=run_prefix,
                output_dir=getattr(config, 'mnn_output_dir', None),
                top_k=top_k_mnn,
                generate_plot=bool(getattr(config, 'mnn_precheck_plot', True)),
                save_details=bool(getattr(config, 'mnn_precheck_save_details', True)),
                detail_limit=detail_limit,
                subject_indices=subject_indices_all,
            )
        except Exception as mnn_err:
            print(f"[MNN-PreCheck] warning: {mnn_err}")
    csls_k_eval = int(getattr(config, 'csls_k', 12) or 12)
    base_logits_csls = base_logits_cpu.clone()
    try:
        base_logits_csls = csls_scores(base_logits_csls, k=csls_k_eval)
    except Exception as base_csls_err:
        print(f"[Ada-CSLS][Debug] base-CSLS failed, fallback to base logits: {base_csls_err}")
        base_logits_csls = base_logits_cpu
    base_logits_csls_cpu = base_logits_csls.detach().clone()

    alignment_strict_flag = bool(alignment_metrics_base.get('aligned_strict', False))

    eval_debug_info = {
        "num_queries": int(labels_cpu_all.numel()),
        "alignment_metrics": alignment_metrics_base,
        "aligned_strict": alignment_strict_flag,
        "evidence_pre_csls": {
            "row_rank_shape": list(pre_csls_evidence["row_rank"].shape) if "row_rank" in pre_csls_evidence else [],
            "col_rank_shape": list(pre_csls_evidence["col_rank"].shape) if "col_rank" in pre_csls_evidence else [],
            "row_rank_dtype": str(pre_csls_evidence["row_rank"].dtype) if "row_rank" in pre_csls_evidence else None,
            "col_rank_dtype": str(pre_csls_evidence["col_rank"].dtype) if "col_rank" in pre_csls_evidence else None,
            "mnn1_mask_shape": list(pre_csls_evidence["mnn1_mask"].shape) if "mnn1_mask" in pre_csls_evidence else [],
            "mnn1_indices_shape": list(pre_csls_evidence["mnn1_indices"].shape) if "mnn1_indices" in pre_csls_evidence else [],
            "mnn1_count": int(pre_csls_evidence["mnn1_indices"].shape[0]) if "mnn1_indices" in pre_csls_evidence else 0,
            "has_mnn1_col_shape": list(pre_csls_evidence["has_mnn1_col"].shape) if "has_mnn1_col" in pre_csls_evidence else [],
            "row_margin_shape": list(pre_csls_evidence["row_margin_top1_top2"].shape) if "row_margin_top1_top2" in pre_csls_evidence else [],
            "row_margin_dtype": str(pre_csls_evidence["row_margin_top1_top2"].dtype) if "row_margin_top1_top2" in pre_csls_evidence else None,
            "col_margin_shape": list(pre_csls_evidence["col_margin_top1_top2"].shape) if "col_margin_top1_top2" in pre_csls_evidence else [],
            "col_margin_dtype": str(pre_csls_evidence["col_margin_top1_top2"].dtype) if "col_margin_top1_top2" in pre_csls_evidence else None,
            "rho_shape": list(pre_csls_evidence["rho_popularity"].shape) if "rho_popularity" in pre_csls_evidence else [],
            "rho_dtype": str(pre_csls_evidence["rho_popularity"].dtype) if "rho_popularity" in pre_csls_evidence else None,
            "hub_level_shape": list(pre_csls_evidence["hub_level"].shape) if "hub_level" in pre_csls_evidence else [],
            "hub_level_dtype": str(pre_csls_evidence["hub_level"].dtype) if "hub_level" in pre_csls_evidence else None,
            "hub_quantiles": pre_csls_evidence.get("hub_quantiles", {}),
            "row_top1_mask_shape": list(pre_csls_evidence["row_top1_mask"].shape) if "row_top1_mask" in pre_csls_evidence else [],
            "row_topk_mask_shape": list(pre_csls_evidence["row_topk_mask"].shape) if "row_topk_mask" in pre_csls_evidence else [],
            "row_top5_mask_shape": list(pre_csls_evidence["row_top5_mask"].shape) if "row_top5_mask" in pre_csls_evidence else [],
            "col_top1_mask_shape": list(pre_csls_evidence["col_top1_mask"].shape) if "col_top1_mask" in pre_csls_evidence else [],
            "col_topk_mask_shape": list(pre_csls_evidence["col_topk_mask"].shape) if "col_topk_mask" in pre_csls_evidence else [],
            "col_top5_mask_shape": list(pre_csls_evidence["col_top5_mask"].shape) if "col_top5_mask" in pre_csls_evidence else [],
            "row_topk_safe_K": int(pre_csls_evidence.get("row_safe_k", getattr(config, 'pre_csls_row_safe_k', 5) or 5)),
            "col_topk_safe_K": int(pre_csls_evidence.get("col_safe_k", getattr(config, 'pre_csls_col_safe_k', 5) or 5)),
            "case2_mask_shape": list(pre_csls_evidence["case2_mask"].shape) if "case2_mask" in pre_csls_evidence else [],
            "case2_tau_row": pre_csls_evidence.get("case2_tau_row", None),
            "case3_mask_shape": list(pre_csls_evidence["case3_mask"].shape) if "case3_mask" in pre_csls_evidence else [],
            "case4_mask_shape": list(pre_csls_evidence["case4_mask"].shape) if "case4_mask" in pre_csls_evidence else [],
            "case5_mask_shape": list(pre_csls_evidence["case5_mask"].shape) if "case5_mask" in pre_csls_evidence else [],
            "strong_protect_mask_shape": list(pre_csls_evidence["strong_protect_mask"].shape) if "strong_protect_mask" in pre_csls_evidence else [],
            "background_mask_shape": list(pre_csls_evidence["background_mask"].shape) if "background_mask" in pre_csls_evidence else [],
            "lock_mask_shape": list(pre_csls_evidence["lock_mask"].shape) if "lock_mask" in pre_csls_evidence else [],
            "penalty_scale_shape": list(pre_csls_evidence["penalty_scale"].shape) if "penalty_scale" in pre_csls_evidence else [],
            "case_id_matrix_shape": list(pre_csls_evidence["case_id_matrix"].shape) if "case_id_matrix" in pre_csls_evidence else [],
        },
        "logit_alignment": {
            "global_temp_scale": float(global_scale),
            "global_temp_bias": float(global_bias),
            "zscore_mean": None if zscore_mean is None else float(zscore_mean),
            "zscore_std": None if zscore_std is None else float(zscore_std),
            "auto_center": auto_center_applied,
            "auto_scale": auto_scale_applied,
            "auto_center_mean": (
                None if getattr(config, '_last_auto_logits_mean', None) is None
                else float(getattr(config, '_last_auto_logits_mean'))
            ),
            "auto_center_std": (
                None if getattr(config, '_last_auto_logits_std', None) is None
                else float(getattr(config, '_last_auto_logits_std'))
            ),
            "aligned_strict": alignment_strict_flag,
        },
    }
    if mnn_result:
        eval_debug_info["mnn_precheck"] = mnn_result
    if ada_metrics:
        eval_debug_info["ada_csls_metrics"] = {k: float(v) for k, v in ada_metrics.items() if isinstance(v, (int, float))}
    try:
        setattr(config, '_last_eval_rank_debug', eval_debug_info)
    except Exception:
        pass

    diag_enabled = bool(getattr(config, 'diag_dump_dir', None))
    if diag_enabled:
        diag_base_logits = base_logits_cpu.clone()
        labels_cpu = labels_cpu_all
        diag_meta = {
            "subject": str(sub),
            "stage": getattr(config, 'stage', None),
            "ks": [int(k) for k in ks],
            "num_queries": int(diag_base_logits.size(0)),
            "num_classes": int(diag_base_logits.size(1)),
            "use_csls": bool(getattr(config, 'use_csls', False)),
            "use_ada_csls": bool(getattr(config, 'use_ada_csls', False)),
            "seed": getattr(config, 'seed', None),
        }
        diag_meta["logit_alignment"] = {
            "global_temp_scale": float(global_scale),
            "global_temp_bias": float(global_bias),
            "zscore_mean": None if zscore_mean is None else float(zscore_mean),
            "zscore_std": None if zscore_std is None else float(zscore_std),
        }
        diag_meta["alignment_metrics"] = alignment_metrics_base
        diag_meta["aligned_strict"] = alignment_strict_flag
        if ada_metrics:
            diag_meta["ada_selective_metrics"] = {
                "rho_row_mean": ada_metrics.get("rho_row_mean"),
                "rho_row_median": ada_metrics.get("rho_row_median"),
                "rho_col_mean": ada_metrics.get("rho_col_mean"),
                "rho_col_median": ada_metrics.get("rho_col_median"),
                "scale_row_mean": ada_metrics.get("scale_row_mean"),
                "scale_row_median": ada_metrics.get("scale_row_median"),
                "scale_col_mean": ada_metrics.get("scale_col_mean"),
                "scale_col_median": ada_metrics.get("scale_col_median"),
                "k_row_min": ada_metrics.get("k_row_min"),
                "k_row_mean": ada_metrics.get("k_row_mean"),
                "k_row_max": ada_metrics.get("k_row_max"),
                "k_col_min": ada_metrics.get("k_col_min"),
                "k_col_mean": ada_metrics.get("k_col_mean"),
                "k_col_max": ada_metrics.get("k_col_max"),
            }
        diag_meta["csls_params"] = {
            "csls_k": int(getattr(config, 'csls_k', ADA_CSLS_DEFAULTS['csls_k'])),
            "csls_kmin": getattr(config, 'csls_kmin', ADA_CSLS_DEFAULTS['csls_kmin']),
            "csls_kmax": getattr(config, 'csls_kmax', ADA_CSLS_DEFAULTS['csls_kmax']),
            "csls_alpha": getattr(config, 'csls_alpha', ADA_CSLS_DEFAULTS['csls_alpha']),
            "csls_m": getattr(config, 'csls_m', ADA_CSLS_DEFAULTS['csls_m']),
            "csls_k_side": getattr(config, 'csls_k_side', ADA_CSLS_DEFAULTS['csls_k_side']),
            "csls_col_alpha": getattr(config, 'csls_col_alpha', ADA_CSLS_DEFAULTS['csls_col_alpha']),
            "csls_col_m": getattr(config, 'csls_col_m', ADA_CSLS_DEFAULTS['csls_col_m']),
            "csls_col_kmin": getattr(config, 'csls_col_kmin', ADA_CSLS_DEFAULTS['csls_col_kmin']),
            "csls_col_kmax": getattr(config, 'csls_col_kmax', ADA_CSLS_DEFAULTS['csls_col_kmax']),
            "pre_csls_bg_penalty": getattr(config, 'pre_csls_bg_penalty', 1.0),
            "pre_csls_case6_penalty_high": getattr(config, 'pre_csls_case6_penalty_high', 0.5),
            "pre_csls_case6_penalty_mid": getattr(config, 'pre_csls_case6_penalty_mid', 0.25),
        }
        diag_payload = {
            "S": diag_base_logits,
            "labels": labels_cpu,
            "meta": diag_meta,
            "evidence_pre_csls": pre_csls_evidence,
        }
        if getattr(config, 'diag_dump_keep_post', False):
            diag_payload["csls_logits"] = logits_all_cpu.clone()
        if ada_metrics:
            diag_payload["ada_metrics"] = {k: float(v) for k, v in ada_metrics.items() if isinstance(v, (int, float))}
        if ada_k_col_values is not None:
            diag_payload["k_col"] = ada_k_col_values.clone()
            k_col_np = ada_k_col_values.numpy()
            diag_meta["k_col_stats"] = {
                "min": float(k_col_np.min()),
                "mean": float(k_col_np.mean()),
                "max": float(k_col_np.max()),
                "p10": float(np.percentile(k_col_np, 10)),
                "p50": float(np.percentile(k_col_np, 50)),
                "p90": float(np.percentile(k_col_np, 90)),
            }
        if k_row_values is not None:
            k_row_cpu = k_row_values.detach().cpu().to(torch.int64)
            diag_payload["k_row"] = k_row_cpu
            k_row_np = k_row_cpu.numpy()
            diag_meta["k_row_stats"] = {
                "min": float(k_row_np.min()),
                "mean": float(k_row_np.mean()),
                "max": float(k_row_np.max()),
                "p10": float(np.percentile(k_row_np, 10)),
                "p50": float(np.percentile(k_row_np, 50)),
                "p90": float(np.percentile(k_row_np, 90)),
            }
        if rT_values is not None:
            diag_payload["rT"] = rT_values.detach().cpu().to(torch.float32)
        if rS_values is not None:
            diag_payload["rS"] = rS_values.detach().cpu().to(torch.float32)
        if base_top5_hits is not None:
            diag_payload["base_top5_hits"] = base_top5_hits.clone()
        if base_topk_indices_cpu is not None:
            diag_payload["base_top5_indices"] = base_topk_indices_cpu.clone()
        if csls_aux is not None:
            def _save_aux_tensor(src_key: str, dst_key: Optional[str] = None, dtype: torch.dtype = torch.float32):
                tensor_val = csls_aux.get(src_key)
                if tensor_val is None:
                    return
                if isinstance(tensor_val, torch.Tensor) and tensor_val.numel() > 0:
                    target_key = dst_key or src_key
                    diag_payload[target_key] = tensor_val.detach().cpu().to(dtype)

            aux_tensor_map = (
                ("rho_row", "rho_row"),
                ("rho_col", "rho_col"),
                ("scale_row", "scale_row"),
                ("scale_col", "scale_col"),
            )
            for src_key, dst_key in aux_tensor_map:
                _save_aux_tensor(src_key, dst_key)
        try:
            setattr(config, '_last_eval_diag', diag_payload)
        except Exception:
            pass
    else:
        try:
            setattr(config, '_last_eval_diag', None)
        except Exception:
            pass
    del base_logits

    num_samples, num_classes = logits_all.shape
    negatives_cache = {}

    candidate_top5_hits = None
    if ada_k_col_values is not None and ada_k_col_values.numel() == num_classes:
        candidate_top5_hits = torch.zeros(num_classes, dtype=torch.int64)

    per_subject_stats = None
    if has_subject_indices and subject_indices_all is not None:
        per_subject_stats = {
            k: defaultdict(lambda: {
                "total": 0,
                "correct_top1": 0,
                "top5_total": 0,
                "correct_top5": 0,
            })
            for k in ks
        }

    metrics_state = {
        k: {
            "correct_total": 0,
            "total_samples": 0,
            "top5_correct_total": 0,
            "top5_total_samples": 0,
        }
        for k in ks
    }

    top5_eval_set = {k for k in ks if (k >= 50) or (k == num_classes)}

    labels_list = labels_all.tolist()
    if subject_indices_all is not None:
        subject_list = subject_indices_all.tolist()
    else:
        subject_list = [None] * num_samples

    for idx in range(num_samples):
        row_logits = logits_all[idx]
        label = labels_list[idx]
        subject_idx_val = subject_list[idx]
        if subject_idx_val is not None and subject_lookup and subject_idx_val in subject_lookup:
            subject_key = subject_lookup[subject_idx_val]
        else:
            subject_key = subject_idx_val if subject_idx_val is not None else None

        if candidate_top5_hits is not None and num_classes > 0:
            top5_k = min(5, num_classes)
            if top5_k > 0:
                top5_indices = torch.topk(row_logits, k=top5_k).indices
                candidate_top5_hits[top5_indices] += 1

        if label not in negatives_cache:
            negatives_cache[label] = [cls for cls in range(num_classes) if cls != label]
        negatives = negatives_cache[label]

        for k in ks:
            state = metrics_state[k]
            state["total_samples"] += 1

            if k == num_classes:
                selected_logits = row_logits
                predicted_label = int(torch.argmax(selected_logits).item())
                correct_flag = 1 if predicted_label == label else 0
                compute_top5 = num_classes >= 5
                if compute_top5:
                    top_count = min(5, num_classes)
                    top_indices = torch.topk(selected_logits, top_count).indices.tolist()
                    top5_flag = 1 if label in top_indices else 0
                else:
                    top5_flag = 0
            else:
                if k > num_classes:
                    raise ValueError(f"Requested k={k} exceeds number of classes {num_classes}.")
                if k == 1:
                    selected_indices = [label]
                else:
                    selected_indices = random.sample(negatives, k - 1) + [label]
                selected_tensor = torch.tensor(selected_indices, dtype=torch.long)
                selected_logits = row_logits.index_select(0, selected_tensor)
                top_idx = int(torch.argmax(selected_logits).item())
                predicted_label = selected_indices[top_idx]
                correct_flag = 1 if predicted_label == label else 0
                compute_top5 = k in top5_eval_set and k >= 5
                if compute_top5:
                    top_count = min(5, len(selected_indices))
                    topk_indices = torch.topk(selected_logits, top_count).indices.tolist()
                    top5_flag = 1 if label in [selected_indices[i] for i in topk_indices] else 0
                else:
                    top5_flag = 0

            state["correct_total"] += correct_flag
            if compute_top5:
                state["top5_total_samples"] += 1
                state["top5_correct_total"] += top5_flag

            if per_subject_stats is not None and subject_key is not None:
                subj_stats = per_subject_stats[k][subject_key]
                subj_stats["total"] += 1
                subj_stats["correct_top1"] += correct_flag
                if compute_top5:
                    subj_stats["top5_total"] += 1
                    subj_stats["correct_top5"] += top5_flag

    macro_average = getattr(config, 'stage', None) == 'tune'
    results_by_k = {}

    for k in ks:
        state = metrics_state[k]
        total = max(1, state["total_samples"])
        top1_accuracy = state["correct_total"] / total

        if state["top5_total_samples"] > 0:
            top5_accuracy = state["top5_correct_total"] / state["top5_total_samples"]
        else:
            top5_accuracy = None

        if macro_average and per_subject_stats is not None:
            subj_stats = per_subject_stats[k]
            subject_top1 = [val["correct_top1"] / val["total"] for val in subj_stats.values() if val["total"] > 0]
            if subject_top1:
                top1_accuracy = sum(subject_top1) / len(subject_top1)
            subject_top5 = [
                val["correct_top5"] / val["top5_total"]
                for val in subj_stats.values()
                if val["top5_total"] > 0
            ]
            if subject_top5:
                top5_accuracy = sum(subject_top5) / len(subject_top5)
            elif state["top5_total_samples"] > 0 and top5_accuracy is None:
                top5_accuracy = state["top5_correct_total"] / state["top5_total_samples"]

        results_by_k[k] = {
            "top1": top1_accuracy,
            "top5": top5_accuracy,
        }

    if ada_k_col_values is not None:
        k_col_tensor = ada_k_col_values.to(torch.float32)

        if base_top5_hits is not None and base_top5_hits.numel() == k_col_tensor.numel():
            base_hits_tensor = base_top5_hits.to(torch.float32)
            base_corr = _spearman_corr_torch(k_col_tensor, base_hits_tensor)
            print(f"[Ada-CSLS][Hub] Pre-CSLS Spearman(k_col, top5_hits)={base_corr:.4f}")

        if candidate_top5_hits is not None:
            hits_tensor = candidate_top5_hits.to(torch.float32)
            spearman_corr = _spearman_corr_torch(k_col_tensor, hits_tensor)
            top_hub_k = min(5, num_classes)
            hub_info = "N/A"
            if top_hub_k > 0:
                top_hits_vals, top_hits_idx = torch.topk(hits_tensor, k=top_hub_k)
                entries = []
                for idx, hit in zip(top_hits_idx.tolist(), top_hits_vals.tolist()):
                    k_val = float(k_col_tensor[int(idx)].item()) if k_col_tensor.numel() > int(idx) else float('nan')
                    entries.append(f"cls={int(idx)} hit={int(hit)} k_col={k_val:.2f}")
                hub_info = ", ".join(entries) if entries else "N/A"
            print(
                f"[Ada-CSLS][Hub] Post-CSLS Spearman(k_col, top5_hits)={spearman_corr:.4f} | top hubs: {hub_info}"
            )

        try:
            csls_scores_tensor = csls_sim_reference
            if csls_scores_tensor is not None and csls_scores_tensor.dim() == 2:
                topk_cap = min(5, csls_scores_tensor.size(1))
                if topk_cap > 0:
                    top5_csls = csls_scores_tensor.topk(k=topk_cap, dim=1, largest=True, sorted=False).indices
                    flat_idx = top5_csls.reshape(-1)
                    hits_post = torch.zeros(csls_scores_tensor.size(1), device=csls_scores_tensor.device, dtype=torch.float32)
                    hits_post.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
                    hits_post = hits_post.cpu()
                    post_corr = _spearman_corr_torch(k_col_tensor, hits_post.to(torch.float32))
                    if base_top5_hits is not None and base_top5_hits.numel() == hits_post.numel():
                        diff_hits = (hits_post - base_top5_hits.to(hits_post.dtype)).abs().sum().item()
                        changed_candidates = int((hits_post != base_top5_hits.to(hits_post.dtype)).sum().item())
                        changed_queries = None
                        if base_topk_indices_cpu is not None:
                            try:
                                csls_topk_cpu = top5_csls.detach().cpu()
                                changed_queries = int((csls_topk_cpu != base_topk_indices_cpu).any(dim=1).sum().item())
                            except Exception:
                                changed_queries = None
                        print(
                            f"[Ada-CSLS][Hub] Post-CSLS (direct) Spearman(k_col, top5_hits)={post_corr:.4f}"
                            f" | Δhits_sum={diff_hits:.0f} | changed_candidates={changed_candidates}"
                            + (f" | changed_queries={changed_queries}" if changed_queries is not None else "")
                        )
                    else:
                        print(f"[Ada-CSLS][Hub] Post-CSLS (direct) Spearman(k_col, top5_hits)={post_corr:.4f}")
        except Exception as post_err:
            print(f"[Ada-CSLS][Hub] Post-CSLS hit recompute failed: {post_err}")

    if single_return:
        only_k = ks[0]
        top5_value = results_by_k[only_k]["top5"]
        if top5_value is None:
            top5_value = 0.0
        return average_loss, results_by_k[only_k]["top1"], top5_value, eval_debug_info

    return average_loss, results_by_k, eval_debug_info

def main_train_loop(sub, current_time, eeg_model, train_dataloader, test_dataloader, optimizer, device, text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=None):
    stage = getattr(config, 'stage', getattr(config, '_phase', 'final'))
    stage = stage if stage in {'tune', 'final'} else 'final'

    subject_str = str(sub)
    subject_tag = re.sub(r"[^0-9A-Za-z_-]+", "-", subject_str).strip('-_') or 'subject'

    logger = wandb_logger(config) if logger else None
    if logger is not None:
        try:
            logger.watch(eeg_model, log='all')
        except Exception:
            pass
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_root = os.path.join(script_dir, "outputs", "analysis_fig")
    exp_name_for_fig = str(getattr(config, 'exp_id', '') or '').strip() or "default_exp"
    analysis_dir = os.path.join(analysis_root, exp_name_for_fig)
    os.makedirs(analysis_dir, exist_ok=True)

    best_accuracy = -float('inf')
    best_top5 = -float('inf')
    best_checkpoint_path = None
    best_epoch_info = None
    best_top5_info = None
    best_accuracy_epoch = None
    best_top5_epoch = None
    results = []  # List to store results for each epoch

    tune_early_stop_enabled = False
    tune_min_epochs = 0
    tune_patience = 0
    tune_min_delta = 0.0
    tune_reset_on_lr_drop = False
    tune_stale_epochs = 0
    tune_best_metric = -float('inf')
    tune_stop_epoch = None
    tune_prev_lr = None
    tune_checkpoint_path = None
    tune_save_best = True
    tune_best_epoch = None
    tune_early_stop_triggered = False
    tune_break_after_epoch = False

    prev_rank_debug = None

    base_dir = None
    if stage == 'final':
        exp_name = str(getattr(config, 'exp_id', '') or '').strip()
        root_dir = os.path.join("./models/contrast", exp_name) if exp_name else "./models/contrast"
        if getattr(config, 'insubject', False):
            base_dir = os.path.join(root_dir, config.encoder_type, str(sub), current_time)
        else:
            base_dir = os.path.join(root_dir, "across", config.encoder_type, current_time)
        os.makedirs(base_dir, exist_ok=True)
        best_checkpoint_path = os.path.join(base_dir, f"{subject_tag}_best_top5.pth")
    else:
        tune_early_stop_enabled = bool(getattr(config, 'tune_early_stop', False))
        if tune_early_stop_enabled:
            tune_min_epochs = max(0, int(getattr(config, 'tune_min_epochs', 0)))
            tune_patience = max(0, int(getattr(config, 'tune_patience', 0)))
            tune_min_delta = float(getattr(config, 'tune_min_delta', 0.0))
            tune_reset_on_lr_drop = bool(getattr(config, 'tune_reset_on_lr_drop', False))
            tune_checkpoint_path = getattr(config, 'tune_best_ckpt_path', None)
            tune_save_best = bool(getattr(config, 'tune_save_best', True))
            if tune_checkpoint_path:
                os.makedirs(os.path.dirname(tune_checkpoint_path), exist_ok=True)
    
    for epoch in range(config.epochs):
        encoder_display = getattr(config, 'chose_eeg_encoder', getattr(config, 'encoder_choice', 'sattc'))
        print(
            f"[{stage.upper()}][Epoch {epoch + 1}/{config.epochs}] 当期选用EEG编码器: {encoder_display}"
        )
        setattr(config, '_current_epoch', int(epoch))
        train_sample_count = None
        train_batch_size = getattr(train_dataloader, 'batch_size', None)
        try:
            if train_batch_size and len(train_dataloader):
                train_sample_count = len(train_dataloader) * train_batch_size
        except Exception:
            train_sample_count = None
        if not train_sample_count:
            try:
                train_sample_count = len(train_dataloader.dataset)
            except Exception:
                train_sample_count = None

        eval_sample_count = None
        eval_batch_size = getattr(test_dataloader, 'batch_size', None)
        try:
            eval_sample_count = len(test_dataloader.dataset)
        except Exception:
            eval_sample_count = None
        if not eval_sample_count and eval_batch_size:
            try:
                eval_sample_count = len(test_dataloader) * eval_batch_size
            except Exception:
                eval_sample_count = None

        extra_eval_runs = 0

        epoch_total_samples = 0
        if train_sample_count:
            epoch_total_samples += train_sample_count
        if eval_sample_count:
            epoch_total_samples += eval_sample_count
            epoch_bar = None
        if epoch_total_samples > 0:
            epoch_label = f"{stage.upper()} Progress"
            if stage == 'final':
                bar_format = "{desc}: |{bar}| {n_fmt}/{total_fmt} ({percentage:3.0f}%) [{elapsed}<{remaining}]"
            else:
                bar_format = "{desc}: |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            epoch_bar = tqdm.tqdm(
                total=epoch_total_samples,
                desc=epoch_label,
                unit="sample",
                leave=False,
                position=0,
                bar_format=bar_format,
            )

        def advance_epoch(samples):
            if epoch_bar is not None:
                epoch_bar.update(samples)

        # Train the model
        train_loss, train_accuracy, _ = train_model(
            sub,
            eeg_model,
            train_dataloader,
            optimizer,
            device,
            text_features_train_all,
            img_features_train_all,
            config=config,
            progress_callback=advance_epoch if epoch_bar is not None else None,
        )
        if stage == 'final' and (epoch + 1) % 5 == 0:
            if base_dir is not None:
                os.makedirs(base_dir, exist_ok=True)
                file_path = os.path.join(base_dir, f"{epoch + 1}.pth")
                torch.save(eeg_model.state_dict(), file_path)
                print(f"model saved in {file_path}!")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)


        # Evaluate the model
        eval_kwargs = dict(
            sub=sub,
            eeg_model=eeg_model,
            dataloader=test_dataloader,
            device=device,
            text_features_all=text_features_test_all,
            img_features_all=img_features_test_all,
            config=config,
            progress_callback=advance_epoch if epoch_bar is not None else None,
        )

        if stage == 'final':
            eval_k_values = [200, 2, 4, 10, 50, 100]
        else:
            eval_k_values = [200]
        test_loss, metrics_by_k, eval_debug = evaluate_model(k_values=eval_k_values, **eval_kwargs)
        metrics_200 = metrics_by_k[200]
        test_accuracy = metrics_200["top1"]
        top5_acc = metrics_200.get("top5") or 0.0

        if stage == 'final':
            v2_acc = metrics_by_k[2]["top1"]
            v4_acc = metrics_by_k[4]["top1"]
            v10_acc = metrics_by_k[10]["top1"]
            v50_acc = metrics_by_k[50]["top1"]
            v100_acc = metrics_by_k[100]["top1"]
            v50_top5_acc = metrics_by_k[50].get("top5") or 0.0
            v100_top5_acc = metrics_by_k[100].get("top5") or 0.0
        else:
            v2_acc = v4_acc = v10_acc = float('nan')
            v50_acc = v100_acc = float('nan')
            v50_top5_acc = v100_top5_acc = float('nan')

        alignment_strict_flag = bool(eval_debug.get("aligned_strict", False))
        ada_metrics_snapshot = eval_debug.get("ada_csls_metrics") if isinstance(eval_debug, dict) else None
        if ada_metrics_snapshot:
            def _fmt_metric(value: Optional[float], precision: int = 2) -> str:
                try:
                    return f"{float(value):.{precision}f}"
                except (TypeError, ValueError):
                    return "N/A"

            print(
                f"[Ada-CSLS][Epoch {epoch + 1:03d}] "
                f"k_row[min/mean/max]={_fmt_metric(ada_metrics_snapshot.get('k_row_min'), 1)}/"
                f"{_fmt_metric(ada_metrics_snapshot.get('k_row_mean'), 2)}/"
                f"{_fmt_metric(ada_metrics_snapshot.get('k_row_max'), 1)}"
                f" | k_col[min/mean/max]={_fmt_metric(ada_metrics_snapshot.get('k_col_min'), 1)}/"
                f"{_fmt_metric(ada_metrics_snapshot.get('k_col_mean'), 2)}/"
                f"{_fmt_metric(ada_metrics_snapshot.get('k_col_max'), 1)}"
                f" | rho_row_med={_fmt_metric(ada_metrics_snapshot.get('rho_row_median'), 4)}"
                f" | rho_col_med={_fmt_metric(ada_metrics_snapshot.get('rho_col_median'), 4)}"
            )
        if epoch_bar is not None:
            epoch_bar.close()
        test_losses.append(test_loss)
        test_accuracies.append(top5_acc)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)
        
        # Append results for this epoch
        epoch_results = {
            "epoch": epoch + 1,
            # "train_loss": train_loss,
            # "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "v2_acc": v2_acc,
            "v4_acc": v4_acc,
            "v10_acc": v10_acc,
            "top5_acc": top5_acc,
            "v50_acc": v50_acc,
            "v100_acc": v100_acc,
            "v50_top5_acc": v50_top5_acc,
            "v100_top5_acc": v100_top5_acc,
            "aligned_strict": alignment_strict_flag,
        }
        epoch_results["stage"] = stage
        results.append(epoch_results)

        tune_epoch_note = None

        diag_payload = getattr(config, '_last_eval_diag', None)
        if diag_payload is not None:
            diag_dump_dir = getattr(config, 'diag_dump_dir', None)
            diag_limit = int(getattr(config, 'diag_dump_limit', 0) or 0)
            should_dump = bool(diag_dump_dir) and (diag_limit <= 0 or (epoch < diag_limit))
            if should_dump:
                meta = diag_payload.get("meta", {})
                meta.update({
                    "epoch": epoch + 1,
                    "eval_top1@200": test_accuracy,
                    "eval_top5@200": top5_acc,
                    "eval_top1@2": v2_acc,
                    "eval_top1@4": v4_acc,
                    "eval_top1@10": v10_acc,
                    "eval_top1@50": v50_acc,
                    "eval_top1@100": v100_acc,
                    "eval_top5@50": v50_top5_acc,
                    "eval_top5@100": v100_top5_acc,
                    "train_top5": train_accuracy,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "timestamp": datetime.datetime.now().isoformat(timespec='seconds'),
                })
                diag_payload["meta"] = meta
                exp_name_raw = str(getattr(config, 'exp_id', '') or '').strip()
                exp_name_tag = re.sub(r"[^0-9A-Za-z_-]+", '-', exp_name_raw).strip('-_') or 'exp'
                stage_tag = stage or 'unknown'
                save_dir = os.path.join(diag_dump_dir, exp_name_tag, stage_tag, subject_tag)
                os.makedirs(save_dir, exist_ok=True)
                prefix = getattr(config, 'diag_dump_prefix', 'epoch') or 'epoch'
                filename = f"{prefix}_{epoch + 1:03d}.pt"
                diag_path = os.path.join(save_dir, filename)
                torch.save(diag_payload, diag_path)
                print(f"             - [Diag] Saved Ada-CSLS snapshot to {diag_path}")
            try:
                setattr(config, '_last_eval_diag', None)
            except Exception:
                pass

        prev_rank_debug = eval_debug

        if tune_early_stop_enabled:
            metric_value = top5_acc if top5_acc is not None else float('-inf')
            try:
                current_lr = float(optimizer.param_groups[0].get('lr', None)) if optimizer.param_groups else None
            except (TypeError, ValueError):
                current_lr = None
            if current_lr is not None and tune_prev_lr is not None and tune_reset_on_lr_drop and current_lr < tune_prev_lr - 1e-12:
                tune_stale_epochs = 0
                tune_epoch_note = "[EarlyStop] LR decreased; patience reset to 0"
            improvement = metric_value >= tune_best_metric + tune_min_delta
            if improvement:
                tune_best_metric = metric_value
                tune_stale_epochs = 0
                if tune_save_best and tune_checkpoint_path:
                    try:
                        torch.save(eeg_model.state_dict(), tune_checkpoint_path)
                    except Exception as save_err:
                        print(f"Warning: failed to store tune checkpoint at {tune_checkpoint_path}: {save_err}")
                tune_best_epoch = epoch + 1
                tune_epoch_note = (f"[EarlyStop] ✅ Top-5@200 improved to {metric_value:.4f} (epoch {epoch + 1}); patience reset")
            else:
                tune_stale_epochs += 1
                if tune_best_metric > -float('inf'):
                    delta_val = metric_value - tune_best_metric
                    delta_text = f"delta={delta_val:.4f}"
                else:
                    delta_text = "delta=N/A"
                patience_text = f"{tune_stale_epochs}/{tune_patience}" if tune_patience else "∞"
                tune_epoch_note = (f"[EarlyStop] ⏳ No improvement ({delta_text}); patience {patience_text}")
            if current_lr is not None:
                tune_prev_lr = current_lr
            if (epoch + 1) >= tune_min_epochs and tune_patience > 0 and tune_stale_epochs >= tune_patience:
                tune_early_stop_triggered = True
                tune_stop_epoch = epoch + 1
                if tune_epoch_note:
                    tune_epoch_note += " -> early-stop condition met; will exit after this epoch"
                else:
                    tune_epoch_note = "[EarlyStop] ⛔ Early-stop condition met; will exit after this epoch"
                tune_break_after_epoch = True

        # If the test accuracy of the current epoch is the best, save the model and related information
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_accuracy_epoch = epoch + 1
            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "v2_acc":v2_acc,
                "v4_acc":v4_acc,
                "v10_acc":v10_acc,
                "aligned_strict": alignment_strict_flag
            }
        if logger is not None:
            log_payload = {
                "Train Loss": train_loss,
                "Train Top-5 Accuracy": train_accuracy,
                "Test Loss": test_loss,
                "Test Accuracy": test_accuracy,
                "Epoch": epoch
            }
            if stage == 'final':
                log_payload.update({
                    "v2 Accuracy": v2_acc,
                    "v4 Accuracy": v4_acc,
                    "v10 Accuracy": v10_acc,
                })
            logger.log(log_payload)
        
        train_sid_mode = getattr(config, 'train_subject_ids', 'spoof')
        test_sid_mode = getattr(config, 'test_subject_ids', 'target')
        sid_dropout = float(getattr(config, 'subject_dropout_p', 0.0) or 0.0)
        saw_enabled = bool(getattr(config, 'use_saw', False))
        saw_shrink = float(getattr(config, 'saw_shrink', 0.2))
        saw_diag = bool(getattr(config, 'saw_diag', False))

        print(
            f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Top-5 Accuracy: {train_accuracy:.4f}"
            f" |  seed : {config.seed} |  stage : [{stage}] |  Subject: {subject_str}"
            f" | subject_ids(train/test)=[{train_sid_mode}/{test_sid_mode}] | subject_dropout_p={sid_dropout:.2f}"
        )
        csls_enabled = bool(getattr(config, 'use_csls', False))
        csls_k = _ada_required_int(config, 'csls_k')
        temp_val = float(getattr(config, 'temp', 1.0) or 1.0)
        ada_enabled = bool(getattr(config, 'use_ada_csls', False))
        if ada_enabled:
            csls_kmin = _ada_required_int(config, 'csls_kmin')
            csls_kmax = _ada_required_int(config, 'csls_kmax')
            csls_alpha = _ada_required_float(config, 'csls_alpha')
            csls_m = _ada_required_int(config, 'csls_m')
            csls_k_side = getattr(config, 'csls_k_side', ADA_CSLS_DEFAULTS['csls_k_side'])
            col_alpha = _ada_optional_float(config, 'csls_col_alpha')
            col_m = _ada_optional_int(config, 'csls_col_m')
            col_kmin = _ada_optional_int(config, 'csls_col_kmin')
            col_kmax = _ada_optional_int(config, 'csls_col_kmax')
            ada_extra = (
                f" | Ada-CSLS:{ada_enabled} (row_k=[{csls_kmin},{csls_kmax}], row_alpha={csls_alpha:.2f}, row_m={csls_m}, k_side={csls_k_side if csls_k_side is not None else 'auto'}; "
                f"col_alpha={f'{col_alpha:.2f}' if col_alpha is not None else 'auto'}, col_m={col_m if col_m is not None else 'auto'}, "
                f"col_k=[{col_kmin if col_kmin is not None else 'auto'},{col_kmax if col_kmax is not None else 'auto'}])"
            )
        else:
            ada_extra = ""
        print(
            f"             - SAW Enabled:{saw_enabled} | saw_shrink={saw_shrink:.3f} | saw_diag={saw_diag}"
            f" | CSLS Enabled:{csls_enabled} | csls_k={csls_k} | temp={temp_val:.3f}{ada_extra}"
        )
        if stage == 'final':
            print(f"             - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc} - v50 Accuracy:{v50_acc} - v100 Accuracy:{v100_acc}")
        
        print(f"             - Test Loss: {test_loss:.4f}, Test Accuracy(Top-1): {test_accuracy:.4f}, 🎯Top-5 Accuracy: {top5_acc:.4f}")
        if stage == 'tune' and tune_early_stop_enabled:
            if tune_best_metric > -float('inf'):
                best_metric_text = f"{tune_best_metric:.4f}"
                best_epoch_text = str(tune_best_epoch)
            else:
                best_metric_text = "未刷新"
                best_epoch_text = "-"
            patience_text = f"{tune_stale_epochs}/{tune_patience}" if tune_patience else "∞"
            early_stop_status_line = (
                f"             - Early-stop status: best Top-5@200 = {best_metric_text} (epoch {best_epoch_text}), "
                f"stale={patience_text}, min_delta={tune_min_delta:.4f}, min_epochs={tune_min_epochs}"
            )
            print(early_stop_status_line)
            if tune_epoch_note:
                print(f"             - {tune_epoch_note}")
        improvement_threshold = tune_min_delta if stage == 'tune' and tune_early_stop_enabled else 0.0
        if stage == 'tune' and tune_early_stop_enabled:
            top5_improved = top5_acc >= best_top5 + improvement_threshold
        else:
            top5_improved = top5_acc > best_top5
        if top5_improved:
            best_top5 = top5_acc
            best_top5_epoch = epoch + 1
            best_top5_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "top5_acc": top5_acc,
                "v2_acc": v2_acc,
                "v4_acc": v4_acc,
                "v10_acc": v10_acc,
                "aligned_strict": alignment_strict_flag,
            }
            if stage == 'final' and best_checkpoint_path is not None:
                torch.save(eeg_model.state_dict(), best_checkpoint_path)
                print(f"             - ✅ Saved best Top-5 checkpoint to {best_checkpoint_path}")
        print(f"            {'─' * 110}")
        if tune_break_after_epoch:
            break
    # # Load the best model weights
    # model.load_state_dict(best_model_weights)

    # # # Save the best model
    # torch.save(model.state_dict(), '{train_pos_img_text}.pth')

    # Create 5 subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # Loss curve
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    # Overall accuracy curve
    axs[0, 1].plot(train_accuracies, label='Train Top-5 Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy (Top-5)')
    chance_top5 = 0.025
    axs[0, 1].axhline(chance_top5, color='gray', linestyle='--', linewidth=1, label='_nolegend_')
    axs[0, 1].text(0.02, chance_top5 + 0.002, 'Top-5 Chance (200-way)', color='gray', fontsize=9, transform=axs[0, 1].get_yaxis_transform())
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    # The following are the three new plots you added, assuming you've already calculated the corresponding accuracies
    # 2-class accuracy plot
    axs[1, 0].plot(v2_accs, label='2-class Accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy Curve")

    # 4-class accuracy plot
    axs[1, 1].plot(v4_accs, label='4-class Accuracy')
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy Curve")

    # 10-class accuracy plot
    axs[2, 0].plot(v10_accs, label='10-class Accuracy')
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy Curve")

    # Construct the string information for annotation
    if best_top5_info is not None:
        info_source = best_top5_info
        metric_label = "Best Top-5"
    elif best_epoch_info is not None:
        info_source = best_epoch_info
        metric_label = "Best Top-1"
    else:
        info_source = None
        metric_label = "No best epoch recorded"

    metadata_lines = [
        f"exp_id: {exp_name_for_fig}",
        f"seed: {getattr(config, 'seed', 'N/A')}",
        f"subject: {subject_str}",
        f"stage: {stage}",
        "",
    ]

    if info_source is not None:
        metric_lines = [f"{metric_label} (Epoch {info_source['epoch']}):"]
        if "top5_acc" in info_source:
            metric_lines.append(f"Test Top-5 Accuracy: {info_source['top5_acc']:.4f}")
        metric_lines.extend([
            f"Train Loss: {info_source['train_loss']:.4f}",
            f"Train Top-5 Accuracy: {info_source['train_accuracy']:.4f}",
            f"Test Loss: {info_source['test_loss']:.4f}",
            f"Test Accuracy(Top-1): {info_source['test_accuracy']:.4f}",
            f"v2_acc:{info_source['v2_acc']:.4f}",
            f"v4_acc:{info_source['v4_acc']:.4f}",
            f"v10_acc:{info_source['v10_acc']:.4f}",
        ])
        aligned_flag = bool(info_source.get('aligned_strict', False))
        metric_lines.append(f"Aligned S_new Strict: {aligned_flag}")
        info_text = "\n".join(metadata_lines + metric_lines)
    else:
        info_text = "\n".join(metadata_lines + ["No valid evaluation metrics recorded."])

    axs[2, 1].axis('off')  
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)

    plt.tight_layout()

    # Add main title
    plot_basename = f"pos_img_text_{stage}_{subject_tag}"
    plt.suptitle(plot_basename, fontsize=16, y=1.05)
    plot_path = os.path.join(analysis_dir, plot_basename)
    plt.savefig(plot_path)
    plt.close(fig)
    if logger is not None:
        logger.finish()

    if stage == 'tune':
        if tune_stop_epoch is None:
            tune_stop_epoch = len(results)
        early_stop_payload = {
            "enabled": tune_early_stop_enabled,
            "triggered": tune_early_stop_triggered,
            "stop_epoch": tune_stop_epoch,
            "best_epoch": best_top5_epoch,
            "best_metric": best_top5,
            "min_epochs": tune_min_epochs,
            "patience": tune_patience,
            "min_delta": tune_min_delta,
            "reset_on_lr_drop": tune_reset_on_lr_drop,
            "saved_checkpoint": bool(tune_checkpoint_path and os.path.isfile(tune_checkpoint_path)),
        }
        try:
            setattr(config, '_early_stop_status', early_stop_payload)
        except Exception:
            pass

    for record in results:
        epoch_id = record.get("epoch")
        record["is_best_top5"] = bool(best_top5_epoch is not None and epoch_id == best_top5_epoch)
        record["is_best_top1"] = bool(best_accuracy_epoch is not None and epoch_id == best_accuracy_epoch)
        record["best_top5_epoch"] = best_top5_epoch if best_top5_epoch is not None else ''
        record["best_top1_epoch"] = best_accuracy_epoch if best_accuracy_epoch is not None else ''

    return results

import datetime

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    import numpy as np
    import torch
    import random
    
    print(f"seed : {seed}")
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def _run_training_pipeline():
    # Use argparse to parse the command-line arguments
    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')
    parser.add_argument('--data_path', type=str, default="../eeg_dataset/preprocessed_data/Preprocessed_data_250Hz", help='Path to the EEG dataset')
    default_output = './outputs/contrast'
    parser.add_argument('--output_dir', type=str, default=default_output, help='Directory to save output results')
    parser.add_argument('--exp_id', type=str, default="run_sattc_loso_conformer_sattc_42", help='Custom experiment identifier for output folder naming')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42,3407,20251)')    
    parser.add_argument('--project', type=str, default="train_pos_img_text_rep", help='WandB project name')
    parser.add_argument('--entity', type=str, default="sustech_rethinkingbci", help='WandB entity name')
    parser.add_argument('--name', type=str, default="lr=5e-4_img_pos_pro_eeg", help='Experiment name')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--eval_batch_size', type=int, default=1024, help='Batch size for evaluation (val/test)')
    parser.add_argument('--logger', type=bool, default=False, help='Enable WandB logging')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to run on (cpu or gpu)')    
    parser.add_argument('--insubject', type=bool, default=False, help='In-subject mode or cross-subject mode')
    parser.add_argument('--encoder_type', type=str, default='SATTC', help='Encoder type')
    # GPU selection
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU device to use')
    # Similarity metric
    parser.add_argument('--sim', type=str, choices=['cos', 'dot'], default='cos', help='Similarity metric for loss/evaluation (cos or dot)')
    parser.add_argument('--train_subject_ids', type=str, choices=['spoof', 'real'], default='real', help='Strategy for subject IDs during training batches (spoof for target-subject tags, real for sample owners)')
    # Evaluation subject ID policy
    parser.add_argument('--test_subject_ids', type=str, choices=['target', 'fixed1', 'unk_avg', 'unk_learned'], default='unk_learned', help='Evaluation subject ID policy')
    parser.add_argument('--subject_dropout_p', type=float, default=0.4, help='Probability of replacing training subject IDs with UNK token when available')
    # saw settings
    parser.add_argument('--use_saw', action='store_true', default=False, help='Enable Subject-Adaptive Whitening (SAW) during evaluation')
    parser.add_argument('--saw_shrink', type=float, default=0.6, help='Shrinkage coefficient for SAW covariance estimation')
    parser.add_argument('--saw_diag', action='store_true', default=False, help='Use diagonal covariance approximation in SAW')
    # Global feature whitening and calibration before CSLS/logit evaluation
    parser.add_argument('--global_align_auto', dest='global_align_auto', action='store_true', default=True, help='Master switch: enable automatic global alignment (auto whiten + logits centering/scaling)')
    parser.add_argument('--disable_global_align_auto', dest='global_align_auto', action='store_false', help='Disable automatic global alignment and revert to legacy protocol defaults')
    
    parser.add_argument('--global_img_whiten_stats', type=str, default=None, help='Path to global whitening stats for candidate/image features (JSON/NPZ/PT)')
    parser.add_argument('--global_text_whiten_stats', type=str, default=None, help='Path to global whitening stats for text features (JSON/NPZ/PT)')
    parser.add_argument('--global_temp_scale', type=float, default=1.0, help='Global temperature multiplier applied prior to CSLS/logit evaluation')
    parser.add_argument('--global_temp_bias', type=float, default=0.0, help='Global additive bias applied prior to CSLS/logit evaluation')
    parser.add_argument('--logits_zscore_path', type=str, default=None, help='Optional stats file containing mean/std for logits z-score calibration')
    parser.add_argument('--logits_zscore_mean', type=float, default=None, help='Override logits z-score mean when no stats file is provided')
    parser.add_argument('--logits_zscore_std', type=float, default=None, help='Override logits z-score std when no stats file is provided')
    parser.add_argument('--global_auto_whiten', dest='global_auto_whiten', action='store_true', default=True, help='Auto-compute whitening stats from current candidate/text features when no external stats provided')
    parser.add_argument('--disable_global_auto_whiten', dest='global_auto_whiten', action='store_false', help='Disable auto-computation of global whitening stats')
    parser.add_argument('--global_whiten_auto_shrink', type=float, default=0.5, help='Shrinkage applied when estimating auto whitening covariance (0-1)')
    parser.add_argument('--global_whiten_auto_diag', action='store_true', default=False, help='Use diagonal covariance during auto whitening estimation')
    parser.add_argument('--logits_auto_center', dest='logits_auto_center', action='store_true', default=True, help='Automatically zero-center base logits when no z-score stats supplied')
    parser.add_argument('--disable_logits_auto_center', dest='logits_auto_center', action='store_false', help='Disable automatic zero-centering of base logits')
    parser.add_argument('--logits_auto_scale', dest='logits_auto_scale', action='store_true', default=True, help='Automatically scale base logits to unit variance when no z-score stats supplied')
    parser.add_argument('--disable_logits_auto_scale', dest='logits_auto_scale', action='store_false', help='Disable automatic variance scaling of base logits')
    register_ada_csls_args(parser, include_use_flags=True)
    register_structural_args(parser)
    register_eeg_encoder_arg(parser)
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature scaling applied before CSLS/logit evaluation')
    # Diagnostic dump settings
    parser.add_argument('--diag_dump_dir', type=str, default="./Retrieval/outputs/diagnosis", help='可选：每个epoch导出原始相似度快照供离线诊断')
    parser.add_argument('--diag_dump_limit', type=int, default=0, help='限制导出epoch数量（0表示全部导出）')
    parser.add_argument('--diag_dump_prefix', type=str, default='epoch', help='诊断快照文件名前缀（默认epoch）')
    parser.add_argument('--diag_dump_keep_post', action='store_true', help='诊断快照中同时保存Ada-CSLS后的相似度矩阵')
    # DataLoader settings
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers for train/val/test')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='DataLoader prefetch factor when num_workers>0')
    parser.add_argument('--subjects', nargs='+', default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'], help='List of subject IDs (default: sub-01 to sub-10)')    
    parser.add_argument('--start_from_subject', type=str, default=None, help='Resume training from specific subject ')
    # Nested-validation split support
    parser.add_argument('--val_subject', type=str, default='sub-06', help='Validation subject for tune stage')
    parser.add_argument('--tune_epochs', type=int, default=None, help='Number of epochs during tune stage (default: same as epochs)')
    parser.add_argument('--skip_tune', action='store_true', help='Skip tune stage and directly run final stage')
    parser.add_argument('--split_file', type=str, default=None, help='Path to outer-fold split JSON (from split_generator.py)')
    parser.add_argument('--split_dir', type=str, default='./Retrieval/splits', help='Directory containing per-fold split JSONs (e.g., outer_sub-*.json); if set, runs all folds')
    parser.add_argument('--mode', type=str, choices=['tune', 'final', 'auto'], default='auto', help='When using split files: tune (inner), final (outer), or auto (tune→final)')
    parser.add_argument('--hparams_file', type=str, default=None, help='Path to tuned hyperparameters JSON to load for final training')
    # tune-stage early stopping
    parser.add_argument('--disable_tune_early_stop', action='store_true', help='Disable early stopping during tune stage')
    parser.add_argument('--tune_min_epochs', type=int, default=15, help='Minimum epochs before early stopping can trigger in tune stage')
    parser.add_argument('--tune_patience', type=int, default=5, help='Patience (in epochs) for tune-stage early stopping')
    parser.add_argument('--tune_min_delta', type=float, default=0.001, help='Minimum Top-5 improvement to reset patience during tune stage')
    parser.add_argument('--tune_reset_on_lr_drop', action='store_true', help='Reset early-stopping patience if learning rate drops during tune stage')
    parser.add_argument('--disable_tune_save_best', action='store_true', help='Skip saving the best tune-stage checkpoint')
    args = parser.parse_args()

    def _log_ada_csls_config(ns):
        use_csls_flag = bool(getattr(ns, "use_csls", False))
        use_ada_flag = bool(getattr(ns, "use_ada_csls", False))
        csls_k = getattr(ns, "csls_k", None)
        csls_kmin = getattr(ns, "csls_kmin", None)
        csls_kmax = getattr(ns, "csls_kmax", None)
        csls_alpha = getattr(ns, "csls_alpha", None)
        csls_m = getattr(ns, "csls_m", None)
        csls_k_side = getattr(ns, "csls_k_side", None)
        col_alpha = getattr(ns, "csls_col_alpha", None)
        col_m = getattr(ns, "csls_col_m", None)
        col_kmin = getattr(ns, "csls_col_kmin", None)
        col_kmax = getattr(ns, "csls_col_kmax", None)
        print(
            f"[Config][Ada-CSLS] use_csls={use_csls_flag} | use_ada_csls={use_ada_flag} | "
            f"row_k=[{csls_kmin},{csls_kmax}] base_k={csls_k} | row_alpha={csls_alpha} | row_m={csls_m} | "
            f"k_side={csls_k_side} | col_alpha={col_alpha} | col_m={col_m} | col_k=[{col_kmin},{col_kmax}]"
        )

    def _log_structural_config(ns):
        auto_pen = getattr(ns, "poe_lambda_pen", None)
        auto_bonus = getattr(ns, "poe_lambda_bonus", None)
        pen_txt = "auto" if auto_pen is None else f"{auto_pen:.4f}"
        bonus_txt = "auto" if auto_bonus is None else f"{auto_bonus:.4f}"
        print(
            "[Config][Struct] row_safe_k={} | col_safe_k={} | row_topL={} | "
            "case2_tau_ratio={:.3f} | case2_col_penalty={:.3f} | hub_quantiles(high={:.3f}, mid={:.3f}) | "
            "poe_enabled={} | poe_beta={:.3f} | poe_lambda_pen={} | poe_lambda_bonus={}".format(
                int(getattr(ns, "pre_csls_row_safe_k", 5)),
                int(getattr(ns, "pre_csls_col_safe_k", 5)),
                int(getattr(ns, "pre_csls_row_topL", 5)),
                float(getattr(ns, "pre_csls_case2_tau_ratio", 0.5)),
                float(getattr(ns, "pre_csls_case2_col_penalty", 0.5)),
                float(getattr(ns, "pre_csls_case5_hub_high_quantile", 0.95)),
                float(getattr(ns, "pre_csls_case5_hub_mid_quantile", 0.8)),
                bool(getattr(ns, "enable_poe", False)),
                float(getattr(ns, "poe_beta", 1.0)),
                pen_txt,
                bonus_txt,
            )
        )

    args.train_subject_ids = (args.train_subject_ids or 'spoof').lower()
    args.test_subject_ids = (args.test_subject_ids or 'target').lower()
    args.subject_dropout_p = max(0.0, min(1.0, float(args.subject_dropout_p)))
    args.chose_eeg_encoder = (getattr(args, 'chose_eeg_encoder', 'sattc') or 'sattc').lower()
    args.tune_early_stop = not getattr(args, 'disable_tune_early_stop', False)
    args.tune_save_best = not getattr(args, 'disable_tune_save_best', False)
    normalize_csls_args(args)
    normalize_structural_args(args)
    master_align_auto = bool(getattr(args, 'global_align_auto', True))
    if master_align_auto:
        args.global_auto_whiten = True
        args.logits_auto_center = True
        args.logits_auto_scale = True
    else:
        args.global_auto_whiten = False
        args.logits_auto_center = False
        args.logits_auto_scale = False
        for cache_attr in (
            '_cache_auto_whiten_text',
            '_cache_auto_whiten_img',
        ):
            if hasattr(args, cache_attr):
                setattr(args, cache_attr, None)
    _log_ada_csls_config(args)
    _log_structural_config(args)

    print(
        "[Config][GlobalAlign] master_auto={} | img_whiten={} | text_whiten={} | auto_whiten={} | temp_scale={:.3f} | temp_bias={:.3f} | zscore_path={} | auto_center={} | auto_scale={}".format(
            master_align_auto,
            bool(args.global_img_whiten_stats),
            bool(args.global_text_whiten_stats),
            bool(getattr(args, 'global_auto_whiten', False)),
            float(args.global_temp_scale),
            float(args.global_temp_bias),
            args.logits_zscore_path or 'None',
            bool(getattr(args, 'logits_auto_center', False)),
            bool(getattr(args, 'logits_auto_scale', False)),
        )
    )
    if getattr(args, 'global_auto_whiten', False):
        print(
            "[Config][GlobalAlign] auto_whiten_shrink={:.3f} | auto_whiten_diag={}".format(
                float(getattr(args, 'global_whiten_auto_shrink', 0.0) or 0.0),
                bool(getattr(args, 'global_whiten_auto_diag', False)),
            )
        )

    set_seed(args.seed)

    args.eval_batch_size = max(1, args.eval_batch_size)

    # Set device based on the argument
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device(args.gpu)
    else:
        device = torch.device('cpu')

    def _load_hparams_into_args(path, args_ns):
        if not path:
            return args_ns
        if not os.path.exists(path):
            raise FileNotFoundError(f"hparams file not found: {path}")
        with open(path, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(args_ns, key):
                setattr(args_ns, key, value)
        print(f"Loaded hparams from {path}")
        return args_ns

    def _build_fold_splits(ns):
        if ns.split_dir:
            if not os.path.isdir(ns.split_dir):
                raise NotADirectoryError(f"split_dir not found: {ns.split_dir}")
            candidates = sorted(p for p in os.listdir(ns.split_dir) if p.startswith('outer_sub-') and p.endswith('.json'))
            if not candidates:
                raise FileNotFoundError(f"No split JSONs (outer_sub-*.json) found in {ns.split_dir}")
            splits = []
            for name in candidates:
                with open(os.path.join(ns.split_dir, name), 'r', encoding='utf-8') as f:
                    splits.append(json.load(f))
            return splits
        if ns.split_file:
            if not os.path.exists(ns.split_file):
                raise FileNotFoundError(f"Split file not found: {ns.split_file}")
            with open(ns.split_file, 'r', encoding='utf-8') as f:
                return [json.load(f)]
        return [None]

    fold_splits = _build_fold_splits(args)

    def _loader_kwargs(ns):
        kwargs = dict(
            num_workers=ns.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
        if ns.num_workers > 0 and ns.prefetch_factor:
            kwargs['prefetch_factor'] = ns.prefetch_factor
        return kwargs

    loader_kwargs = _loader_kwargs(args)
    use_real_subject_ids = args.train_subject_ids == 'real'
    subject_numeric_values = []
    for name in args.subjects:
        sid = extract_id_from_string(name)
        if sid is not None:
            subject_numeric_values.append(int(sid))
    max_subject_numeric = max(subject_numeric_values) if subject_numeric_values else len(args.subjects)
    base_subject_slots = max(len(args.subjects), max_subject_numeric + 1)
    use_subject_unk = args.test_subject_ids in {'unk_avg', 'unk_learned'} or args.subject_dropout_p > 0
    total_subject_slots = base_subject_slots + (1 if use_subject_unk else 0)
    args.use_subject_unk = use_subject_unk
    args._subject_slots = total_subject_slots
    args._subject_numeric_max = max_subject_numeric

    for split in fold_splits:
        if split is not None:
            test_subject = split['test_subject']
            dev_subjects = split.get('dev_subjects', [])
            train_subjects_fold = split.get('train_subjects', [])
            val_unseen = split.get('val_unseen_classes', [])
            fold_subjects = sorted(set(train_subjects_fold + dev_subjects + [test_subject]))
            print("\n📋 Experiment Configuration (split mode):")
            print(f"   Fold test subject: {test_subject}")
            print(f"   Train subjects (tune core): {train_subjects_fold}")
            print(f"   Dev pack subjects: {dev_subjects}")
            print(f"   Val-unseen classes: N={len(val_unseen)}")
            if args.tune_early_stop and not args.insubject:
                print(
                    f"   Tune early stopping: enabled (min_epochs={args.tune_min_epochs}, "
                    f"patience={args.tune_patience}, min_delta={args.tune_min_delta:.4f})"
                )

            if args.start_from_subject and args.start_from_subject not in fold_subjects:
                print(f"Start subject {args.start_from_subject} not part of this fold; skipping.")
                continue
            if args.start_from_subject and args.start_from_subject != test_subject:
                print(f"Skipping fold {test_subject}; start_from_subject={args.start_from_subject}.")
                continue

            subjects_to_train = [test_subject]
            base_subjects = fold_subjects
        else:
            base_subjects = list(args.subjects)
            if args.val_subject in base_subjects and not args.insubject:
                print(f"Warning: removing validation subject {args.val_subject} from training pool to avoid leakage.")
                base_subjects = [s for s in base_subjects if s != args.val_subject]

            if args.start_from_subject:
                if args.start_from_subject not in base_subjects:
                    print(f"Error: {args.start_from_subject} not found in subjects list: {base_subjects}")
                    return
                start_index = base_subjects.index(args.start_from_subject)
                subjects_to_train = base_subjects[start_index:]
                print(f"Resuming training from {args.start_from_subject}")
            else:
                subjects_to_train = base_subjects

            print(f"\n📋 Experiment Configuration (LOSO mode):")
            print(f"   Train/Test subjects: {subjects_to_train}")
            if not args.insubject:
                print(f"   Validation subject (tune stage): {args.val_subject}")
            if args.tune_early_stop and not args.insubject:
                print(
                    f"   Tune early stopping: enabled (min_epochs={args.tune_min_epochs}, "
                    f"patience={args.tune_patience}, min_delta={args.tune_min_delta:.4f})"
                )

        for sub in subjects_to_train:
            current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
            print(f"\n{'='*60}")
            print(f"Training pipeline for subject: {sub}")
            print(f"{'='*60}")

            if args.exp_id:
                results_dir = os.path.join(args.output_dir, args.exp_id)
            else:
                results_dir = os.path.join(args.output_dir, args.encoder_type, sub, current_time)
            os.makedirs(results_dir, exist_ok=True)
            subject_start_time = time.time()
            tune_summary_path = os.path.join(results_dir, "tune_summary.json")
            final_summary_path = os.path.join(results_dir, "final_summary.json")

            tune_results = []
            final_results = []
            best_tune_epoch = None
            best_val_top5 = None
            tune_epoch_limit = None

            try:
                if split is not None:
                    if sub != test_subject:
                        print(f"Subject {sub} not equal to fold test subject {test_subject}; skipping.")
                        continue
                    tune_train_subjects = train_subjects_fold
                    tune_val_subjects = dev_subjects
                    final_train_subjects = sorted(set(train_subjects_fold + dev_subjects))
                    final_test_subjects = [test_subject]
                    tune_val_uses_train_split = True
                    train_classes_tune = sorted(c for c in range(1654) if c not in set(val_unseen))
                else:
                    tune_train_subjects = [s for s in base_subjects if s != sub and s != args.val_subject]
                    if not tune_train_subjects and not args.insubject:
                        tune_train_subjects = [s for s in base_subjects if s != sub]
                    tune_val_subjects = [args.val_subject] if (args.val_subject and not args.insubject) else []
                    final_train_subjects = [sub] if args.insubject else [s for s in base_subjects if s != sub]
                    final_test_subjects = [sub]
                    tune_val_uses_train_split = False
                    train_classes_tune = None

                mode_lower = args.mode.lower() if split is not None else 'auto'
                run_tune_stage = (
                    not args.skip_tune
                    and not args.insubject
                    and ((split is None and tune_val_subjects)
                         or (split is not None and mode_lower in {'tune', 'auto'}))
                )
                run_final_stage = True
                if split is not None:
                    if mode_lower == 'tune':
                        run_final_stage = False
                    elif mode_lower == 'final':
                        run_tune_stage = False

                if run_tune_stage:
                    tune_stage_start = time.time()
                    if not tune_train_subjects:
                        print("No training subjects available for tune stage; skipping tune.")
                    elif not tune_val_subjects:
                        print("No validation subjects available for tune stage; skipping tune.")
                    else:
                        print(f"[Tune] Train subjects: {tune_train_subjects}")
                        print(f"[Tune] Validation subjects: {tune_val_subjects}")

                        tune_model = globals()[args.encoder_type](
                            num_subjects=args._subject_slots,
                            use_subject_unk=use_subject_unk,
                            encoder_choice=getattr(args, 'chose_eeg_encoder', 'sattc'),
                        )
                        tune_model.to(device)
                        optimizer_tune = AdamW(itertools.chain(tune_model.parameters()), lr=args.lr)

                        tune_train_kwargs = dict(
                            data_path=args.data_path,
                            subjects=tune_train_subjects,
                            train=True,
                        )
                        if use_real_subject_ids:
                            tune_train_kwargs['return_subject_ids'] = True
                        if train_classes_tune:
                            tune_train_kwargs['classes'] = train_classes_tune
                        tune_train_dataset = EEGDataset(**tune_train_kwargs)

                        tune_val_kwargs = dict(
                            data_path=args.data_path,
                            subjects=tune_val_subjects,
                            train=tune_val_uses_train_split if tune_val_uses_train_split else False,
                        )
                        if split is not None and val_unseen:
                            tune_val_kwargs['classes'] = val_unseen
                            tune_val_kwargs['anchor_mode'] = 'first_per_class'
                            tune_val_kwargs['return_subject_ids'] = True
                        tune_val_dataset = EEGDataset(**tune_val_kwargs)

                        if split is not None:
                            loaded_val_subjects = list(getattr(tune_val_dataset, 'subjects', []))
                            missing_dev = sorted(set(dev_subjects) - set(loaded_val_subjects))
                            if missing_dev:
                                raise ValueError(
                                    f"Tune-stage dev subjects mismatch: expected {dev_subjects}, got {loaded_val_subjects}"
                                )
                            if val_unseen:
                                feature_class_count = int(tune_val_dataset.text_features.size(0))
                                if feature_class_count != len(val_unseen):
                                    raise ValueError(
                                        f"Tune-stage dev unseen classes mismatch: expected {len(val_unseen)}, got {feature_class_count}"
                                    )

                        if train_classes_tune:
                            expected_train_samples = len(train_classes_tune) * 10 * 4 * len(tune_train_subjects)
                            print(f"[Tune] Train dataset samples: {len(tune_train_dataset)} (expected {expected_train_samples})")
                        if split is not None and val_unseen:
                            expected_val_samples = len(val_unseen) * 4 * len(tune_val_subjects)
                            print(f"[Tune] Val dataset samples: {len(tune_val_dataset)} (expected {expected_val_samples})")

                        tune_train_loader = DataLoader(
                            tune_train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            **loader_kwargs,
                        )
                        tune_val_loader = DataLoader(
                            tune_val_dataset,
                            batch_size=args.eval_batch_size,
                            shuffle=False,
                            drop_last=False,
                            **loader_kwargs,
                        )

                        config_tune = copy.deepcopy(args)
                        config_tune.stage = 'tune'
                        config_tune.epochs = args.tune_epochs if args.tune_epochs else args.epochs
                        config_tune.tune_early_stop = getattr(args, 'tune_early_stop', True)
                        config_tune.tune_min_epochs = args.tune_min_epochs
                        config_tune.tune_patience = args.tune_patience
                        config_tune.tune_min_delta = args.tune_min_delta
                        config_tune.tune_reset_on_lr_drop = args.tune_reset_on_lr_drop
                        config_tune.tune_save_best = getattr(args, 'tune_save_best', True)
                        config_tune.use_subject_unk = use_subject_unk
                        config_tune.subject_dropout_p = args.subject_dropout_p
                        config_tune.test_subject_ids = args.test_subject_ids
                        tune_subject_tag = re.sub(r"[^0-9A-Za-z_-]+", '-', str(sub)).strip('-_') or 'subject'
                        exp_name_tune = str(args.exp_id or '').strip()
                        tune_model_root = os.path.join("./models/contrast", exp_name_tune) if exp_name_tune else "./models/contrast"
                        if args.insubject:
                            tune_ckpt_dir = os.path.join(tune_model_root, "tune", args.encoder_type, sub, current_time)
                        else:
                            tune_ckpt_dir = os.path.join(tune_model_root, "tune", "across", args.encoder_type, current_time)
                        os.makedirs(tune_ckpt_dir, exist_ok=True)
                        config_tune.tune_best_ckpt_path = os.path.join(tune_ckpt_dir, f"{tune_subject_tag}_tune_best_top5.pth")

                        print(
                            f"[Tune] Early-stop guard: min_epochs={config_tune.tune_min_epochs}, "
                            f"patience={config_tune.tune_patience}, min_delta={config_tune.tune_min_delta}"
                        )

                        tune_results = main_train_loop(
                            sub,
                            current_time,
                            tune_model,
                            tune_train_loader,
                            tune_val_loader,
                            optimizer_tune,
                            device,
                            tune_train_dataset.text_features,
                            tune_val_dataset.text_features,
                            tune_train_dataset.img_features,
                            tune_val_dataset.img_features,
                            config=config_tune,
                            logger=config_tune.logger,
                        )

                        if tune_results:
                            tune_stage_duration = time.time() - tune_stage_start
                            tune_duration_str = format_duration_hm(tune_stage_duration)
                            early_stop_status = getattr(config_tune, '_early_stop_status', {})
                            tune_stop_epoch = early_stop_status.get('stop_epoch', len(tune_results)) if isinstance(early_stop_status, dict) else len(tune_results)
                            tune_early_stop_triggered = bool(early_stop_status.get('triggered', False)) if isinstance(early_stop_status, dict) else False
                            top5_values = [entry.get('top5_acc') for entry in tune_results if entry.get('top5_acc') is not None]
                            if top5_values:
                                best_val_top5 = max(top5_values)
                                top5_candidates = [
                                    entry for entry in tune_results
                                    if entry.get('top5_acc') is not None and math.isclose(entry.get('top5_acc'), best_val_top5, rel_tol=1e-6, abs_tol=1e-9)
                                ]
                                best_entry = min(top5_candidates, key=lambda x: x.get('epoch', float('inf')))
                            else:
                                best_entry = max(
                                    tune_results,
                                    key=lambda x: x.get('test_accuracy', -1)
                                )
                                best_val_top5 = best_entry.get('top5_acc', None)
                            epoch_value = best_entry.get('epoch', config_tune.epochs)
                            try:
                                best_tune_epoch = int(epoch_value)
                            except (TypeError, ValueError):
                                best_tune_epoch = None

                            epoch_candidates = []
                            if isinstance(best_tune_epoch, (int, float)) and best_tune_epoch > 0:
                                epoch_candidates.append(int(best_tune_epoch))
                            min_epoch_guard = getattr(args, 'tune_min_epochs', None)
                            if isinstance(min_epoch_guard, (int, float)) and min_epoch_guard > 0:
                                epoch_candidates.append(int(min_epoch_guard))
                            if epoch_candidates:
                                tune_epoch_limit = max(epoch_candidates)

                            tune_results_file = os.path.join(
                                results_dir, f"{args.encoder_type}_{sub}_tune_seed{args.seed}.csv"
                            )
                            with open(tune_results_file, 'w', newline='') as file:
                                writer = csv.DictWriter(file, fieldnames=tune_results[0].keys())
                                writer.writeheader()
                                writer.writerows(tune_results)
                            print(f"Tune-stage results saved to {tune_results_file}")

                            tune_primary = {
                                "Best Val Top-5": best_val_top5,
                                "Best Val Top-1": best_entry.get('test_accuracy'),
                                "Best Epoch": best_tune_epoch,
                                "Duration": tune_duration_str,
                                "Stop Epoch": tune_stop_epoch,
                            }
                            print_stage_summary("Tune", sub, tune_primary)

                            tune_summary = {
                                "subject": sub,
                                "stage": "tune",
                                "best_epoch": best_tune_epoch,
                                "best_val_top5": best_val_top5,
                                "epochs_run": config_tune.epochs,
                                "train_subjects": tune_train_subjects,
                                "val_subjects": tune_val_subjects,
                                "val_split_source": 'dev_unseen' if split is not None else ('train_set' if tune_val_uses_train_split else 'test_set'),
                                "val_unseen_classes": val_unseen if split is not None else None,
                                "stage_duration": tune_duration_str,
                                "stage_duration_seconds": round(tune_stage_duration, 2),
                                "stop_epoch": tune_stop_epoch,
                                "early_stop_enabled": bool(early_stop_status.get('enabled', False)) if isinstance(early_stop_status, dict) else bool(getattr(args, 'tune_early_stop', True)),
                                "early_stop_triggered": tune_early_stop_triggered,
                                "early_stop_min_epochs": early_stop_status.get('min_epochs', args.tune_min_epochs) if isinstance(early_stop_status, dict) else args.tune_min_epochs,
                                "early_stop_patience": early_stop_status.get('patience', args.tune_patience) if isinstance(early_stop_status, dict) else args.tune_patience,
                                "early_stop_min_delta": early_stop_status.get('min_delta', args.tune_min_delta) if isinstance(early_stop_status, dict) else args.tune_min_delta,
                                "early_stop_reset_on_lr_drop": early_stop_status.get('reset_on_lr_drop', args.tune_reset_on_lr_drop) if isinstance(early_stop_status, dict) else args.tune_reset_on_lr_drop,
                                "tune_best_checkpoint": config_tune.tune_best_ckpt_path,
                                "early_stop_saved_checkpoint": early_stop_status.get('saved_checkpoint', False) if isinstance(early_stop_status, dict) else False,
                            }
                            with open(tune_summary_path, 'w') as f:
                                json.dump(tune_summary, f, indent=2)
                        else:
                            print("Tune stage completed but no results were returned.")

                        shutdown_dataloader(tune_train_loader)
                        shutdown_dataloader(tune_val_loader)

                        clear_memory(
                            tune_model,
                            tune_train_dataset,
                            tune_val_dataset,
                            tune_train_loader,
                            tune_val_loader,
                        )
                        del tune_model
                        del tune_train_loader
                        del tune_val_loader
                        del tune_train_dataset
                        del tune_val_dataset
                        force_ram_cleanup()
                        print("[Tune] 已强制回收内存与显存缓存。")
                else:
                    print("Tune stage skipped based on configuration.")

                if not run_final_stage:
                    force_ram_cleanup()
                    continue

                force_ram_cleanup()
                final_model = globals()[args.encoder_type](
                    num_subjects=args._subject_slots,
                    use_subject_unk=use_subject_unk,
                    encoder_choice=getattr(args, 'chose_eeg_encoder', 'sattc'),
                )
                final_model.to(device)
                optimizer_final = AdamW(itertools.chain(final_model.parameters()), lr=args.lr)

                print(f"\n{'='*60}\n开始Final阶段训练\n{'='*60}")
                final_stage_start = time.time()

                if args.insubject:
                    final_train_dataset = EEGDataset(
                        args.data_path,
                        subjects=[sub],
                        train=True,
                        return_subject_ids=use_real_subject_ids,
                    )
                    final_test_dataset = EEGDataset(args.data_path, subjects=[sub], train=False)
                else:
                    if split is not None:
                        final_train_dataset = EEGDataset(
                            args.data_path,
                            subjects=final_train_subjects,
                            train=True,
                            return_subject_ids=use_real_subject_ids,
                        )
                        final_test_dataset = EEGDataset(args.data_path, subjects=final_test_subjects, train=False)
                    else:
                        final_train_dataset = EEGDataset(
                            args.data_path,
                            subjects=final_train_subjects,
                            exclude_subject=None,
                            train=True,
                            return_subject_ids=use_real_subject_ids,
                        )
                        final_test_dataset = EEGDataset(
                            args.data_path,
                            subjects=final_test_subjects,
                            train=False,
                        )

                final_train_loader = DataLoader(
                    final_train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=True,
                    **loader_kwargs,
                )
                final_test_loader = DataLoader(
                    final_test_dataset,
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                    drop_last=False,
                    **loader_kwargs,
                )

                config_final = copy.deepcopy(args)
                config_final.stage = 'final'
                if split is not None and args.mode.lower() == 'final' and args.hparams_file:
                    config_final = _load_hparams_into_args(args.hparams_file, config_final)
                if tune_epoch_limit is not None:
                    config_final.epochs = min(args.epochs, max(int(tune_epoch_limit), 1))
                else:
                    config_final.epochs = args.epochs
                config_final.use_subject_unk = use_subject_unk
                config_final.subject_dropout_p = args.subject_dropout_p
                config_final.test_subject_ids = args.test_subject_ids

                final_results = main_train_loop(
                    sub,
                    current_time,
                    final_model,
                    final_train_loader,
                    final_test_loader,
                    optimizer_final,
                    device,
                    final_train_dataset.text_features,
                    final_test_dataset.text_features,
                    final_train_dataset.img_features,
                    final_test_dataset.img_features,
                    config=config_final,
                    logger=config_final.logger,
                )

                if final_results:
                    final_stage_duration = time.time() - final_stage_start
                    final_stage_duration_str = format_duration_hm(final_stage_duration)
                    subject_tag = re.sub(r"[^0-9A-Za-z_-]+", "-", str(sub)).strip('-_') or 'subject'
                    final_results_file = os.path.join(
                        results_dir,
                        f"final_{args.encoder_type}_{subject_tag}_seed{args.seed}.csv",
                    )

                    with open(final_results_file, 'w', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=final_results[0].keys())
                        writer.writeheader()
                        writer.writerows(final_results)
                    print(f"Final-stage results saved to {final_results_file}")

                    best_final = max(
                        final_results,
                        key=lambda x: (x.get('top5_acc', -1), x.get('test_accuracy', -1))
                    )
                    exp_name = str(getattr(args, 'exp_id', '') or '').strip()
                    model_root = os.path.join("./models/contrast", exp_name) if exp_name else "./models/contrast"
                    if args.insubject:
                        model_base_dir = os.path.join(model_root, args.encoder_type, str(sub), current_time)
                    else:
                        model_base_dir = os.path.join(model_root, "across", args.encoder_type, current_time)
                    os.makedirs(model_base_dir, exist_ok=True)
                    best_checkpoint_path = os.path.join(model_base_dir, f"{subject_tag}_best_top5.pth")

                    final_primary = {
                        "Best Test Top-5": best_final.get('top5_acc'),
                        "Best Test Top-1": best_final.get('test_accuracy'),
                        "Duration": final_stage_duration_str,
                    }
                    final_auxiliary = {
                        "Epoch": best_final.get('epoch'),
                        "Top-1@2": best_final.get('v2_acc'),
                        "Top-1@4": best_final.get('v4_acc'),
                        "Top-1@10": best_final.get('v10_acc'),
                        "Top-1@50": best_final.get('v50_acc'),
                        "Top-1@100": best_final.get('v100_acc'),
                        "Top-5@50": best_final.get('v50_top5_acc'),
                        "Top-5@100": best_final.get('v100_top5_acc'),
                        "Checkpoint": best_checkpoint_path,
                    }
                    print_stage_summary("Final", sub, final_primary, final_auxiliary)

                    subject_elapsed_seconds = time.time() - subject_start_time
                    total_duration_str = format_duration_hm(subject_elapsed_seconds)

                    final_summary = {
                        "subject": sub,
                        "stage": "final",
                        "best_epoch": best_final.get('epoch'),
                        "test_top1": best_final.get('test_accuracy'),
                        "best_test_top5": best_final.get('top5_acc'),
                        "epochs_run": config_final.epochs,
                        "best_tune_epoch": best_tune_epoch,
                        "best_val_top5": best_val_top5,
                        "best_checkpoint": best_checkpoint_path,
                        "final_train_subjects": final_train_subjects if not args.insubject else [sub],
                        "final_test_subjects": final_test_subjects,
                        "split_mode": bool(split is not None),
                        "total_duration": total_duration_str,
                        "total_duration_seconds": round(subject_elapsed_seconds, 2),
                        "stage_duration": final_stage_duration_str,
                        "stage_duration_seconds": round(final_stage_duration, 2),
                    }
                    with open(final_summary_path, 'w') as f:
                        json.dump(final_summary, f, indent=2)
                    update_summary_duration(tune_summary_path, total_duration_str, subject_elapsed_seconds)
                else:
                    print("Final stage completed but no results were returned.")

                shutdown_dataloader(final_train_loader)
                shutdown_dataloader(final_test_loader)

                clear_memory(
                    final_model,
                    final_train_dataset,
                    final_test_dataset,
                    final_train_loader,
                    final_test_loader,
                )
                del final_model
                del final_train_loader
                del final_test_loader
                del final_train_dataset
                del final_test_dataset
                force_ram_cleanup()
                print("[Final] 已强制回收内存与显存缓存。")
            except Exception as e:
                print(f"Error during training for {sub}: {str(e)}")
                raise
            except KeyboardInterrupt:
                print("⚠️ KeyboardInterrupt detected. Clearing memory before exit...")
                clear_memory()
                raise
            finally:
                subject_total_seconds = time.time() - subject_start_time
                subject_duration_str = format_duration_hm(subject_total_seconds)
                update_summary_duration(tune_summary_path, subject_duration_str, subject_total_seconds)
                update_summary_duration(final_summary_path, subject_duration_str, subject_total_seconds)
                clear_memory()
                force_ram_cleanup()

    print("\nAll subjects training completed!")


def main():
    overall_start_time = time.time()
    status_label = "完成"
    try:
        _run_training_pipeline()
    except KeyboardInterrupt:
        status_label = "中断"
        raise
    except Exception:
        status_label = "异常"
        raise
    finally:
        elapsed = time.time() - overall_start_time
        print(f"\n⏱️ 总运行时长（{status_label}）: {format_duration_hm(elapsed)}")


if __name__ == '__main__':
    main()
