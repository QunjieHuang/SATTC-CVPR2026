"""Scoring utilities for CSLS variants."""
from __future__ import annotations

import torch


def csls_fixed(similarities: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Apply fixed-k CSLS re-ranking on an [N, C] similarity matrix."""
    if similarities.dim() != 2:
        raise ValueError("CSLS expects an [N, C] 2D similarity matrix")
    n, c = similarities.shape
    if n == 0 or c == 0:
        return similarities
    k_eff = max(1, min(int(k), n, c))
    rx = similarities.topk(k_eff, dim=1).values.mean(dim=1, keepdim=True)
    ry = similarities.topk(k_eff, dim=0).values.mean(dim=0, keepdim=True)
    return 2 * similarities - rx - ry


@torch.no_grad()
def csls_adaptive(
    S: torch.Tensor,
    k0: int = 10,
    kmin: int = 5,
    kmax: int = 20,
    alpha: float = 1.0,
    m: int = 10,
    k_side: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ada-CSLS adaptive re-ranking that adjusts k based on query-side density."""
    if S.dim() != 2:
        raise ValueError("Ada-CSLS expects an [N, C] 2D similarity matrix")
    n_q, n_c = S.shape
    if n_q == 0 or n_c == 0:
        empty_float = torch.empty(0, device=S.device, dtype=S.dtype)
        empty_long = torch.empty(0, device=S.device, dtype=torch.long)
        return S, empty_long, empty_float, empty_float

    kmin = max(1, min(int(kmin), n_c))
    kmax = max(kmin, min(int(kmax), n_c))
    k0 = max(kmin, min(int(k0), kmax))
    m_eff = max(1, min(int(m), n_c))

    rho = S.topk(m_eff, dim=1).values.mean(dim=1)
    med = rho.median()
    med_safe = med + 1e-9
    scale = (rho / med_safe).clamp_min(1e-6).pow(float(alpha))

    k_row = (k0 * scale).round().clamp(kmin, kmax).to(torch.int64)

    rT = torch.empty(n_q, device=S.device, dtype=S.dtype)
    for k_val in torch.unique(k_row, sorted=True).tolist():
        k_int = max(1, int(k_val))
        mask = k_row == k_int
        if not mask.any():
            continue
        vals = S[mask].topk(k_int, dim=1).values
        rT[mask] = vals.mean(dim=1)

    ks = int(k0) if k_side is None else int(k_side)
    ks = max(1, min(ks, n_q))
    rS = S.topk(ks, dim=0).values.mean(dim=0)

    S_csls = 2 * S - rT.unsqueeze(1) - rS.unsqueeze(0)
    return S_csls, k_row, rT, rS
