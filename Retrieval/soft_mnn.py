"""Soft-MNN helper utilities.

This module provides reusable building blocks to move from aTop-K
hubSoft-MNN

1. / (percentile)/
2. “”“Top-L”hub
3. γsoft

/
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def percentile_from_rank(ranks: torch.Tensor, total: int) -> torch.Tensor:
	"""Convert 1-based ranks into [0, 1] percentiles.

	Args:
		ranks: 1-based rank tensor (any shape).
		total: Total number of candidates ranked (>= 1).

	Returns:
		Tensor of the same shape with values in [0, 1]. Lower values represent
		better (smaller) ranks. Degenerates to zeros when ``total <= 1``.
	"""

	ranks = ranks.to(torch.float32)
	if total is None or total <= 1:
		return torch.zeros_like(ranks)
	denom = float(total - 1)
	return torch.clamp((ranks - 1.0) / denom, min=0.0, max=1.0)


def estimate_class_popularity(
	sim_matrix: torch.Tensor,
	top_l: int,
	*,
	weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
	"""Estimate per-class popularity based on Top-L hits across queries.

	Args:
		sim_matrix: Normalised similarity/logit matrix of shape [Q, C]. Each row
			corresponds to one query; columns are candidate classes.
		top_l: Number of top entries per query to consider (>= 1).
		weights: Optional per-query weights [Q]1

	Returns:
		Popularity tensor ``rho`` of shape [C] representing per-class Top-L hit probability.
	"""

	if sim_matrix.dim() != 2:
		raise ValueError("sim_matrix must be rank-2")
	num_queries, num_classes = sim_matrix.shape
	if num_queries == 0 or num_classes == 0 or top_l <= 0:
		return torch.zeros(num_classes, dtype=torch.float32)

	top_l = min(int(top_l), num_classes)
	top_indices = torch.topk(sim_matrix, k=top_l, dim=1, largest=True, sorted=False).indices
	flat_idx = top_indices.reshape(-1)
	if weights is None:
		scatter_weights = torch.ones_like(flat_idx, dtype=torch.float32)
		normaliser = float(num_queries)
	else:
		if weights.numel() != num_queries:
			raise ValueError("weights tensor must match number of queries")
		expanded = weights.unsqueeze(1).expand(-1, top_l).reshape(-1)
		scatter_weights = expanded.to(torch.float32)
		normaliser = float(weights.sum().item()) if weights.sum() > 0 else 1.0

	hits = torch.zeros(num_classes, dtype=torch.float32)
	hits.scatter_add_(0, flat_idx, scatter_weights)
	if normaliser <= 0:
		normaliser = 1.0
	return hits / normaliser


def adjust_reverse_percentile(
	reverse_percentiles: torch.Tensor,
	labels: torch.Tensor,
	popularity: torch.Tensor,
	gamma: float,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Apply hub-aware discounting to reverse percentiles.

	Implements :math:`\tilde{\pi}_r = \pi_r / (1 - \gamma\,\rho(c))_+`,
	where ``rho`` is class popularity and ``gamma`` is the discount strength.
	"""

	if popularity.numel() == 0:
		return reverse_percentiles

	gamma = float(gamma)
	gamma = max(0.0, gamma)
	if gamma == 0.0:
		return reverse_percentiles

	labels = labels.to(torch.long)
	gathered = torch.zeros_like(reverse_percentiles)
	valid_mask = (labels >= 0) & (labels < popularity.numel())
	if valid_mask.any():
		gathered_vals = popularity[labels[valid_mask]]
		denom = 1.0 - gamma * gathered_vals
		denom = torch.clamp(denom, min=eps)
		adjusted = reverse_percentiles[valid_mask] / denom
		gathered[valid_mask] = adjusted
	if (~valid_mask).any():
		gathered[~valid_mask] = reverse_percentiles[~valid_mask]
	return torch.clamp(gathered, min=0.0)


def tolerance_weight(
	forward_percentiles: torch.Tensor,
	reverse_percentiles_soft: torch.Tensor,
	*,
	delta_tol: float,
	tau_delta: float,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Compute tolerance weights based on percentile gap.

	Args:
		forward_percentiles: pi_f tensor.
		reverse_percentiles_soft: \~π_r tensor ( hub )
		delta_tol: Tolerance window delta.
		tau_delta: Decay scale tau_delta.
	"""

	delta = reverse_percentiles_soft - forward_percentiles
	delta_tol = max(0.0, float(delta_tol))
	tau_delta = max(eps, float(tau_delta))
	excess = torch.clamp(delta - delta_tol, min=0.0)
	return torch.exp(-excess / tau_delta)


def soft_mnn_bundle(
	forward_percentiles: torch.Tensor,
	reverse_percentiles: torch.Tensor,
	labels: torch.Tensor,
	popularity: torch.Tensor,
	*,
	gamma: float,
	tau_f: float,
	tau_r: float,
	delta_tol: float,
	tau_delta: float,
	eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
	"""Aggregate Soft-MNN components for a batch of samples."""

	if forward_percentiles.numel() == 0:
		zero = forward_percentiles.to(torch.float32)
		return {
			"pi_r_soft": zero.clone(),
			"w_forward": torch.ones_like(zero),
			"w_reverse": torch.ones_like(zero),
			"w_tol": torch.ones_like(zero),
			"w_mnn": torch.ones_like(zero),
			"delta": zero.clone(),
		}

	gamma = float(gamma)
	tau_f = max(eps, float(tau_f))
	tau_r = max(eps, float(tau_r))
	delta_tol = float(delta_tol)
	tau_delta = float(tau_delta)

	pi_r_soft = adjust_reverse_percentile(reverse_percentiles, labels, popularity, gamma=gamma, eps=eps)
	w_forward = torch.exp(-forward_percentiles / tau_f)
	w_reverse = torch.exp(-pi_r_soft / tau_r)
	w_tol = tolerance_weight(forward_percentiles, pi_r_soft, delta_tol=delta_tol, tau_delta=max(eps, tau_delta), eps=eps)
	w_mnn = w_forward * w_reverse * w_tol
	delta = pi_r_soft - forward_percentiles

	return {
		"pi_r_soft": pi_r_soft,
		"w_forward": w_forward,
		"w_reverse": w_reverse,
		"w_tol": w_tol,
		"w_mnn": w_mnn,
		"delta": delta,
	}


def boundary_gate(
	margin: torch.Tensor,
	*,
	m0: float,
	scale: float,
) -> torch.Tensor:
	"""Sigmoid boundary gate based on unlabeled margin.

	Args:
		margin: Tensor of boundary gaps (e.g. S5 - S6).
		m0: Target margin; larger values indicate a safer keep decision.
		scale: Sigmoid temperature (smaller = sharper transition).
	"""

	m0 = float(m0)
	scale = max(1e-6, float(scale))
	return torch.sigmoid((m0 - margin) / scale)
