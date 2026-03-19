"""Shared CLI helpers for Selective Ada-CSLS knobs."""

from __future__ import annotations

import argparse
import math
from argparse import ArgumentParser, Namespace, SUPPRESS
from typing import Any, Dict, Optional

import numpy as np


_CANONICAL_DEFAULTS: Dict[str, Any] = {
    "use_csls": False,
    "use_ada_csls": False,
    "csls_k": 12,
    "csls_kmin": 5,
    "csls_kmax": 20,
    "csls_alpha": 1.0,
    "csls_m": 10,

    # Ada CSLS column-side params
    "csls_k_side": 12,
    "csls_col_alpha": 1.2,
    "csls_col_m": 20,
    "csls_col_kmin": 8,
    "csls_col_kmax": 24,
}

_STRUCTURAL_DEFAULTS: Dict[str, Any] = {
    "pre_csls_row_safe_k": 35,
    "pre_csls_col_safe_k": 35,
    "pre_csls_row_topL": 5,
    "pre_csls_case2_tau_ratio": 0.5,
    "pre_csls_case2_col_penalty": 0.5,
    "pre_csls_case5_hub_high_quantile": 0.85,
    "pre_csls_case5_hub_mid_quantile": 0.70,
    "pre_csls_bg_penalty": 0.3,
    "pre_csls_case6_penalty_high": 0.5,
    "pre_csls_case6_penalty_mid": 0.25,
    # PoE
    "enable_poe": False,
    "poe_beta": 1.9,
    "poe_lambda_pen": None,
    "poe_lambda_bonus": None,
}

_EEG_ENCODER_CHOICES = ("atm", "eegnet", "shallow", "conformer")


def get_ada_csls_defaults(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a defensive copy of the canonical Ada-CSLS defaults."""
    defaults = dict(_CANONICAL_DEFAULTS)
    if overrides:
        for key, value in overrides.items():
            if key in defaults:
                defaults[key] = value
    return defaults


def register_eeg_encoder_arg(
    parser: ArgumentParser,
    *,
    default: str = "shallow",
) -> ArgumentParser:
    """Register the EEG encoder selection argument.

    The value controls which EEG backbone is used while keeping the rest of the
    retrieval pipeline unchanged.
    """

    parser.add_argument(
        "--chose_eeg_encoder",
        type=lambda value: str(value).strip().lower(),
        choices=_EEG_ENCODER_CHOICES,
        default=default,
        help="Select EEG encoder backbone (atm, eegnet, shallow, conformer)",
    )
    return parser


def register_ada_csls_args(
    parser: ArgumentParser,
    *,
    include_use_flags: bool = True,
    defaults: Optional[Dict[str, Any]] = None,
    legacy_aliases: bool = False,
) -> ArgumentParser:
    """Attach common Ada-CSLS arguments to a parser."""
    merged = get_ada_csls_defaults(defaults)

    if include_use_flags:
        parser.add_argument(
            "--use_csls",
            action="store_true",
            default=merged["use_csls"],
            help="Enable CSLS re-ranking during evaluation",
        )
        parser.add_argument(
            "--use_ada_csls",
            action="store_true",
            default=merged["use_ada_csls"],
            help="Enable Selective Ada-CSLS (requires --use_csls)",
        )

    parser.add_argument(
        "--csls_k",
        type=int,
        default=merged["csls_k"],
        help="Base neighborhood size for CSLS/Ada-CSLS",
    )
    parser.add_argument(
        "--csls_kmin",
        type=int,
        default=merged["csls_kmin"],
        help="Minimum adaptive neighborhood size",
    )
    parser.add_argument(
        "--csls_kmax",
        type=int,
        default=merged["csls_kmax"],
        help="Maximum adaptive neighborhood size",
    )
    parser.add_argument(
        "--csls_alpha",
        type=float,
        default=merged["csls_alpha"],
        help="Row-side density exponent alpha",
    )
    parser.add_argument(
        "--csls_m",
        type=int,
        default=merged["csls_m"],
        help="Row-side density window size m",
    )
    parser.add_argument(
        "--csls_k_side",
        type=int,
        default=merged["csls_k_side"],
        help="Column-side base neighborhood size",
    )
    parser.add_argument(
        "--csls_col_alpha",
        type=float,
        default=merged["csls_col_alpha"],
        help="Column-side density exponent beta (None mirrors alpha)",
    )
    parser.add_argument(
        "--csls_col_m",
        type=int,
        default=merged["csls_col_m"],
        help="Column-side density window size m",
    )
    parser.add_argument(
        "--csls_col_kmin",
        type=int,
        default=merged["csls_col_kmin"],
        help="Column-side minimum adaptive neighborhood",
    )
    parser.add_argument(
        "--csls_col_kmax",
        type=int,
        default=merged["csls_col_kmax"],
        help="Column-side maximum adaptive neighborhood",
    )
    if legacy_aliases:
        pass

    return parser


def normalize_csls_args(args: Namespace) -> Namespace:
    """Best-effort sanitisation for Ada-CSLS related CLI arguments."""
    if hasattr(args, "use_ada_csls"):
        args.use_ada_csls = bool(getattr(args, "use_ada_csls", False))
    return args


def register_structural_args(
    parser: ArgumentParser,
    *,
    defaults: Optional[Dict[str, Any]] = None,
) -> ArgumentParser:
    """Attach pre-CSLS evidence & PoE related controls to a parser."""

    merged = dict(_STRUCTURAL_DEFAULTS)
    if defaults:
        for key, value in defaults.items():
            if key in merged:
                merged[key] = value

    parser.add_argument(
        "--pre_csls_row_safe_k",
        type=int,
        default=merged["pre_csls_row_safe_k"],
        help="Row-side safe band (Top-K) retained as protection before penalties (default: 5)",
    )
    parser.add_argument(
        "--pre_csls_col_safe_k",
        type=int,
        default=merged["pre_csls_col_safe_k"],
        help="Column-side safe band (Top-K) retained as protection before penalties (default: 5)",
    )
    parser.add_argument(
        "--pre_csls_row_topL",
        type=int,
        default=merged["pre_csls_row_topL"],
        help="Row-side L used for popularity (rho) statistics (default: 5)",
    )
    parser.add_argument(
        "--pre_csls_case2_tau_ratio",
        type=float,
        default=merged["pre_csls_case2_tau_ratio"],
        help="Tau ratio for Case2 margin filtering (default: 0.5)",
    )
    parser.add_argument(
        "--pre_csls_case2_col_penalty",
        type=float,
        default=merged["pre_csls_case2_col_penalty"],
        help="Column penalty strength used when Case2 locks a candidate (default: 0.5)",
    )
    parser.add_argument(
        "--pre_csls_case5_hub_high_quantile",
        type=float,
        default=merged["pre_csls_case5_hub_high_quantile"],
        help="Quantile threshold for high hubness level in Case5 (default: 0.95)",
    )
    parser.add_argument(
        "--pre_csls_case5_hub_mid_quantile",
        type=float,
        default=merged["pre_csls_case5_hub_mid_quantile"],
        help="Quantile threshold for mid hubness level in Case5 (default: 0.80)",
    )
    parser.add_argument(
        "--pre_csls_bg_penalty",
        type=float,
        default=merged["pre_csls_bg_penalty"],
        help="Penalty scale applied to background region (default: 1.0)",
    )
    parser.add_argument(
        "--pre_csls_case6_penalty_high",
        type=float,
        default=merged["pre_csls_case6_penalty_high"],
        help="Penalty scale applied to Case6 high-hub entries (default: 0.5)",
    )
    parser.add_argument(
        "--pre_csls_case6_penalty_mid",
        type=float,
        default=merged["pre_csls_case6_penalty_mid"],
        help="Penalty scale applied to Case6 mid-hub entries (default: 0.25)",
    )
    parser.add_argument(
        "--enable_poe",
        action="store_true",
        default=merged["enable_poe"],
        help="Enable Product-of-Experts fusion between CSLS logits and structural expert",
    )
    parser.add_argument(
        "--poe_beta",
        type=float,
        default=merged["poe_beta"],
        help="Fusion weight beta applied to structural expert when PoE is enabled (default: 1.0)",
    )
    parser.add_argument(
        "--poe_lambda_pen",
        type=float,
        default=merged["poe_lambda_pen"],
        help="Override structural penalty scale (default: auto 0.5 * std(S_geom))",
    )
    parser.add_argument(
        "--poe_lambda_bonus",
        type=float,
        default=merged["poe_lambda_bonus"],
        help="Override structural bonus scale (default: auto 0.25 * std(S_geom))",
    )
    return parser


def normalize_structural_args(args: Namespace) -> Namespace:
    """Clamp structural hyper-parameters to safe ranges."""

    def _clamp_quantile(value: Optional[float]) -> float:
        if value is None:
            return 0.0
        try:
            return float(min(1.0, max(0.0, value)))
        except (TypeError, ValueError):
            return 0.0

    if hasattr(args, "pre_csls_row_safe_k"):
        args.pre_csls_row_safe_k = max(1, int(getattr(args, "pre_csls_row_safe_k", 5) or 5))
    if hasattr(args, "pre_csls_col_safe_k"):
        args.pre_csls_col_safe_k = max(1, int(getattr(args, "pre_csls_col_safe_k", 5) or 5))
    if hasattr(args, "pre_csls_row_topL"):
        args.pre_csls_row_topL = max(1, int(getattr(args, "pre_csls_row_topL", 5) or 5))

    high_q = _clamp_quantile(getattr(args, "pre_csls_case5_hub_high_quantile", 0.95))
    mid_q = _clamp_quantile(getattr(args, "pre_csls_case5_hub_mid_quantile", 0.80))
    if mid_q > high_q:
        mid_q = high_q
    args.pre_csls_case5_hub_high_quantile = high_q
    args.pre_csls_case5_hub_mid_quantile = mid_q

    if hasattr(args, "poe_beta"):
        try:
            args.poe_beta = float(getattr(args, "poe_beta", 1.0) or 0.0)
        except (TypeError, ValueError):
            args.poe_beta = 1.0

    if hasattr(args, "pre_csls_bg_penalty"):
        try:
            bg_pen = float(getattr(args, "pre_csls_bg_penalty", 1.0))
        except (TypeError, ValueError):
            bg_pen = 1.0
        args.pre_csls_bg_penalty = max(0.0, bg_pen)

    for attr in ("pre_csls_case6_penalty_high", "pre_csls_case6_penalty_mid"):
        if hasattr(args, attr):
            try:
                value = float(getattr(args, attr))
            except (TypeError, ValueError):
                value = _STRUCTURAL_DEFAULTS[attr]
            setattr(args, attr, max(0.0, value))

    for attr in ("poe_lambda_pen", "poe_lambda_bonus"):
        value = getattr(args, attr, None)
        if value is None:
            continue
        try:
            setattr(args, attr, float(value))
        except (TypeError, ValueError):
            setattr(args, attr, None)

    return args
