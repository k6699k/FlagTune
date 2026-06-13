#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

FLAGTUNE_DIR = __import__('pathlib').Path(__file__).resolve().parents[2]
BENCHMARK_SPEEDUP_COL = "benchmark_best_speedup_pct_vs_summary_default_gems"
GAP_COL = "speedup_gap_pct"
DEFAULT_LATENCY_COL = "summary_default_gems_latency_ms"
BENCHMARK_LATENCY_COL = "benchmark_best_measured_p50"
RUN_LATENCY_COL = "run_latency_ms"
BENCHMARK_TO_RUN_LATENCY_PCT_COL = "benchmark_best_latency_pct_of_run"
RUN_SPEEDUP_COL = "run_speedup_pct_vs_summary_default_gems"
PLOT_DATA_COLS = [
    "shape",
    DEFAULT_LATENCY_COL,
    BENCHMARK_LATENCY_COL,
    RUN_LATENCY_COL,
    BENCHMARK_TO_RUN_LATENCY_PCT_COL,
    BENCHMARK_SPEEDUP_COL,
    RUN_SPEEDUP_COL,
    GAP_COL,
]


def normalize_shape_for_merge(value: Any) -> str:
    return ",".join(part.strip() for part in str(value).split(","))


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def plot_row_merge_key(row: Dict[str, Any]) -> Optional[str]:
    for col in ["shape", "summary_shape_b_m_n_k", "shape_key"]:
        value = row.get(col)
        if not is_missing_value(value):
            return normalize_shape_for_merge(value)
    return None


def parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def parse_float(value: Any) -> float:
    if value is None:
        return np.nan
    text = str(value).strip().replace(",", "")
    if text.endswith("%"):
        text = text[:-1]
    try:
        parsed = float(text) if text else np.nan
    except Exception:
        return np.nan
    return parsed if np.isfinite(parsed) else np.nan


def normalize_dtype_name(dtype: Any) -> str:
    return str(dtype or "bfloat16").replace("torch.", "")


def _speedup(default_latency: pd.Series, latency: pd.Series) -> pd.Series:
    default_latency = pd.to_numeric(default_latency, errors="coerce")
    latency = pd.to_numeric(latency, errors="coerce")
    speedup = ((default_latency / latency) - 1.0) * 100.0
    return speedup.where((default_latency > 0) & (latency > 0))


def _percent_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    ratio = (numerator / denominator) * 100.0
    return ratio.where((numerator > 0) & (denominator > 0))
