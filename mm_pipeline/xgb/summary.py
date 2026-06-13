#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from FlagTune.mm_pipeline.xgb.common import normalize_dtype_name, parse_float, parse_int


def parse_summary_markdown(summary_md: str, dtype: str) -> pd.DataFrame:
    """
    Parse rows like:
    | 1, 136, 64, 7168 | 244 | 0.013024 | 0.022400 | 0.581 | 0.013056 | 0.016512 | 0.791 | 36.14% |

    Output shape_key is:
      M,N,K,K,N,dtype
    for core mm where A=[M,K], B=[K,N], stride_am=K, stride_bk=N.
    """
    path = Path(summary_md)
    if not path.exists():
        raise FileNotFoundError(f"summary markdown not found: {path}")

    rows: List[Dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue

        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) != 9:
            continue

        shape_text = parts[0]
        if not shape_text or shape_text in {"Shape (B, M, N, K)", "Min", "Max", "Avg"}:
            continue
        if shape_text.startswith("---"):
            continue
        if "Torch Latency" in line or "Default Configuration" in line:
            continue

        shape_values = [x.strip() for x in shape_text.split(",")]
        if len(shape_values) != 4:
            continue

        b = parse_int(shape_values[0])
        m = parse_int(shape_values[1])
        n = parse_int(shape_values[2])
        k = parse_int(shape_values[3])
        if b is None or m is None or n is None or k is None:
            continue

        rows.append(
            {
                "summary_shape_b_m_n_k": f"{b}, {m}, {n}, {k}",
                "summary_B": b,
                "summary_M": m,
                "summary_N": n,
                "summary_K": k,
                "shape_key": f"{m},{n},{k},{k},{n},{dtype}",
                "summary_count": parse_int(parts[1]),
                "summary_default_torch_latency_ms": parse_float(parts[2]),
                "summary_default_gems_latency_ms": parse_float(parts[3]),
                "summary_default_gems_speedup": parse_float(parts[4]),
                "summary_expand_torch_latency_ms": parse_float(parts[5]),
                "summary_expand_gems_latency_ms": parse_float(parts[6]),
                "summary_expand_gems_speedup": parse_float(parts[7]),
                "summary_speedup_gain_pct": parse_float(parts[8]),
            }
        )

    if not rows:
        raise RuntimeError(f"No valid summary rows parsed from {summary_md}")

    # The markdown usually contains two sections with duplicated rows.
    return pd.DataFrame(rows).drop_duplicates(subset=["shape_key"], keep="first").reset_index(drop=True)


def merge_summary_default_latency(best_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    merge_cols = [
        "shape_key",
        "summary_shape_b_m_n_k",
        "summary_count",
        "summary_default_torch_latency_ms",
        "summary_default_gems_latency_ms",
        "summary_default_gems_speedup",
        "summary_expand_torch_latency_ms",
        "summary_expand_gems_latency_ms",
        "summary_expand_gems_speedup",
        "summary_speedup_gain_pct",
    ]

    out = best_df.merge(summary_df[merge_cols], on="shape_key", how="left")
    out["has_summary_default_gems_latency"] = out["summary_default_gems_latency_ms"].notna()
    out["has_summary_expand_gems_latency"] = out["summary_expand_gems_latency_ms"].notna()
    # Keep the historical output column name, but use the report's Expand
    # Configuration Gems latency instead of the raw BenchmarkCache oracle p50.
    out["benchmark_best_measured_p50"] = out["summary_expand_gems_latency_ms"]

    out["predicted_speedup_pct_vs_summary_default_gems"] = np.where(
        (out["summary_default_gems_latency_ms"] > 0) & (
            out["predicted_best_measured_p50"] > 0),
        (out["summary_default_gems_latency_ms"] /
         out["predicted_best_measured_p50"] - 1.0) * 100.0,
        np.nan,
    )

    out["oracle_speedup_pct_vs_summary_default_gems"] = np.where(
        (out["summary_default_gems_latency_ms"] > 0) & (
            out["benchmark_best_measured_p50"] > 0),
        (out["summary_default_gems_latency_ms"] /
         out["benchmark_best_measured_p50"] - 1.0) * 100.0,
        np.nan,
    )

    out["speedup_gap_pct"] = (
        out["oracle_speedup_pct_vs_summary_default_gems"]
        - out["predicted_speedup_pct_vs_summary_default_gems"]
    )
    return out
