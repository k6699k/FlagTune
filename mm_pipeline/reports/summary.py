#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from FlagTune.mm_pipeline.reports.common import DEFAULT_LATENCY_COL, BENCHMARK_LATENCY_COL, FLAGTUNE_DIR, normalize_dtype_name, normalize_shape_for_merge, parse_float, parse_int


def parse_summary_markdown(summary_md: Path, dtype: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    dtype_name = normalize_dtype_name(dtype)
    for raw_line in summary_md.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 9:
            continue
        shape_text = parts[0]
        if not shape_text or shape_text.startswith("---"):
            continue
        if "Torch Latency" in line or "Default Configuration" in line:
            continue
        shape_values = [value.strip() for value in shape_text.split(",")]
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
                "shape_key": f"{m},{n},{k},{k},{n},{dtype_name}",
                "summary_default_gems_latency_ms": parse_float(parts[3]),
                "summary_expand_gems_latency_ms": parse_float(parts[6]),
            }
        )
    if not rows:
        raise RuntimeError(f"No valid summary rows parsed from {summary_md}")
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["shape_key"], keep="first")
        .reset_index(drop=True)
    )


def resolve_summary_md(
    model: Optional[str],
    op: str,
    summary_md: Optional[str],
) -> Optional[Path]:
    if summary_md:
        return Path(summary_md)
    if not model:
        return None
    candidates = [
        FLAGTUNE_DIR / "reports" / f"{model}_{op}.md",
        FLAGTUNE_DIR / "processing" / f"{model}_{op}.md",
        FLAGTUNE_DIR / "processing" / f"{model}_mm.md",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_summary_dataframe(
    out_dir: Path,
    model: Optional[str],
    op: str,
    summary_md: Optional[str],
    dtype: str,
) -> Optional[pd.DataFrame]:
    summary_path = resolve_summary_md(model, op, summary_md)
    if summary_path is not None:
        if not summary_path.is_file():
            raise FileNotFoundError(f"summary markdown not found: {summary_path}")
        summary_df = parse_summary_markdown(summary_path, dtype=dtype)
        parsed_csv = out_dir / "parsed_summary_md.csv"
        parsed_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(parsed_csv, index=False)
        print(f"[INFO] parsed summary md: {summary_path}")
        return summary_df

    summary_csv = out_dir / "parsed_summary_md.csv"
    if not summary_csv.exists():
        if model:
            print(
                f"[WARN] No summary markdown found for model '{model}' "
                f"under {FLAGTUNE_DIR / 'reports'}"
            )
        return None

    summary_df = pd.read_csv(summary_csv)
    needed = {DEFAULT_LATENCY_COL, "summary_expand_gems_latency_ms"}
    if not needed.issubset(summary_df.columns):
        return None
    return summary_df


def _summary_latency_by_shape(
    summary_df: pd.DataFrame,
    latency_col: str,
) -> Dict[str, float]:
    if latency_col not in summary_df.columns:
        return {}
    latency_by_shape: Dict[str, float] = {}
    latencies = pd.to_numeric(summary_df[latency_col], errors="coerce")
    for shape_col in ["summary_shape_b_m_n_k", "shape", "shape_key"]:
        if shape_col not in summary_df.columns:
            continue
        for shape, latency in zip(summary_df[shape_col], latencies):
            if pd.isna(shape) or pd.isna(latency):
                continue
            latency_by_shape[normalize_shape_for_merge(shape)] = float(latency)
    return latency_by_shape


def _map_run_latency_by_shape(
    plot_df: pd.DataFrame,
    latency_by_shape: Dict[str, float],
) -> pd.Series:
    mapped = pd.Series(pd.NA, index=plot_df.index, dtype="Float64")
    for col in ["summary_shape_b_m_n_k", "shape", "shape_key"]:
        if col not in plot_df.columns:
            continue
        keys = plot_df[col].map(
            lambda value: normalize_shape_for_merge(
                value) if pd.notna(value) else None
        )
        col_latency = keys.map(latency_by_shape)
        mapped = mapped.where(mapped.notna(), col_latency)
    return mapped


def merge_summary_latency_columns(
    plot_df: pd.DataFrame,
    summary_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if summary_df is None or summary_df.empty:
        return plot_df

    out = plot_df.copy()
    default_latency_by_shape = _summary_latency_by_shape(
        summary_df, DEFAULT_LATENCY_COL
    )
    expand_latency_by_shape = _summary_latency_by_shape(
        summary_df, "summary_expand_gems_latency_ms"
    )
    if default_latency_by_shape:
        out[DEFAULT_LATENCY_COL] = _map_run_latency_by_shape(
            out, default_latency_by_shape
        )
    if expand_latency_by_shape:
        out[BENCHMARK_LATENCY_COL] = _map_run_latency_by_shape(
            out, expand_latency_by_shape
        )
    return out
