#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from FlagTune.mm_pipeline.reports.common import BENCHMARK_LATENCY_COL, BENCHMARK_SPEEDUP_COL, BENCHMARK_TO_RUN_LATENCY_PCT_COL, DEFAULT_LATENCY_COL, GAP_COL, PLOT_DATA_COLS, RUN_LATENCY_COL, RUN_SPEEDUP_COL, _percent_ratio, _speedup, is_missing_value, normalize_shape_for_merge, plot_row_merge_key
from FlagTune.mm_pipeline.reports.summary import load_summary_dataframe, merge_summary_latency_columns


def prepare_plot_dataframe(best_df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    needed_cols = [
        "shape_key",
        "shape_index",
        "summary_shape_b_m_n_k",
        DEFAULT_LATENCY_COL,
    ]
    for col in needed_cols:
        if col not in best_df.columns:
            raise RuntimeError(f"Missing required plot column: {col}")

    plot_df = best_df.copy()
    if "summary_expand_gems_latency_ms" in plot_df.columns:
        plot_df[BENCHMARK_LATENCY_COL] = plot_df["summary_expand_gems_latency_ms"]
    elif BENCHMARK_LATENCY_COL not in plot_df.columns:
        if "oracle_best_measured_p50" in plot_df.columns:
            plot_df[BENCHMARK_LATENCY_COL] = plot_df["oracle_best_measured_p50"]
        else:
            raise RuntimeError(f"Missing required plot column: {BENCHMARK_LATENCY_COL}")

    plot_df[BENCHMARK_SPEEDUP_COL] = _speedup(
        plot_df[DEFAULT_LATENCY_COL], plot_df[BENCHMARK_LATENCY_COL]
    )
    plot_df = plot_df[plot_df[BENCHMARK_SPEEDUP_COL].notna()].copy()
    if plot_df.empty:
        raise RuntimeError(
            "No rows available for plotting after filtering NaN benchmark speedups."
        )

    # Keep CLI compatibility, but plot data is always benchmark-speedup sorted.
    plot_df = plot_df.sort_values(
        [BENCHMARK_SPEEDUP_COL, "shape_index"], ascending=[True, True]
    ).reset_index(drop=True)
    plot_df["plot_rank"] = np.arange(1, len(plot_df) + 1)
    return plot_df


def compact_plot_dataframe(plot_df: pd.DataFrame) -> pd.DataFrame:
    df = plot_df.copy()
    if "shape" not in df.columns:
        if "summary_shape_b_m_n_k" in df.columns:
            df["shape"] = df["summary_shape_b_m_n_k"]
            if "shape_key" in df.columns:
                df["shape"] = df["shape"].where(
                    df["shape"].notna(), df["shape_key"])
        elif "shape_key" in df.columns:
            df["shape"] = df["shape_key"]

    if "summary_expand_gems_latency_ms" in df.columns:
        df[BENCHMARK_LATENCY_COL] = df["summary_expand_gems_latency_ms"]
    elif BENCHMARK_LATENCY_COL not in df.columns:
        if "oracle_best_measured_p50" in df.columns:
            df[BENCHMARK_LATENCY_COL] = df["oracle_best_measured_p50"]

    if DEFAULT_LATENCY_COL in df.columns:
        default_latency = df[DEFAULT_LATENCY_COL]
        if BENCHMARK_LATENCY_COL in df.columns:
            df[BENCHMARK_SPEEDUP_COL] = _speedup(
                default_latency, df[BENCHMARK_LATENCY_COL])

        if RUN_LATENCY_COL in df.columns:
            df[RUN_SPEEDUP_COL] = _speedup(
                default_latency, df[RUN_LATENCY_COL])

    if BENCHMARK_LATENCY_COL in df.columns and RUN_LATENCY_COL in df.columns:
        df[BENCHMARK_TO_RUN_LATENCY_PCT_COL] = _percent_ratio(
            df[BENCHMARK_LATENCY_COL], df[RUN_LATENCY_COL])

    if BENCHMARK_SPEEDUP_COL in df.columns and RUN_SPEEDUP_COL in df.columns:
        df[GAP_COL] = df[BENCHMARK_SPEEDUP_COL] - df[RUN_SPEEDUP_COL]

    if BENCHMARK_SPEEDUP_COL in df.columns:
        sort_cols = [BENCHMARK_SPEEDUP_COL]
        if "shape_index" in df.columns:
            sort_cols.append("shape_index")
        df = df.sort_values(sort_cols, ascending=True).reset_index(drop=True)

    for col in PLOT_DATA_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[PLOT_DATA_COLS]


def write_speedup_plot_data(
    best_df: pd.DataFrame,
    out_dir: Path,
    sort_by: str,
    plot_format: str,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    plot_df = prepare_plot_dataframe(best_df, sort_by=sort_by)
    plot_csv = out_dir / "speedup_plot_data.csv"
    compact_plot_dataframe(plot_df).to_csv(plot_csv, index=False)

    return plot_csv, None, None


def _collect_run_latency_by_shape(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    latency_by_shape: Dict[str, float] = {}
    for row in rows:
        if not row.get("success"):
            continue
        latency_ms = row.get(RUN_LATENCY_COL, row.get("latency_ms"))
        if latency_ms is None:
            continue
        latency = float(latency_ms)
        for shape in [row.get("shape"), row.get("shape_key")]:
            if shape is None:
                continue
            key = normalize_shape_for_merge(shape)
            old_latency = latency_by_shape.get(key)
            if old_latency is None or latency < old_latency:
                latency_by_shape[key] = latency
    return latency_by_shape


def _collect_topk_run_latency_by_shape(out_dir: Path) -> Dict[str, float]:
    latency_by_shape: Dict[str, float] = {}

    def collect_from_frame(frame: pd.DataFrame) -> None:
        latency_col = (
            RUN_LATENCY_COL
            if RUN_LATENCY_COL in frame.columns
            else "latency_ms"
            if "latency_ms" in frame.columns
            else None
        )
        if latency_col is None:
            return
        run_latency = pd.to_numeric(frame[latency_col], errors="coerce")
        for shape_col in ["shape", "shape_key", "summary_shape_b_m_n_k"]:
            if shape_col not in frame.columns:
                continue
            for shape, latency in zip(frame[shape_col], run_latency):
                if pd.isna(shape) or pd.isna(latency):
                    continue
                key = normalize_shape_for_merge(shape)
                old_latency = latency_by_shape.get(key)
                latency = float(latency)
                if old_latency is None or latency < old_latency:
                    latency_by_shape[key] = latency

    for topk_csv in sorted(out_dir.glob("top*_predicted_config_by_shape.csv")):
        collect_from_frame(pd.read_csv(topk_csv))

    latency_csv_paths = sorted(out_dir.glob("predicted_shape_configs_top*latency.csv"))
    for latency_csv in latency_csv_paths:
        collect_from_frame(pd.read_csv(latency_csv))

    latency_yaml_paths = sorted(out_dir.glob("predicted_shape_configs_top*latency.yaml"))
    for latency_yaml in latency_yaml_paths:
        with latency_yaml.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        rows = None
        if isinstance(payload, dict):
            rows = payload.get("items")
            if rows is None:
                rows = payload.get("results")
        if isinstance(rows, list) and rows:
            collect_from_frame(pd.DataFrame(rows))

    if latency_by_shape:
        print(
            f"[INFO] collected run latency for {len(latency_by_shape)} shapes "
            f"from {len(latency_csv_paths)} csv and {len(latency_yaml_paths)} yaml files"
        )
    else:
        print(
            f"[WARN] no top-k run latency found under {out_dir}; "
            "speedup_plot_data.csv will not get run_latency_ms"
        )
    return latency_by_shape


def _build_plot_data_from_topk_run_latency(out_dir: Path) -> Optional[pd.DataFrame]:
    rows: List[Dict[str, Any]] = []

    def collect_from_frame(frame: pd.DataFrame) -> None:
        latency_col = (
            RUN_LATENCY_COL
            if RUN_LATENCY_COL in frame.columns
            else "latency_ms"
            if "latency_ms" in frame.columns
            else None
        )
        if latency_col is None:
            return
        run_latency = pd.to_numeric(frame[latency_col], errors="coerce")
        for idx, latency in enumerate(run_latency):
            if pd.isna(latency):
                continue
            row = frame.iloc[idx]
            shape = row.get("shape")
            shape_key = row.get("shape_key")
            if pd.isna(shape) and pd.isna(shape_key):
                continue
            rows.append(
                {
                    "shape": shape if pd.notna(shape) else shape_key,
                    "shape_key": shape_key if pd.notna(shape_key) else shape,
                    RUN_LATENCY_COL: float(latency),
                }
            )

    for latency_csv in sorted(out_dir.glob("predicted_shape_configs_top*latency.csv")):
        collect_from_frame(pd.read_csv(latency_csv))

    for latency_yaml in sorted(out_dir.glob("predicted_shape_configs_top*latency.yaml")):
        with latency_yaml.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        yaml_rows = None
        if isinstance(payload, dict):
            yaml_rows = payload.get("items") or payload.get("results")
        if isinstance(yaml_rows, list) and yaml_rows:
            collect_from_frame(pd.DataFrame(yaml_rows))

    if not rows:
        return None

    best_by_shape: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = normalize_shape_for_merge(row.get("shape_key") or row.get("shape"))
        old = best_by_shape.get(key)
        if old is None or row[RUN_LATENCY_COL] < old[RUN_LATENCY_COL]:
            best_by_shape[key] = row

    plot_df = pd.DataFrame(best_by_shape.values()).reset_index(drop=True)
    plot_df["shape_index"] = np.arange(len(plot_df), dtype=int)
    return compact_plot_dataframe(plot_df)


def _build_plot_data_from_summary(summary_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if summary_df is None or summary_df.empty:
        return None

    needed = {DEFAULT_LATENCY_COL, "summary_expand_gems_latency_ms"}
    if not needed.issubset(summary_df.columns):
        return None

    plot_df = summary_df.copy()
    default_latency = pd.to_numeric(plot_df[DEFAULT_LATENCY_COL], errors="coerce")
    expand_latency = pd.to_numeric(
        plot_df["summary_expand_gems_latency_ms"], errors="coerce"
    )
    plot_df = plot_df[(default_latency > 0) | (expand_latency > 0)].copy()
    if plot_df.empty:
        return None

    if "shape_index" not in plot_df.columns:
        plot_df["shape_index"] = np.arange(len(plot_df), dtype=int)
    if "shape" not in plot_df.columns and "summary_shape_b_m_n_k" in plot_df.columns:
        plot_df["shape"] = plot_df["summary_shape_b_m_n_k"]

    return compact_plot_dataframe(plot_df)


def merge_topk_plot_rows(plot_df: pd.DataFrame, topk_plot_df: pd.DataFrame) -> pd.DataFrame:
    rows = plot_df.copy().to_dict("records")
    key_to_index: Dict[str, int] = {}
    for idx, row in enumerate(rows):
        key = plot_row_merge_key(row)
        if key is not None and key not in key_to_index:
            key_to_index[key] = idx

    for topk_row in topk_plot_df.copy().to_dict("records"):
        key = plot_row_merge_key(topk_row)
        if key is None or key not in key_to_index:
            if key is not None:
                key_to_index[key] = len(rows)
            rows.append(topk_row)
            continue

        row = rows[key_to_index[key]]
        for col, value in topk_row.items():
            if is_missing_value(value):
                continue
            if col == RUN_LATENCY_COL:
                row[col] = value
            elif is_missing_value(row.get(col)):
                row[col] = value

    return pd.DataFrame(rows)


def merge_latency_into_plot_outputs(
    rows: Iterable[Dict[str, Any]],
    out_dir: Path,
) -> List[Path]:
    latency_by_shape = _collect_run_latency_by_shape(rows)
    if not latency_by_shape:
        return []

    plot_csv = out_dir / "speedup_plot_data.csv"
    if not plot_csv.exists():
        return []

    plot_df = merge_summary_latency_columns(
        pd.read_csv(plot_csv),
        load_summary_dataframe(
            out_dir,
            model=None,
            op="mm",
            summary_md=None,
            dtype="bfloat16",
        ),
    )

    run_latency = _map_run_latency_by_shape(plot_df, latency_by_shape)

    old_run_latency = (
        pd.to_numeric(plot_df[RUN_LATENCY_COL], errors="coerce")
        if RUN_LATENCY_COL in plot_df.columns
        else None
    )
    if old_run_latency is not None:
        plot_df[RUN_LATENCY_COL] = run_latency.where(
            run_latency.notna(), old_run_latency
        )
    else:
        plot_df[RUN_LATENCY_COL] = run_latency

    compact_plot_dataframe(plot_df).to_csv(plot_csv, index=False)
    return [plot_csv]
