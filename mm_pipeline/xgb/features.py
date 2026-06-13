#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from FlagTune.mm_pipeline.xgb.common import CANONICAL_SHAPE_COLS, DEFAULT_CONFIG_COLS, SHAPE_GROUP_COLS, TRAIN_MODE_RATIOS, normalize_dtype_name, parse_float, parse_int, stable_int_hash


def get_config_cols(df: pd.DataFrame) -> List[str]:
    config_cols = [c for c in DEFAULT_CONFIG_COLS if c in df.columns]
    required = ["BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required config columns: {missing}")
    return config_cols


def kernel_kind_from_table(table: Any) -> str:
    text = str(table)
    if "gemv_kernel" in text:
        return "gemv"
    if "mm_kernel_general_host_tma" in text:
        return "mm"
    return "other"


def add_common_columns(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.copy()
    for col in ["p50", "p20", "p80"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [f"key_{idx}" for idx in range(7)]:
        if col not in df.columns:
            df[col] = np.nan

    for col in DEFAULT_CONFIG_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df["kernel_kind"] = df["benchmark_table"].map(kernel_kind_from_table)
    for col in CANONICAL_SHAPE_COLS:
        df[col] = np.nan
    df["dtype"] = "unknown"

    mm_mask = df["kernel_kind"].eq("mm")
    gemv_mask = df["kernel_kind"].eq("gemv")

    df.loc[mm_mask, "M"] = pd.to_numeric(df.loc[mm_mask, "key_0"], errors="coerce")
    df.loc[mm_mask, "N"] = pd.to_numeric(df.loc[mm_mask, "key_1"], errors="coerce")
    df.loc[mm_mask, "K"] = pd.to_numeric(df.loc[mm_mask, "key_2"], errors="coerce")
    df.loc[mm_mask, "stride_am"] = pd.to_numeric(
        df.loc[mm_mask, "key_3"], errors="coerce"
    )
    df.loc[mm_mask, "stride_bk"] = pd.to_numeric(
        df.loc[mm_mask, "key_4"], errors="coerce"
    )
    df.loc[mm_mask, "dtype"] = df.loc[mm_mask, "key_5"].map(normalize_dtype_name)

    df.loc[gemv_mask, "M"] = pd.to_numeric(df.loc[gemv_mask, "key_0"], errors="coerce")
    df.loc[gemv_mask, "N"] = 1
    df.loc[gemv_mask, "K"] = pd.to_numeric(df.loc[gemv_mask, "key_1"], errors="coerce")
    df.loc[gemv_mask, "stride_am"] = pd.to_numeric(
        df.loc[gemv_mask, "key_2"], errors="coerce"
    )
    df.loc[gemv_mask, "stride_bk"] = pd.to_numeric(
        df.loc[gemv_mask, "key_3"], errors="coerce"
    )
    df.loc[gemv_mask, "dtype"] = df.loc[gemv_mask, "key_4"].map(normalize_dtype_name)

    other_mask = ~(mm_mask | gemv_mask)
    df.loc[other_mask, "M"] = pd.to_numeric(
        df.loc[other_mask, "key_0"], errors="coerce"
    )
    df.loc[other_mask, "N"] = pd.to_numeric(
        df.loc[other_mask, "key_1"], errors="coerce"
    )
    df.loc[other_mask, "K"] = pd.to_numeric(
        df.loc[other_mask, "key_2"], errors="coerce"
    )
    df.loc[other_mask, "stride_am"] = pd.to_numeric(
        df.loc[other_mask, "key_3"], errors="coerce"
    )
    df.loc[other_mask, "stride_bk"] = pd.to_numeric(
        df.loc[other_mask, "key_4"], errors="coerce"
    )
    df.loc[other_mask, "dtype"] = df.loc[other_mask, "key_5"].map(
        normalize_dtype_name
    )

    df["BLOCK_N"] = pd.to_numeric(df["BLOCK_N"], errors="coerce")
    df.loc[gemv_mask & df["BLOCK_N"].isna(), "BLOCK_N"] = 1
    df["GROUP_M"] = pd.to_numeric(df["GROUP_M"], errors="coerce")
    df.loc[df["GROUP_M"].isna(), "GROUP_M"] = 8

    for col in CANONICAL_SHAPE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def shape_key(row: pd.Series) -> str:
        parts: List[str] = []
        for col in CANONICAL_SHAPE_COLS:
            value = row.get(col)
            if pd.isna(value):
                parts.append("")
            else:
                parts.append(str(int(value)))
        parts.append(normalize_dtype_name(row.get("dtype")))
        return ",".join(parts)

    df["shape_key"] = df.apply(shape_key, axis=1)
    target_values = pd.to_numeric(
        df[target], errors="coerce").to_numpy(dtype=float)
    df["is_finite_perf"] = np.isfinite(target_values) & (target_values > 0)

    group_cols = SHAPE_GROUP_COLS
    df = df.sort_values(group_cols + ["sqlite_rowid"]).reset_index(drop=True)
    df["config_order_in_shape"] = df.groupby(
        group_cols, dropna=False).cumcount()
    return df


def key_count_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    counts = df.groupby(group_cols, dropna=False).size(
    ).reset_index(name="num_configs_in_cache")
    meta_cols = ["M", "N", "K", "stride_am", "stride_bk", "dtype"]
    meta = df.groupby(group_cols, dropna=False)[meta_cols].first().reset_index()
    counts = counts.merge(meta, on=group_cols, how="left")
    return counts


def table_kernel_kind(table: Any, table_counts: pd.DataFrame) -> str:
    if "kernel_kind" in table_counts.columns:
        kinds = {
            str(kind).strip().lower()
            for kind in table_counts["kernel_kind"].dropna().unique()
        }
        if "gemv" in kinds:
            return "gemv"
        if "mm" in kinds:
            return "mm"

    table_name = str(table).lower()
    if "gemv_kernel" in table_name:
        return "gemv"
    if "mm_kernel" in table_name:
        return "mm"
    return "other"


def select_expand_df(
    df: pd.DataFrame,
    counts: pd.DataFrame,
    group_cols: List[str],
    expand_count: Optional[int],
    gemv_expand_count: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, int], pd.DataFrame]:
    selected_parts: List[pd.DataFrame] = []
    selected_count_by_table: Dict[str, int] = {}

    for table, table_counts in counts.groupby("benchmark_table", dropna=False):
        available = sorted(table_counts["num_configs_in_cache"].unique().tolist())
        if not available:
            continue
        kernel_kind = table_kernel_kind(table, table_counts)
        if kernel_kind == "gemv":
            requested_count = gemv_expand_count
            requested_label = "gemv-expand-count"
        else:
            requested_count = expand_count
            requested_label = "expand-count"

        if requested_count is None:
            selected_count = int(max(available))
        elif int(requested_count) in available:
            selected_count = int(requested_count)
        else:
            selected_count = int(max(available))
            print(
                f"[WARN] {requested_label}={requested_count} not found for table {table}; "
                f"use table max count={selected_count}. Available: {available}"
            )
        selected_count_by_table[str(table)] = selected_count
        selected_parts.append(
            table_counts[table_counts["num_configs_in_cache"] == selected_count].copy()
        )

    selected_counts = (
        pd.concat(selected_parts, ignore_index=True, sort=False)
        if selected_parts
        else pd.DataFrame()
    )
    if selected_counts.empty:
        available = sorted(counts["num_configs_in_cache"].unique().tolist())
        raise RuntimeError(
            f"No expand shapes found. Available config counts: {available}")

    expand_df = df.merge(selected_counts[group_cols], on=group_cols, how="inner")
    return expand_df, selected_count_by_table, selected_counts


def build_rank_labels(
    target_values: np.ndarray,
    shape_groups: Dict[Tuple[Any, ...], np.ndarray],
) -> np.ndarray:
    labels = np.full(len(target_values), np.nan, dtype=float)
    for positions in shape_groups.values():
        positions = np.asarray(positions, dtype=int)
        if len(positions) == 0:
            continue
        group_targets = target_values[positions]
        order = np.argsort(group_targets, kind="stable")
        group_labels = np.zeros(len(positions), dtype=float)
        group_labels[order] = np.arange(len(positions) - 1, -1, -1, dtype=float)
        labels[positions] = group_labels
    return labels


def validate_ratio(name: str, value: float) -> float:
    value = float(value)
    if not (0.0 < value <= 1.0):
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def resolve_train_ratios(args: argparse.Namespace) -> Tuple[float, float]:
    shape_ratio, config_ratio = TRAIN_MODE_RATIOS[args.train_mode]
    if args.train_ratio is not None and args.config_train_ratio is None:
        config_ratio = args.train_ratio
    if args.shape_train_ratio is not None:
        shape_ratio = args.shape_train_ratio
    if args.config_train_ratio is not None:
        config_ratio = args.config_train_ratio
    return (
        validate_ratio("--shape-train-ratio", shape_ratio),
        validate_ratio("--config-train-ratio", config_ratio),
    )


def ratio_count(total: int, ratio: float) -> int:
    if total <= 0:
        return 0
    if ratio >= 1.0:
        return total
    return max(1, int(math.ceil(total * ratio)))


def build_feature_frame(df: pd.DataFrame, config_cols: List[str]) -> pd.DataFrame:
    x = df[config_cols].copy()
    for col in config_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0)

    if {"BLOCK_M", "BLOCK_N"}.issubset(x.columns):
        x["tile_mn"] = x["BLOCK_M"] * x["BLOCK_N"]
        x["log2_tile_mn"] = np.log2(np.maximum(x["tile_mn"], 1))
    if {"BLOCK_M", "BLOCK_K"}.issubset(x.columns):
        x["tile_mk"] = x["BLOCK_M"] * x["BLOCK_K"]
        x["log2_tile_mk"] = np.log2(np.maximum(x["tile_mk"], 1))
    if {"BLOCK_N", "BLOCK_K"}.issubset(x.columns):
        x["tile_nk"] = x["BLOCK_N"] * x["BLOCK_K"]
        x["log2_tile_nk"] = np.log2(np.maximum(x["tile_nk"], 1))
    for col in ["BLOCK_M", "BLOCK_N", "BLOCK_K"]:
        if col in x.columns:
            x[f"log2_{col}"] = np.log2(np.maximum(x[col], 1))
    if "GROUP_M" in x.columns:
        group_m = np.maximum(x["GROUP_M"], 1)
        x["log2_GROUP_M"] = np.log2(group_m)
    return x


def build_global_feature_frame(df: pd.DataFrame, config_cols: List[str]) -> pd.DataFrame:
    feature_cols = CANONICAL_SHAPE_COLS + config_cols
    x = df.reindex(columns=feature_cols).copy()
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0)

    for col in CANONICAL_SHAPE_COLS:
        if col in x.columns:
            x[f"log2_{col}"] = np.log2(np.maximum(x[col], 1))

    if {"M", "N"}.issubset(x.columns):
        x["shape_mn"] = x["M"] * x["N"]
        x["log2_shape_mn"] = np.log2(np.maximum(x["shape_mn"], 1))
    if {"M", "K"}.issubset(x.columns):
        x["shape_mk"] = x["M"] * x["K"]
        x["log2_shape_mk"] = np.log2(np.maximum(x["shape_mk"], 1))
    if {"N", "K"}.issubset(x.columns):
        x["shape_nk"] = x["N"] * x["K"]
        x["log2_shape_nk"] = np.log2(np.maximum(x["shape_nk"], 1))

    config_features = build_feature_frame(df, config_cols)
    for col in config_features.columns:
        if col not in x.columns:
            x[col] = config_features[col]

    one = pd.Series(1.0, index=x.index)
    block_m = np.maximum(x.get("BLOCK_M", one), 1)
    block_n = np.maximum(x.get("BLOCK_N", one), 1)
    block_k = np.maximum(x.get("BLOCK_K", one), 1)
    group_m = np.maximum(x.get("GROUP_M", one), 1)
    shape_m = np.maximum(x.get("M", one), 1)
    shape_n = np.maximum(x.get("N", one), 1)
    shape_k = np.maximum(x.get("K", one), 1)

    x["grid_m"] = np.ceil(shape_m / block_m)
    x["grid_n"] = np.ceil(shape_n / block_n)
    x["grid_k"] = np.ceil(shape_k / block_k)
    x["grid_mn"] = x["grid_m"] * x["grid_n"]
    x["grid_work"] = x["grid_mn"] * x["grid_k"]
    x["log2_grid_mn"] = np.log2(np.maximum(x["grid_mn"], 1))
    x["log2_grid_work"] = np.log2(np.maximum(x["grid_work"], 1))
    x["m_mod_block_m"] = np.mod(shape_m, block_m)
    x["n_mod_block_n"] = np.mod(shape_n, block_n)
    x["k_mod_block_k"] = np.mod(shape_k, block_k)
    x["block_m_ratio"] = block_m / shape_m
    x["block_n_ratio"] = block_n / shape_n
    x["block_k_ratio"] = block_k / shape_k
    x["tile_volume"] = block_m * block_n * block_k
    x["log2_tile_volume"] = np.log2(np.maximum(x["tile_volume"], 1))
    x["group_tile_m"] = group_m * block_m
    x["log2_group_tile_m"] = np.log2(np.maximum(x["group_tile_m"], 1))
    x["group_tiles_m"] = np.ceil(x["grid_m"] / group_m)
    x["grid_m_per_group"] = x["grid_m"] / group_m
    x["group_m_ratio"] = group_m / np.maximum(x["grid_m"], 1)
    x["m_mod_group_tile_m"] = np.mod(shape_m, np.maximum(x["group_tile_m"], 1))

    kind_map = {"mm": 0, "gemv": 1, "other": 2}
    x["kernel_kind_code"] = df["kernel_kind"].map(kind_map).fillna(2).astype(int)
    x["dtype_code"] = df["dtype"].map(lambda value: stable_int_hash(str(value)) % 997)
    return x
