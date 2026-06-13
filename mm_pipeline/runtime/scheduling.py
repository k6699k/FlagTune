#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from FlagTune.mm_pipeline.runtime.common import normalize_existing_path, normalize_shape_for_merge, parse_float
from FlagTune.mm_pipeline.runtime.payloads import group_shape_keys, parse_shape


def resolve_balance_latency_csv(input_path: Path, args: argparse.Namespace) -> Optional[Path]:
    if args.no_report_latency_balance:
        return None
    if args.balance_latency_csv:
        path = normalize_existing_path(args.balance_latency_csv)
        if not path.exists():
            raise FileNotFoundError(f"--balance-latency-csv not found: {path}")
        return path
    for candidate in [
        input_path.parent / "speedup_plot_data.csv",
        input_path.parent / "parsed_summary_md.csv",
    ]:
        if candidate.exists():
            return candidate
    return None


def load_balance_latency_by_shape(
    input_path: Path, args: argparse.Namespace
) -> tuple[Dict[str, float], Optional[Path], Optional[str]]:
    csv_path = resolve_balance_latency_csv(input_path, args)
    if csv_path is None:
        return {}, None, None

    latency_by_shape: Dict[str, float] = {}
    preferred_latency_cols = (
        [args.balance_latency_col]
        if args.balance_latency_col
        else [
            "summary_expand_gems_latency_ms",
            "benchmark_best_measured_p50",
            "expand_gems_latency_ms",
        ]
    )
    shape_cols = ["summary_shape_b_m_n_k", "shape", "shape_key"]
    used_latency_col: Optional[str] = None

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        latency_col = next(
            (col for col in preferred_latency_cols if col and col in fieldnames),
            None,
        )
        if latency_col is None:
            raise RuntimeError(
                f"No report latency column found in {csv_path}; tried {preferred_latency_cols}"
            )
        used_latency_col = latency_col

        available_shape_cols = [col for col in shape_cols if col in fieldnames]
        if not available_shape_cols:
            raise RuntimeError(
                f"No shape column found in {csv_path}; tried {shape_cols}"
            )

        for row in reader:
            latency = parse_float(row.get(latency_col))
            if latency is None:
                continue
            for shape_col in available_shape_cols:
                value = row.get(shape_col)
                if not value:
                    continue
                key = normalize_shape_for_merge(value)
                old_latency = latency_by_shape.get(key)
                if old_latency is None or latency > old_latency:
                    latency_by_shape[key] = latency

    return latency_by_shape, csv_path, used_latency_col


def report_balance_latency_ms(
    group: Dict[str, Any], latency_by_shape: Dict[str, float]
) -> Optional[float]:
    for key in group_shape_keys(group):
        latency = latency_by_shape.get(key)
        if latency is not None:
            return latency

    entry = group["entries"][0]
    for col in [
        "summary_expand_gems_latency_ms",
        "benchmark_best_measured_p50",
        "report_expand_gems_latency_ms",
    ]:
        latency = parse_float(entry.get(col))
        if latency is not None:
            return latency
    return None


def estimate_group_cost(group: Dict[str, Any]) -> float:
    try:
        shape = parse_shape(group["entries"][0])
        m, n, k = int(shape["M"]), int(shape["N"]), int(shape["K"])
        shape_cost = (
            math.log2(max(2, m))
            * math.log2(max(2, n))
            * math.log2(max(2, k))
        )
        return shape_cost * max(1, len(group["entries"]))
    except Exception:
        return float(max(1, len(group.get("entries", []))))


def select_groups(
    groups: List[Dict[str, Any]],
    args: argparse.Namespace,
    latency_by_shape: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    if args.start_shape < 0:
        raise ValueError("--start-shape must be >= 0")
    if args.top_cost_shapes is not None and args.top_cost_shapes <= 0:
        raise ValueError("--top-cost-shapes must be > 0")

    indexed = list(enumerate(groups))
    selected_pairs = indexed[args.start_shape:]
    if args.limit_shapes is not None:
        selected_pairs = selected_pairs[: args.limit_shapes]

    selected = []
    for original_index, group in selected_pairs:
        group = dict(group)
        if args.max_configs_per_shape is not None:
            group["entries"] = group["entries"][: args.max_configs_per_shape]
        group["shape_group_index"] = original_index
        balance_latency_ms = report_balance_latency_ms(
            group, latency_by_shape or {}
        )
        if balance_latency_ms is not None:
            group["balance_latency_ms"] = balance_latency_ms
            group["estimated_cost"] = balance_latency_ms
            group["cost_source"] = "report_expand_gems_latency_ms"
        else:
            group["estimated_cost"] = float(estimate_group_cost(group))
            group["cost_source"] = "shape_log_mnk_config_count"
        selected.append(group)

    if args.top_cost_shapes is not None:
        selected = sorted(
            selected,
            key=lambda group: group["estimated_cost"],
            reverse=True,
        )[: args.top_cost_shapes]
        selected.sort(key=lambda group: group["shape_group_index"])

    return selected
