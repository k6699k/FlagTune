#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from FlagTune.mm_pipeline.xgb.common import parse_float, parse_int, to_py_scalar
from FlagTune.mm_pipeline.xgb.config_format import build_config_string_from_entry, build_force_config_entry
from FlagTune.mm_pipeline.xgb.features import resolve_train_ratios


def parse_shape_key_fields(shape_key: Any) -> Dict[str, Any]:
    parts = [part.strip() for part in str(shape_key).split(",")]
    if len(parts) < 6:
        return {}
    values: Dict[str, Any] = {
        "M": parse_int(parts[0]),
        "N": parse_int(parts[1]),
        "K": parse_int(parts[2]),
        "stride_am": parse_int(parts[3]),
        "stride_bk": parse_int(parts[4]),
        "dtype": parts[5],
    }
    return {k: v for k, v in values.items() if v is not None}


def build_predicted_shape_config_entries(best_df: pd.DataFrame) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for _, row in best_df.iterrows():
        shape = row.get("summary_shape_b_m_n_k")
        if shape is None or pd.isna(shape):
            shape = row.get("shape_key")
        parsed_shape = parse_shape_key_fields(row.get("shape_key"))
        dtype = parsed_shape.get("dtype")

        entry: Dict[str, Any] = {
            "shape": to_py_scalar(shape),
            "shape_key": to_py_scalar(row.get("shape_key")),
            "kernel_kind": to_py_scalar(row.get("kernel_kind")),
            "shape_index": to_py_scalar(row.get("shape_index")),
            "M": to_py_scalar(row.get("M")),
            "N": to_py_scalar(row.get("N")),
            "K": to_py_scalar(row.get("K")),
            "dtype": dtype,
            "config": build_force_config_entry(row),
        }
        try:
            entry["benchmark_best_config"] = build_force_config_entry(
                row, prefix="oracle"
            )
        except ValueError:
            pass
        if "candidate_rank" in row and pd.notna(row.get("candidate_rank")):
            entry["candidate_rank"] = to_py_scalar(row.get("candidate_rank"))
        for col in [
            "summary_default_gems_latency_ms",
            "summary_expand_gems_latency_ms",
            "benchmark_best_measured_p50",
            "has_summary_default_gems_latency",
            "has_summary_expand_gems_latency",
            "matches_benchmark_best_config",
            "benchmark_best_in_topk_for_shape",
            "oracle_best_config_order",
            "predicted_best_config_order",
            "predicted_best_xgb_rank_score",
        ]:
            if col in row and pd.notna(row.get(col)):
                entry[col] = to_py_scalar(row.get(col))
        entries.append(entry)
    return entries


def write_predicted_shape_config_outputs(
    best_df: pd.DataFrame,
    out_dir: Path,
    args: argparse.Namespace,
    stem: str = "predicted_shape_configs",
    source_summary_file: Optional[str] = None,
) -> Tuple[Path, Path]:
    entries = build_predicted_shape_config_entries(best_df)
    shape_train_ratio, config_train_ratio = resolve_train_ratios(args)

    yaml_path = out_dir / f"{stem}.yaml"
    md_path = out_dir / f"{stem}.md"
    exported_shape_count = (
        int(best_df["shape_key"].nunique()) if "shape_key" in best_df.columns else len(entries)
    )

    payload = {
        "metadata": {
            "source_db": str(args.db),
            "predict_model": getattr(args, "predict_model", None),
            "predict_db": str(args.db),
            "train_db": list(getattr(args, "train_db", []) or [args.db]),
            "summary_md": str(args.summary_md) if args.summary_md else None,
            "source_summary_file": source_summary_file,
            "total_best_shape_count": int(len(best_df)),
            "exported_shape_count": exported_shape_count,
            "train_mode": args.train_mode,
            "shape_train_ratio": float(shape_train_ratio),
            "config_train_ratio": float(config_train_ratio),
            "top_k": int(args.top_k),
            "entry_count": len(entries),
            "exported_kernel_kinds": sorted(
                str(kind) for kind in best_df["kernel_kind"].dropna().unique()
            )
            if "kernel_kind" in best_df.columns
            else [],
            "usage": "python FlagTune/processing/run_mm_predicted_configs.py --input this_yaml_path",
        },
        "items": entries,
    }

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    with md_path.open("w", encoding="utf-8") as f:
        has_rank = any(item.get("candidate_rank") is not None for item in entries)
        if has_rank:
            f.write("| rank | kernel_kind | shape | dtype | config |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
        else:
            f.write("| kernel_kind | shape | dtype | config |\n")
            f.write("| --- | --- | --- | --- |\n")
        for item in entries:
            if has_rank:
                f.write(
                    "| "
                    f"{item.get('candidate_rank', '')} | "
                    f"{item.get('kernel_kind')} | "
                    f"{item.get('shape')} | "
                    f"{item.get('dtype')} | "
                    f"{build_config_string_from_entry(item.get('config', {}))} |\n"
                )
            else:
                f.write(
                    "| "
                    f"{item.get('kernel_kind')} | "
                    f"{item.get('shape')} | "
                    f"{item.get('dtype')} | "
                    f"{build_config_string_from_entry(item.get('config', {}))} |\n"
                )

    return yaml_path, md_path
