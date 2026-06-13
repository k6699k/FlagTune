#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from FlagTune.mm_pipeline.xgb.common import CANONICAL_SHAPE_COLS, SHAPE_GROUP_COLS, normalize_dtype_name, parse_float, to_py_scalar
from FlagTune.mm_pipeline.xgb.config_format import build_force_config_entry


def build_benchmark_best_by_shape(
    df: pd.DataFrame,
    config_cols: List[str],
    target: str,
) -> Dict[str, Dict[str, Any]]:
    if df.empty:
        return {}

    target_values = pd.to_numeric(df[target], errors="coerce")
    finite_mask = np.isfinite(target_values.to_numpy(dtype=float)) & (
        target_values.to_numpy(dtype=float) > 0
    )
    work = df.loc[finite_mask].copy()
    if work.empty:
        return {}
    work["_benchmark_target"] = pd.to_numeric(work[target], errors="coerce")

    best_by_shape: Dict[str, Dict[str, Any]] = {}
    for _, group in work.groupby(["shape_key"], dropna=False, sort=False):
        sort_cols = ["_benchmark_target"]
        if "config_order_in_shape" in group.columns:
            sort_cols.append("config_order_in_shape")
        best = group.sort_values(sort_cols, ascending=True).iloc[0]

        prefixed: Dict[str, Any] = {
            "kernel_kind": best.get("kernel_kind"),
            "shape_key": best.get("shape_key"),
        }
        for col in config_cols:
            prefixed[f"oracle_{col}"] = best.get(col)

        try:
            config = build_force_config_entry(pd.Series(prefixed), "oracle")
        except Exception:
            continue

        shape_key = str(best.get("shape_key"))
        record = {
            "kernel_kind": to_py_scalar(best.get("kernel_kind")),
            "benchmark_table": to_py_scalar(best.get("benchmark_table")),
            "shape_key": shape_key,
            "shape": (
                f"1, {int(best.get('M'))}, {int(best.get('N'))}, {int(best.get('K'))}"
                if not pd.isna(best.get("M"))
                and not pd.isna(best.get("N"))
                and not pd.isna(best.get("K"))
                else shape_key
            ),
            "M": to_py_scalar(best.get("M")),
            "N": to_py_scalar(best.get("N")),
            "K": to_py_scalar(best.get("K")),
            "dtype": normalize_dtype_name(best.get("dtype")),
            "benchmark_best_measured_p50": to_py_scalar(best.get(target)),
            "oracle_best_config_order": to_py_scalar(
                best.get("config_order_in_shape")
            ),
            "config": config,
        }
        best_by_shape[shape_key] = record
    return best_by_shape


def export_xgboost_model_artifact(
    train_state: Dict[str, Any],
    args: argparse.Namespace,
    config_cols: List[str],
    train_expand_df: Optional[pd.DataFrame] = None,
) -> None:
    if not args.export_model_dir:
        return

    export_dir = Path(args.export_model_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    model_path = export_dir / "xgboost_ranker.json"
    schema_path = export_dir / "feature_schema.json"

    train_state["model"].save_model(str(model_path))

    schema = {
        "artifact_version": 1,
        "model_file": model_path.name,
        "feature_builder": "build_global_feature_frame",
        "feature_cols": list(train_state["feature_cols"]),
        "config_cols": list(config_cols),
        "canonical_shape_cols": list(CANONICAL_SHAPE_COLS),
        "shape_group_cols": list(SHAPE_GROUP_COLS),
        "rank_score_direction": "higher_is_better",
        "kernel_kind_codes": {"mm": 0, "gemv": 1, "other": 2},
        "dtype_code": "stable_int_hash(dtype) % 997",
        "benchmark_best_by_shape": build_benchmark_best_by_shape(
            train_expand_df if train_expand_df is not None else pd.DataFrame(),
            config_cols,
            args.target,
        ),
    }
    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"[INFO] Exported XGBoost model to {model_path}")
    print(f"[INFO] Exported XGBoost feature schema to {schema_path}")
