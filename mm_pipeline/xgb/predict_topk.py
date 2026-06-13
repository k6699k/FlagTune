#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

try:
    from xgboost import XGBRanker
except ImportError as exc:
    raise SystemExit("Please install xgboost first: pip install xgboost") from exc

from FlagTune.mm_pipeline.core.routing import route_mm_kernel_kind
from FlagTune.mm_pipeline.core.shape_io import output_shape_for, resolve_shape_config, shape_key_for
from FlagTune.mm_pipeline.ga import search as mm_ga_search
from FlagTune.mm_pipeline.xgb.common import normalize_dtype_name, safe_filename, to_py_scalar
from FlagTune.mm_pipeline.xgb.features import build_global_feature_frame


def iter_flat_configs(kernel_kind: str, schema: Dict[str, Any]) -> Iterable[Dict[str, int]]:
    values = mm_ga_search.legal_value_space(kernel_kind)
    fields = list(values.keys())
    for combo in itertools.product(*(values[field] for field in fields)):
        flat = {field: int(value) for field, value in zip(fields, combo)}
        yield flat


def flat_to_config(flat: Dict[str, int], kernel_kind: str) -> Dict[str, Any]:
    config = mm_ga_search.config_from_flat(flat, kernel_kind)
    config.setdefault("num_ctas", 1)
    return config


def benchmark_best_for_shape(
    schema: Dict[str, Any], shape_key: str
) -> Optional[Dict[str, Any]]:
    by_shape = schema.get("benchmark_best_by_shape") or {}
    if not isinstance(by_shape, dict):
        return None
    record = by_shape.get(shape_key)
    return record if isinstance(record, dict) else None


def config_matches_benchmark_best(
    config: Dict[str, Any],
    benchmark_best: Optional[Dict[str, Any]],
    kernel_kind: str,
) -> bool:
    if not benchmark_best or not isinstance(benchmark_best.get("config"), dict):
        return False
    left = mm_ga_search.config_key(
        mm_ga_search.flatten_config(config), kernel_kind)
    right = mm_ga_search.config_key(
        mm_ga_search.flatten_config(benchmark_best["config"]), kernel_kind)
    return bool(left and right and left == right)


def flat_to_feature_values(flat: Dict[str, int], kernel_kind: str, config_cols: List[str]) -> Dict[str, int]:
    values: Dict[str, int] = {
        "BLOCK_M": int(flat.get("BLOCK_M", 0)),
        "BLOCK_N": int(flat.get("BLOCK_N", 1 if kernel_kind == "gemv" else 0)),
        "BLOCK_K": int(flat.get("BLOCK_K", 0)),
        "GROUP_M": int(flat.get("GROUP_M", 8)),
        "SPLIT_K": int(flat.get("SPLIT_K", 0)),
        "num_warps": int(flat.get("num_warps", 0)),
        "num_ctas": int(flat.get("num_ctas", 1)),
        "num_stages": int(flat.get("num_stages", 0)),
    }
    return {col: values.get(col, 0) for col in config_cols}


def build_candidate_frame(
    shape: Dict[str, Any],
    dtype: str,
    config_cols: List[str],
    schema: Dict[str, Any],
) -> pd.DataFrame:
    m, n, k = int(shape["M"]), int(shape["N"]), int(shape["K"])
    kernel_kind = route_mm_kernel_kind(m, n, k, dtype)
    feature_kernel_kind = "gemv" if kernel_kind == "gemv" else "mm"
    rows: List[Dict[str, Any]] = []
    for config_order, flat in enumerate(iter_flat_configs(kernel_kind, schema)):
        row: Dict[str, Any] = {
            "M": m,
            "N": n,
            "K": k,
            "stride_am": k,
            "stride_bk": n,
            "dtype": normalize_dtype_name(dtype),
            "kernel_kind": feature_kernel_kind,
            "runtime_kernel_kind": kernel_kind,
            "shape_key": shape_key_for(m, n, k, dtype),
            "config_order_in_shape": config_order,
            "config": flat_to_config(flat, kernel_kind),
        }
        row.update(flat_to_feature_values(flat, kernel_kind, config_cols))
        rows.append(row)
    return pd.DataFrame(rows)


def predict_topk_for_shape(
    model: XGBRanker,
    schema: Dict[str, Any],
    shape: Dict[str, Any],
    dtype: str,
    top_k: int,
) -> pd.DataFrame:
    config_cols = list(schema["config_cols"])
    feature_cols = list(schema["feature_cols"])
    candidates = build_candidate_frame(shape, dtype, config_cols, schema)
    x_pred = build_global_feature_frame(candidates, config_cols)
    x_pred = x_pred.reindex(columns=feature_cols, fill_value=0)
    candidates["xgb_rank_score"] = model.predict(x_pred)
    return candidates.sort_values(
        ["xgb_rank_score", "config_order_in_shape"],
        ascending=[False, True],
    ).head(int(top_k))


def build_yaml_items(
    model: XGBRanker,
    schema: Dict[str, Any],
    model_name: str,
    shapes: List[Dict[str, Any]],
    dtype: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for shape_index, shape in enumerate(shapes):
        b, m, n, k = (
            int(shape["B"]),
            int(shape["M"]),
            int(shape["N"]),
            int(shape["K"]),
        )
        shape_key = shape_key_for(m, n, k, dtype)
        benchmark_best = benchmark_best_for_shape(schema, shape_key)
        shape_items: List[Dict[str, Any]] = []
        topk = predict_topk_for_shape(model, schema, shape, dtype, top_k)
        for rank, (_, row) in enumerate(topk.iterrows(), start=1):
            config = row["config"]
            kernel_kind = to_py_scalar(row.get("runtime_kernel_kind"))
            entry = {
                "shape": output_shape_for(b, m, n, k),
                "shape_key": shape_key,
                "kernel_kind": kernel_kind,
                "shape_index": shape_index,
                "candidate_rank": rank,
                "M": m,
                "N": n,
                "K": k,
                "dtype": normalize_dtype_name(dtype),
                "shape_count": int(shape.get("count", 1)),
                "predicted_best_config_order": int(row["config_order_in_shape"]),
                "predicted_best_xgb_rank_score": float(row["xgb_rank_score"]),
                "config": config,
            }
            if benchmark_best:
                entry["benchmark_best_config"] = benchmark_best.get("config")
                entry["benchmark_best_measured_p50"] = benchmark_best.get(
                    "benchmark_best_measured_p50"
                )
                entry["oracle_best_config_order"] = benchmark_best.get(
                    "oracle_best_config_order"
                )
                entry["matches_benchmark_best_config"] = (
                    config_matches_benchmark_best(config, benchmark_best, kernel_kind)
                )
            shape_items.append(entry)

        benchmark_best_in_topk = any(
            bool(item.get("matches_benchmark_best_config")) for item in shape_items
        )
        for item in shape_items:
            if benchmark_best:
                item["benchmark_best_in_topk_for_shape"] = benchmark_best_in_topk
            items.append(item)
    return items


def load_exported_model(model_dir: Path) -> Tuple[XGBRanker, Dict[str, Any]]:
    schema_path = model_dir / "feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"feature_schema.json not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    model_path = model_dir / schema.get("model_file", "xgboost_ranker.json")
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model file not found: {model_path}")

    model = XGBRanker()
    model.load_model(str(model_path))
    force_xgboost_cpu(model)
    return model, schema


def force_xgboost_cpu(model: XGBRanker) -> None:
    """Keep exported-model prediction from initializing a CUDA context."""
    try:
        model.set_params(device="cpu")
    except Exception:
        pass

    try:
        booster = model.get_booster()
    except Exception:
        return

    try:
        booster.set_param({"device": "cpu"})
    except Exception:
        pass


def write_model_yaml(
    out_dir: Path,
    model_name: str,
    shape_config_path: Path,
    items: List[Dict[str, Any]],
    schema: Dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / f"{safe_filename(model_name)}_predicted_shape_configs_top{int(args.top_k)}.yaml"
    payload = {
        "metadata": {
            "predict_model": model_name,
            "shape_config": str(shape_config_path),
            "top_k": int(args.top_k),
            "entry_count": len(items),
            "shape_count": len({item["shape_key"] for item in items}),
            "dtype": normalize_dtype_name(args.dtype),
            "model_dir": str(args.model_dir),
            "model_file": schema.get("model_file"),
            "feature_schema": "feature_schema.json",
            "rank_score_direction": schema.get("rank_score_direction", "higher_is_better"),
        },
        "items": items,
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return yaml_path
