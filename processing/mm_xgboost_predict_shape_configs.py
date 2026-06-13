#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Predict mm top-k configs from an exported XGBoost model and shape-config files."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

try:
    from xgboost import XGBRanker
except ImportError as exc:
    raise SystemExit("Please install xgboost first: pip install xgboost") from exc

try:
    from . import mm_ga_search
    from .mm_hopper_routing import route_mm_kernel_kind
    from .mm_xgboost_from_benchmark_cache import (
        build_global_feature_frame,
        normalize_dtype_name,
        safe_filename,
        to_py_scalar,
    )
except ImportError:
    import mm_ga_search
    from mm_hopper_routing import route_mm_kernel_kind
    from mm_xgboost_from_benchmark_cache import (
        build_global_feature_frame,
        normalize_dtype_name,
        safe_filename,
        to_py_scalar,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
FLAGTUNE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SHAPE_CONFIG_DIR = FLAGTUNE_DIR / "shape-config"
DEFAULT_OUTPUT_DIR = FLAGTUNE_DIR / "mm_xgb_outputs" / "predicted_topk_yamls"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an exported mm XGBoost ranker and write one top-k YAML per "
            "prediction model from FlagTune/shape-config."
        )
    )
    parser.add_argument("--model-dir", required=True, help="Directory with xgboost_ranker.json and feature_schema.json.")
    parser.add_argument("--model", "--predict-model", action="append", dest="models", required=True)
    parser.add_argument("--shape-config-dir", default=str(DEFAULT_SHAPE_CONFIG_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--op", default="mm")
    return parser.parse_args()


def parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def normalize_text(text: str) -> str:
    printable = "".join(ch if ch.isprintable() else " " for ch in text)
    return re.sub(r"\s+", " ", printable).strip()


def parse_shape_line(line: str) -> Optional[Tuple[str, List[int], int]]:
    line = line.strip()
    if "[shape info]:" not in line:
        return None

    op_part = line.split("[shape info]:", 1)[0].strip().rstrip(", ")
    op_name = normalize_text(op_part)
    if not op_name.startswith("flag_gems.ops.mm."):
        return None

    shape_part = line.split("[shape info]:", 1)[1]
    left = shape_part.find("[")
    right = shape_part.find("]", left + 1) if left != -1 else -1
    if left == -1 or right == -1:
        return None

    dims = []
    for token in shape_part[left + 1 : right].split(","):
        token = token.strip()
        dims.append(1 if token == "-" else int(token))

    count = 1
    match = re.search(r"\[count\]\s*:\s*(\d+)", line)
    if match:
        count = int(match.group(1))
    return op_name, dims, count


def normalize_mm_shape(op_name: str, dims: List[int]) -> Optional[Tuple[int, int, int, int]]:
    if op_name.endswith(".gemv_mm") or op_name.endswith(".general_mm"):
        if len(dims) == 3:
            m, k, n = dims
            return 1, int(m), int(n), int(k)
        if len(dims) == 4:
            b, m, n, k = dims
            return int(b), int(m), int(n), int(k)
    return None


def load_txt_shapes(path: Path) -> List[Dict[str, Any]]:
    by_shape: Dict[Tuple[int, int, int, int], int] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_shape_line(line)
            if parsed is None:
                continue
            op_name, dims, count = parsed
            shape = normalize_mm_shape(op_name, dims)
            if shape is None:
                continue
            by_shape[shape] = by_shape.get(shape, 0) + int(count)

    return [
        {"B": b, "M": m, "N": n, "K": k, "count": count}
        for (b, m, n, k), count in sorted(
            by_shape.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def load_yaml_shapes(path: Path, op: str) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported shape YAML format: {path}")

    candidate_keys = [op, "mm", "BlasBenchmark"]
    shape_items = None
    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, dict) and isinstance(value.get("shapes"), list):
            shape_items = value["shapes"]
            break
    if shape_items is None:
        raise RuntimeError(f"No mm shapes found in {path}")

    shapes = []
    for item in shape_items:
        dims = [int(value) for value in item]
        if len(dims) == 4:
            b, m, n, k = dims
        elif len(dims) == 3:
            b, m, n, k = 1, dims[0], dims[1], dims[2]
        else:
            continue
        shapes.append({"B": b, "M": m, "N": n, "K": k, "count": 1})
    return shapes


def resolve_shape_config(model: str, shape_config_dir: Path, op: str) -> Path:
    candidates = [
        shape_config_dir / f"{model}.txt",
        shape_config_dir / f"{model}_{op}.yaml",
        shape_config_dir / f"{model}.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No shape config found for model '{model}'. Tried:\n  - {tried}")


def load_model_shapes(model: str, shape_config_dir: Path, op: str) -> List[Dict[str, Any]]:
    path = resolve_shape_config(model, shape_config_dir, op)
    if path.suffix == ".txt":
        shapes = load_txt_shapes(path)
    else:
        shapes = load_yaml_shapes(path, op)
    if not shapes:
        raise RuntimeError(f"No mm shapes parsed from {path}")
    return shapes


def shape_key_for(m: int, n: int, k: int, dtype: str) -> str:
    return f"{m},{n},{k},{k},{n},{normalize_dtype_name(dtype)}"


def output_shape_for(b: int, m: int, n: int, k: int) -> str:
    return f"{b}, {m}, {n}, {k}"


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


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    model_dir = Path(args.model_dir)
    shape_config_dir = Path(args.shape_config_dir)
    out_dir = Path(args.out_dir)
    model, schema = load_exported_model(model_dir)

    for model_name in args.models:
        shape_config_path = resolve_shape_config(model_name, shape_config_dir, args.op)
        shapes = load_model_shapes(model_name, shape_config_dir, args.op)
        items = build_yaml_items(model, schema, model_name, shapes, args.dtype, args.top_k)
        yaml_path = write_model_yaml(out_dir, model_name, shape_config_path, items, schema, args)
        print(
            f"[OK] {model_name}: shapes={len(shapes)} entries={len(items)} "
            f"topk_yaml={yaml_path}"
        )


if __name__ == "__main__":
    main()
