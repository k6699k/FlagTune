#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train one XGBoost model across mm/gemv expand shapes from FlagGems BenchmarkCache.

Outputs:
- all_benchmark_cache_configs.csv
- benchmark_cache_key_counts.csv
- expand_benchmark_cache_configs.csv
- top{K}_predicted_config_by_shape.xlsx / .csv
- top{K}_benchmark_best_hit_by_shape.xlsx / .csv
- predicted_shape_configs_top{K}.yaml / .md
- optional speedup plot data

Example:
python  mm_xgboost_from_benchmark_cache.py  \
  --db /home/secure/.flaggems/config_cache/TunedConfig_NVIDIA_H800_triton_3_6.db \
  --summary-md ./Deepseek-3.2-p1024d1024_mm.md \
  --summary-dtype bfloat16 \
  --out-dir ./mm_xgb_outputs \
  --expand-count 480 \
  --make-plots

Use mm_speedup_plot.py to draw figures from speedup_plot_data.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    from .mm_speedup_plot import write_speedup_plot_data
except ImportError:
    from mm_speedup_plot import write_speedup_plot_data

try:
    from xgboost import XGBRanker
except ImportError as exc:
    raise SystemExit(
        "Please install xgboost first: pip install xgboost") from exc


DEFAULT_CONFIG_COLS = [
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "GROUP_M",
    "num_warps",
    "num_ctas",
    "num_stages",
]

DEFAULT_KERNEL_SUBSTRS = ["mm_kernel_general_host_tma", "gemv_kernel"]
DEFAULT_GEMV_EXPAND_COUNT = 168
SHAPE_GROUP_COLS = ["source_db", "benchmark_table", "kernel_kind", "shape_key"]
DISPLAY_SHAPE_GROUP_COLS = ["benchmark_table", "kernel_kind", "shape_key"]
CANONICAL_SHAPE_COLS = ["M", "N", "K", "stride_am", "stride_bk"]
TRAIN_MODE_RATIOS = {
    "shape100_config100": (1.0, 1.0),
    "shape50_config100": (0.5, 1.0),
    "shape50_config50": (0.5, 0.5),
    "shape25_config50": (0.25, 0.5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        default="/home/secure/.flaggems/done",
        help=(
            "DB file or directory used to resolve --predict-model. Also used as "
            "the default training source when --train-db is omitted. In --export-only "
            "mode this must be one DB file."
        ),
    )
    parser.add_argument(
        "--predict-model",
        action="append",
        default=None,
        help=(
            "Model name to predict. Can be passed multiple times. The prediction DB is resolved from --db using "
            "<model>_TunedConfig*.db or <model>_YYYYMMDD_HHMMSS_TunedConfig*.db."
        ),
    )
    parser.add_argument(
        "--op",
        default="mm",
        help="Operation name used when auto-resolving summary markdown files.",
    )
    parser.add_argument(
        "--train-db",
        action="append",
        default=None,
        help=(
            "Training DB file or directory. Can be passed multiple times; all resolved "
            "DB files are concatenated to train one ranking model. Relative file names "
            "are also searched under --db when --db is a directory. In --export-only "
            "mode pass exactly one DB file."
        ),
    )
    parser.add_argument(
        "--summary-md",
        default=None,
        help="Optional performance summary markdown. Used to merge default Gems latency.",
    )
    parser.add_argument("--summary-dtype", default="bfloat16")
    parser.add_argument("--out-dir", default="mm_xgb_outputs")
    parser.add_argument(
        "--benchmark-table",
        action="append",
        default=None,
        help=(
            "Exact BenchmarkCache table name to read. Can be passed multiple times. "
            "If omitted, benchmark tables are discovered by --benchmark-table-like "
            "and --kernel-substr."
        ),
    )
    parser.add_argument("--benchmark-table-like", default="%benchmark%")
    parser.add_argument(
        "--kernel-substr",
        action="append",
        default=None,
        help=(
            "Benchmark table name substring to include. Can be passed multiple times. "
            f"Default: {', '.join(DEFAULT_KERNEL_SUBSTRS)}. Use an empty string to include all benchmark tables."
        ),
    )
    parser.add_argument(
        "--expand-count",
        type=int,
        default=None,
        help=(
            "Config count per shape used to identify expand shapes. "
            "If omitted, use each benchmark table's max count. "
            "For gemv tables, --gemv-expand-count is used instead when set. "
            "If a table does not have the requested count, that table falls back to its max count."
        ),
    )
    parser.add_argument(
        "--gemv-expand-count",
        type=int,
        default=DEFAULT_GEMV_EXPAND_COUNT,
        help=(
            "Config count per shape for gemv BenchmarkCache tables. "
            f"Default: {DEFAULT_GEMV_EXPAND_COUNT}. Use 0 to fall back to each gemv table's max count."
        ),
    )
    parser.add_argument("--target", default="p50",
                        choices=["p50", "p20", "p80"])
    parser.add_argument(
        "--train-mode",
        default="shape100_config100",
        choices=sorted(TRAIN_MODE_RATIOS),
        help=(
            "Training sample preset. Prediction is always run over all shapes/configs. "
            "shape100_config100 uses every finite benchmark row; shape50_config100 "
            "uses all configs from half of shapes; shape50_config50 uses half configs "
            "from half of shapes; shape25_config50 uses half configs from one quarter "
            "of shapes."
        ),
    )
    parser.add_argument(
        "--shape-train-ratio",
        type=float,
        default=None,
        help="Override shape sampling ratio after --train-mode is applied.",
    )
    parser.add_argument(
        "--config-train-ratio",
        type=float,
        default=None,
        help="Override per-selected-shape config sampling ratio after --train-mode is applied.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Backward-compatible alias for --config-train-ratio.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-train-rows", type=int, default=8)
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help=(
            "Export the top K predicted configs per shape, ranked by xgb_rank_score."
        ),
    )
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.95)
    parser.add_argument("--colsample-bytree", type=float, default=0.95)
    parser.add_argument("--reg-lambda", type=float, default=1.5)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--max-bin", type=int, default=512)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--export-model-dir",
        default=None,
        help=(
            "Optional directory to export the trained XGBoost ranker. "
            "Writes xgboost_ranker.json and feature_schema.json."
        ),
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Train and export the XGBoost model, then exit before prediction outputs.",
    )

    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument(
        "--plot-sort-by",
        default="oracle",
        choices=["oracle", "predicted", "gap", "shape_index"],
    )
    parser.add_argument("--plot-format", default="png",
                        choices=["png", "pdf", "svg"])
    parser.add_argument("--plot-dpi", type=int, default=220)
    parser.add_argument("--plot-topk-annotate", type=int, default=10)
    return parser.parse_args()


def safe_filename(text: str, max_len: int = 180) -> str:
    text = re.sub(r"[^0-9a-zA-Z._=-]+", "_", str(text)).strip("_")
    return (text or "shape")[:max_len]


def stable_int_hash(text: str) -> int:
    return int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16)


def parse_float(value: Any) -> float:
    if value is None:
        return np.nan
    text = str(value).strip().replace(",", "")
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text) if text else np.nan
    except Exception:
        return np.nan


def parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def db_search_dirs(db_arg: Optional[str]) -> List[Path]:
    if not db_arg:
        return []
    path = Path(db_arg)
    if path.is_dir():
        return [path]
    if path.is_file():
        return [path.parent]
    return []


def resolve_db_paths(
    paths: Iterable[str],
    arg_name: str,
    search_dirs: Optional[Iterable[Path]] = None,
) -> List[Path]:
    resolved: List[Path] = []
    search_dirs = list(search_dirs or [])
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() and not path.is_absolute():
            for search_dir in search_dirs:
                candidate = search_dir / raw_path
                if candidate.exists():
                    path = candidate
                    break
        if path.is_file():
            if path.suffix != ".db":
                raise ValueError(f"{arg_name} must point to a .db file: {path}")
            resolved.append(path)
            continue
        if path.is_dir():
            candidates = sorted(path.glob("*.db"))
            if not candidates:
                raise FileNotFoundError(f"No *.db files found under {arg_name}: {path}")
            resolved.extend(candidates)
            continue
        searched = (
            ", searched under: "
            + ", ".join(str(search_dir) for search_dir in search_dirs)
            if search_dirs and not Path(raw_path).is_absolute()
            else ""
        )
        raise FileNotFoundError(f"{arg_name} path not found: {raw_path}{searched}")

    unique: List[Path] = []
    seen = set()
    for path in resolved:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if not unique:
        raise ValueError(f"No DB files resolved from {arg_name}")
    return unique


def model_name_from_db_path(path: Path) -> str:
    stem = path.name
    if stem.endswith(".db"):
        stem = stem[:-3]
    timestamp_match = re.match(r"^(.+)_([0-9]{8})_([0-9]{6})_TunedConfig.*$", stem)
    if timestamp_match:
        return timestamp_match.group(1)
    tuned_match = re.match(r"^(.+)_TunedConfig.*$", stem)
    if tuned_match:
        return tuned_match.group(1)
    return stem.split("_", 1)[0]


def resolve_predict_db_from_model(model: str, candidates: Iterable[Path]) -> Path:
    candidate_list = list(candidates)
    matches = [
        path
        for path in candidate_list
        if model_name_from_db_path(path) == model
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(str(path) for path in matches)
        raise RuntimeError(f"Multiple DB files match --predict-model {model!r}: {names}")

    if len(candidate_list) == 1:
        return candidate_list[0]

    names = ", ".join(path.name for path in candidate_list[:10])
    suffix = " ..." if len(candidate_list) > 10 else ""
    raise FileNotFoundError(
        f"No DB matched --predict-model {model!r}. "
        f"Expected filenames like {model}_TunedConfig*.db under --db. "
        f"Candidates: {names}{suffix}"
    )


def resolve_predict_targets(
    args: argparse.Namespace,
    train_db_paths: List[Path],
) -> List[Dict[str, Any]]:
    predict_models = list(args.predict_model or [])
    targets: List[Dict[str, Any]] = []

    if predict_models:
        predict_candidates = (
            resolve_db_paths([args.db], "--db") if args.db else train_db_paths
        )
        for model in predict_models:
            db_path = resolve_predict_db_from_model(model, predict_candidates)
            targets.append({"model": model, "db_path": db_path})
    elif args.db:
        for db_path in resolve_db_paths([args.db], "--db"):
            targets.append(
                {"model": model_name_from_db_path(db_path), "db_path": db_path}
            )
    elif len(train_db_paths) == 1:
        db_path = train_db_paths[0]
        targets.append({"model": model_name_from_db_path(db_path), "db_path": db_path})
    else:
        raise ValueError(
            "--predict-model is required when multiple --train-db inputs are used without --db."
        )

    unique: List[Dict[str, Any]] = []
    seen = set()
    for target in targets:
        key = (str(Path(target["db_path"]).resolve()), str(target["model"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(target)
    if not unique:
        raise ValueError("No prediction targets resolved.")
    return unique


def output_dir_for_predict_target(
    base_out_dir: Path,
    target_count: int,
    model: str,
    db_path: Path,
    train_mode: str,
    top_k: int,
) -> Path:
    if target_count == 1:
        return base_out_dir
    model_name = model or model_name_from_db_path(db_path)
    return base_out_dir / (
        f"{safe_filename(model_name)}_{safe_filename(train_mode)}_top{int(top_k)}"
    )


def resolve_summary_md_for_model(model: str, args: argparse.Namespace) -> Optional[Path]:
    if args.summary_md:
        return Path(args.summary_md)

    processing_dir = Path(__file__).resolve().parent
    flagtune_dir = processing_dir.parent
    candidates = [
        flagtune_dir / "reports" / f"{model}_{args.op}.md",
        processing_dir / f"{model}_{args.op}.md",
        processing_dir / f"{model}_mm.md",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def list_benchmark_tables(
    conn: sqlite3.Connection,
    table_like: str,
    kernel_substr: Optional[List[str]],
    explicit_tables: Optional[List[str]] = None,
) -> List[str]:
    all_tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'",
        ).fetchall()
    }
    if explicit_tables:
        missing = [t for t in explicit_tables if t not in all_tables]
        if missing:
            raise RuntimeError(
                f"Benchmark table(s) not found in DB: {missing}")
        return list(explicit_tables)

    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
        (table_like,),
    ).fetchall()
    tables = [r[0] for r in rows]
    kernel_substrs = (
        DEFAULT_KERNEL_SUBSTRS
        if kernel_substr is None
        else [s for s in kernel_substr if s]
    )
    if kernel_substrs:
        tables = [t for t in tables if any(substr in t for substr in kernel_substrs)]
    if not tables:
        raise RuntimeError(
            f"No benchmark tables found. LIKE={table_like!r}, kernel_substr={kernel_substrs!r}"
        )
    return sorted(tables)


def read_benchmark_tables(conn: sqlite3.Connection, tables: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for table in tables:
        df = pd.read_sql_query(
            f'SELECT rowid AS sqlite_rowid, * FROM "{table}"', conn)
        df.insert(0, "benchmark_table", table)
        frames.append(df)
    if not frames:
        raise RuntimeError("No benchmark rows loaded.")
    return pd.concat(frames, ignore_index=True, sort=False)


def read_benchmark_cache_db(
    db_path: Path,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, List[str]]:
    conn = sqlite3.connect(str(db_path))
    try:
        tables = list_benchmark_tables(
            conn,
            table_like=args.benchmark_table_like,
            kernel_substr=args.kernel_substr,
            explicit_tables=args.benchmark_table,
        )
        raw_df = read_benchmark_tables(conn, tables)
    finally:
        conn.close()

    raw_df.insert(0, "source_db", str(db_path))
    raw_df.insert(1, "source_db_name", db_path.name)
    return raw_df, tables


def get_config_cols(df: pd.DataFrame) -> List[str]:
    config_cols = [c for c in DEFAULT_CONFIG_COLS if c in df.columns]
    required = ["BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required config columns: {missing}")
    return config_cols


def normalize_dtype_name(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("torch."):
        text = text.split(".", 1)[1]
    return text


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


def make_model(args: argparse.Namespace, seed: int) -> XGBRanker:
    return XGBRanker(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        objective="rank:pairwise",
        eval_metric="ndcg",
        random_state=seed,
        n_jobs=args.n_jobs,
        tree_method="hist",
        max_bin=args.max_bin,
    )


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


def to_py_scalar(v: Any) -> Any:
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        value = float(v)
        return value if math.isfinite(value) else None
    if isinstance(v, np.bool_):
        return bool(v)
    if pd.isna(v):
        return None
    return v


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


def train_global_ranker(
    train_expand_df: pd.DataFrame,
    config_cols: List[str],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    train_out = train_expand_df.sort_values(
        SHAPE_GROUP_COLS + ["config_order_in_shape"]
    ).reset_index(
        drop=True
    ).copy()
    train_out["used_for_train"] = False
    train_out["xgb_rank_label"] = np.nan

    target_values = pd.to_numeric(
        train_out[args.target], errors="coerce"
    ).to_numpy(dtype=float)
    finite_mask = np.isfinite(target_values) & (target_values > 0)
    finite_positions = np.flatnonzero(finite_mask)
    if len(finite_positions) < args.min_train_rows:
        raise RuntimeError(
            f"Not enough finite benchmark rows to train one global model: {len(finite_positions)}"
        )

    shape_ratio, config_ratio = resolve_train_ratios(args)
    rng = np.random.default_rng(args.seed)
    finite_df = train_out.iloc[finite_positions].copy()
    finite_df["_row_pos"] = finite_positions
    shape_groups = (
        finite_df.groupby(SHAPE_GROUP_COLS, dropna=False, sort=True)["_row_pos"]
        .apply(lambda values: values.to_numpy(dtype=int))
        .to_dict()
    )
    rank_labels = build_rank_labels(target_values, shape_groups)
    train_out["xgb_rank_label"] = rank_labels
    shape_keys = list(shape_groups.keys())
    total_shape_count = len(shape_keys)
    if total_shape_count == 0:
        raise RuntimeError("No finite benchmark shape groups available for training.")

    selected_shape_count = ratio_count(total_shape_count, shape_ratio)
    selected_shape_indexes = rng.choice(
        np.arange(total_shape_count),
        size=selected_shape_count,
        replace=False,
    )
    selected_shape_keys = [shape_keys[int(idx)] for idx in selected_shape_indexes]

    train_position_parts: List[np.ndarray] = []
    for shape_key in selected_shape_keys:
        positions = np.asarray(shape_groups[shape_key], dtype=int)
        config_count = ratio_count(len(positions), config_ratio)
        if config_count >= len(positions):
            selected_positions = positions
        else:
            selected_positions = rng.choice(
                positions,
                size=config_count,
                replace=False,
            )
        selected_positions = np.sort(np.asarray(selected_positions, dtype=int))
        if len(selected_positions) >= 2:
            train_position_parts.append(selected_positions)

    train_positions = (
        np.concatenate(train_position_parts)
        if train_position_parts
        else np.asarray([], dtype=int)
    )
    train_group_sizes = [int(len(part)) for part in train_position_parts]
    if len(train_positions) < args.min_train_rows:
        raise RuntimeError(
            f"Not enough sampled training rows: {len(train_positions)} < {args.min_train_rows}. "
            "Increase --shape-train-ratio/--config-train-ratio or lower --min-train-rows. "
            "Ranking needs at least two sampled configs per train shape."
        )

    train_out.loc[train_positions, "used_for_train"] = True
    train_row_key_cols = ["source_db", "benchmark_table", "sqlite_rowid"]
    train_keys = set(
        tuple(row)
        for row in train_out.loc[train_positions, train_row_key_cols]
        .astype(str)
        .to_numpy()
        .tolist()
    )

    x_train = build_global_feature_frame(train_out.iloc[train_positions], config_cols)
    y_train = rank_labels[train_positions]

    model = make_model(args, args.seed)
    xgboost_fit_start = time.perf_counter()
    model.fit(x_train, y_train, group=train_group_sizes)
    xgboost_fit_elapsed_s = time.perf_counter() - xgboost_fit_start

    train_state = {
        "model": model,
        "feature_cols": list(x_train.columns),
        "train_keys": train_keys,
        "train_row_key_cols": train_row_key_cols,
    }
    train_info = {
        "train_mode": args.train_mode,
        "shape_train_ratio": float(shape_ratio),
        "config_train_ratio": float(config_ratio),
        "xgboost_objective": "rank:pairwise",
        "xgboost_eval_metric": "ndcg",
        "total_finite_config_count": int(len(finite_positions)),
        "total_shape_count": int(total_shape_count),
        "train_shape_count": int(selected_shape_count),
        "train_ranking_group_count": int(len(train_group_sizes)),
        "global_train_config_count": int(len(train_positions)),
        "xgboost_fit_elapsed_s": float(xgboost_fit_elapsed_s),
        "xgboost_params": {
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "learning_rate": float(args.learning_rate),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "reg_lambda": float(args.reg_lambda),
            "reg_alpha": float(args.reg_alpha),
            "min_child_weight": float(args.min_child_weight),
            "gamma": float(args.gamma),
            "max_bin": int(args.max_bin),
            "n_jobs": int(args.n_jobs),
            "objective": "rank:pairwise",
            "eval_metric": "ndcg",
        },
    }
    return train_state, train_info


def predict_with_global_ranker(
    predict_expand_df: pd.DataFrame,
    config_cols: List[str],
    args: argparse.Namespace,
    train_state: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    predict_out = predict_expand_df.sort_values(
        SHAPE_GROUP_COLS + ["config_order_in_shape"]
    ).reset_index(drop=True).copy()
    predict_out["used_for_train"] = False
    predict_out["xgb_rank_label"] = np.nan
    predict_out["xgb_rank_score"] = np.nan

    train_keys = train_state.get("train_keys") or set()
    train_row_key_cols = train_state.get("train_row_key_cols") or [
        "source_db",
        "benchmark_table",
        "sqlite_rowid",
    ]
    if train_keys and all(col in predict_out.columns for col in train_row_key_cols):
        predict_keys = predict_out[train_row_key_cols].astype(str).apply(tuple, axis=1)
        predict_out["used_for_train"] = predict_keys.map(lambda key: key in train_keys)

    predict_target_values = pd.to_numeric(
        predict_out[args.target], errors="coerce"
    ).to_numpy(dtype=float)
    predict_finite_mask = np.isfinite(predict_target_values) & (
        predict_target_values > 0
    )
    predict_finite_positions = np.flatnonzero(predict_finite_mask)
    if len(predict_finite_positions):
        predict_finite_df = predict_out.iloc[predict_finite_positions].copy()
        predict_finite_df["_row_pos"] = predict_finite_positions
        predict_shape_groups = (
            predict_finite_df.groupby(SHAPE_GROUP_COLS, dropna=False, sort=True)[
                "_row_pos"
            ]
            .apply(lambda values: values.to_numpy(dtype=int))
            .to_dict()
        )
        predict_out["xgb_rank_label"] = build_rank_labels(
            predict_target_values, predict_shape_groups
        )

    x_pred = build_global_feature_frame(predict_out, config_cols)
    feature_cols = train_state["feature_cols"]
    x_pred = x_pred.reindex(columns=feature_cols, fill_value=0)

    xgboost_predict_start = time.perf_counter()
    rank_score = train_state["model"].predict(x_pred)
    xgboost_predict_elapsed_s = time.perf_counter() - xgboost_predict_start
    predict_out["xgb_rank_score"] = rank_score

    predict_info = {
        "predict_shape_count": int(
            predict_expand_df.groupby(SHAPE_GROUP_COLS, dropna=False).ngroups
        ),
        "predict_config_count": int(len(predict_out)),
        "xgboost_predict_elapsed_s": float(xgboost_predict_elapsed_s),
    }
    return predict_out, predict_info


def summarize_predicted_shape(
    g: pd.DataFrame,
    config_cols: List[str],
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    finite_pred = (
        g["is_finite_perf"].to_numpy(dtype=bool)
        & np.isfinite(pd.to_numeric(g["xgb_rank_score"], errors="coerce"))
    )
    finite_positions = np.flatnonzero(finite_pred)
    if len(finite_positions) == 0:
        return None

    pred_values = pd.to_numeric(
        g.iloc[finite_positions]["xgb_rank_score"], errors="coerce"
    ).to_numpy(dtype=float)
    best_pos = int(finite_positions[int(np.argmax(pred_values))])
    best_row = g.iloc[best_pos].copy()

    target_values = pd.to_numeric(
        g.iloc[finite_positions][args.target], errors="coerce"
    ).to_numpy(dtype=float)
    oracle_pos = int(finite_positions[int(np.argmin(target_values))])
    oracle_row = g.iloc[oracle_pos].copy()

    oracle_p50 = float(oracle_row["p50"])
    pred_best_p50 = float(best_row["p50"])

    summary: Dict[str, Any] = {
        "shape_key": str(best_row["shape_key"]),
        "benchmark_table": str(best_row["benchmark_table"]),
        "kernel_kind": str(best_row["kernel_kind"]),
        "num_configs_in_shape": int(len(g)),
        "finite_config_count": int(len(finite_positions)),
        "train_config_count": int(g["used_for_train"].sum()),
        "predicted_best_config_order": int(best_row["config_order_in_shape"]),
        "predicted_best_xgb_rank_score": float(best_row["xgb_rank_score"]),
        "predicted_best_measured_p50": pred_best_p50,
        "predicted_best_measured_p20": float(best_row["p20"])
        if "p20" in g.columns and pd.notna(best_row["p20"])
        else np.nan,
        "predicted_best_measured_p80": float(best_row["p80"])
        if "p80" in g.columns and pd.notna(best_row["p80"])
        else np.nan,
        "oracle_best_config_order": int(oracle_row["config_order_in_shape"]),
        "oracle_best_measured_p50": oracle_p50,
        "delta_p50_vs_oracle": float(pred_best_p50 - oracle_p50),
        "delta_pct_vs_oracle": float((pred_best_p50 - oracle_p50) / oracle_p50 * 100.0)
        if oracle_p50 > 0
        else np.nan,
        "predicted_best_is_train": bool(best_row["used_for_train"]),
    }

    for col in CANONICAL_SHAPE_COLS + ["dtype"]:
        summary[col] = to_py_scalar(best_row[col])

    for col in config_cols:
        summary[f"pred_{col}"] = to_py_scalar(best_row[col])
        summary[f"oracle_{col}"] = to_py_scalar(oracle_row[col])

    return summary


def summarize_topk_predicted_shape(
    g: pd.DataFrame,
    config_cols: List[str],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    finite_pred = (
        g["is_finite_perf"].to_numpy(dtype=bool)
        & np.isfinite(pd.to_numeric(g["xgb_rank_score"], errors="coerce"))
    )
    finite_positions = np.flatnonzero(finite_pred)
    if len(finite_positions) == 0:
        return []

    candidate_df = g.iloc[finite_positions].copy()
    candidate_df = candidate_df.sort_values(
        ["xgb_rank_score", "config_order_in_shape"],
        ascending=[False, True],
    )
    top_k = min(int(args.top_k), len(candidate_df))
    candidate_df = candidate_df.head(top_k)

    target_values = pd.to_numeric(
        g.iloc[finite_positions][args.target], errors="coerce"
    ).to_numpy(dtype=float)
    oracle_pos = int(finite_positions[int(np.argmin(target_values))])
    oracle_row = g.iloc[oracle_pos].copy()
    oracle_p50 = float(oracle_row["p50"])

    rows: List[Dict[str, Any]] = []
    for rank, (_, candidate_row) in enumerate(candidate_df.iterrows(), start=1):
        pred_p50 = float(candidate_row["p50"])
        row: Dict[str, Any] = {
            "candidate_rank": int(rank),
            "shape_key": str(candidate_row["shape_key"]),
            "benchmark_table": str(candidate_row["benchmark_table"]),
            "kernel_kind": str(candidate_row["kernel_kind"]),
            "num_configs_in_shape": int(len(g)),
            "finite_config_count": int(len(finite_positions)),
            "train_config_count": int(g["used_for_train"].sum()),
            "predicted_best_config_order": int(candidate_row["config_order_in_shape"]),
            "predicted_best_xgb_rank_score": float(candidate_row["xgb_rank_score"]),
            "predicted_best_measured_p50": pred_p50,
            "predicted_best_measured_p20": float(candidate_row["p20"])
            if "p20" in g.columns and pd.notna(candidate_row["p20"])
            else np.nan,
            "predicted_best_measured_p80": float(candidate_row["p80"])
            if "p80" in g.columns and pd.notna(candidate_row["p80"])
            else np.nan,
            "oracle_best_config_order": int(oracle_row["config_order_in_shape"]),
            "oracle_best_measured_p50": oracle_p50,
            "delta_p50_vs_oracle": float(pred_p50 - oracle_p50),
            "delta_pct_vs_oracle": float((pred_p50 - oracle_p50) / oracle_p50 * 100.0)
            if oracle_p50 > 0
            else np.nan,
            "predicted_best_is_train": bool(candidate_row["used_for_train"]),
        }

        for col in CANONICAL_SHAPE_COLS + ["dtype"]:
            row[col] = to_py_scalar(candidate_row[col])

        for col in config_cols:
            row[f"pred_{col}"] = to_py_scalar(candidate_row[col])
            row[f"oracle_{col}"] = to_py_scalar(oracle_row[col])

        rows.append(row)

    return rows


def build_force_config_entry(row: pd.Series, prefix: str = "pred") -> Dict[str, Any]:
    kernel_kind = row.get("kernel_kind")
    block_m = row.get(f"{prefix}_BLOCK_M")
    block_n = row.get(f"{prefix}_BLOCK_N")
    block_k = row.get(f"{prefix}_BLOCK_K")
    if pd.isna(block_m) or pd.isna(block_k):
        raise ValueError(
            f"Missing {prefix}_BLOCK_M/BLOCK_K for shape={row.get('shape_key')}")
    if kernel_kind != "gemv" and pd.isna(block_n):
        raise ValueError(
            f"Missing {prefix}_BLOCK_N for shape={row.get('shape_key')}")

    group_m = row.get(f"{prefix}_GROUP_M")
    if pd.isna(group_m):
        group_m = 8

    num_warps = row.get(f"{prefix}_num_warps")
    num_stages = row.get(f"{prefix}_num_stages")
    num_ctas = row.get(f"{prefix}_num_ctas")
    if pd.isna(num_warps) or pd.isna(num_stages):
        raise ValueError(
            f"Missing {prefix}_num_warps/stages for shape={row.get('shape_key')}")

    config: Dict[str, Any] = {
        "META": {
            "BLOCK_M": int(block_m),
            "BLOCK_K": int(block_k),
        },
        "num_warps": int(num_warps),
        "num_stages": int(num_stages),
    }
    if kernel_kind == "gemv":
        if not pd.isna(block_n):
            config["META"]["BLOCK_N"] = int(block_n)
    else:
        config["META"]["BLOCK_N"] = int(block_n)
        config["META"]["GROUP_M"] = int(group_m)
    if not pd.isna(num_ctas):
        config["num_ctas"] = int(num_ctas)
    return config


def build_config_string_from_prefix(row: pd.Series, prefix: str) -> str:
    fields = ["BLOCK_M", "BLOCK_N", "BLOCK_K",
              "GROUP_M", "num_warps", "num_ctas", "num_stages"]
    parts = []
    for field in fields:
        col = f"{prefix}_{field}"
        if col not in row:
            continue
        value = row[col]
        if pd.isna(value):
            continue
        try:
            value = int(value)
        except Exception:
            pass
        parts.append(f"{field}={value}")
    return ", ".join(parts)


def build_config_string_from_entry(config: Dict[str, Any]) -> str:
    meta = config.get("META", {}) if isinstance(config, dict) else {}
    fields = [
        ("BLOCK_M", meta.get("BLOCK_M")),
        ("BLOCK_N", meta.get("BLOCK_N")),
        ("BLOCK_K", meta.get("BLOCK_K")),
        ("GROUP_M", meta.get("GROUP_M")),
        ("num_warps", config.get("num_warps")
         if isinstance(config, dict) else None),
        ("num_ctas", config.get("num_ctas") if isinstance(config, dict) else None),
        ("num_stages", config.get("num_stages")
         if isinstance(config, dict) else None),
    ]
    parts = []
    for field, value in fields:
        if value is None or pd.isna(value):
            continue
        try:
            value = int(value)
        except Exception:
            pass
        parts.append(f"{field}={value}")
    return ", ".join(parts)


def remove_stale_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def normalize_config_compare_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    parsed_int = parse_int(value)
    if parsed_int is not None:
        return parsed_int
    return str(value)


def topk_group_cols(df: pd.DataFrame) -> List[str]:
    cols = [
        col
        for col in ["benchmark_table", "kernel_kind", "shape_key"]
        if col in df.columns
    ]
    return cols or ["shape_key"]


def topk_row_matches_oracle(row: pd.Series, config_cols: List[str]) -> bool:
    for col in config_cols:
        pred_value = normalize_config_compare_value(row.get(f"pred_{col}"))
        oracle_value = normalize_config_compare_value(row.get(f"oracle_{col}"))
        if pred_value != oracle_value:
            return False
    return True


def annotate_topk_benchmark_best_hits(
    topk_df: pd.DataFrame,
    config_cols: List[str],
) -> pd.DataFrame:
    out = topk_df.copy()
    out["matches_benchmark_best_config"] = out.apply(
        lambda row: topk_row_matches_oracle(row, config_cols), axis=1
    )
    group_cols = topk_group_cols(out)
    out["benchmark_best_in_topk_for_shape"] = out.groupby(
        group_cols,
        dropna=False,
    )["matches_benchmark_best_config"].transform("any")
    return out


def build_topk_benchmark_best_hit_dataframe(
    topk_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = topk_group_cols(topk_df)

    for _, group in topk_df.groupby(group_cols, dropna=False, sort=False):
        group = group.sort_values("candidate_rank")
        first = group.iloc[0]
        hit_rows = group[group["matches_benchmark_best_config"]]
        hit_ranks = [
            str(int(rank))
            for rank in hit_rows["candidate_rank"].dropna().tolist()
        ]
        rank1 = group.iloc[0]

        shape = first.get("summary_shape_b_m_n_k")
        if shape is None or pd.isna(shape):
            shape = first.get("shape_key")

        rows.append(
            {
                "kernel_kind": first.get("kernel_kind"),
                "benchmark_table": first.get("benchmark_table"),
                "shape": shape,
                "shape_key": first.get("shape_key"),
                "M": first.get("M"),
                "N": first.get("N"),
                "K": first.get("K"),
                "dtype": first.get("dtype"),
                "requested_top_k": int(args.top_k),
                "exported_topk_count": int(len(group)),
                "benchmark_best_in_topk": bool(len(hit_rows) > 0),
                "benchmark_best_hit_rank": ",".join(hit_ranks) if hit_ranks else "",
                "benchmark_best_measured_p50": first.get("benchmark_best_measured_p50"),
                "benchmark_best_config_order": first.get("oracle_best_config_order"),
                "benchmark_best_config": build_config_string_from_prefix(first, "oracle"),
                "rank1_predicted_config": build_config_string_from_prefix(rank1, "pred"),
                "rank1_predicted_measured_p50": rank1.get("predicted_best_measured_p50"),
                "rank1_delta_pct_vs_oracle": rank1.get("delta_pct_vs_oracle"),
            }
        )

    return pd.DataFrame(rows)


def topk_candidate_specific_column(col: str) -> bool:
    if col.startswith("pred_"):
        return True
    return col in {
        "candidate_rank",
        "predicted_best_config_order",
        "predicted_best_xgb_rank_score",
        "predicted_best_measured_p50",
        "predicted_best_measured_p20",
        "predicted_best_measured_p80",
        "delta_p50_vs_oracle",
        "delta_pct_vs_oracle",
        "predicted_best_is_train",
        "matches_benchmark_best_config",
    }


def excel_merge_compare_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def series_has_single_display_value(series: pd.Series) -> bool:
    values = [excel_merge_compare_value(value) for value in series.tolist()]
    if not values:
        return False
    first = values[0]
    return all(value == first for value in values[1:])


def write_topk_outputs(
    topk_df: pd.DataFrame,
    out_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Path, Path, Path, Path]:
    topk_label = f"top{int(args.top_k)}"
    topk_csv = out_dir / f"{topk_label}_predicted_config_by_shape.csv"
    topk_xlsx = out_dir / f"{topk_label}_predicted_config_by_shape.xlsx"
    hit_csv = out_dir / f"{topk_label}_benchmark_best_hit_by_shape.csv"
    hit_xlsx = out_dir / f"{topk_label}_benchmark_best_hit_by_shape.xlsx"

    remove_stale_file(topk_csv)
    remove_stale_file(hit_csv)
    topk_df.to_csv(topk_csv, index=False)

    hit_df = build_topk_benchmark_best_hit_dataframe(topk_df, args)
    hit_df.to_csv(hit_csv, index=False)

    from openpyxl.styles import Alignment

    with pd.ExcelWriter(topk_xlsx, engine="openpyxl") as writer:
        topk_df.to_excel(writer, sheet_name="topk_by_shape",
                         index=False, na_rep="", inf_rep="inf")
        ws = writer.sheets["topk_by_shape"]
        ws.freeze_panes = "A2"
        group_cols = topk_group_cols(topk_df)
        merge_cols = [
            col for col in topk_df.columns if not topk_candidate_specific_column(col)
        ]
        for _, group in topk_df.groupby(group_cols, dropna=False, sort=False):
            if len(group) <= 1:
                continue
            row_start = int(group.index.min()) + 2
            row_end = int(group.index.max()) + 2
            if row_end - row_start + 1 != len(group):
                continue
            for col in merge_cols:
                if not series_has_single_display_value(group[col]):
                    continue
                col_idx = int(topk_df.columns.get_loc(col)) + 1
                ws.merge_cells(
                    start_row=row_start,
                    start_column=col_idx,
                    end_row=row_end,
                    end_column=col_idx,
                )
                ws.cell(row=row_start, column=col_idx).alignment = Alignment(
                    vertical="center"
                )

    with pd.ExcelWriter(hit_xlsx, engine="openpyxl") as writer:
        hit_df.to_excel(writer, sheet_name="topk_hit",
                        index=False, na_rep="", inf_rep="inf")

    return topk_csv, topk_xlsx, hit_csv, hit_xlsx


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


def write_run_summary_yaml(out_path: Path, summary: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)


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


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def write_prediction_target_outputs(
    target_args: argparse.Namespace,
    out_dir: Path,
    predict_db_path: Path,
    predict_tables: List[str],
    predict_df: pd.DataFrame,
    counts: pd.DataFrame,
    expand_df: pd.DataFrame,
    expand_count_by_table: Dict[str, int],
    expand_counts: pd.DataFrame,
    train_db_paths: List[Path],
    train_tables_by_db: Dict[str, List[str]],
    train_df: pd.DataFrame,
    train_counts: pd.DataFrame,
    train_expand_df: pd.DataFrame,
    train_expand_count_by_table: Dict[str, int],
    train_expand_counts: pd.DataFrame,
    gemv_expand_count: Optional[int],
    db_read_elapsed_s: float,
    train_info: Dict[str, Any],
    predict_info: Dict[str, Any],
    predicted_df: pd.DataFrame,
    config_cols: List[str],
    xgboost_elapsed_s: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cache_csv = out_dir / "all_benchmark_cache_configs.csv"
    counts_csv = out_dir / "benchmark_cache_key_counts.csv"
    expand_perf_csv = out_dir / "expand_benchmark_cache_configs.csv"

    predict_df.to_csv(all_cache_csv, index=False)
    counts.to_csv(counts_csv, index=False)
    predicted_df.to_csv(expand_perf_csv, index=False)

    best_rows: List[Dict[str, Any]] = []
    topk_rows: List[Dict[str, Any]] = []
    for shape_idx, (_, group) in enumerate(
        predicted_df.groupby(SHAPE_GROUP_COLS, dropna=False, sort=True)
    ):
        shape_key = str(group["shape_key"].iloc[0])
        shape_pred_df = group.copy()
        summary = summarize_predicted_shape(
            shape_pred_df,
            config_cols=config_cols,
            args=target_args,
        )
        topk_summaries = summarize_topk_predicted_shape(
            shape_pred_df,
            config_cols=config_cols,
            args=target_args,
        )

        if summary is not None:
            summary["shape_index"] = shape_idx
            summary["global_train_config_count"] = int(
                train_info["global_train_config_count"]
            )
            summary["shape_used_for_train"] = bool(
                shape_pred_df["used_for_train"].any()
            )
            best_rows.append(summary)
            for topk_summary in topk_summaries:
                topk_summary["shape_index"] = shape_idx
                topk_summary["global_train_config_count"] = int(
                    train_info["global_train_config_count"]
                )
                topk_summary["shape_used_for_train"] = bool(
                    shape_pred_df["used_for_train"].any()
                )
                topk_rows.append(topk_summary)
        else:
            print(f"[WARN] skipped training for shape: {shape_key}")

    if not best_rows:
        raise RuntimeError("No shape produced a valid XGBoost model.")

    best_df = pd.DataFrame(best_rows).sort_values(
        "shape_index").reset_index(drop=True)
    topk_df = pd.DataFrame(topk_rows).sort_values(
        ["shape_index", "candidate_rank"]).reset_index(drop=True)

    summary_md_csv = None
    if target_args.summary_md:
        summary_df = parse_summary_markdown(
            target_args.summary_md, dtype=target_args.summary_dtype
        )
        summary_md_csv = out_dir / "parsed_summary_md.csv"
        summary_df.to_csv(summary_md_csv, index=False)
        best_df = merge_summary_default_latency(best_df, summary_df)
        topk_df = merge_summary_default_latency(topk_df, summary_df)
        matched = int(best_df["has_summary_default_gems_latency"].sum())
        print(
            f"[INFO] merged summary md default gems latency: {matched}/{len(best_df)} shapes matched"
        )
    else:
        best_df["has_summary_default_gems_latency"] = False
        best_df["has_summary_expand_gems_latency"] = False
        best_df["summary_default_gems_latency_ms"] = np.nan
        best_df["summary_expand_gems_latency_ms"] = np.nan
        best_df["benchmark_best_measured_p50"] = np.nan
        best_df["predicted_speedup_pct_vs_summary_default_gems"] = np.nan
        best_df["oracle_speedup_pct_vs_summary_default_gems"] = np.nan
        best_df["speedup_gap_pct"] = np.nan
        topk_df["has_summary_default_gems_latency"] = False
        topk_df["has_summary_expand_gems_latency"] = False
        topk_df["summary_default_gems_latency_ms"] = np.nan
        topk_df["summary_expand_gems_latency_ms"] = np.nan
        topk_df["benchmark_best_measured_p50"] = np.nan
        topk_df["predicted_speedup_pct_vs_summary_default_gems"] = np.nan
        topk_df["oracle_speedup_pct_vs_summary_default_gems"] = np.nan
        topk_df["speedup_gap_pct"] = np.nan

    topk_df = annotate_topk_benchmark_best_hits(topk_df, config_cols)

    topk_csv, topk_xlsx, topk_hit_csv, topk_hit_xlsx = write_topk_outputs(
        topk_df,
        out_dir,
        target_args,
    )
    topk_label = f"top{int(target_args.top_k)}"
    topk_shape_config_yaml, topk_shape_config_md = (
        write_predicted_shape_config_outputs(
            topk_df,
            out_dir,
            target_args,
            stem=f"predicted_shape_configs_{topk_label}",
            source_summary_file=f"{topk_label}_predicted_config_by_shape.xlsx",
        )
    )

    plot_csv = None
    plot_comparison_path = None
    plot_gap_path = None
    if target_args.make_plots:
        if not target_args.summary_md:
            print(
                "[WARN] --make-plots is set, but summary markdown is missing. Skip plot data."
            )
        else:
            plot_csv, plot_comparison_path, plot_gap_path = write_speedup_plot_data(
                best_df=best_df,
                out_dir=out_dir,
                sort_by=target_args.plot_sort_by,
                plot_format=target_args.plot_format,
            )
            print("[INFO] plot data generated. Plot image generation is disabled.")

    train_predict_info = {**train_info, **predict_info}
    summary_payload = {
        "source_db": str(predict_db_path),
        "predict_model": target_args.predict_model,
        "predict_db": str(predict_db_path),
        "train_db": [str(path) for path in train_db_paths],
        "summary_md": str(target_args.summary_md) if target_args.summary_md else None,
        "summary_dtype": target_args.summary_dtype,
        "out_dir": str(out_dir),
        "benchmark_tables": predict_tables,
        "predict_benchmark_tables": predict_tables,
        "train_benchmark_tables_by_db": train_tables_by_db,
        "all_benchmark_cache_rows": int(len(predict_df)),
        "predict_benchmark_cache_rows": int(len(predict_df)),
        "train_benchmark_cache_rows": int(len(train_df)),
        "expand_config_count_by_table": expand_count_by_table,
        "train_expand_config_count_by_table": train_expand_count_by_table,
        "requested_expand_count": int(target_args.expand_count)
        if target_args.expand_count is not None
        else None,
        "requested_gemv_expand_count": int(gemv_expand_count)
        if gemv_expand_count is not None
        else None,
        "expand_shape_count": int(expand_counts.shape[0]),
        "expand_rows": int(len(expand_df)),
        "train_expand_shape_count": int(train_expand_counts.shape[0]),
        "train_expand_rows": int(len(train_expand_df)),
        "trained_shape_count": int(len(best_df)),
        "top_k": int(target_args.top_k),
        "topk_entry_count": int(len(topk_df)),
        "xgboost_model_granularity": "one_xgboost_ranker_for_all_predict_targets",
        "database_read_elapsed_s": float(db_read_elapsed_s),
        **train_predict_info,
        "xgboost_train_predict_elapsed_s": float(xgboost_elapsed_s),
        "all_cache_csv": str(all_cache_csv),
        "counts_csv": str(counts_csv),
        "expand_perf_csv": str(expand_perf_csv),
        "xgb_scored_all_configs_csv": str(expand_perf_csv),
        "xgb_predicted_all_configs_csv": str(expand_perf_csv),
        "parsed_summary_md_csv": str(summary_md_csv) if summary_md_csv else None,
        "topk_csv": str(topk_csv),
        "topk_xlsx": str(topk_xlsx),
        "topk_benchmark_best_hit_csv": str(topk_hit_csv),
        "topk_benchmark_best_hit_xlsx": str(topk_hit_xlsx),
        "topk_shape_config_yaml": str(topk_shape_config_yaml),
        "topk_shape_config_md": str(topk_shape_config_md),
        "plot_csv": str(plot_csv) if plot_csv else None,
        "plot_comparison": str(plot_comparison_path) if plot_comparison_path else None,
        "plot_gap": str(plot_gap_path) if plot_gap_path else None,
        "plot_sort_by": target_args.plot_sort_by,
        "plot_format": target_args.plot_format,
        "plot_dpi": int(target_args.plot_dpi),
        "plot_topk_annotate": int(target_args.plot_topk_annotate),
    }

    run_summary_yaml = out_dir / "run_summary.yaml"
    write_run_summary_yaml(run_summary_yaml, summary_payload)

    print("[OK] outputs:")
    print(f"  predict BenchmarkCache config data: {all_cache_csv}")
    print(f"  key counts: {counts_csv}")
    print(f"  expand BenchmarkCache config data: {expand_perf_csv}")
    if summary_md_csv:
        print(f"  parsed summary md: {summary_md_csv}")
    print(f"  top-k summary csv: {topk_csv}")
    print(f"  top-k summary xlsx: {topk_xlsx}")
    print(f"  top-k benchmark-best hit csv: {topk_hit_csv}")
    print(f"  top-k benchmark-best hit xlsx: {topk_hit_xlsx}")
    print(f"  top-k predicted shape configs yaml: {topk_shape_config_yaml}")
    print(f"  top-k predicted shape configs md: {topk_shape_config_md}")
    if plot_csv:
        print(f"  plot data csv: {plot_csv}")
    print(f"  run summary yaml: {run_summary_yaml}")
    return topk_shape_config_yaml


def main() -> None:
    args = parse_args()
    resolve_train_ratios(args)
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.export_only and not args.export_model_dir:
        raise ValueError("--export-only requires --export-model-dir")
    gemv_expand_count = (
        int(args.gemv_expand_count)
        if args.gemv_expand_count and args.gemv_expand_count > 0
        else None
    )

    train_db_inputs = list(args.train_db or [])
    if not train_db_inputs and args.db:
        train_db_inputs = [args.db]
    if not train_db_inputs:
        raise ValueError("Pass --train-db, or use backward-compatible --db.")

    if args.export_only:
        if len(train_db_inputs) != 1:
            raise ValueError("--export-only requires exactly one training DB file")
        train_db_path = Path(train_db_inputs[0])
        if not train_db_path.is_file():
            raise ValueError(
                f"--export-only training DB must be a file, not a directory or missing path: {train_db_path}"
            )
        train_db_paths = [train_db_path]
    else:
        train_db_paths = resolve_db_paths(
            train_db_inputs,
            "--train-db/--db",
            search_dirs=db_search_dirs(args.db),
        )
    predict_targets = [] if args.export_only else resolve_predict_targets(args, train_db_paths)
    args.train_db = [str(path) for path in train_db_paths]

    base_out_dir = Path(args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    if len(predict_targets) > 1 and args.summary_md:
        print(
            "[WARN] one --summary-md was provided for multiple predict targets; "
            "the same summary file will be used for all targets."
        )

    db_read_start = time.perf_counter()
    train_raw_parts: List[pd.DataFrame] = []
    train_tables_by_db: Dict[str, List[str]] = {}
    for train_db_path in train_db_paths:
        raw_part, tables = read_benchmark_cache_db(train_db_path, args)
        train_raw_parts.append(raw_part)
        train_tables_by_db[str(train_db_path)] = tables
    train_raw_df = pd.concat(train_raw_parts, ignore_index=True, sort=False)

    predict_contexts: List[Dict[str, Any]] = []
    if not args.export_only:
        for target in predict_targets:
            predict_db_path = Path(target["db_path"])
            predict_raw_df, predict_tables = read_benchmark_cache_db(predict_db_path, args)
            predict_contexts.append(
                {
                    "model": str(target["model"]),
                    "db_path": predict_db_path,
                    "raw_df": predict_raw_df,
                    "tables": predict_tables,
                }
            )
    db_read_elapsed_s = time.perf_counter() - db_read_start
    print(
        "[TIME] database read time: "
        f"{format_duration(db_read_elapsed_s)} ({db_read_elapsed_s:.3f}s)"
    )

    train_df = add_common_columns(train_raw_df, args.target)
    all_config_frames = [train_df]
    for context in predict_contexts:
        predict_df = add_common_columns(context["raw_df"], args.target)
        context["predict_df"] = predict_df
        all_config_frames.append(predict_df)
    config_cols = get_config_cols(
        pd.concat(all_config_frames, ignore_index=True, sort=False)
    )

    train_counts = key_count_table(train_df, SHAPE_GROUP_COLS)

    train_expand_df, train_expand_count_by_table, train_expand_counts = select_expand_df(
        train_df,
        train_counts,
        group_cols=SHAPE_GROUP_COLS,
        expand_count=args.expand_count,
        gemv_expand_count=gemv_expand_count,
    )
    for context in predict_contexts:
        predict_df = context["predict_df"]
        counts = key_count_table(predict_df, SHAPE_GROUP_COLS)
        expand_df, expand_count_by_table, expand_counts = select_expand_df(
            predict_df,
            counts,
            group_cols=SHAPE_GROUP_COLS,
            expand_count=args.expand_count,
            gemv_expand_count=gemv_expand_count,
        )
        context["counts"] = counts
        context["expand_df"] = expand_df
        context["expand_count_by_table"] = expand_count_by_table
        context["expand_counts"] = expand_counts

    print(f"[INFO] train DB files: {len(train_db_paths)}")
    for path in train_db_paths:
        print(f"  - {path}")
    print(f"[INFO] predict targets: {len(predict_contexts)}")
    print(f"[INFO] train BenchmarkCache rows: {len(train_df)}")
    print("[INFO] train expand config count per table:")
    for table, count in train_expand_count_by_table.items():
        print(f"  - {table}: {count}")
    print(f"[INFO] train expand shape count: {train_expand_counts.shape[0]}")
    print(f"[INFO] train expand rows: {len(train_expand_df)}")

    xgboost_train_start = time.perf_counter()
    train_state, train_info = train_global_ranker(
        train_expand_df,
        config_cols=config_cols,
        args=args,
    )
    xgboost_train_elapsed_s = time.perf_counter() - xgboost_train_start
    print(
        "[INFO] xgboost training sample: "
        f"objective={train_info['xgboost_objective']}, "
        f"mode={train_info['train_mode']}, "
        f"shape_ratio={train_info['shape_train_ratio']}, "
        f"config_ratio={train_info['config_train_ratio']}, "
        f"train_shapes={train_info['train_shape_count']}/{train_info['total_shape_count']}, "
        f"train_groups={train_info['train_ranking_group_count']}, "
        f"train_configs={train_info['global_train_config_count']}/"
        f"{train_info['total_finite_config_count']}"
    )
    print(
        "[TIME] xgboost training time: "
        f"{format_duration(train_info['xgboost_fit_elapsed_s'])} "
        f"({train_info['xgboost_fit_elapsed_s']:.3f}s)"
    )
    print(
        "[TIME] xgboost one-time training overhead: "
        f"{format_duration(xgboost_train_elapsed_s)} ({xgboost_train_elapsed_s:.3f}s)"
    )
    export_xgboost_model_artifact(
        train_state=train_state,
        args=args,
        config_cols=config_cols,
        train_expand_df=train_expand_df,
    )
    if args.export_only:
        print("[INFO] --export-only set; skip prediction outputs.")
        return

    for context in predict_contexts:
        model = context["model"]
        predict_db_path = Path(context["db_path"])
        target_args = argparse.Namespace(**vars(args))
        target_args.db = str(predict_db_path)
        target_args.predict_model = model
        target_args.train_db = [str(path) for path in train_db_paths]
        summary_md = resolve_summary_md_for_model(model, target_args)
        target_args.summary_md = str(summary_md) if summary_md else None

        out_dir = output_dir_for_predict_target(
            base_out_dir,
            len(predict_contexts),
            model,
            predict_db_path,
            args.train_mode,
            args.top_k,
        )
        print(f"[INFO] Processing predict target: {model}")
        print(f"[INFO] Predict DB: {predict_db_path}")
        print(f"[INFO] Predict benchmark tables: {len(context['tables'])}")
        for table in context["tables"]:
            print(f"  - {table}")
        print(f"[INFO] Predict BenchmarkCache rows: {len(context['predict_df'])}")
        print("[INFO] predict expand config count per table:")
        for table, count in context["expand_count_by_table"].items():
            print(f"  - {table}: {count}")
        print(f"[INFO] expand shape count: {context['expand_counts'].shape[0]}")
        print(f"[INFO] expand rows: {len(context['expand_df'])}")
        if target_args.summary_md:
            print(f"[INFO] Summary md: {target_args.summary_md}")
        else:
            print(
                f"[WARN] No summary markdown found for model '{model}'; speedup plot data may be skipped."
            )

        xgboost_predict_start = time.perf_counter()
        predicted_df, predict_info = predict_with_global_ranker(
            context["expand_df"],
            config_cols=config_cols,
            args=target_args,
            train_state=train_state,
        )
        xgboost_predict_elapsed_s = time.perf_counter() - xgboost_predict_start
        xgboost_elapsed_s = (
            float(train_info["xgboost_fit_elapsed_s"])
            + float(predict_info["xgboost_predict_elapsed_s"])
        )
        print(
            "[INFO] xgboost full prediction: "
            f"predict_shapes={predict_info['predict_shape_count']}, "
            f"predict_configs={predict_info['predict_config_count']}"
        )
        print(
            "[TIME] xgboost prediction time: "
            f"{format_duration(predict_info['xgboost_predict_elapsed_s'])} "
            f"({predict_info['xgboost_predict_elapsed_s']:.3f}s)"
        )
        print(
            "[TIME] xgboost target output prep excluding file writes: "
            f"{format_duration(xgboost_predict_elapsed_s)} ({xgboost_predict_elapsed_s:.3f}s)"
        )

        write_prediction_target_outputs(
            target_args=target_args,
            out_dir=out_dir,
            predict_db_path=predict_db_path,
            predict_tables=context["tables"],
            predict_df=context["predict_df"],
            counts=context["counts"],
            expand_df=context["expand_df"],
            expand_count_by_table=context["expand_count_by_table"],
            expand_counts=context["expand_counts"],
            train_db_paths=train_db_paths,
            train_tables_by_db=train_tables_by_db,
            train_df=train_df,
            train_counts=train_counts,
            train_expand_df=train_expand_df,
            train_expand_count_by_table=train_expand_count_by_table,
            train_expand_counts=train_expand_counts,
            gemv_expand_count=gemv_expand_count,
            db_read_elapsed_s=db_read_elapsed_s,
            train_info=train_info,
            predict_info=predict_info,
            predicted_df=predicted_df,
            config_cols=config_cols,
            xgboost_elapsed_s=xgboost_elapsed_s,
        )


if __name__ == "__main__":
    main()
