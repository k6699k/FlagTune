#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train/export an XGBoost ranker from MM BenchmarkCache data."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from FlagTune.mm_pipeline.xgb.common import DEFAULT_CONFIG_COLS, DEFAULT_GEMV_EXPAND_COUNT, DEFAULT_KERNEL_SUBSTRS, SHAPE_GROUP_COLS, TRAIN_MODE_RATIOS, format_duration
from FlagTune.mm_pipeline.xgb.features import add_common_columns, get_config_cols, key_count_table, resolve_train_ratios, select_expand_df
from FlagTune.mm_pipeline.xgb.export import export_xgboost_model_artifact
from FlagTune.mm_pipeline.xgb.target_outputs import write_prediction_target_outputs
from FlagTune.mm_pipeline.xgb.paths_db import db_search_dirs, output_dir_for_predict_target, read_benchmark_cache_db, resolve_db_paths, resolve_predict_targets, resolve_summary_md_for_model
from FlagTune.mm_pipeline.xgb.prediction import predict_with_global_ranker
from FlagTune.mm_pipeline.xgb.train import train_global_ranker


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
