#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from FlagTune.mm_pipeline.reports.speedup_data import write_speedup_plot_data
from FlagTune.mm_pipeline.xgb.common import DISPLAY_SHAPE_GROUP_COLS, SHAPE_GROUP_COLS
from FlagTune.mm_pipeline.xgb.predicted_configs import write_predicted_shape_config_outputs
from FlagTune.mm_pipeline.xgb.prediction import summarize_predicted_shape, summarize_topk_predicted_shape
from FlagTune.mm_pipeline.xgb.summary import merge_summary_default_latency, parse_summary_markdown
from FlagTune.mm_pipeline.xgb.topk_outputs import annotate_topk_benchmark_best_hits, build_topk_benchmark_best_hit_dataframe, write_topk_outputs


def write_run_summary_yaml(out_path: Path, summary: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)


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
