#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from FlagTune.mm_pipeline.reports.speedup_data import _build_plot_data_from_summary, _build_plot_data_from_topk_run_latency, _collect_topk_run_latency_by_shape, compact_plot_dataframe, merge_topk_plot_rows
from FlagTune.mm_pipeline.reports.summary import _map_run_latency_by_shape, load_summary_dataframe, merge_summary_latency_columns
from FlagTune.mm_pipeline.reports.common import RUN_LATENCY_COL


def draw_plots_from_out_dir(
    out_dir: Path,
    plot_format: Optional[str] = None,
    plot_dpi: Optional[int] = None,
    sort_by: Optional[str] = None,
    topk_annotate: Optional[int] = None,
    model: Optional[str] = None,
    op: str = "mm",
    summary_md: Optional[str] = None,
    summary_dtype: str = "bfloat16",
) -> List[Path]:
    _ = (plot_format, plot_dpi, sort_by, topk_annotate)
    summary_df = load_summary_dataframe(
        out_dir,
        model=model,
        op=op,
        summary_md=summary_md,
        dtype=summary_dtype,
    )

    plot_csv = out_dir / "speedup_plot_data.csv"
    if not plot_csv.exists():
        plot_df = _build_plot_data_from_topk_run_latency(out_dir)
        if plot_df is None:
            plot_df = _build_plot_data_from_summary(summary_df)
        if plot_df is None:
            raise RuntimeError(f"Missing plot data csv: {plot_csv}")
        plot_df = compact_plot_dataframe(
            merge_summary_latency_columns(plot_df, summary_df)
        )
        plot_df.to_csv(plot_csv, index=False)
        print(f"[INFO] created plot data: {plot_csv}")
        return [plot_csv]

    plot_df = merge_summary_latency_columns(pd.read_csv(plot_csv), summary_df)

    topk_plot_df = _build_plot_data_from_topk_run_latency(out_dir)
    if topk_plot_df is not None:
        topk_plot_df = merge_summary_latency_columns(topk_plot_df, summary_df)
        before_count = len(plot_df)
        plot_df = merge_topk_plot_rows(plot_df, topk_plot_df)
        topk_latency_by_shape = _collect_topk_run_latency_by_shape(out_dir)
        topk_run_latency = _map_run_latency_by_shape(plot_df, topk_latency_by_shape)
        mapped_count = int(topk_run_latency.notna().sum())
        print(
            f"[INFO] matched run latency to {mapped_count}/{len(plot_df)} "
            "speedup plot rows"
        )
        if len(plot_df) != before_count:
            print(
                f"[INFO] expanded speedup plot rows from {before_count} "
                f"to {len(plot_df)} using top-k latency outputs"
            )
        old_run_latency = (
            pd.to_numeric(plot_df[RUN_LATENCY_COL], errors="coerce")
            if RUN_LATENCY_COL in plot_df.columns
            else None
        )
        if old_run_latency is not None:
            plot_df[RUN_LATENCY_COL] = topk_run_latency.where(
                topk_run_latency.notna(), old_run_latency
            )
        else:
            plot_df[RUN_LATENCY_COL] = topk_run_latency

    plot_df = compact_plot_dataframe(plot_df)
    plot_df.to_csv(plot_csv, index=False)

    return [plot_csv]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update mm speedup plot data from run latency outputs.")
    parser.add_argument("--out-dir", default="./mm_xgb_outputs")
    parser.add_argument("--plot-format", default=None,
                        choices=["png", "pdf", "svg"])
    parser.add_argument("--plot-dpi", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--op", default="mm")
    parser.add_argument("--summary-md", default=None)
    parser.add_argument("--summary-dtype", default="bfloat16")
    parser.add_argument(
        "--plot-sort-by",
        default=None,
        choices=["oracle", "predicted", "gap", "shape_index"],
    )
    parser.add_argument("--plot-topk-annotate", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = draw_plots_from_out_dir(
        out_dir=Path(args.out_dir),
        plot_format=args.plot_format,
        plot_dpi=args.plot_dpi,
        sort_by=args.plot_sort_by,
        topk_annotate=args.plot_topk_annotate,
        model=args.model,
        op=args.op,
        summary_md=args.summary_md,
        summary_dtype=args.summary_dtype,
    )
    for path in written:
        print(f"[DONE] plot data: {path}")


if __name__ == "__main__":
    main()
