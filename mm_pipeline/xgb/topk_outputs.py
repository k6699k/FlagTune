#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from FlagTune.mm_pipeline.xgb.common import excel_merge_compare_value, normalize_config_compare_value, parse_int, remove_stale_file, series_has_single_display_value
from FlagTune.mm_pipeline.xgb.config_format import build_config_string_from_entry, build_config_string_from_prefix, build_force_config_entry


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
