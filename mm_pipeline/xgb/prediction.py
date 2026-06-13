#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from FlagTune.mm_pipeline.xgb.common import CANONICAL_SHAPE_COLS, SHAPE_GROUP_COLS, parse_float, to_py_scalar
from FlagTune.mm_pipeline.xgb.features import build_global_feature_frame, build_rank_labels


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
