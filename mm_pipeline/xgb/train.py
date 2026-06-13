#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRanker
except ImportError as exc:
    raise SystemExit("Please install xgboost first: pip install xgboost") from exc

from FlagTune.mm_pipeline.xgb.common import SHAPE_GROUP_COLS, parse_float
from FlagTune.mm_pipeline.xgb.features import build_global_feature_frame, build_rank_labels, ratio_count, resolve_train_ratios


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
