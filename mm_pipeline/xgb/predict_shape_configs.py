#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Predict mm top-k configs from an exported XGBoost model and shape-config files."""

from __future__ import annotations

import argparse
from pathlib import Path

from FlagTune.mm_pipeline.core.shape_io import load_model_shapes, resolve_shape_config
from FlagTune.mm_pipeline.xgb.predict_topk import build_yaml_items, load_exported_model, write_model_yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
FLAGTUNE_DIR = Path(__file__).resolve().parents[2]
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
