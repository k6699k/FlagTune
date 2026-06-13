#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from FlagTune.mm_pipeline.xgb.common import to_py_scalar


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
