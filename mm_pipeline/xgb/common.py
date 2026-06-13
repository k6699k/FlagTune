#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

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


def normalize_dtype_name(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("torch."):
        text = text.split(".", 1)[1]
    return text


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


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
