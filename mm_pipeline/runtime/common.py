#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
TMA_OVERRIDE_ENV = "FLAGGEMS_MM_TMA_CONFIG_OVERRIDE"
GEMV_OVERRIDE_ENV = "FLAGGEMS_MM_GEMV_CONFIG_OVERRIDE"
DEBUG_ENV = "FLAGGEMS_DEBUG_LIBTUNER_CONFIGS"


def parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(str(value).strip())
    except Exception:
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def normalize_kernel_kind(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in ("gemv", "mv", "matrix_vector"):
        return "gemv"
    if text in ("mm_splitk", "splitk", "split_k"):
        return "mm_splitk"
    if text in ("mm_general", "general", "non_tma", "mm_general_non_tma"):
        return "mm_general"
    if text in ("mm", "matmul", "tma", "mm_general_tma", "general_tma"):
        return "mm_general_tma"
    return text


def override_env_for_kernel_kind(kernel_kind: str) -> str:
    if kernel_kind == "gemv":
        return GEMV_OVERRIDE_ENV
    if kernel_kind == "mm_general_tma":
        return TMA_OVERRIDE_ENV
    if kernel_kind in ("mm_general", "mm_splitk"):
        return "LIBTUNER_RUNTIME"
    raise RuntimeError(
        f"Unsupported kernel_kind for LibTuner override: {kernel_kind}")


def normalize_existing_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists() or "\\" not in path_text:
        return path

    slash_path = Path(path_text.replace("\\", "/"))
    if slash_path.exists():
        return slash_path
    return path


def normalize_shape_for_merge(value: Any) -> str:
    return ",".join(part.strip() for part in str(value).split(","))
