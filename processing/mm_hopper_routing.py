#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shape-only routing helpers for Hopper mm FlagTune pipelines."""

from __future__ import annotations

from typing import Any


def normalize_dtype_name(dtype: Any) -> str:
    name = str(dtype or "bfloat16").strip().lower()
    if name.startswith("torch."):
        name = name.split(".", 1)[1]
    aliases = {
        "half": "float16",
        "fp16": "float16",
        "bf16": "bfloat16",
        "float": "float32",
        "fp32": "float32",
    }
    return aliases.get(name, name)


def is_tma_compatible_dtype(dtype: Any, n: int, k: int) -> bool:
    name = normalize_dtype_name(dtype)
    if name in ("float16", "bfloat16"):
        return n % 8 == 0 and k % 8 == 0
    if name == "float32":
        return n % 4 == 0 and k % 4 == 0
    return False


def route_mm_kernel_kind(m: int, n: int, k: int, dtype: Any) -> str:
    """Mirror the supported Hopper ``mm.py`` branches used by FlagTune."""
    if int(n) == 1:
        return "gemv"
    if int(m) < 2048 and int(n) < 2048 and int(k) >= 4096:
        return "mm_splitk"
    if is_tma_compatible_dtype(dtype, int(n), int(k)):
        return "mm_general_tma"
    return "mm_general"
