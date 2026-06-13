#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from FlagTune.mm_pipeline.xgb.common import normalize_dtype_name


def parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def normalize_text(text: str) -> str:
    printable = "".join(ch if ch.isprintable() else " " for ch in text)
    return re.sub(r"\s+", " ", printable).strip()


def parse_shape_line(line: str) -> Optional[Tuple[str, List[int], int]]:
    line = line.strip()
    if "[shape info]:" not in line:
        return None

    op_part = line.split("[shape info]:", 1)[0].strip().rstrip(", ")
    op_name = normalize_text(op_part)
    if not op_name.startswith("flag_gems.ops.mm."):
        return None

    shape_part = line.split("[shape info]:", 1)[1]
    left = shape_part.find("[")
    right = shape_part.find("]", left + 1) if left != -1 else -1
    if left == -1 or right == -1:
        return None

    dims = []
    for token in shape_part[left + 1 : right].split(","):
        token = token.strip()
        dims.append(1 if token == "-" else int(token))

    count = 1
    match = re.search(r"\[count\]\s*:\s*(\d+)", line)
    if match:
        count = int(match.group(1))
    return op_name, dims, count


def normalize_mm_shape(op_name: str, dims: List[int]) -> Optional[Tuple[int, int, int, int]]:
    if op_name.endswith(".gemv_mm") or op_name.endswith(".general_mm"):
        if len(dims) == 3:
            m, k, n = dims
            return 1, int(m), int(n), int(k)
        if len(dims) == 4:
            b, m, n, k = dims
            return int(b), int(m), int(n), int(k)
    return None


def load_txt_shapes(path: Path) -> List[Dict[str, Any]]:
    by_shape: Dict[Tuple[int, int, int, int], int] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_shape_line(line)
            if parsed is None:
                continue
            op_name, dims, count = parsed
            shape = normalize_mm_shape(op_name, dims)
            if shape is None:
                continue
            by_shape[shape] = by_shape.get(shape, 0) + int(count)

    return [
        {"B": b, "M": m, "N": n, "K": k, "count": count}
        for (b, m, n, k), count in sorted(
            by_shape.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def load_yaml_shapes(path: Path, op: str) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported shape YAML format: {path}")

    candidate_keys = [op, "mm", "BlasBenchmark"]
    shape_items = None
    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, dict) and isinstance(value.get("shapes"), list):
            shape_items = value["shapes"]
            break
    if shape_items is None:
        raise RuntimeError(f"No mm shapes found in {path}")

    shapes = []
    for item in shape_items:
        dims = [int(value) for value in item]
        if len(dims) == 4:
            b, m, n, k = dims
        elif len(dims) == 3:
            b, m, n, k = 1, dims[0], dims[1], dims[2]
        else:
            continue
        shapes.append({"B": b, "M": m, "N": n, "K": k, "count": 1})
    return shapes


def resolve_shape_config(model: str, shape_config_dir: Path, op: str) -> Path:
    candidates = [
        shape_config_dir / f"{model}.txt",
        shape_config_dir / f"{model}_{op}.yaml",
        shape_config_dir / f"{model}.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No shape config found for model '{model}'. Tried:\n  - {tried}")


def load_model_shapes(model: str, shape_config_dir: Path, op: str) -> List[Dict[str, Any]]:
    path = resolve_shape_config(model, shape_config_dir, op)
    if path.suffix == ".txt":
        shapes = load_txt_shapes(path)
    else:
        shapes = load_yaml_shapes(path, op)
    if not shapes:
        raise RuntimeError(f"No mm shapes parsed from {path}")
    return shapes


def shape_key_for(m: int, n: int, k: int, dtype: str) -> str:
    return f"{m},{n},{k},{k},{n},{normalize_dtype_name(dtype)}"


def output_shape_for(b: int, m: int, n: int, k: int) -> str:
    return f"{b}, {m}, {n}, {k}"
