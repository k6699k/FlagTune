#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from FlagTune.mm_pipeline.core.routing import route_mm_kernel_kind
from FlagTune.mm_pipeline.runtime.common import normalize_kernel_kind, normalize_shape_for_merge, parse_int


def load_entries(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return list(payload["items"])
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict) and "mm" in payload:
        configs = payload.get("mm", {}).get("configs", {})
        return [
            {"shape": shape_key, "shape_key": shape_key, "config": config}
            for shape_key, config in configs.items()
        ]
    raise RuntimeError(f"Unsupported input yaml format: {path}")


def load_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported payload format: {path}")
    return payload


def parse_shape(entry: Dict[str, Any]) -> Dict[str, Any]:
    summary_shape = entry.get("summary_shape_b_m_n_k")
    if summary_shape:
        parts = [part.strip() for part in str(summary_shape).split(",")]
        if len(parts) >= 4:
            m = parse_int(parts[1])
            n = parse_int(parts[2])
            k = parse_int(parts[3])
            if m is not None and n is not None and k is not None:
                return {"M": m, "N": n, "K": k, "dtype": entry.get("dtype")}

    dtype = entry.get("dtype")

    shape = entry.get("shape")
    if shape:
        parts = [part.strip() for part in str(shape).split(",")]
        if len(parts) >= 6:
            # mm_xgboost summary shape is B,M,N,K,count,dtype.
            m = parse_int(parts[1])
            n = parse_int(parts[2])
            k = parse_int(parts[3])
            dtype = parts[5]
        elif len(parts) == 4:
            # Compact matmul shape is B,M,N,K.
            m = parse_int(parts[1])
            n = parse_int(parts[2])
            k = parse_int(parts[3])
        else:
            m = parse_int(parts[0]) if len(parts) > 0 else None
            n = parse_int(parts[1]) if len(parts) > 1 else None
            k = parse_int(parts[2]) if len(parts) > 2 else None
        if m is not None and n is not None and k is not None:
            return {"M": m, "N": n, "K": k, "dtype": dtype}

    m = parse_int(entry.get("M"))
    n = parse_int(entry.get("N"))
    k = parse_int(entry.get("K"))
    if m is not None and n is not None and k is not None:
        return {"M": m, "N": n, "K": k, "dtype": dtype}

    shape_key = entry.get("shape_key")
    if shape_key:
        parts = [part.strip() for part in str(shape_key).split(",")]
        if len(parts) >= 3:
            m = parse_int(parts[0])
            n = parse_int(parts[1])
            k = parse_int(parts[2])
            if len(parts) >= 6:
                dtype = parts[5]
            if m is not None and n is not None and k is not None:
                return {"M": m, "N": n, "K": k, "dtype": dtype}

    raise RuntimeError(f"Cannot parse shape from entry: {entry}")


def split_shape_parts(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value]
    return [part.strip() for part in str(value).split(",")]


def format_shape_parts(parts: List[str]) -> str:
    formatted = []
    for part in parts[:4]:
        parsed = parse_int(part)
        formatted.append(str(parsed) if parsed is not None else part)
    return ", ".join(formatted)


def output_shape(entry: Dict[str, Any], normalized_shape: Optional[Dict[str, Any]] = None) -> str:
    summary_shape = split_shape_parts(entry.get("summary_shape_b_m_n_k"))
    if len(summary_shape) >= 4:
        return format_shape_parts(summary_shape)

    shape = split_shape_parts(entry.get("shape"))
    if len(shape) >= 4:
        return format_shape_parts(shape)

    parsed = normalized_shape or parse_shape(entry)
    return f"1, {int(parsed['M'])}, {int(parsed['N'])}, {int(parsed['K'])}"


def group_shape_keys(group: Dict[str, Any]) -> List[str]:
    entry = group["entries"][0]
    keys: List[str] = []
    for value in [
        entry.get("summary_shape_b_m_n_k"),
        entry.get("shape"),
        output_shape(entry),
        entry.get("shape_key"),
        group.get("shape_key"),
    ]:
        if value is None:
            continue
        key = normalize_shape_for_merge(value)
        if key not in keys:
            keys.append(key)
    return keys


def route_kernel_kind_from_shape(shape: Dict[str, Any]) -> str:
    return route_mm_kernel_kind(
        int(shape["M"]),
        int(shape["N"]),
        int(shape["K"]),
        shape.get("dtype") or "bfloat16",
    )


def route_kernel_kind_from_entry(entry: Dict[str, Any]) -> str:
    return route_kernel_kind_from_shape(parse_shape(entry))


def infer_kernel_kind(entry: Dict[str, Any]) -> str:
    kernel_kind = normalize_kernel_kind(entry.get("kernel_kind"))
    try:
        return route_kernel_kind_from_entry(entry)
    except Exception:
        if kernel_kind is not None:
            return kernel_kind
        raise


def payload_kernel_kind(payload: Dict[str, Any]) -> str:
    kernel_kind = normalize_kernel_kind(payload.get("kernel_kind"))

    entry = payload.get("shape_entry") or {}
    if isinstance(entry, dict):
        try:
            shape = payload.get("normalized_shape") or parse_shape(entry)
            return route_kernel_kind_from_shape(shape)
        except Exception:
            pass

    if kernel_kind is not None:
        return kernel_kind

    if isinstance(entry, dict):
        kernel_kind = normalize_kernel_kind(entry.get("kernel_kind"))
        if kernel_kind is not None:
            return kernel_kind

    configs = payload.get("configs") or []
    for config in configs:
        if isinstance(config, dict):
            kernel_kind = normalize_kernel_kind(config.get("kernel_kind"))
            if kernel_kind is not None:
                return kernel_kind

    raise RuntimeError("Cannot infer kernel_kind from payload")


def shape_group_key(entry: Dict[str, Any]) -> str:
    if entry.get("shape_key") is not None:
        return str(entry["shape_key"])
    if entry.get("shape") is not None:
        return str(entry["shape"])
    shape = parse_shape(entry)
    return f"{shape['M']},{shape['N']},{shape['K']},{shape.get('dtype') or ''}"


def config_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    config = entry.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"Missing config in entry: {entry}")
    record = {
        "shape_index": entry.get("shape_index"),
        "shape_key": entry.get("shape_key"),
        "shape": entry.get("shape"),
        "candidate_rank": entry.get("candidate_rank"),
        "kernel_kind": entry.get("kernel_kind"),
        "config": config,
    }
    for key, value in entry.items():
        if str(key).startswith("_metadata_"):
            continue
        if key not in record:
            record[key] = value
    return record


def group_entries(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for entry in entries:
        grouped.setdefault(shape_group_key(entry), []).append(entry)

    groups = []
    for key, group in grouped.items():
        try:
            group = sorted(
                group,
                key=lambda item: (
                    parse_int(item.get("candidate_rank")) is None,
                    parse_int(item.get("candidate_rank")) or 0,
                ),
            )
        except Exception:
            pass
        groups.append({"shape_key": key, "entries": group})
    return groups


def write_group_payload(group: Dict[str, Any], path: Path) -> None:
    payload = group_payload(group)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def group_payload(group: Dict[str, Any]) -> Dict[str, Any]:
    entries = group["entries"]
    normalized_shape = parse_shape(entries[0])
    kernel_kind = infer_kernel_kind(entries[0])
    payload = {
        "shape": output_shape(entries[0], normalized_shape),
        "shape_key": group["shape_key"],
        "kernel_kind": kernel_kind,
        "normalized_shape": normalized_shape,
        "shape_entry": entries[0],
        "configs": [config_entry(entry) for entry in entries],
    }
    return payload
