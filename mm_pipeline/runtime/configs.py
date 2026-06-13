#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional

from FlagTune.mm_pipeline.runtime.common import parse_float


CONFIG_COMPARE_FIELDS = (
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "GROUP_M",
    "SPLIT_K",
    "num_warps",
    "num_stages",
    "num_ctas",
)


def triton_config_record(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    record = dict(getattr(config, "kwargs", {}))
    for attr in ("num_warps", "num_stages", "num_ctas"):
        if hasattr(config, attr):
            record[attr] = getattr(config, attr)
    return record


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def flatten_config_record(config: Any) -> Dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    meta = config.get("META")
    if not isinstance(meta, dict):
        meta = {}
    record: Dict[str, Any] = {}
    for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "SPLIT_K"):
        value = config.get(key, meta.get(key))
        if value is not None:
            record[key] = value
    for key in ("num_warps", "num_stages", "num_ctas"):
        value = config.get(key)
        if value is not None:
            record[key] = value
    return record


def normalize_config_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        parsed = float(str(value))
    except Exception:
        return value
    if parsed.is_integer():
        return int(parsed)
    return parsed


def configs_match(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    compared = False
    for field in CONFIG_COMPARE_FIELDS:
        left_value = normalize_config_value(left.get(field))
        right_value = normalize_config_value(right.get(field))
        if left_value is None or right_value is None:
            continue
        compared = True
        if left_value != right_value:
            return False
    return compared


def find_matching_config_entry(
    best_record: Dict[str, Any], config_entries: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    for entry in config_entries:
        if configs_match(best_record, flatten_config_record(entry.get("config"))):
            return entry
    return None


def benchmark_best_config_record(
    payload: Dict[str, Any], config_entries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    sources: List[Any] = [payload.get("shape_entry")] + list(config_entries)
    for source in sources:
        if not isinstance(source, dict):
            continue
        config = source.get("benchmark_best_config")
        if isinstance(config, dict):
            return flatten_config_record(config)

    for source in sources:
        if not isinstance(source, dict):
            continue
        if parse_bool(source.get("matches_benchmark_best_config")) is True:
            return flatten_config_record(source.get("config"))
    return {}
