#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from FlagTune.mm_pipeline.runtime.common import parse_int
from FlagTune.mm_pipeline.runtime.configs import CONFIG_COMPARE_FIELDS, flatten_config_record, normalize_config_value, parse_bool


def output_paths(args: argparse.Namespace, input_path: Path) -> tuple[Path, Path]:
    out_csv = (
        Path(args.output_csv)
        if args.output_csv
        else input_path.with_name(f"{input_path.stem}_libtuner_latency.csv")
    )
    out_yaml = (
        Path(args.output_yaml)
        if args.output_yaml
        else input_path.with_name(f"{input_path.stem}_libtuner_latency.yaml")
    )
    return out_csv, out_yaml


def ga_selected_output_paths(
    args: argparse.Namespace, input_path: Path
) -> Tuple[Path, Path]:
    out_csv = (
        Path(args.ga_selected_output_csv)
        if args.ga_selected_output_csv
        else input_path.with_name(f"{input_path.stem}_ga_selected_configs.csv")
    )
    out_yaml = (
        Path(args.ga_selected_output_yaml)
        if args.ga_selected_output_yaml
        else input_path.with_name(f"{input_path.stem}_ga_selected_configs.yaml")
    )
    return out_csv, out_yaml


def write_results(rows: List[Dict[str, Any]], out_csv: Path, out_yaml: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)

    preferred_keys = [
        "shape",
        "shape_key",
        "latency_ms",
        "topk_runtime_best_config",
        "topk_runtime_best_candidate_rank",
        "topk_runtime_best_ga_source",
        "topk_runtime_best_ga_generation",
        "benchmark_best_config",
        "benchmark_best_available",
        "topk_runtime_best_same_as_benchmark_best",
        "benchmark_best_in_topk_for_shape",
    ]
    if any(row.get("success") is False or row.get("error") for row in rows):
        preferred_keys.extend(["success", "error", "kernel_kind", "config_count"])
    keys = [key for key in preferred_keys if any(key in row for row in rows)]
    output_rows = [{key: row.get(key) for key in keys} for row in rows]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(output_rows)
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"items": output_rows}, f, sort_keys=False)


def collect_ga_selected_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in rows:
        items = row.get("_ga_selected_configs")
        if isinstance(items, list):
            selected.extend(item for item in items if isinstance(item, dict))
    return selected


def write_ga_selected_results(
    rows: List[Dict[str, Any]], out_csv: Path, out_yaml: Path
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    preferred_keys = [
        "shape",
        "shape_key",
        "candidate_rank",
        "ga_generation",
        "ga_source",
        "ga_rank_by_latency",
        "ga_is_best",
        "ga_is_benchmark_best",
        "latency_ms",
        "config",
        "success",
        "error",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_M",
        "SPLIT_K",
        "num_warps",
        "num_ctas",
        "num_stages",
    ]
    keys = [key for key in preferred_keys if any(key in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in rows)
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"items": rows}, f, sort_keys=False)


def config_string(config: Any) -> str:
    flat = flatten_config_record(config)
    parts = []
    for field in CONFIG_COMPARE_FIELDS:
        value = flat.get(field)
        if value is None:
            continue
        value = normalize_config_value(value)
        parts.append(f"{field}={value}")
    return ", ".join(parts)


def write_benchmark_hit_results(
    groups: List[Dict[str, Any]],
    out_csv: Path,
    out_xlsx: Path,
) -> None:
    rows: List[Dict[str, Any]] = []
    for group in groups:
        entries = list(group.get("entries", []))
        if not entries:
            continue
        entries = sorted(
            entries,
            key=lambda item: parse_int(item.get("candidate_rank")) or 0,
        )
        first = entries[0]
        hit_entries = [
            entry
            for entry in entries
            if parse_bool(entry.get("matches_benchmark_best_config")) is True
        ]
        hit_ranks = [
            str(parse_int(entry.get("candidate_rank")))
            for entry in hit_entries
            if parse_int(entry.get("candidate_rank")) is not None
        ]
        benchmark_best_config = first.get("benchmark_best_config")
        rows.append(
            {
                "kernel_kind": first.get("kernel_kind"),
                "shape": first.get("shape"),
                "shape_key": first.get("shape_key") or group.get("shape_key"),
                "M": first.get("M"),
                "N": first.get("N"),
                "K": first.get("K"),
                "dtype": first.get("dtype"),
                "requested_top_k": len(entries),
                "exported_topk_count": len(entries),
                "benchmark_best_in_topk": bool(hit_entries),
                "benchmark_best_hit_rank": ",".join(hit_ranks),
                "benchmark_best_measured_p50": first.get("benchmark_best_measured_p50"),
                "benchmark_best_config_order": first.get("oracle_best_config_order"),
                "benchmark_best_config": config_string(benchmark_best_config),
                "rank1_predicted_config": config_string(first.get("config")),
                "rank1_predicted_xgb_rank_score": first.get(
                    "predicted_best_xgb_rank_score"
                ),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        frame = pd.DataFrame(rows)
        frame.to_csv(out_csv, index=False)
        frame.to_excel(out_xlsx, index=False)
    except Exception:
        keys = list(rows[0].keys()) if rows else [
            "shape",
            "shape_key",
            "benchmark_best_in_topk",
        ]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in keys} for row in rows)
        if out_xlsx.exists():
            out_xlsx.unlink()


def write_parent_outputs(
    rows: List[Dict[str, Any]],
    out_csv: Path,
    out_yaml: Path,
    args: argparse.Namespace,
    input_path: Path,
) -> None:
    write_results(rows, out_csv, out_yaml)
    if args.ga_generations > 0:
        selected_csv, selected_yaml = ga_selected_output_paths(args, input_path)
        write_ga_selected_results(
            collect_ga_selected_rows(rows), selected_csv, selected_yaml
        )


def row_sort_key(row: Dict[str, Any]) -> tuple[int, int]:
    index = parse_int(row.get("shape_group_index"))
    return (0 if index is not None else 1, index if index is not None else 0)


def sorted_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=row_sort_key)
