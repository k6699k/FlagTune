#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Measure predicted top-k mm configs through LibTuner config injection.

The input is a YAML file produced by ``mm_xgboost_from_benchmark_cache.py`` such
as ``predicted_shape_configs_topK.yaml``. For each shape, this script writes the
shape's top-k configs to a temporary YAML file. Sequential mode can still launch
one long-lived worker process; parallel mode launches one long-lived worker per
GPU. Each worker injects the shape's configs into the loaded LibTuner object
before benchmarking that shape.

Latency is measured by calling the backend LibTuner policy with direct
``LibTuner._bench`` kernel launches.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

try:
    from . import mm_ga_libtuner_runtime
    from .mm_hopper_routing import route_mm_kernel_kind
except ImportError:
    import mm_ga_libtuner_runtime
    from mm_hopper_routing import route_mm_kernel_kind


REPO_ROOT = Path(__file__).resolve().parents[2]
TMA_OVERRIDE_ENV = "FLAGGEMS_MM_TMA_CONFIG_OVERRIDE"
GEMV_OVERRIDE_ENV = "FLAGGEMS_MM_GEMV_CONFIG_OVERRIDE"
DEBUG_ENV = "FLAGGEMS_DEBUG_LIBTUNER_CONFIGS"
PRINT_RUNTIME_CONFIGS_ENV = "FLAGGEMS_PRINT_RUNTIME_CONFIGS"
PRINT_RUNTIME_CONFIGS_FILE_ENV = "FLAGGEMS_PRINT_RUNTIME_CONFIGS_FILE"
DEFAULT_RUNTIME_CONFIGS_LOG_PATH = Path(
    "/home/secure/autotune/FlagGems/FlagTune/scripts/selected_runtime_configs.log"
)


def write_runtime_config_log(message: str) -> None:
    log_path = Path(
        os.getenv(
            PRINT_RUNTIME_CONFIGS_FILE_ENV,
            str(DEFAULT_RUNTIME_CONFIGS_LOG_PATH),
        )
    )
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message)
            f.write("\n")
    except Exception as exc:
        print(
            "[FlagTune][run_topk] failed to write runtime config log "
            f"to {log_path}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run each shape's top-k predicted mm configs through LibTuner."
    )
    parser.add_argument("--input", default=None,
                        help="predicted top-k config YAML")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-yaml", default=None)
    parser.add_argument("--ga-selected-output-csv", default=None)
    parser.add_argument("--ga-selected-output-yaml", default=None)
    parser.add_argument("--benchmark-hit-output-csv", default=None)
    parser.add_argument("--benchmark-hit-output-xlsx", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None,
                        help="Override dtype, e.g. bfloat16")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=["kernel", "operator", "wrapper"],
        default="kernel",
        help="Benchmark.get_latency mode",
    )
    parser.add_argument("--start-shape", type=int, default=0)
    parser.add_argument("--limit-shapes", type=int, default=None)
    parser.add_argument(
        "--top-cost-shapes",
        type=int,
        default=None,
        help=(
            "After start/limit filtering, keep only the N shapes with the "
            "largest scheduling cost. Report expand Gems latency is used when available."
        ),
    )
    parser.add_argument("--max-configs-per-shape", type=int, default=None)
    parser.add_argument(
        "--balance-latency-csv",
        default=None,
        help=(
            "Optional CSV with report latency by shape. If omitted, the script "
            "auto-detects speedup_plot_data.csv or parsed_summary_md.csv next to --input."
        ),
    )
    parser.add_argument(
        "--balance-latency-col",
        default=None,
        help=(
            "Latency column to use from --balance-latency-csv. Defaults to "
            "summary_expand_gems_latency_ms or benchmark_best_measured_p50."
        ),
    )
    parser.add_argument(
        "--no-report-latency-balance",
        action="store_true",
        help="Disable report-latency-based scheduling and fall back to M*N*K*config_count.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ga-generations", type=int, default=0)
    parser.add_argument("--ga-population-size", type=int, default=10)
    parser.add_argument("--ga-elite-size", type=int, default=4)
    parser.add_argument("--ga-offspring-per-generation", type=int, default=10)
    parser.add_argument("--ga-mutation-rate", type=float, default=0.35)
    parser.add_argument("--ga-random-rate", type=float, default=0.15)
    parser.add_argument(
        "--ga-max-evaluations-per-shape",
        type=int,
        default=0,
        help="0 means no cap beyond initial top-k plus generated GA offspring.",
    )
    parser.add_argument("--debug-libtuner", action="store_true")
    parser.add_argument("--show-worker-output", action="store_true")
    parser.add_argument(
        "--no-print-runtime-configs",
        action="store_true",
        help="Do not print LibTuner runtime configs before launching them.",
    )
    parser.add_argument("--keep-override-files", action="store_true")
    parser.add_argument("--override-dir", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Run shapes in parallel across N visible devices, e.g. --parallel 8.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated device ids for --parallel. Defaults to 0..N-1.",
    )
    parser.add_argument(
        "--visible-devices-env",
        default=None,
        help="Override visible device env var, e.g. CUDA_VISIBLE_DEVICES.",
    )

    # Internal worker-process mode. Keep this hidden from normal help output.
    parser.add_argument("--_worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--_manifest", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


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


def normalize_existing_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists() or "\\" not in path_text:
        return path

    slash_path = Path(path_text.replace("\\", "/"))
    if slash_path.exists():
        return slash_path
    return path


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


def normalize_shape_for_merge(value: Any) -> str:
    return ",".join(part.strip() for part in str(value).split(","))


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


def resolve_balance_latency_csv(input_path: Path, args: argparse.Namespace) -> Optional[Path]:
    if args.no_report_latency_balance:
        return None
    if args.balance_latency_csv:
        path = normalize_existing_path(args.balance_latency_csv)
        if not path.exists():
            raise FileNotFoundError(f"--balance-latency-csv not found: {path}")
        return path
    for candidate in [
        input_path.parent / "speedup_plot_data.csv",
        input_path.parent / "parsed_summary_md.csv",
    ]:
        if candidate.exists():
            return candidate
    return None


def load_balance_latency_by_shape(
    input_path: Path, args: argparse.Namespace
) -> tuple[Dict[str, float], Optional[Path], Optional[str]]:
    csv_path = resolve_balance_latency_csv(input_path, args)
    if csv_path is None:
        return {}, None, None

    latency_by_shape: Dict[str, float] = {}
    preferred_latency_cols = (
        [args.balance_latency_col]
        if args.balance_latency_col
        else [
            "summary_expand_gems_latency_ms",
            "benchmark_best_measured_p50",
            "expand_gems_latency_ms",
        ]
    )
    shape_cols = ["summary_shape_b_m_n_k", "shape", "shape_key"]
    used_latency_col: Optional[str] = None

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        latency_col = next(
            (col for col in preferred_latency_cols if col and col in fieldnames),
            None,
        )
        if latency_col is None:
            raise RuntimeError(
                f"No report latency column found in {csv_path}; tried {preferred_latency_cols}"
            )
        used_latency_col = latency_col

        available_shape_cols = [col for col in shape_cols if col in fieldnames]
        if not available_shape_cols:
            raise RuntimeError(
                f"No shape column found in {csv_path}; tried {shape_cols}"
            )

        for row in reader:
            latency = parse_float(row.get(latency_col))
            if latency is None:
                continue
            for shape_col in available_shape_cols:
                value = row.get(shape_col)
                if not value:
                    continue
                key = normalize_shape_for_merge(value)
                old_latency = latency_by_shape.get(key)
                if old_latency is None or latency > old_latency:
                    latency_by_shape[key] = latency

    return latency_by_shape, csv_path, used_latency_col


def report_balance_latency_ms(
    group: Dict[str, Any], latency_by_shape: Dict[str, float]
) -> Optional[float]:
    for key in group_shape_keys(group):
        latency = latency_by_shape.get(key)
        if latency is not None:
            return latency

    entry = group["entries"][0]
    for col in [
        "summary_expand_gems_latency_ms",
        "benchmark_best_measured_p50",
        "report_expand_gems_latency_ms",
    ]:
        latency = parse_float(entry.get(col))
        if latency is not None:
            return latency
    return None


def estimate_group_cost(group: Dict[str, Any]) -> float:
    try:
        shape = parse_shape(group["entries"][0])
        m, n, k = int(shape["M"]), int(shape["N"]), int(shape["K"])
        shape_cost = (
            math.log2(max(2, m))
            * math.log2(max(2, n))
            * math.log2(max(2, k))
        )
        return shape_cost * max(1, len(group["entries"]))
    except Exception:
        return float(max(1, len(group.get("entries", []))))


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


def select_groups(
    groups: List[Dict[str, Any]],
    args: argparse.Namespace,
    latency_by_shape: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    if args.start_shape < 0:
        raise ValueError("--start-shape must be >= 0")
    if args.top_cost_shapes is not None and args.top_cost_shapes <= 0:
        raise ValueError("--top-cost-shapes must be > 0")

    indexed = list(enumerate(groups))
    selected_pairs = indexed[args.start_shape:]
    if args.limit_shapes is not None:
        selected_pairs = selected_pairs[: args.limit_shapes]

    selected = []
    for original_index, group in selected_pairs:
        group = dict(group)
        if args.max_configs_per_shape is not None:
            group["entries"] = group["entries"][: args.max_configs_per_shape]
        group["shape_group_index"] = original_index
        balance_latency_ms = report_balance_latency_ms(
            group, latency_by_shape or {}
        )
        if balance_latency_ms is not None:
            group["balance_latency_ms"] = balance_latency_ms
            group["estimated_cost"] = balance_latency_ms
            group["cost_source"] = "report_expand_gems_latency_ms"
        else:
            group["estimated_cost"] = float(estimate_group_cost(group))
            group["cost_source"] = "shape_log_mnk_config_count"
        selected.append(group)

    if args.top_cost_shapes is not None:
        selected = sorted(
            selected,
            key=lambda group: group["estimated_cost"],
            reverse=True,
        )[: args.top_cost_shapes]
        selected.sort(key=lambda group: group["shape_group_index"])

    return selected


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


def device_type(device_name: str) -> str:
    return str(device_name).split(":", 1)[0]


def visible_devices_env_name(args: argparse.Namespace) -> Optional[str]:
    if args.visible_devices_env:
        return args.visible_devices_env
    return {
        "cuda": "CUDA_VISIBLE_DEVICES",
        "musa": "MUSA_VISIBLE_DEVICES",
    }.get(device_type(args.device))


def worker_device_arg(args: argparse.Namespace, gpu_id: Optional[str] = None) -> str:
    if gpu_id is None:
        return args.device
    dev_type = device_type(args.device)
    if dev_type in ("cuda", "musa"):
        return dev_type
    return args.device


def parse_gpu_ids(args: argparse.Namespace) -> List[str]:
    if args.parallel <= 0:
        return []
    if args.gpus:
        gpu_ids = [part.strip() for part in str(
            args.gpus).split(",") if part.strip()]
    else:
        gpu_ids = [str(idx) for idx in range(args.parallel)]
    if not gpu_ids:
        raise RuntimeError("--parallel requires at least one GPU id")
    if len(gpu_ids) < args.parallel:
        raise RuntimeError(
            f"--parallel {args.parallel} requires at least {args.parallel} GPU ids, "
            f"got {len(gpu_ids)} from --gpus"
        )
    return gpu_ids[: args.parallel]


def worker_env(args: argparse.Namespace, gpu_id: Optional[str] = None) -> Dict[str, str]:
    env = os.environ.copy()
    for override_env in (TMA_OVERRIDE_ENV, GEMV_OVERRIDE_ENV):
        env.pop(override_env, None)
    env.setdefault("USE_FLAGTUNE", "1")
    if args.debug_libtuner:
        env[DEBUG_ENV] = "1"
        env.setdefault("TRITON_PRINT_AUTOTUNING", "1")
    if args.no_print_runtime_configs:
        env.pop(PRINT_RUNTIME_CONFIGS_ENV, None)
    else:
        env[PRINT_RUNTIME_CONFIGS_ENV] = "1"

    if gpu_id is not None:
        visible_env = visible_devices_env_name(args)
        if visible_env is None:
            raise RuntimeError(
                f"--parallel is not supported for device '{args.device}' unless "
                "--visible-devices-env is provided"
            )
        env[visible_env] = str(gpu_id)

    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not old_pythonpath
        else str(REPO_ROOT) + os.pathsep + old_pythonpath
    )
    return env


def worker_command(
    manifest_path: Path, args: argparse.Namespace, gpu_id: Optional[str] = None
) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--_manifest",
        str(manifest_path),
        "--device",
        worker_device_arg(args, gpu_id),
        "--warmup",
        str(args.warmup),
        "--rep",
        str(args.rep),
        "--mode",
        args.mode,
        "--seed",
        str(args.seed),
        "--ga-generations",
        str(args.ga_generations),
        "--ga-population-size",
        str(args.ga_population_size),
        "--ga-elite-size",
        str(args.ga_elite_size),
        "--ga-offspring-per-generation",
        str(args.ga_offspring_per_generation),
        "--ga-mutation-rate",
        str(args.ga_mutation_rate),
        "--ga-random-rate",
        str(args.ga_random_rate),
        "--ga-max-evaluations-per-shape",
        str(args.ga_max_evaluations_per_shape),
    ]
    if args.dtype:
        cmd.extend(["--dtype", args.dtype])
    if args.keep_override_files:
        cmd.append("--keep-override-files")
    if args.fail_fast:
        cmd.append("--fail-fast")
    return cmd


def write_worker_manifest(
    jobs: List[Dict[str, Any]], manifest_path: Path
) -> None:
    manifest = {
        "jobs": [
            {
                "payload_path": str(job["payload_path"]),
                "shape_group_index": job["shape_group_index"],
                "shape": job["shape"],
                "shape_key": job["group"]["shape_key"],
                "kernel_kind": job["kernel_kind"],
                "config_count": job["config_count"],
                "balance_latency_ms": job.get("balance_latency_ms"),
                "estimated_cost": job["cost"],
                "cost_source": job.get("cost_source"),
            }
            for job in jobs
        ]
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def run_worker_process(
    jobs: List[Dict[str, Any]],
    manifest_path: Path,
    args: argparse.Namespace,
    gpu_id: Optional[str],
) -> List[Dict[str, Any]]:
    write_worker_manifest(jobs, manifest_path)
    cmd = worker_command(manifest_path, args, gpu_id)
    if args.ga_generations > 0:
        print(
            f"[parallel] launching worker gpu={gpu_id} "
            f"jobs={len(jobs)} ga_generations={args.ga_generations} "
            f"ga_offspring_per_generation={args.ga_offspring_per_generation}",
            flush=True,
        )
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=worker_env(args, gpu_id),
        text=True,
        capture_output=True,
    )
    if args.show_worker_output or proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
    elif not args.no_print_runtime_configs and proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker process failed with exit code {proc.returncode}")
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        raise RuntimeError(
            f"Worker did not emit JSON result: {proc.stdout!r}") from exc
    rows = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError(
            f"Worker JSON result does not contain items: {payload!r}")
    for row in rows:
        if isinstance(row, dict) and row.get("gpu_id") is None:
            row["gpu_id"] = gpu_id
    return rows


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


def build_jobs(groups: List[Dict[str, Any]], override_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    jobs = []
    for fallback_index, group in enumerate(groups, start=args.start_shape):
        index = parse_int(group.get("shape_group_index"))
        if index is None:
            index = fallback_index
        payload_path = override_dir / f"shape_{index:06d}_topk_configs.yaml"
        kernel_kind = infer_kernel_kind(group["entries"][0])
        display_shape = output_shape(group["entries"][0])
        estimated_cost = parse_float(group.get("estimated_cost"))
        if estimated_cost is None:
            estimated_cost = float(estimate_group_cost(group))
        write_group_payload(group, payload_path)
        jobs.append(
            {
                "shape_group_index": index,
                "group": group,
                "payload_path": payload_path,
                "kernel_kind": kernel_kind,
                "shape": display_shape,
                "config_count": len(group["entries"]),
                "cost": estimated_cost,
                "balance_latency_ms": group.get("balance_latency_ms"),
                "cost_source": group.get("cost_source"),
            }
        )
    return jobs


def split_jobs_evenly(jobs: List[Dict[str, Any]], num_buckets: int) -> List[List[Dict[str, Any]]]:
    if num_buckets <= 0:
        return []
    sorted_jobs = sorted(jobs, key=lambda job: job["cost"], reverse=True)
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(num_buckets)]
    bucket_costs = [0.0] * num_buckets

    for job in sorted_jobs:
        target = min(
            range(num_buckets),
            key=lambda idx: (len(buckets[idx]), bucket_costs[idx], idx),
        )
        buckets[target].append(job)
        bucket_costs[target] += job["cost"]

    for bucket in buckets:
        bucket.sort(key=lambda job: job["shape_group_index"])
    return [bucket for bucket in buckets if bucket]


def make_error_row(job: Dict[str, Any], exc: Exception, gpu_id: Optional[str] = None) -> Dict[str, Any]:
    row = {
        "shape": job["shape"],
        "shape_key": job["group"]["shape_key"],
        "kernel_kind": job["kernel_kind"],
        "gpu_id": gpu_id,
        "shape_group_index": job["shape_group_index"],
        "balance_latency_ms": job.get("balance_latency_ms"),
        "estimated_cost": job["cost"],
        "cost_source": job.get("cost_source"),
        "config_count": job["config_count"],
        "success": False,
        "error": repr(exc),
    }
    return row


def run_parent(args: argparse.Namespace) -> None:
    if not args.input:
        raise RuntimeError("--input is required")
    input_path = normalize_existing_path(args.input)
    all_groups = group_entries(load_entries(input_path))
    latency_by_shape, latency_csv, latency_col = load_balance_latency_by_shape(
        input_path, args
    )
    if latency_csv is not None:
        print(
            f"Loaded report balance latency for {len(latency_by_shape)} shapes "
            f"from {latency_csv} column={latency_col}"
        )
    else:
        print(
            "No report balance latency CSV found; will use latency embedded in input YAML when available"
        )
    groups = select_groups(all_groups, args, latency_by_shape)
    report_cost_groups = sum(
        1 for group in groups if group.get("cost_source") == "report_expand_gems_latency_ms"
    )
    if report_cost_groups:
        print(
            f"Using report expand Gems latency cost for {report_cost_groups}/{len(groups)} shapes"
        )
    else:
        print(
            "No report expand Gems latency matched; using "
            "log2(M)*log2(N)*log2(K)*config_count cost"
        )
    if args.top_cost_shapes is not None:
        print(
            f"Selected {len(groups)} highest-cost shapes from {len(all_groups)} groups "
            f"(top_cost_shapes={args.top_cost_shapes})"
        )
    if args.ga_generations > 0:
        print(
            f"GA enabled in worker search mode: generations={args.ga_generations} "
            f"offspring_per_generation={args.ga_offspring_per_generation}"
        )
    out_csv, out_yaml = output_paths(args, input_path)
    if args.benchmark_hit_output_csv or args.benchmark_hit_output_xlsx:
        hit_csv = (
            Path(args.benchmark_hit_output_csv)
            if args.benchmark_hit_output_csv
            else out_csv.with_name("benchmark_best_hit_by_shape.csv")
        )
        hit_xlsx = (
            Path(args.benchmark_hit_output_xlsx)
            if args.benchmark_hit_output_xlsx
            else hit_csv.with_suffix(".xlsx")
        )
        write_benchmark_hit_results(groups, hit_csv, hit_xlsx)
        print(f"Wrote benchmark-best hit csv to {hit_csv}")
        if hit_xlsx.exists():
            print(f"Wrote benchmark-best hit xlsx to {hit_xlsx}")
    gpu_ids = parse_gpu_ids(args)
    if gpu_ids and visible_devices_env_name(args) is None:
        raise RuntimeError(
            f"--parallel is not supported for device '{args.device}' unless "
            "--visible-devices-env is provided"
        )

    if args.override_dir:
        override_dir = Path(args.override_dir)
        override_dir.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="mm_topk_libtuner_")
        override_dir = Path(temp_ctx.name)

    rows: List[Dict[str, Any]] = []
    try:
        jobs = build_jobs(groups, override_dir, args)
        if not jobs:
            write_parent_outputs(rows, out_csv, out_yaml, args, input_path)
            print(f"Wrote {len(rows)} rows to {out_csv}")
            print(f"Wrote YAML summary to {out_yaml}")
            return

        if gpu_ids:
            buckets = split_jobs_evenly(jobs, len(gpu_ids))
            total_cost = sum(job["cost"] for job in jobs)
            print(
                f"Running {len(jobs)} shapes across {len(buckets)} GPU worker processes "
                f"({visible_devices_env_name(args) or 'visible devices'}={','.join(gpu_ids)}), "
                f"estimated_total_cost={total_cost}"
            )
            bucket_summary = ", ".join(
                f"{gpu_ids[idx]}:{len(bucket)}"
                for idx, bucket in enumerate(buckets)
            )
            print(f"[parallel] bucket job counts: {bucket_summary}")
            future_to_assignment: Dict[Any,
                                       tuple[str, List[Dict[str, Any]]]] = {}
            completed_count = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(buckets)) as ex:
                for gpu_id, bucket in zip(gpu_ids, buckets):
                    safe_gpu_id = str(gpu_id).replace(
                        ":", "_").replace("/", "_")
                    manifest_path = override_dir / \
                        f"worker_{safe_gpu_id}_manifest.yaml"
                    future = ex.submit(
                        run_worker_process, bucket, manifest_path, args, gpu_id
                    )
                    future_to_assignment[future] = (gpu_id, bucket)

                for future in concurrent.futures.as_completed(future_to_assignment):
                    gpu_id, bucket = future_to_assignment[future]
                    try:
                        worker_rows = future.result()
                    except Exception as exc:
                        worker_rows = [
                            make_error_row(job, exc, gpu_id) for job in bucket
                        ]
                        if args.fail_fast:
                            rows.extend(worker_rows)
                            write_parent_outputs(sorted_rows(rows), out_csv, out_yaml, args, input_path)
                            raise
                    rows.extend(worker_rows)
                    completed_count += len(worker_rows)
                    print(
                        f"[parallel] gpu={gpu_id} finished {len(worker_rows)} shapes "
                        f"({completed_count}/{len(jobs)})"
                    )
                    write_parent_outputs(sorted_rows(rows), out_csv, out_yaml, args, input_path)
        else:
            print(f"Running {len(jobs)} shapes in one worker process")
            manifest_path = override_dir / "worker_single_manifest.yaml"
            try:
                rows.extend(run_worker_process(
                    jobs, manifest_path, args, None))
            except Exception as exc:
                rows.extend(make_error_row(job, exc) for job in jobs)
                write_parent_outputs(sorted_rows(rows), out_csv, out_yaml, args, input_path)
                if args.fail_fast:
                    raise
            write_parent_outputs(sorted_rows(rows), out_csv, out_yaml, args, input_path)
    finally:
        if temp_ctx is not None and not args.keep_override_files:
            temp_ctx.cleanup()

    rows = sorted_rows(rows)
    write_parent_outputs(rows, out_csv, out_yaml, args, input_path)
    print(f"Wrote {len(rows)} rows to {out_csv}")
    print(f"Wrote YAML summary to {out_yaml}")
    if args.ga_generations > 0:
        selected_csv, selected_yaml = ga_selected_output_paths(args, input_path)
        print(f"Wrote GA selected configs to {selected_csv}")
        print(f"Wrote GA selected config YAML to {selected_yaml}")


def torch_dtype(dtype_name: Optional[str]):
    import torch

    name = (dtype_name or "bfloat16").replace("torch.", "")
    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if name not in mapping:
        raise RuntimeError(f"Unsupported dtype: {dtype_name}")
    return mapping[name]


def is_tma_compatible_dtype(dtype: Any, n: int, k: int) -> bool:
    import torch

    return (
        dtype in (torch.float16, torch.bfloat16)
        and n % 8 == 0
        and k % 8 == 0
    ) or (dtype in (torch.float32,) and n % 4 == 0 and k % 4 == 0)


def unwrap_tuner(kernel: Any) -> Any:
    missing = object()
    fn = kernel
    visited = set()
    while fn is not None and id(fn) not in visited:
        visited.add(id(fn))
        if (
            safe_getattr(fn, "configs", missing) is not missing
            and safe_getattr(fn, "policy", missing) is not missing
            and safe_getattr(fn, "prune_configs", missing) is not missing
        ):
            return fn
        fn = safe_getattr(fn, "fn", None)
    return None


def is_libentry_kernel(kernel: Any) -> bool:
    missing = object()
    return (
        safe_getattr(kernel, "kernel_cache", missing) is not missing
        and safe_getattr(kernel, "key", missing) is not missing
        and safe_getattr(kernel, "fn", missing) is not missing
    )


KERNEL_ATTR_BY_KIND = {
    "gemv": "gemv_kernel",
    "mm_general_tma": "mm_kernel_general_host_tma",
    "mm_general": "mm_kernel_general",
    "mm_splitk": "mm_kernel_splitk",
}


def kernel_attr_for_kind(kernel_kind: str) -> str:
    kind = normalize_kernel_kind(kernel_kind) or kernel_kind
    try:
        return KERNEL_ATTR_BY_KIND[kind]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported kernel kind: {kernel_kind}") from exc


def find_kernel_libentry_with_module(
    kernel_kind: str, preferred_module: Any
) -> tuple[Any, Any, Optional[str], Any]:
    kernel_attr = kernel_attr_for_kind(kernel_kind)
    candidates: List[tuple[Any, Any, str, Any]] = []
    for module_name, module in iter_loaded_mm_modules(preferred_module):
        kernel = safe_getattr(module, kernel_attr, None)
        if not is_libentry_kernel(kernel):
            continue
        tuner = unwrap_tuner(kernel)
        if tuner is None:
            continue
        candidates.append((kernel, tuner, module_name, module))
        if safe_getattr(tuner, "best_config", None) is not None:
            return kernel, tuner, module_name, module

    if candidates:
        return candidates[0]
    return None, None, None, None


def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def runtime_config_dict(config_entry: Dict[str, Any]) -> Dict[str, Any]:
    config = config_entry.get("config", config_entry)
    if not isinstance(config, dict):
        raise ValueError(f"Missing config object in runtime config: {config_entry!r}")
    return config


def runtime_config_int(
    config: Dict[str, Any],
    meta: Dict[str, Any],
    name: str,
    *,
    required: bool = False,
) -> Optional[int]:
    value = config.get(name, meta.get(name))
    if value is None:
        if required:
            raise ValueError(f"runtime config requires {name}")
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name} in runtime config: {value!r}") from exc


def make_triton_config(
    meta: Dict[str, Any],
    *,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    num_ctas: Optional[int] = None,
    pre_hook: Any = None,
) -> Any:
    import triton

    kwargs: Dict[str, Any] = {}
    if num_warps is not None:
        kwargs["num_warps"] = num_warps
    if num_stages is not None:
        kwargs["num_stages"] = num_stages
    if pre_hook is not None:
        kwargs["pre_hook"] = pre_hook
    if num_ctas is not None:
        kwargs["num_ctas"] = num_ctas
    try:
        return triton.Config(meta, **kwargs)
    except TypeError:
        if "num_ctas" not in kwargs:
            raise
        kwargs.pop("num_ctas", None)
        return triton.Config(meta, **kwargs)


def make_mm_tma_runtime_configs(
    config_entries: List[Dict[str, Any]],
    *,
    pre_hook: Any,
) -> List[Any]:
    if pre_hook is None:
        raise RuntimeError("mm_general_tma LibTuner does not expose a FlagTune pre-hook")

    configs = []
    for entry in config_entries:
        config = runtime_config_dict(entry)
        meta = config.get("META") or {}
        block_m = runtime_config_int(config, meta, "BLOCK_M", required=True)
        block_n = runtime_config_int(config, meta, "BLOCK_N", required=True)
        block_k = runtime_config_int(config, meta, "BLOCK_K", required=True)
        group_m = runtime_config_int(config, meta, "GROUP_M") or 8
        configs.append(
            make_triton_config(
                {
                    "BLOCK_M": block_m,
                    "BLOCK_N": block_n,
                    "BLOCK_K": block_k,
                    "GROUP_M": group_m,
                },
                num_warps=runtime_config_int(config, meta, "num_warps", required=True),
                num_stages=runtime_config_int(
                    config, meta, "num_stages", required=True
                ),
                num_ctas=runtime_config_int(config, meta, "num_ctas"),
                pre_hook=pre_hook,
            )
        )
    return configs


def make_mm_general_runtime_configs(config_entries: List[Dict[str, Any]]) -> List[Any]:
    configs = []
    for entry in config_entries:
        config = runtime_config_dict(entry)
        meta = config.get("META") or {}
        configs.append(
            make_triton_config(
                {
                    "BLOCK_M": runtime_config_int(config, meta, "BLOCK_M", required=True),
                    "BLOCK_N": runtime_config_int(config, meta, "BLOCK_N", required=True),
                    "BLOCK_K": runtime_config_int(config, meta, "BLOCK_K", required=True),
                    "GROUP_M": runtime_config_int(config, meta, "GROUP_M") or 8,
                },
                num_warps=runtime_config_int(config, meta, "num_warps", required=True),
                num_stages=runtime_config_int(
                    config, meta, "num_stages", required=True
                ),
                num_ctas=runtime_config_int(config, meta, "num_ctas"),
            )
        )
    return configs


def make_mm_splitk_runtime_configs(config_entries: List[Dict[str, Any]]) -> List[Any]:
    configs = []
    for entry in config_entries:
        config = runtime_config_dict(entry)
        meta = config.get("META") or {}
        configs.append(
            make_triton_config(
                {
                    "BLOCK_M": runtime_config_int(config, meta, "BLOCK_M", required=True),
                    "BLOCK_N": runtime_config_int(config, meta, "BLOCK_N", required=True),
                    "BLOCK_K": runtime_config_int(config, meta, "BLOCK_K", required=True),
                    "SPLIT_K": runtime_config_int(config, meta, "SPLIT_K") or 4,
                },
                num_warps=runtime_config_int(config, meta, "num_warps", required=True),
                num_stages=runtime_config_int(
                    config, meta, "num_stages", required=True
                ),
            )
        )
    return configs


def make_gemv_runtime_configs(config_entries: List[Dict[str, Any]]) -> List[Any]:
    configs = []
    for entry in config_entries:
        config = runtime_config_dict(entry)
        meta = config.get("META") or {}
        configs.append(
            make_triton_config(
                {
                    "BLOCK_M": runtime_config_int(config, meta, "BLOCK_M", required=True),
                    "BLOCK_K": runtime_config_int(config, meta, "BLOCK_K", required=True),
                },
                num_warps=runtime_config_int(config, meta, "num_warps"),
                num_stages=runtime_config_int(config, meta, "num_stages"),
            )
        )
    return configs


def iter_loaded_mm_modules(preferred_module: Any) -> Iterable[tuple[str, Any]]:
    seen = set()
    if preferred_module is not None:
        seen.add(id(preferred_module))
        yield safe_getattr(preferred_module, "__name__", "<preferred>"), preferred_module

    for name, module in list(sys.modules.items()):
        if not isinstance(name, str) or not name.endswith(".ops.mm"):
            continue
        if id(module) in seen:
            continue
        if (
            safe_getattr(module, "gemv_kernel", None) is None
            and safe_getattr(module, "mm_kernel_general_host_tma", None) is None
            and safe_getattr(module, "mm_kernel_general", None) is None
            and safe_getattr(module, "mm_kernel_splitk", None) is None
        ):
            continue
        seen.add(id(module))
        yield name, module


def find_kernel_tuner_with_module(
    kernel_kind: str, preferred_module: Any
) -> tuple[Any, Optional[str], Any]:
    kernel_attr = kernel_attr_for_kind(kernel_kind)
    candidates: List[tuple[Any, str, Any]] = []
    for module_name, module in iter_loaded_mm_modules(preferred_module):
        kernel = safe_getattr(module, kernel_attr, None)
        tuner = unwrap_tuner(kernel)
        if tuner is None:
            continue
        candidates.append((tuner, module_name, module))
        if safe_getattr(tuner, "best_config", None) is not None:
            return tuner, module_name, module

    if candidates:
        return candidates[0]
    return None, None, None


def make_runtime_configs(
    kernel_kind: str,
    payload: Dict[str, Any],
    *,
    pre_hook: Any = None,
) -> List[Any]:
    config_entries = payload.get("configs")
    if not isinstance(config_entries, list) or not config_entries:
        raise RuntimeError("Worker payload does not contain configs")

    if kernel_kind == "gemv":
        return make_gemv_runtime_configs(config_entries)
    if kernel_kind == "mm_general_tma":
        return make_mm_tma_runtime_configs(config_entries, pre_hook=pre_hook)
    if kernel_kind == "mm_general":
        return make_mm_general_runtime_configs(config_entries)
    if kernel_kind == "mm_splitk":
        return make_mm_splitk_runtime_configs(config_entries)
    raise RuntimeError(f"Unsupported kernel kind for runtime configs: {kernel_kind}")


def libtuner_pre_hook(tuner: Any) -> Any:
    pre_hook = safe_getattr(tuner, "_flagtune_pre_hook", None)
    if pre_hook is not None:
        return pre_hook
    for config in safe_getattr(tuner, "configs", []) or []:
        pre_hook = safe_getattr(config, "pre_hook", None)
        if pre_hook is not None:
            return pre_hook
    return None


def install_runtime_configs(kernel_kind: str, payload: Dict[str, Any]) -> tuple[Any, Optional[str]]:
    tuner, tuner_source, module = find_kernel_tuner_with_module(
        kernel_kind, None)
    if tuner is None or module is None:
        raise RuntimeError(
            f"Could not find LibTuner for kernel kind: {kernel_kind}")
    pre_hook = libtuner_pre_hook(tuner)
    tuner._flagtune_runtime_configs = make_runtime_configs(
        kernel_kind, payload, pre_hook=pre_hook)
    return tuner, tuner_source


def ga_libtuner_runtime_helpers() -> mm_ga_libtuner_runtime.LibTunerRuntimeHelpers:
    return mm_ga_libtuner_runtime.LibTunerRuntimeHelpers(
        install_runtime_configs=install_runtime_configs,
        triton_config_record=triton_config_record,
        find_matching_config_entry=find_matching_config_entry,
        flatten_config_record=flatten_config_record,
        configs_match=configs_match,
        parse_float=parse_float,
        parse_int=parse_int,
        safe_getattr=safe_getattr,
        write_runtime_config_log=write_runtime_config_log,
        print_runtime_configs_env=PRINT_RUNTIME_CONFIGS_ENV,
    )


def _snapshot_libentry_kernel_cache(libentry_kernel: Any) -> tuple[List[Dict[Any, Any]], Optional[Dict[Any, Any]]]:
    kernel_cache = list(safe_getattr(libentry_kernel, "kernel_cache", ()) or ())
    kernel_cache_snapshot = [dict(cache) for cache in kernel_cache]
    cpu_cache = safe_getattr(libentry_kernel, "_cpu_cache", None)
    cpu_cache_snapshot = dict(cpu_cache) if isinstance(cpu_cache, dict) else None
    return kernel_cache_snapshot, cpu_cache_snapshot


def _clear_libentry_kernel_cache(libentry_kernel: Any) -> None:
    for cache in safe_getattr(libentry_kernel, "kernel_cache", ()) or ():
        if hasattr(cache, "clear"):
            cache.clear()
    cpu_cache = safe_getattr(libentry_kernel, "_cpu_cache", None)
    if isinstance(cpu_cache, dict):
        cpu_cache.clear()


def _restore_libentry_kernel_cache(
    libentry_kernel: Any,
    kernel_cache_snapshot: List[Dict[Any, Any]],
    cpu_cache_snapshot: Optional[Dict[Any, Any]],
) -> None:
    for cache, snapshot in zip(
        safe_getattr(libentry_kernel, "kernel_cache", ()) or (),
        kernel_cache_snapshot,
    ):
        if hasattr(cache, "clear") and hasattr(cache, "update"):
            cache.clear()
            cache.update(snapshot)
    cpu_cache = safe_getattr(libentry_kernel, "_cpu_cache", None)
    if isinstance(cpu_cache, dict) and cpu_cache_snapshot is not None:
        cpu_cache.clear()
        cpu_cache.update(cpu_cache_snapshot)


def _temporary_benchmark_repetition(warmup: int, rep: int):
    from benchmark import conftest as bench_conftest

    old_warmup = bench_conftest.Config.warm_up
    old_rep = bench_conftest.Config.repetition
    bench_conftest.Config.warm_up = warmup
    bench_conftest.Config.repetition = rep

    def restore() -> None:
        bench_conftest.Config.warm_up = old_warmup
        bench_conftest.Config.repetition = old_rep

    return restore


def benchmark_ga_best_config_with_bench_get_latency(
    kernel_kind: str,
    tuner: Any,
    best_config: Any,
    bench: Any,
    mm_args: tuple[Any, ...],
    mm_kwargs: Dict[str, Any],
    torch_device_fn: Any,
    warmup: int = 1000,
    rep: int = 100,
) -> Dict[str, Any]:
    libentry_kernel, found_tuner, libentry_source, _ = find_kernel_libentry_with_module(
        kernel_kind,
        None,
    )
    if libentry_kernel is None or found_tuner is None:
        raise RuntimeError(f"Could not find LibEntry for kernel kind: {kernel_kind}")
    if found_tuner is not tuner:
        tuner = found_tuner

    old_configs = safe_getattr(tuner, "configs", None)
    old_best_config = safe_getattr(tuner, "best_config", None)
    old_runtime_configs = safe_getattr(tuner, "_flagtune_runtime_configs", None)
    old_has_flagtune_tuner = safe_getattr(libentry_kernel, "_has_flagtune_tuner", None)
    kernel_cache_snapshot, cpu_cache_snapshot = _snapshot_libentry_kernel_cache(
        libentry_kernel
    )
    restore_benchmark_repetition = _temporary_benchmark_repetition(warmup, rep)

    import flag_gems

    try:
        # Force LibEntry.run() to compile and cache this exact GA best config.
        libentry_kernel._has_flagtune_tuner = False
        tuner.configs = [best_config]
        tuner.best_config = best_config
        tuner._flagtune_runtime_configs = [best_config]
        _clear_libentry_kernel_cache(libentry_kernel)
        with flag_gems.use_gems(exclude=["zero_"]):
            latency_ms = bench.get_latency(bench.torch_op, *mm_args, **mm_kwargs)
        torch_device_fn.synchronize()
        latency_ms = parse_float(latency_ms)
        if latency_ms is None:
            raise RuntimeError(
                "GA final best bench.get_latency produced a non-finite or invalid latency"
            )
    finally:
        restore_benchmark_repetition()
        _restore_libentry_kernel_cache(
            libentry_kernel,
            kernel_cache_snapshot,
            cpu_cache_snapshot,
        )
        if old_has_flagtune_tuner is not None:
            libentry_kernel._has_flagtune_tuner = old_has_flagtune_tuner
        tuner.configs = old_configs
        tuner.best_config = old_best_config
        if old_runtime_configs is None:
            try:
                delattr(tuner, "_flagtune_runtime_configs")
            except AttributeError:
                pass
        else:
            tuner._flagtune_runtime_configs = old_runtime_configs

    return {
        "latency_ms": float(latency_ms),
        "best_config": best_config,
        "runtime_configs": [best_config],
        "warmup": warmup,
        "rep": rep,
        "libentry_source": libentry_source,
    }


def triton_config_record(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    record = dict(getattr(config, "kwargs", {}))
    for attr in ("num_warps", "num_stages", "num_ctas"):
        if hasattr(config, attr):
            record[attr] = getattr(config, attr)
    return record


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


def configure_benchmark(warmup: int, rep: int, mode: str) -> None:
    from benchmark import conftest as bench_conftest
    from benchmark import consts

    bench_conftest.Config = bench_conftest.BenchConfig()
    bench_conftest.Config.warm_up = warmup
    bench_conftest.Config.repetition = rep
    bench_conftest.Config.mode = consts.BenchMode(mode)


def mm_input_fn(
    b: int,
    m: int,
    n: int,
    k: int,
    cur_dtype: Any,
    device: Any,
    b_column_major: bool,
) -> Iterable[tuple[Any, Any]]:
    # Keep this in lockstep with benchmark/test_blas_perf_parallel.py:mm_input_fn.
    import torch

    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        yield inp1, inp2.t()
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        yield inp1, inp2


def measure_payload(
    payload: Dict[str, Any],
    args: argparse.Namespace,
    bench: Any,
    torch: Any,
    torch_device_fn: Any,
) -> Dict[str, Any]:
    entry = payload["shape_entry"]
    configs = payload["configs"]
    shape = payload.get("normalized_shape") or parse_shape(entry)
    dtype = torch_dtype(args.dtype or shape.get("dtype"))
    device = torch.device(args.device)
    m, n, k = int(shape["M"]), int(shape["N"]), int(shape["K"])
    kernel_kind = route_mm_kernel_kind(m, n, k, dtype)
    override_env = override_env_for_kernel_kind(kernel_kind)

    shape_index = parse_int(entry.get("shape_index")) or 0
    torch.manual_seed(args.seed + shape_index)
    input_item = next(mm_input_fn(1, m, n, k, dtype, device, False))
    mm_args, mm_kwargs = bench.unpack_to_args_kwargs(input_item)

    if kernel_kind == "mm_general_tma" and not is_tma_compatible_dtype(dtype, n, k):
        raise RuntimeError(
            f"Shape is not TMA compatible: M={m}, N={n}, K={k}, dtype={dtype}"
        )

    import flag_gems

    ga_measured_entries: Optional[List[Dict[str, Any]]] = None
    if args.ga_generations > 0:
        ga_helpers = ga_libtuner_runtime_helpers()
        if os.getenv(PRINT_RUNTIME_CONFIGS_ENV, None) == "1":
            write_runtime_config_log(
                "[FlagTune][run_topk] GA branch enter "
                f"shape={payload.get('shape') or output_shape(entry, shape)} "
                f"shape_key={payload.get('shape_key')} "
                f"kernel_kind={kernel_kind} "
                f"generations={args.ga_generations} "
                f"initial_config_count={len(configs)}"
            )
        kernel_args, kernel_kwargs = mm_ga_libtuner_runtime.make_kernel_bench_call(
            kernel_kind, mm_args, torch
        )
        ga_result = mm_ga_libtuner_runtime.run_ga_search(
            payload,
            args,
            kernel_kind,
            kernel_args,
            kernel_kwargs,
            mm_args[0].device,
            torch_device_fn,
            ga_helpers,
        )
        tuner = ga_result["tuner"]
        tuner_source = ga_result["tuner_source"]
        best_config = ga_result["best_config"]
        timings = ga_result["timings"]
        runtime_configs = ga_result["runtime_configs"]
        configs = ga_result["measured_entries"]
        ga_measured_entries = configs
        ga_generated_count = sum(
            1
            for config_entry_item in configs
            if parse_int(config_entry_item.get("ga_generation")) not in (None, 0)
        )
        if best_config is None:
            raise RuntimeError("GA generated configs but LibTuner did not select a best config")
        final_bench = benchmark_ga_best_config_with_bench_get_latency(
            kernel_kind,
            tuner,
            best_config,
            bench,
            mm_args,
            mm_kwargs,
            torch_device_fn,
            warmup=args.warmup,
            rep=args.rep,
        )
        latency_ms = final_bench["latency_ms"]
        best_config = final_bench["best_config"]
        runtime_configs = final_bench["runtime_configs"]
        if os.getenv(PRINT_RUNTIME_CONFIGS_ENV, None) == "1":
            write_runtime_config_log(
                "[FlagTune][run_topk] GA final best bench.get_latency measured "
                f"shape={payload.get('shape') or output_shape(entry, shape)} "
                f"shape_key={payload.get('shape_key')} "
                f"kernel_kind={kernel_kind} "
                f"libentry_source={final_bench['libentry_source']} "
                f"warmup={final_bench['warmup']} "
                f"rep={final_bench['rep']} "
                f"latency_ms={float(latency_ms)} "
                f"best_config={best_config}"
            )
    else:
        policy_helpers = ga_libtuner_runtime_helpers()
        if os.getenv(PRINT_RUNTIME_CONFIGS_ENV, None) == "1":
            write_runtime_config_log(
                "[FlagTune][run_topk] top-k policy branch enter "
                f"shape={payload.get('shape') or output_shape(entry, shape)} "
                f"shape_key={payload.get('shape_key')} "
                f"kernel_kind={kernel_kind} "
                f"config_count={len(configs)}"
            )
        kernel_args, kernel_kwargs = mm_ga_libtuner_runtime.make_kernel_bench_call(
            kernel_kind, mm_args, torch
        )
        policy_result = mm_ga_libtuner_runtime.evaluate_ga_generation(
            payload,
            configs,
            kernel_kind,
            args,
            kernel_args,
            kernel_kwargs,
            mm_args[0].device,
            torch_device_fn,
            policy_helpers,
            log_label="top-k policy measured",
        )
        tuner = policy_result["tuner"]
        tuner_source = policy_result["tuner_source"]
        best_config = policy_result["best_config"]
        timings = policy_result["timings"]
        runtime_configs = policy_result["runtime_configs"]
        configs = policy_result["measured_entries"]
        latency_ms = policy_result["generation_best_latency_ms"]
        if latency_ms is None:
            raise RuntimeError("top-k policy measurement did not produce a latency")
        ga_generated_count = 0

    runtime_best_config = triton_config_record(best_config)
    benchmark_best_config = benchmark_best_config_record(payload, configs)
    selected_entry = find_matching_config_entry(runtime_best_config, configs)
    selected_candidate_rank = (
        selected_entry.get("candidate_rank") if isinstance(selected_entry, dict) else None
    )
    selected_ga_source = (
        selected_entry.get("ga_source") if isinstance(selected_entry, dict) else None
    )
    selected_ga_generation = (
        selected_entry.get("ga_generation") if isinstance(selected_entry, dict) else None
    )
    selected_entry_matches_benchmark_best = (
        parse_bool(selected_entry.get("matches_benchmark_best_config"))
        if isinstance(selected_entry, dict)
        else None
    )
    if selected_entry_matches_benchmark_best is None:
        best_config_same_as_benchmark_best = (
            configs_match(runtime_best_config, benchmark_best_config)
            if benchmark_best_config
            else None
        )
    else:
        best_config_same_as_benchmark_best = selected_entry_matches_benchmark_best
    if os.getenv(PRINT_RUNTIME_CONFIGS_ENV, None) == "1":
        write_runtime_config_log(
            "[FlagTune][run_topk] after get_latency "
            f"shape={payload.get('shape') or output_shape(entry, shape)} "
            f"shape_key={payload.get('shape_key')} "
            f"kernel_kind={kernel_kind} latency_ms={float(latency_ms)} "
            f"best_config={best_config} "
            f"benchmark_best_config={benchmark_best_config} "
            f"same_as_benchmark_best={best_config_same_as_benchmark_best} "
            f"ga_generated={ga_generated_count} "
            f"final_config_count={len(configs)} "
            f"timings={timings}"
        )

    benchmark_best_in_configs = (
        find_matching_config_entry(benchmark_best_config, configs) is not None
        if benchmark_best_config
        else entry.get("benchmark_best_in_topk_for_shape")
    )

    row = {
        "shape": payload.get("shape") or output_shape(entry, shape),
        "shape_key": payload.get("shape_key"),
        "dtype": str(dtype).replace("torch.", ""),
        "kernel_kind": kernel_kind,
        "override_env": override_env,
        "tuner_source": tuner_source,
        "config_count": len(configs),
        "latency_ms": float(latency_ms),
        "warmup": args.warmup,
        "rep": args.rep,
        "mode": args.mode,
        "best_config": runtime_best_config,
        "topk_runtime_best_config": runtime_best_config,
        "topk_runtime_best_candidate_rank": selected_candidate_rank,
        "topk_runtime_best_ga_source": selected_ga_source,
        "topk_runtime_best_ga_generation": selected_ga_generation,
        "benchmark_best_config": benchmark_best_config,
        "benchmark_best_available": bool(benchmark_best_config),
        "topk_runtime_best_same_as_benchmark_best": best_config_same_as_benchmark_best,
        "benchmark_best_in_topk_for_shape": benchmark_best_in_configs,
        "oracle_best_config_order": entry.get("oracle_best_config_order"),
        "timings": str(timings) if timings is not None else None,
    }
    if args.ga_generations > 0 and ga_measured_entries is not None:
        row["_ga_selected_configs"] = mm_ga_libtuner_runtime.build_ga_measured_config_rows(
            ga_measured_entries,
            kernel_kind,
            runtime_best_config,
            benchmark_best_config,
            ga_libtuner_runtime_helpers(),
        )
    return row


def make_manifest_error_row(
    job: Dict[str, Any], exc: Exception, args: argparse.Namespace
) -> Dict[str, Any]:
    return {
        "shape": job.get("shape"),
        "shape_key": job.get("shape_key"),
        "kernel_kind": job.get("kernel_kind"),
        "shape_group_index": job.get("shape_group_index"),
        "balance_latency_ms": job.get("balance_latency_ms"),
        "estimated_cost": job.get("estimated_cost"),
        "cost_source": job.get("cost_source"),
        "config_count": job.get("config_count"),
        "success": False,
        "error": repr(exc),
        "warmup": args.warmup,
        "rep": args.rep,
        "mode": args.mode,
    }


def finalize_worker_row(
    row: Dict[str, Any], job: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    row["shape_group_index"] = job.get("shape_group_index")
    row["balance_latency_ms"] = job.get("balance_latency_ms")
    row["estimated_cost"] = job.get("estimated_cost")
    row["cost_source"] = job.get("cost_source")
    row["success"] = True
    row["override_yaml"] = (
        str(job.get("payload_path")) if args.keep_override_files else None
    )
    return row


def run_worker(args: argparse.Namespace) -> None:
    if not args._manifest:
        raise RuntimeError("worker mode requires --_manifest")

    with Path(args._manifest).open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise RuntimeError(
            f"Unsupported worker manifest format: {args._manifest}")

    configure_benchmark(args.warmup, args.rep, args.mode)

    import torch

    from benchmark.base import Benchmark
    from flag_gems.runtime import torch_device_fn

    bench = Benchmark("mm", torch.Tensor.mm)
    rows: List[Dict[str, Any]] = []
    for job in jobs:
        try:
            payload = load_payload(Path(job["payload_path"]))
            row = measure_payload(payload, args, bench, torch, torch_device_fn)
            rows.append(finalize_worker_row(row, job, args))
        except Exception as exc:
            row = make_manifest_error_row(job, exc, args)
            rows.append(row)
            if args.fail_fast:
                raise

    print(json.dumps({"items": rows}, ensure_ascii=False, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.ga_generations < 0:
        raise ValueError("--ga-generations must be >= 0")
    if args.ga_population_size < 1:
        raise ValueError("--ga-population-size must be >= 1")
    if args.ga_elite_size < 1:
        raise ValueError("--ga-elite-size must be >= 1")
    if args.ga_offspring_per_generation < 0:
        raise ValueError("--ga-offspring-per-generation must be >= 0")
    if not 0.0 <= args.ga_mutation_rate <= 1.0:
        raise ValueError("--ga-mutation-rate must be in [0, 1]")
    if not 0.0 <= args.ga_random_rate <= 1.0:
        raise ValueError("--ga-random-rate must be in [0, 1]")
    if args.ga_max_evaluations_per_shape < 0:
        raise ValueError("--ga-max-evaluations-per-shape must be >= 0")
    if args._worker:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
