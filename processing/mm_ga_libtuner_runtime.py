#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""LibTuner-backed GA evaluation helpers for mm top-k runtime search."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

try:
    from . import mm_ga_search
except ImportError:
    import mm_ga_search


GA_STOP_IMPROVEMENT_THRESHOLD = 0.05
GA_STOP_PATIENCE = 3


@dataclass(frozen=True)
class LibTunerRuntimeHelpers:
    install_runtime_configs: Callable[[str, Dict[str, Any]], tuple[Any, Optional[str]]]
    triton_config_record: Callable[[Any], Dict[str, Any]]
    find_matching_config_entry: Callable[
        [Dict[str, Any], List[Dict[str, Any]]], Optional[Dict[str, Any]]
    ]
    flatten_config_record: Callable[[Any], Dict[str, Any]]
    configs_match: Callable[[Dict[str, Any], Dict[str, Any]], bool]
    parse_float: Callable[[Any], Optional[float]]
    parse_int: Callable[[Any], Optional[int]]
    safe_getattr: Callable[..., Any]
    write_runtime_config_log: Callable[[str], None]
    print_runtime_configs_env: str


def ga_options_from_args(args: Any) -> mm_ga_search.GAOptions:
    return mm_ga_search.GAOptions(
        generations=args.ga_generations,
        population_size=args.ga_population_size,
        elite_size=args.ga_elite_size,
        offspring_per_generation=args.ga_offspring_per_generation,
        mutation_rate=args.ga_mutation_rate,
        random_rate=args.ga_random_rate,
        max_evaluations=args.ga_max_evaluations_per_shape,
    )


def timing_for_runtime_config(
    timings: Any, runtime_config: Any, helpers: LibTunerRuntimeHelpers
) -> Any:
    if not isinstance(timings, dict) or runtime_config is None:
        return None
    try:
        target_kwargs = runtime_config.all_kwargs()
    except Exception:
        target_kwargs = helpers.triton_config_record(runtime_config)

    for config, timing in timings.items():
        try:
            config_kwargs = config.all_kwargs()
        except Exception:
            config_kwargs = helpers.triton_config_record(config)
        if config_kwargs == target_kwargs:
            return timing
    return None


def timing_to_latency_ms(timing: Any, helpers: LibTunerRuntimeHelpers) -> Optional[float]:
    if isinstance(timing, (list, tuple)):
        if not timing:
            return None
        return helpers.parse_float(timing[0])
    return helpers.parse_float(timing)


def payload_with_config_entries(
    payload: Dict[str, Any], config_entries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    out = dict(payload)
    out["configs"] = config_entries
    return out


def higher_torch_dtype(torch: Any, left: Any, right: Any) -> Any:
    ordered = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    if left == right:
        return left
    left_index = ordered.index(left) if left in ordered else -1
    right_index = ordered.index(right) if right in ordered else -1
    return left if left_index >= right_index else right


def make_kernel_bench_call(
    kernel_kind: str,
    mm_args: tuple[Any, ...],
    torch: Any,
) -> tuple[tuple[Any, ...], Dict[str, Any]]:
    """Build direct backend kernel args used by LibTuner._bench()."""
    import triton

    a, b = mm_args[:2]
    m, k = a.shape
    _, n = b.shape
    c_dtype = higher_torch_dtype(torch, a.dtype, b.dtype)
    c = torch.empty((m, n), device=a.device, dtype=c_dtype)

    if kernel_kind == "gemv":

        def grid(meta):
            return (triton.cdiv(m, meta["BLOCK_M"]),)

        return (
            a,
            b,
            c,
            m,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
        ), {
            "grid": grid,
            "IS_FP64": a.dtype == torch.float64,
            "warmup": False,
        }

    if kernel_kind in ("mm_general", "mm_splitk"):

        def grid(meta):
            tile_count = (
                triton.cdiv(m, meta["BLOCK_M"])
                * triton.cdiv(n, meta["BLOCK_N"])
            )
            if kernel_kind == "mm_splitk":
                return (tile_count, meta["SPLIT_K"])
            return (tile_count,)

        if kernel_kind == "mm_general":

            def alloc_fn(size: int, align: int, stream: Optional[int]):
                return torch.empty(size, dtype=torch.int8, device=a.device)

            triton.set_allocator(alloc_fn)

        kwargs: Dict[str, Any] = {
            "grid": grid,
            "warmup": False,
        }
        if kernel_kind == "mm_general":
            kwargs["IS_FP64"] = a.dtype == torch.float64
        return (
            a,
            b,
            c,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
        ), kwargs

    if kernel_kind == "mm_general_tma":
        a_row_major = a.stride(1) == 1
        b_row_major = b.stride(1) == 1
        dummy_block = [1, 1]
        from triton.tools.tensor_descriptor import TensorDescriptor

        if a_row_major:
            a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
        else:
            a_desc = TensorDescriptor(a, a.T.shape, a.T.stride(), dummy_block)
        if b_row_major:
            b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
        else:
            b_desc = TensorDescriptor(b, b.T.shape, b.T.stride(), dummy_block)
        c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
        dtype_str = str(a.dtype).split(".")[-1]

        def grid(meta):
            return (
                triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            )

        return (
            a_desc,
            b_desc,
            c_desc,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
        ), {
            "grid": grid,
            "A_ROW_MAJOR": a_row_major,
            "B_ROW_MAJOR": b_row_major,
            "dtype": dtype_str,
            "warmup": False,
        }

    raise RuntimeError(f"Unsupported kernel_kind for direct LibTuner bench: {kernel_kind}")


def set_libtuner_runtime_key(
    tuner: Any, kernel_args: tuple[Any, ...], kernel_kwargs: Dict[str, Any]
) -> Any:
    nargs = dict(zip(tuner.arg_names, kernel_args))
    all_args = {**nargs, **kernel_kwargs}
    key_args = {key: value for key, value in all_args.items() if key in tuner.arg_names}
    key = tuner.get_key(key_args)
    tuner._flagtune_last_key = key
    return key


def evaluate_ga_generation(
    payload: Dict[str, Any],
    config_entries: List[Dict[str, Any]],
    kernel_kind: str,
    args: Any,
    kernel_args: tuple[Any, ...],
    kernel_kwargs: Dict[str, Any],
    bench_device: Any,
    torch_device_fn: Any,
    helpers: LibTunerRuntimeHelpers,
    log_label: str = "GA generation measured",
) -> Dict[str, Any]:
    generation_payload = payload_with_config_entries(payload, config_entries)
    tuner, tuner_source = helpers.install_runtime_configs(kernel_kind, generation_payload)
    runtime_configs = list(getattr(tuner, "_flagtune_runtime_configs", []) or [])
    set_libtuner_runtime_key(tuner, kernel_args, kernel_kwargs)

    old_nargs = helpers.safe_getattr(tuner, "nargs", None)
    tuner.nargs = dict(zip(tuner.arg_names, kernel_args))

    def bench_config(config: Any) -> List[float]:
        with torch_device_fn.device(bench_device):
            ret = tuner._bench(*kernel_args, config=config, **kernel_kwargs)
        return list(ret)

    try:
        best_config, timings = tuner.policy(
            bench_config,
            runtime_configs,
            kernel_args,
            kernel_kwargs,
        )
    finally:
        tuner.nargs = old_nargs
    tuner.best_config = best_config
    tuner.configs_timings = timings

    measured_entries: List[Dict[str, Any]] = []
    for idx, entry in enumerate(config_entries):
        measured_entry = dict(entry)
        runtime_config = runtime_configs[idx] if idx < len(runtime_configs) else None
        timing = timing_for_runtime_config(timings, runtime_config, helpers)
        measured_latency = timing_to_latency_ms(timing, helpers)
        measured_entry["ga_latency_ms"] = measured_latency
        measured_entry["ga_policy_timing"] = str(timing) if timing is not None else None
        measured_entries.append(measured_entry)

    generation_best_config = best_config
    generation_best_record = helpers.triton_config_record(generation_best_config)
    generation_best_entry = helpers.find_matching_config_entry(
        generation_best_record, measured_entries
    )
    generation_best_latency = None
    if generation_best_entry is not None:
        generation_best_latency = helpers.parse_float(
            generation_best_entry.get("ga_latency_ms")
        )
    measured_latencies = [
        latency
        for latency in (
            helpers.parse_float(entry.get("ga_latency_ms"))
            for entry in measured_entries
        )
        if latency is not None
    ]
    if generation_best_latency is None and measured_latencies:
        generation_best_latency = min(measured_latencies)

    if os.getenv(helpers.print_runtime_configs_env, None) == "1":
        generation = config_entries[0].get("ga_generation") if config_entries else None
        generation_text = f"generation={generation} " if generation is not None else ""
        helpers.write_runtime_config_log(
            f"[FlagTune][run_topk] {log_label} "
            f"{generation_text}"
            f"shape={payload.get('shape')} "
            f"shape_key={payload.get('shape_key')} "
            f"kernel_kind={kernel_kind} "
            f"config_count={len(config_entries)} "
            f"generation_best_latency_ms={generation_best_latency} "
            f"generation_best_config={generation_best_config}"
        )

    return {
        "tuner": tuner,
        "tuner_source": tuner_source,
        "best_config": generation_best_config,
        "timings": timings,
        "runtime_configs": runtime_configs,
        "measured_entries": measured_entries,
        "generation_best_latency_ms": generation_best_latency,
    }


def build_ga_measured_config_rows(
    measured_entries: List[Dict[str, Any]],
    kernel_kind: str,
    runtime_best_config: Dict[str, Any],
    benchmark_best_config: Dict[str, Any],
    helpers: LibTunerRuntimeHelpers,
) -> List[Dict[str, Any]]:
    sorted_entries = sorted(
        measured_entries,
        key=lambda entry: (
            helpers.parse_float(entry.get("ga_latency_ms")) is None,
            helpers.parse_float(entry.get("ga_latency_ms")) or 0.0,
        ),
    )
    rank_by_key = {
        mm_ga_search.entry_key(entry, kernel_kind): idx + 1
        for idx, entry in enumerate(sorted_entries)
    }

    rows: List[Dict[str, Any]] = []
    for entry in measured_entries:
        config = entry.get("config")
        flat = helpers.flatten_config_record(config)
        row: Dict[str, Any] = {
            "shape": entry.get("shape"),
            "shape_key": entry.get("shape_key"),
            "candidate_rank": entry.get("candidate_rank"),
            "ga_generation": entry.get("ga_generation"),
            "ga_source": entry.get("ga_source")
            or ("topk" if entry.get("candidate_rank") is not None else "input"),
            "ga_rank_by_latency": rank_by_key.get(
                mm_ga_search.entry_key(entry, kernel_kind)
            ),
            "ga_is_best": helpers.configs_match(flat, runtime_best_config),
            "ga_is_benchmark_best": helpers.configs_match(
                flat, benchmark_best_config
            )
            if benchmark_best_config
            else None,
            "latency_ms": entry.get("ga_latency_ms"),
            "policy_timing": entry.get("ga_policy_timing"),
            "config": config,
            "success": True,
        }
        row.update({key: value for key, value in flat.items() if value is not None})
        rows.append(row)
    return rows


def log_ga_generation_end(
    payload: Dict[str, Any],
    kernel_kind: str,
    generation: int,
    measured_entries: List[Dict[str, Any]],
    generation_best_latency: Optional[float],
    generation_best_config: Any,
    global_best_latency: Optional[float],
    global_best_config: Any,
    improvement: Optional[float],
    small_improvement_count: int,
    will_stop: bool,
    helpers: LibTunerRuntimeHelpers,
) -> None:
    if os.getenv(helpers.print_runtime_configs_env, None) != "1":
        return

    helpers.write_runtime_config_log(
        "[FlagTune][run_topk] GA generation end "
        f"generation={generation} "
        f"shape={payload.get('shape')} "
        f"shape_key={payload.get('shape_key')} "
        f"kernel_kind={kernel_kind} "
        f"candidate_count={len(measured_entries)} "
        f"generation_best_latency_ms={generation_best_latency} "
        f"generation_best_config={generation_best_config} "
        f"global_best_latency_ms={global_best_latency} "
        f"global_best_config={global_best_config} "
        f"improvement={improvement} "
        f"small_improvement_count={small_improvement_count} "
        f"will_stop={will_stop}"
    )

    candidates = []
    for idx, entry in enumerate(measured_entries):
        candidates.append(
            {
                "idx": idx,
                "candidate_rank": entry.get("candidate_rank"),
                "ga_generation": entry.get("ga_generation"),
                "ga_source": entry.get("ga_source")
                or ("topk" if entry.get("candidate_rank") is not None else "input"),
                "latency_ms": entry.get("ga_latency_ms"),
                "policy_timing": entry.get("ga_policy_timing"),
                "config": helpers.flatten_config_record(entry.get("config")),
            }
        )
    helpers.write_runtime_config_log(
        "[FlagTune][run_topk] GA generation candidates "
        f"generation={generation} "
        f"shape_key={payload.get('shape_key')} "
        f"items={json.dumps(candidates, ensure_ascii=False, sort_keys=True, default=str)}"
    )


def run_ga_search(
    payload: Dict[str, Any],
    args: Any,
    kernel_kind: str,
    kernel_args: tuple[Any, ...],
    kernel_kwargs: Dict[str, Any],
    bench_device: Any,
    torch_device_fn: Any,
    helpers: LibTunerRuntimeHelpers,
) -> Dict[str, Any]:
    options = ga_options_from_args(args)
    initial_entries = mm_ga_search.initial_population(payload["configs"], kernel_kind)
    if not initial_entries:
        raise RuntimeError("GA enabled but initial top-k population is empty")

    rng_seed = args.seed + (
        helpers.parse_int(payload.get("shape_entry", {}).get("shape_index")) or 0
    )
    rng = random.Random(rng_seed)
    known_entries: List[Dict[str, Any]] = []
    known_keys = set()
    measured_entries: List[Dict[str, Any]] = []
    best_latency: Optional[float] = None
    best_config = None
    best_tuner = None
    best_tuner_source = None
    best_runtime_configs = None
    best_timings = None
    small_improvement_count = 0

    def add_measured(entries: List[Dict[str, Any]]) -> None:
        for entry in entries:
            key = mm_ga_search.entry_key(entry, kernel_kind)
            if key in known_keys:
                continue
            known_keys.add(key)
            known_entries.append(entry)
            measured_entries.append(entry)

    eval_result = evaluate_ga_generation(
        payload,
        initial_entries,
        kernel_kind,
        args,
        kernel_args,
        kernel_kwargs,
        bench_device,
        torch_device_fn,
        helpers,
    )
    add_measured(eval_result["measured_entries"])
    best_latency = eval_result["generation_best_latency_ms"]
    best_config = eval_result["best_config"]
    best_tuner = eval_result["tuner"]
    best_tuner_source = eval_result["tuner_source"]
    best_runtime_configs = eval_result["runtime_configs"]
    best_timings = eval_result["timings"]
    log_ga_generation_end(
        payload,
        kernel_kind,
        0,
        eval_result["measured_entries"],
        eval_result["generation_best_latency_ms"],
        eval_result["best_config"],
        best_latency,
        best_config,
        None,
        small_improvement_count,
        False,
        helpers,
    )

    base_entry = payload["configs"][0]
    for generation in range(1, options.generations + 1):
        if options.max_evaluations:
            remaining = options.max_evaluations - len(known_entries)
            if remaining <= 0:
                break
            target_count = min(options.offspring_per_generation, remaining)
        else:
            target_count = options.offspring_per_generation
        if target_count <= 0:
            break

        offspring = mm_ga_search.next_generation(
            base_entry,
            known_entries,
            known_keys,
            kernel_kind,
            generation,
            target_count,
            options,
            rng,
        )
        if not offspring:
            break

        previous_best = best_latency
        eval_result = evaluate_ga_generation(
            payload,
            offspring,
            kernel_kind,
            args,
            kernel_args,
            kernel_kwargs,
            bench_device,
            torch_device_fn,
            helpers,
        )
        add_measured(eval_result["measured_entries"])

        generation_best = eval_result["generation_best_latency_ms"]
        improved = (
            generation_best is not None
            and (best_latency is None or generation_best < best_latency)
        )
        if improved:
            best_latency = generation_best
            best_config = eval_result["best_config"]
            best_tuner = eval_result["tuner"]
            best_tuner_source = eval_result["tuner_source"]
            best_runtime_configs = eval_result["runtime_configs"]
            best_timings = eval_result["timings"]

        improvement = None
        will_stop = False
        if previous_best is not None and best_latency is not None:
            improvement = max(0.0, (previous_best - best_latency) / previous_best)
            if improvement <= GA_STOP_IMPROVEMENT_THRESHOLD:
                small_improvement_count += 1
            else:
                small_improvement_count = 0
            if os.getenv(helpers.print_runtime_configs_env, None) == "1":
                helpers.write_runtime_config_log(
                    "[FlagTune][run_topk] GA generation improvement "
                    f"generation={generation} "
                    f"shape={payload.get('shape')} "
                    f"shape_key={payload.get('shape_key')} "
                    f"best_latency_ms={best_latency} "
                    f"improvement={improvement} "
                    f"small_improvement_count={small_improvement_count}"
                )
            will_stop = small_improvement_count >= GA_STOP_PATIENCE
        log_ga_generation_end(
            payload,
            kernel_kind,
            generation,
            eval_result["measured_entries"],
            eval_result["generation_best_latency_ms"],
            eval_result["best_config"],
            best_latency,
            best_config,
            improvement,
            small_improvement_count,
            will_stop,
            helpers,
        )
        if will_stop:
            break

    if best_tuner is None or best_config is None:
        raise RuntimeError("GA search did not produce a best config")

    return {
        "tuner": best_tuner,
        "tuner_source": best_tuner_source,
        "best_config": best_config,
        "timings": best_timings,
        "runtime_configs": best_runtime_configs,
        "latency_ms": best_latency,
        "measured_entries": measured_entries,
    }
