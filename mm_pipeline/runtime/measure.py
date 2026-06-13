#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from typing import Any, Dict, Iterable, List, Optional

from FlagTune.mm_pipeline.core.routing import route_mm_kernel_kind
from FlagTune.mm_pipeline.ga import libtuner_runtime as mm_ga_libtuner_runtime
from FlagTune.mm_pipeline.runtime.common import override_env_for_kernel_kind, parse_float, parse_int
from FlagTune.mm_pipeline.runtime.configs import benchmark_best_config_record, configs_match, find_matching_config_entry, parse_bool, triton_config_record
from FlagTune.mm_pipeline.runtime.libtuner import find_kernel_libentry_with_module, ga_libtuner_runtime_helpers, safe_getattr
from FlagTune.mm_pipeline.runtime.payloads import output_shape, parse_shape


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
    else:
        policy_helpers = ga_libtuner_runtime_helpers()
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
