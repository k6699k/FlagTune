#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from FlagTune.mm_pipeline.runtime.common import DEBUG_ENV, GEMV_OVERRIDE_ENV, REPO_ROOT, TMA_OVERRIDE_ENV, normalize_existing_path, parse_float, parse_int
from FlagTune.mm_pipeline.runtime.measure import configure_benchmark, finalize_worker_row, make_manifest_error_row, measure_payload
from FlagTune.mm_pipeline.runtime.outputs import ga_selected_output_paths, output_paths, sorted_rows, write_benchmark_hit_results, write_parent_outputs
from FlagTune.mm_pipeline.runtime.payloads import group_entries, infer_kernel_kind, load_entries, load_payload, output_shape, write_group_payload
from FlagTune.mm_pipeline.runtime.scheduling import estimate_group_cost, load_balance_latency_by_shape, select_groups


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
