#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse


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
