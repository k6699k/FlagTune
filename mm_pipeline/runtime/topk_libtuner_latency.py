#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CLI orchestration for MM top-k LibTuner latency measurement."""

from __future__ import annotations

from FlagTune.mm_pipeline.runtime.args import parse_args
from FlagTune.mm_pipeline.runtime.worker import run_parent, run_worker


def main() -> None:
    args = parse_args()
    if args._worker:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
