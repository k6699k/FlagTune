#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from FlagTune.mm_pipeline.xgb.common import DEFAULT_KERNEL_SUBSTRS, safe_filename


def db_search_dirs(db_arg: Optional[str]) -> List[Path]:
    if not db_arg:
        return []
    path = Path(db_arg)
    if path.is_dir():
        return [path]
    if path.is_file():
        return [path.parent]
    return []


def resolve_db_paths(
    paths: Iterable[str],
    arg_name: str,
    search_dirs: Optional[Iterable[Path]] = None,
) -> List[Path]:
    resolved: List[Path] = []
    search_dirs = list(search_dirs or [])
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() and not path.is_absolute():
            for search_dir in search_dirs:
                candidate = search_dir / raw_path
                if candidate.exists():
                    path = candidate
                    break
        if path.is_file():
            if path.suffix != ".db":
                raise ValueError(f"{arg_name} must point to a .db file: {path}")
            resolved.append(path)
            continue
        if path.is_dir():
            candidates = sorted(path.glob("*.db"))
            if not candidates:
                raise FileNotFoundError(f"No *.db files found under {arg_name}: {path}")
            resolved.extend(candidates)
            continue
        searched = (
            ", searched under: "
            + ", ".join(str(search_dir) for search_dir in search_dirs)
            if search_dirs and not Path(raw_path).is_absolute()
            else ""
        )
        raise FileNotFoundError(f"{arg_name} path not found: {raw_path}{searched}")

    unique: List[Path] = []
    seen = set()
    for path in resolved:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if not unique:
        raise ValueError(f"No DB files resolved from {arg_name}")
    return unique


def model_name_from_db_path(path: Path) -> str:
    stem = path.name
    if stem.endswith(".db"):
        stem = stem[:-3]
    timestamp_match = re.match(r"^(.+)_([0-9]{8})_([0-9]{6})_TunedConfig.*$", stem)
    if timestamp_match:
        return timestamp_match.group(1)
    tuned_match = re.match(r"^(.+)_TunedConfig.*$", stem)
    if tuned_match:
        return tuned_match.group(1)
    return stem.split("_", 1)[0]


def resolve_predict_db_from_model(model: str, candidates: Iterable[Path]) -> Path:
    candidate_list = list(candidates)
    matches = [
        path
        for path in candidate_list
        if model_name_from_db_path(path) == model
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(str(path) for path in matches)
        raise RuntimeError(f"Multiple DB files match --predict-model {model!r}: {names}")

    if len(candidate_list) == 1:
        return candidate_list[0]

    names = ", ".join(path.name for path in candidate_list[:10])
    suffix = " ..." if len(candidate_list) > 10 else ""
    raise FileNotFoundError(
        f"No DB matched --predict-model {model!r}. "
        f"Expected filenames like {model}_TunedConfig*.db under --db. "
        f"Candidates: {names}{suffix}"
    )


def resolve_predict_targets(
    args: argparse.Namespace,
    train_db_paths: List[Path],
) -> List[Dict[str, Any]]:
    predict_models = list(args.predict_model or [])
    targets: List[Dict[str, Any]] = []

    if predict_models:
        predict_candidates = (
            resolve_db_paths([args.db], "--db") if args.db else train_db_paths
        )
        for model in predict_models:
            db_path = resolve_predict_db_from_model(model, predict_candidates)
            targets.append({"model": model, "db_path": db_path})
    elif args.db:
        for db_path in resolve_db_paths([args.db], "--db"):
            targets.append(
                {"model": model_name_from_db_path(db_path), "db_path": db_path}
            )
    elif len(train_db_paths) == 1:
        db_path = train_db_paths[0]
        targets.append({"model": model_name_from_db_path(db_path), "db_path": db_path})
    else:
        raise ValueError(
            "--predict-model is required when multiple --train-db inputs are used without --db."
        )

    unique: List[Dict[str, Any]] = []
    seen = set()
    for target in targets:
        key = (str(Path(target["db_path"]).resolve()), str(target["model"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(target)
    if not unique:
        raise ValueError("No prediction targets resolved.")
    return unique


def output_dir_for_predict_target(
    base_out_dir: Path,
    target_count: int,
    model: str,
    db_path: Path,
    train_mode: str,
    top_k: int,
) -> Path:
    if target_count == 1:
        return base_out_dir
    model_name = model or model_name_from_db_path(db_path)
    return base_out_dir / (
        f"{safe_filename(model_name)}_{safe_filename(train_mode)}_top{int(top_k)}"
    )


def resolve_summary_md_for_model(model: str, args: argparse.Namespace) -> Optional[Path]:
    if args.summary_md:
        return Path(args.summary_md)

    pipeline_dir = Path(__file__).resolve().parents[1]
    flagtune_dir = Path(__file__).resolve().parents[2]
    legacy_processing_dir = flagtune_dir / "processing"
    candidates = [
        flagtune_dir / "reports" / f"{model}_{args.op}.md",
        pipeline_dir / "reports" / f"{model}_{args.op}.md",
        pipeline_dir / "reports" / f"{model}_mm.md",
        legacy_processing_dir / f"{model}_{args.op}.md",
        legacy_processing_dir / f"{model}_mm.md",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def list_benchmark_tables(
    conn: sqlite3.Connection,
    table_like: str,
    kernel_substr: Optional[List[str]],
    explicit_tables: Optional[List[str]] = None,
) -> List[str]:
    all_tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'",
        ).fetchall()
    }
    if explicit_tables:
        missing = [t for t in explicit_tables if t not in all_tables]
        if missing:
            raise RuntimeError(
                f"Benchmark table(s) not found in DB: {missing}")
        return list(explicit_tables)

    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
        (table_like,),
    ).fetchall()
    tables = [r[0] for r in rows]
    kernel_substrs = (
        DEFAULT_KERNEL_SUBSTRS
        if kernel_substr is None
        else [s for s in kernel_substr if s]
    )
    if kernel_substrs:
        tables = [t for t in tables if any(substr in t for substr in kernel_substrs)]
    if not tables:
        raise RuntimeError(
            f"No benchmark tables found. LIKE={table_like!r}, kernel_substr={kernel_substrs!r}"
        )
    return sorted(tables)


def read_benchmark_tables(conn: sqlite3.Connection, tables: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for table in tables:
        df = pd.read_sql_query(
            f'SELECT rowid AS sqlite_rowid, * FROM "{table}"', conn)
        df.insert(0, "benchmark_table", table)
        frames.append(df)
    if not frames:
        raise RuntimeError("No benchmark rows loaded.")
    return pd.concat(frames, ignore_index=True, sort=False)


def read_benchmark_cache_db(
    db_path: Path,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, List[str]]:
    conn = sqlite3.connect(str(db_path))
    try:
        tables = list_benchmark_tables(
            conn,
            table_like=args.benchmark_table_like,
            kernel_substr=args.kernel_substr,
            explicit_tables=args.benchmark_table,
        )
        raw_df = read_benchmark_tables(conn, tables)
    finally:
        conn.close()

    raw_df.insert(0, "source_db", str(db_path))
    raw_df.insert(1, "source_db_name", db_path.name)
    return raw_df, tables
