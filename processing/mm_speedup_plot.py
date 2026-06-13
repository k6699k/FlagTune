#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

FLAGTUNE_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_SPEEDUP_COL = "benchmark_best_speedup_pct_vs_summary_default_gems"
GAP_COL = "speedup_gap_pct"
DEFAULT_LATENCY_COL = "summary_default_gems_latency_ms"
BENCHMARK_LATENCY_COL = "benchmark_best_measured_p50"
RUN_LATENCY_COL = "run_latency_ms"
BENCHMARK_TO_RUN_LATENCY_PCT_COL = "benchmark_best_latency_pct_of_run"
RUN_SPEEDUP_COL = "run_speedup_pct_vs_summary_default_gems"
PLOT_DATA_COLS = [
    "shape",
    DEFAULT_LATENCY_COL,
    BENCHMARK_LATENCY_COL,
    RUN_LATENCY_COL,
    BENCHMARK_TO_RUN_LATENCY_PCT_COL,
    BENCHMARK_SPEEDUP_COL,
    RUN_SPEEDUP_COL,
    GAP_COL,
]


def normalize_shape_for_merge(value: Any) -> str:
    return ",".join(part.strip() for part in str(value).split(","))


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def plot_row_merge_key(row: Dict[str, Any]) -> Optional[str]:
    for col in ["shape", "summary_shape_b_m_n_k", "shape_key"]:
        value = row.get(col)
        if not is_missing_value(value):
            return normalize_shape_for_merge(value)
    return None


def parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def parse_float(value: Any) -> float:
    if value is None:
        return np.nan
    text = str(value).strip().replace(",", "")
    if text.endswith("%"):
        text = text[:-1]
    try:
        parsed = float(text) if text else np.nan
    except Exception:
        return np.nan
    return parsed if np.isfinite(parsed) else np.nan


def normalize_dtype_name(dtype: Any) -> str:
    return str(dtype or "bfloat16").replace("torch.", "")


def prepare_plot_dataframe(best_df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    needed_cols = [
        "shape_key",
        "shape_index",
        "summary_shape_b_m_n_k",
        DEFAULT_LATENCY_COL,
    ]
    for col in needed_cols:
        if col not in best_df.columns:
            raise RuntimeError(f"Missing required plot column: {col}")

    plot_df = best_df.copy()
    if "summary_expand_gems_latency_ms" in plot_df.columns:
        plot_df[BENCHMARK_LATENCY_COL] = plot_df["summary_expand_gems_latency_ms"]
    elif BENCHMARK_LATENCY_COL not in plot_df.columns:
        if "oracle_best_measured_p50" in plot_df.columns:
            plot_df[BENCHMARK_LATENCY_COL] = plot_df["oracle_best_measured_p50"]
        else:
            raise RuntimeError(f"Missing required plot column: {BENCHMARK_LATENCY_COL}")

    plot_df[BENCHMARK_SPEEDUP_COL] = _speedup(
        plot_df[DEFAULT_LATENCY_COL], plot_df[BENCHMARK_LATENCY_COL]
    )
    plot_df = plot_df[plot_df[BENCHMARK_SPEEDUP_COL].notna()].copy()
    if plot_df.empty:
        raise RuntimeError(
            "No rows available for plotting after filtering NaN benchmark speedups."
        )

    # Keep CLI compatibility, but plot data is always benchmark-speedup sorted.
    plot_df = plot_df.sort_values(
        [BENCHMARK_SPEEDUP_COL, "shape_index"], ascending=[True, True]
    ).reset_index(drop=True)
    plot_df["plot_rank"] = np.arange(1, len(plot_df) + 1)
    return plot_df


def compact_plot_dataframe(plot_df: pd.DataFrame) -> pd.DataFrame:
    df = plot_df.copy()
    if "shape" not in df.columns:
        if "summary_shape_b_m_n_k" in df.columns:
            df["shape"] = df["summary_shape_b_m_n_k"]
            if "shape_key" in df.columns:
                df["shape"] = df["shape"].where(
                    df["shape"].notna(), df["shape_key"])
        elif "shape_key" in df.columns:
            df["shape"] = df["shape_key"]

    if "summary_expand_gems_latency_ms" in df.columns:
        df[BENCHMARK_LATENCY_COL] = df["summary_expand_gems_latency_ms"]
    elif BENCHMARK_LATENCY_COL not in df.columns:
        if "oracle_best_measured_p50" in df.columns:
            df[BENCHMARK_LATENCY_COL] = df["oracle_best_measured_p50"]

    if DEFAULT_LATENCY_COL in df.columns:
        default_latency = df[DEFAULT_LATENCY_COL]
        if BENCHMARK_LATENCY_COL in df.columns:
            df[BENCHMARK_SPEEDUP_COL] = _speedup(
                default_latency, df[BENCHMARK_LATENCY_COL])

        if RUN_LATENCY_COL in df.columns:
            df[RUN_SPEEDUP_COL] = _speedup(
                default_latency, df[RUN_LATENCY_COL])

    if BENCHMARK_LATENCY_COL in df.columns and RUN_LATENCY_COL in df.columns:
        df[BENCHMARK_TO_RUN_LATENCY_PCT_COL] = _percent_ratio(
            df[BENCHMARK_LATENCY_COL], df[RUN_LATENCY_COL])

    if BENCHMARK_SPEEDUP_COL in df.columns and RUN_SPEEDUP_COL in df.columns:
        df[GAP_COL] = df[BENCHMARK_SPEEDUP_COL] - df[RUN_SPEEDUP_COL]

    if BENCHMARK_SPEEDUP_COL in df.columns:
        sort_cols = [BENCHMARK_SPEEDUP_COL]
        if "shape_index" in df.columns:
            sort_cols.append("shape_index")
        df = df.sort_values(sort_cols, ascending=True).reset_index(drop=True)

    for col in PLOT_DATA_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[PLOT_DATA_COLS]


def _speedup(default_latency: pd.Series, latency: pd.Series) -> pd.Series:
    default_latency = pd.to_numeric(default_latency, errors="coerce")
    latency = pd.to_numeric(latency, errors="coerce")
    speedup = ((default_latency / latency) - 1.0) * 100.0
    return speedup.where((default_latency > 0) & (latency > 0))


def _percent_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    ratio = (numerator / denominator) * 100.0
    return ratio.where((numerator > 0) & (denominator > 0))


def write_speedup_plot_data(
    best_df: pd.DataFrame,
    out_dir: Path,
    sort_by: str,
    plot_format: str,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    plot_df = prepare_plot_dataframe(best_df, sort_by=sort_by)
    plot_csv = out_dir / "speedup_plot_data.csv"
    compact_plot_dataframe(plot_df).to_csv(plot_csv, index=False)

    return plot_csv, None, None


def _collect_run_latency_by_shape(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    latency_by_shape: Dict[str, float] = {}
    for row in rows:
        if not row.get("success"):
            continue
        latency_ms = row.get(RUN_LATENCY_COL, row.get("latency_ms"))
        if latency_ms is None:
            continue
        latency = float(latency_ms)
        for shape in [row.get("shape"), row.get("shape_key")]:
            if shape is None:
                continue
            key = normalize_shape_for_merge(shape)
            old_latency = latency_by_shape.get(key)
            if old_latency is None or latency < old_latency:
                latency_by_shape[key] = latency
    return latency_by_shape


def _collect_topk_run_latency_by_shape(out_dir: Path) -> Dict[str, float]:
    latency_by_shape: Dict[str, float] = {}

    def collect_from_frame(frame: pd.DataFrame) -> None:
        latency_col = (
            RUN_LATENCY_COL
            if RUN_LATENCY_COL in frame.columns
            else "latency_ms"
            if "latency_ms" in frame.columns
            else None
        )
        if latency_col is None:
            return
        run_latency = pd.to_numeric(frame[latency_col], errors="coerce")
        for shape_col in ["shape", "shape_key", "summary_shape_b_m_n_k"]:
            if shape_col not in frame.columns:
                continue
            for shape, latency in zip(frame[shape_col], run_latency):
                if pd.isna(shape) or pd.isna(latency):
                    continue
                key = normalize_shape_for_merge(shape)
                old_latency = latency_by_shape.get(key)
                latency = float(latency)
                if old_latency is None or latency < old_latency:
                    latency_by_shape[key] = latency

    for topk_csv in sorted(out_dir.glob("top*_predicted_config_by_shape.csv")):
        collect_from_frame(pd.read_csv(topk_csv))

    latency_csv_paths = sorted(out_dir.glob("predicted_shape_configs_top*latency.csv"))
    for latency_csv in latency_csv_paths:
        collect_from_frame(pd.read_csv(latency_csv))

    latency_yaml_paths = sorted(out_dir.glob("predicted_shape_configs_top*latency.yaml"))
    for latency_yaml in latency_yaml_paths:
        with latency_yaml.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        rows = None
        if isinstance(payload, dict):
            rows = payload.get("items")
            if rows is None:
                rows = payload.get("results")
        if isinstance(rows, list) and rows:
            collect_from_frame(pd.DataFrame(rows))

    if latency_by_shape:
        print(
            f"[INFO] collected run latency for {len(latency_by_shape)} shapes "
            f"from {len(latency_csv_paths)} csv and {len(latency_yaml_paths)} yaml files"
        )
    else:
        print(
            f"[WARN] no top-k run latency found under {out_dir}; "
            "speedup_plot_data.csv will not get run_latency_ms"
        )
    return latency_by_shape


def _build_plot_data_from_topk_run_latency(out_dir: Path) -> Optional[pd.DataFrame]:
    rows: List[Dict[str, Any]] = []

    def collect_from_frame(frame: pd.DataFrame) -> None:
        latency_col = (
            RUN_LATENCY_COL
            if RUN_LATENCY_COL in frame.columns
            else "latency_ms"
            if "latency_ms" in frame.columns
            else None
        )
        if latency_col is None:
            return
        run_latency = pd.to_numeric(frame[latency_col], errors="coerce")
        for idx, latency in enumerate(run_latency):
            if pd.isna(latency):
                continue
            row = frame.iloc[idx]
            shape = row.get("shape")
            shape_key = row.get("shape_key")
            if pd.isna(shape) and pd.isna(shape_key):
                continue
            rows.append(
                {
                    "shape": shape if pd.notna(shape) else shape_key,
                    "shape_key": shape_key if pd.notna(shape_key) else shape,
                    RUN_LATENCY_COL: float(latency),
                }
            )

    for latency_csv in sorted(out_dir.glob("predicted_shape_configs_top*latency.csv")):
        collect_from_frame(pd.read_csv(latency_csv))

    for latency_yaml in sorted(out_dir.glob("predicted_shape_configs_top*latency.yaml")):
        with latency_yaml.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        yaml_rows = None
        if isinstance(payload, dict):
            yaml_rows = payload.get("items") or payload.get("results")
        if isinstance(yaml_rows, list) and yaml_rows:
            collect_from_frame(pd.DataFrame(yaml_rows))

    if not rows:
        return None

    best_by_shape: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = normalize_shape_for_merge(row.get("shape_key") or row.get("shape"))
        old = best_by_shape.get(key)
        if old is None or row[RUN_LATENCY_COL] < old[RUN_LATENCY_COL]:
            best_by_shape[key] = row

    plot_df = pd.DataFrame(best_by_shape.values()).reset_index(drop=True)
    plot_df["shape_index"] = np.arange(len(plot_df), dtype=int)
    return compact_plot_dataframe(plot_df)


def _build_plot_data_from_summary(summary_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if summary_df is None or summary_df.empty:
        return None

    needed = {DEFAULT_LATENCY_COL, "summary_expand_gems_latency_ms"}
    if not needed.issubset(summary_df.columns):
        return None

    plot_df = summary_df.copy()
    default_latency = pd.to_numeric(plot_df[DEFAULT_LATENCY_COL], errors="coerce")
    expand_latency = pd.to_numeric(
        plot_df["summary_expand_gems_latency_ms"], errors="coerce"
    )
    plot_df = plot_df[(default_latency > 0) | (expand_latency > 0)].copy()
    if plot_df.empty:
        return None

    if "shape_index" not in plot_df.columns:
        plot_df["shape_index"] = np.arange(len(plot_df), dtype=int)
    if "shape" not in plot_df.columns and "summary_shape_b_m_n_k" in plot_df.columns:
        plot_df["shape"] = plot_df["summary_shape_b_m_n_k"]

    return compact_plot_dataframe(plot_df)


def parse_summary_markdown(summary_md: Path, dtype: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    dtype_name = normalize_dtype_name(dtype)
    for raw_line in summary_md.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 9:
            continue
        shape_text = parts[0]
        if not shape_text or shape_text.startswith("---"):
            continue
        if "Torch Latency" in line or "Default Configuration" in line:
            continue
        shape_values = [value.strip() for value in shape_text.split(",")]
        if len(shape_values) != 4:
            continue
        b = parse_int(shape_values[0])
        m = parse_int(shape_values[1])
        n = parse_int(shape_values[2])
        k = parse_int(shape_values[3])
        if b is None or m is None or n is None or k is None:
            continue
        rows.append(
            {
                "summary_shape_b_m_n_k": f"{b}, {m}, {n}, {k}",
                "shape_key": f"{m},{n},{k},{k},{n},{dtype_name}",
                "summary_default_gems_latency_ms": parse_float(parts[3]),
                "summary_expand_gems_latency_ms": parse_float(parts[6]),
            }
        )
    if not rows:
        raise RuntimeError(f"No valid summary rows parsed from {summary_md}")
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["shape_key"], keep="first")
        .reset_index(drop=True)
    )


def resolve_summary_md(
    model: Optional[str],
    op: str,
    summary_md: Optional[str],
) -> Optional[Path]:
    if summary_md:
        return Path(summary_md)
    if not model:
        return None
    candidates = [
        FLAGTUNE_DIR / "reports" / f"{model}_{op}.md",
        FLAGTUNE_DIR / "processing" / f"{model}_{op}.md",
        FLAGTUNE_DIR / "processing" / f"{model}_mm.md",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_summary_dataframe(
    out_dir: Path,
    model: Optional[str],
    op: str,
    summary_md: Optional[str],
    dtype: str,
) -> Optional[pd.DataFrame]:
    summary_path = resolve_summary_md(model, op, summary_md)
    if summary_path is not None:
        if not summary_path.is_file():
            raise FileNotFoundError(f"summary markdown not found: {summary_path}")
        summary_df = parse_summary_markdown(summary_path, dtype=dtype)
        parsed_csv = out_dir / "parsed_summary_md.csv"
        parsed_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(parsed_csv, index=False)
        print(f"[INFO] parsed summary md: {summary_path}")
        return summary_df

    summary_csv = out_dir / "parsed_summary_md.csv"
    if not summary_csv.exists():
        if model:
            print(
                f"[WARN] No summary markdown found for model '{model}' "
                f"under {FLAGTUNE_DIR / 'reports'}"
            )
        return None

    summary_df = pd.read_csv(summary_csv)
    needed = {DEFAULT_LATENCY_COL, "summary_expand_gems_latency_ms"}
    if not needed.issubset(summary_df.columns):
        return None
    return summary_df


def _summary_latency_by_shape(
    summary_df: pd.DataFrame,
    latency_col: str,
) -> Dict[str, float]:
    if latency_col not in summary_df.columns:
        return {}
    latency_by_shape: Dict[str, float] = {}
    latencies = pd.to_numeric(summary_df[latency_col], errors="coerce")
    for shape_col in ["summary_shape_b_m_n_k", "shape", "shape_key"]:
        if shape_col not in summary_df.columns:
            continue
        for shape, latency in zip(summary_df[shape_col], latencies):
            if pd.isna(shape) or pd.isna(latency):
                continue
            latency_by_shape[normalize_shape_for_merge(shape)] = float(latency)
    return latency_by_shape


def _map_run_latency_by_shape(
    plot_df: pd.DataFrame,
    latency_by_shape: Dict[str, float],
) -> pd.Series:
    mapped = pd.Series(pd.NA, index=plot_df.index, dtype="Float64")
    for col in ["summary_shape_b_m_n_k", "shape", "shape_key"]:
        if col not in plot_df.columns:
            continue
        keys = plot_df[col].map(
            lambda value: normalize_shape_for_merge(
                value) if pd.notna(value) else None
        )
        col_latency = keys.map(latency_by_shape)
        mapped = mapped.where(mapped.notna(), col_latency)
    return mapped


def merge_summary_latency_columns(
    plot_df: pd.DataFrame,
    summary_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if summary_df is None or summary_df.empty:
        return plot_df

    out = plot_df.copy()
    default_latency_by_shape = _summary_latency_by_shape(
        summary_df, DEFAULT_LATENCY_COL
    )
    expand_latency_by_shape = _summary_latency_by_shape(
        summary_df, "summary_expand_gems_latency_ms"
    )
    if default_latency_by_shape:
        out[DEFAULT_LATENCY_COL] = _map_run_latency_by_shape(
            out, default_latency_by_shape
        )
    if expand_latency_by_shape:
        out[BENCHMARK_LATENCY_COL] = _map_run_latency_by_shape(
            out, expand_latency_by_shape
        )
    return out


def merge_topk_plot_rows(plot_df: pd.DataFrame, topk_plot_df: pd.DataFrame) -> pd.DataFrame:
    rows = plot_df.copy().to_dict("records")
    key_to_index: Dict[str, int] = {}
    for idx, row in enumerate(rows):
        key = plot_row_merge_key(row)
        if key is not None and key not in key_to_index:
            key_to_index[key] = idx

    for topk_row in topk_plot_df.copy().to_dict("records"):
        key = plot_row_merge_key(topk_row)
        if key is None or key not in key_to_index:
            if key is not None:
                key_to_index[key] = len(rows)
            rows.append(topk_row)
            continue

        row = rows[key_to_index[key]]
        for col, value in topk_row.items():
            if is_missing_value(value):
                continue
            if col == RUN_LATENCY_COL:
                row[col] = value
            elif is_missing_value(row.get(col)):
                row[col] = value

    return pd.DataFrame(rows)


def merge_latency_into_plot_outputs(
    rows: Iterable[Dict[str, Any]],
    out_dir: Path,
) -> List[Path]:
    latency_by_shape = _collect_run_latency_by_shape(rows)
    if not latency_by_shape:
        return []

    plot_csv = out_dir / "speedup_plot_data.csv"
    if not plot_csv.exists():
        return []

    plot_df = merge_summary_latency_columns(
        pd.read_csv(plot_csv),
        load_summary_dataframe(
            out_dir,
            model=None,
            op="mm",
            summary_md=None,
            dtype="bfloat16",
        ),
    )

    run_latency = _map_run_latency_by_shape(plot_df, latency_by_shape)

    old_run_latency = (
        pd.to_numeric(plot_df[RUN_LATENCY_COL], errors="coerce")
        if RUN_LATENCY_COL in plot_df.columns
        else None
    )
    if old_run_latency is not None:
        plot_df[RUN_LATENCY_COL] = run_latency.where(
            run_latency.notna(), old_run_latency
        )
    else:
        plot_df[RUN_LATENCY_COL] = run_latency

    compact_plot_dataframe(plot_df).to_csv(plot_csv, index=False)
    return [plot_csv]


def draw_plots_from_out_dir(
    out_dir: Path,
    plot_format: Optional[str] = None,
    plot_dpi: Optional[int] = None,
    sort_by: Optional[str] = None,
    topk_annotate: Optional[int] = None,
    model: Optional[str] = None,
    op: str = "mm",
    summary_md: Optional[str] = None,
    summary_dtype: str = "bfloat16",
) -> List[Path]:
    _ = (plot_format, plot_dpi, sort_by, topk_annotate)
    summary_df = load_summary_dataframe(
        out_dir,
        model=model,
        op=op,
        summary_md=summary_md,
        dtype=summary_dtype,
    )

    plot_csv = out_dir / "speedup_plot_data.csv"
    if not plot_csv.exists():
        plot_df = _build_plot_data_from_topk_run_latency(out_dir)
        if plot_df is None:
            plot_df = _build_plot_data_from_summary(summary_df)
        if plot_df is None:
            raise RuntimeError(f"Missing plot data csv: {plot_csv}")
        plot_df = compact_plot_dataframe(
            merge_summary_latency_columns(plot_df, summary_df)
        )
        plot_df.to_csv(plot_csv, index=False)
        print(f"[INFO] created plot data: {plot_csv}")
        return [plot_csv]

    plot_df = merge_summary_latency_columns(pd.read_csv(plot_csv), summary_df)

    topk_plot_df = _build_plot_data_from_topk_run_latency(out_dir)
    if topk_plot_df is not None:
        topk_plot_df = merge_summary_latency_columns(topk_plot_df, summary_df)
        before_count = len(plot_df)
        plot_df = merge_topk_plot_rows(plot_df, topk_plot_df)
        topk_latency_by_shape = _collect_topk_run_latency_by_shape(out_dir)
        topk_run_latency = _map_run_latency_by_shape(plot_df, topk_latency_by_shape)
        mapped_count = int(topk_run_latency.notna().sum())
        print(
            f"[INFO] matched run latency to {mapped_count}/{len(plot_df)} "
            "speedup plot rows"
        )
        if len(plot_df) != before_count:
            print(
                f"[INFO] expanded speedup plot rows from {before_count} "
                f"to {len(plot_df)} using top-k latency outputs"
            )
        old_run_latency = (
            pd.to_numeric(plot_df[RUN_LATENCY_COL], errors="coerce")
            if RUN_LATENCY_COL in plot_df.columns
            else None
        )
        if old_run_latency is not None:
            plot_df[RUN_LATENCY_COL] = topk_run_latency.where(
                topk_run_latency.notna(), old_run_latency
            )
        else:
            plot_df[RUN_LATENCY_COL] = topk_run_latency

    plot_df = compact_plot_dataframe(plot_df)
    plot_df.to_csv(plot_csv, index=False)

    return [plot_csv]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update mm speedup plot data from run latency outputs.")
    parser.add_argument("--out-dir", default="./mm_xgb_outputs")
    parser.add_argument("--plot-format", default=None,
                        choices=["png", "pdf", "svg"])
    parser.add_argument("--plot-dpi", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--op", default="mm")
    parser.add_argument("--summary-md", default=None)
    parser.add_argument("--summary-dtype", default="bfloat16")
    parser.add_argument(
        "--plot-sort-by",
        default=None,
        choices=["oracle", "predicted", "gap", "shape_index"],
    )
    parser.add_argument("--plot-topk-annotate", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = draw_plots_from_out_dir(
        out_dir=Path(args.out_dir),
        plot_format=args.plot_format,
        plot_dpi=args.plot_dpi,
        sort_by=args.plot_sort_by,
        topk_annotate=args.plot_topk_annotate,
        model=args.model,
        op=args.op,
        summary_md=args.summary_md,
        summary_dtype=args.summary_dtype,
    )
    for path in written:
        print(f"[DONE] plot data: {path}")


if __name__ == "__main__":
    main()
