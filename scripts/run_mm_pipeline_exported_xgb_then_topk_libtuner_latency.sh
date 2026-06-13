#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  ./run_mm_pipeline_exported_xgb_then_topk_libtuner_latency.sh --export-model-dir /path/to/exported_xgb_model [args]

Example:
  bash FlagTune/scripts/run_mm_exported_xgb_then_topk_libtuner_latency.sh \
    --export-model-dir /home/secure/autotune/FlagGems/FlagTune/mm_xgb_outputs/xgboost_model_shape50_config100_top10 \
    --predict-model Qwen3.5-35B-A3B-p32768d1024 \
    --train-mode shape50_config100 \
    --top-k 10 \
    --warmup 1000 \
    --rep 100 \
    --parallel 8 \
    --gpus 0,1,2,3,4,5,6,7

This script skips XGBoost training. It runs the new FlagTune.mm_pipeline module entry points:
  1. FlagTune.mm_pipeline.cli.predict_topk from an exported model directory
  2. FlagTune.mm_pipeline.cli.run_topk_latency for each generated top-k YAML
  3. FlagTune.mm_pipeline.cli.speedup_plot for each model output directory

Required:
  --export-model-dir, --model-dir
    Directory containing xgboost_ranker.json and feature_schema.json.

Prediction targets:
  --predict-model, --model
    Can be passed multiple times. If omitted, every *.txt file under
    --shape-config-dir is used as a model name.

Global options:
  --predict-model, --model, --op, --summary-md, --summary-dtype,
  --out-dir, --export-model-dir, --model-dir, --shape-config-dir,
  --topk-yaml-dir, --train-mode, --top-k, --warmup, --rep

Run-topk options can be passed directly, for example:
  --parallel 8 --gpus 0,1,2,3,4,5,6,7 --ga-generations 0
  --start-shape 0 --limit-shapes 10 --top-cost-shapes 30
EOF
}

is_run_value_opt() {
  case "$1" in
    --input|--output-csv|--output-yaml|--ga-selected-output-csv|--ga-selected-output-yaml|\
    --benchmark-hit-output-csv|--benchmark-hit-output-xlsx|\
    --device|--dtype|--warmup|--rep|--mode|\
    --start|--limit|--start-shape|--limit-shapes|--top-cost-shapes|--max-configs-per-shape|\
    --seed|--ga-generations|--ga-population-size|--ga-elite-size|\
    --ga-offspring-per-generation|--ga-mutation-rate|--ga-random-rate|\
    --ga-max-evaluations-per-shape|--override-dir|--progress-every|--parallel|--gpus|--visible-devices-env|\
    --balance-latency-csv|--balance-latency-col)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_run_flag_opt() {
  case "$1" in
    --no-compile-once|--debug-libtuner|--show-child-output|--keep-override-files|\
    --fail-fast|--no-report-latency-balance|--no-print-runtime-configs)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_global_value_opt() {
  case "$1" in
    --predict-model|--model|--op|--summary-md|--summary-dtype|\
    --out-dir|--export-model-dir|--model-dir|--shape-config-dir|\
    --topk-yaml-dir|--train-mode|--top-k|--warmup|--rep)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

append_run_opt() {
  local opt="$1"
  local value="${2:-}"
  case "$opt" in
    --start)
      USER_RUN_ARGS+=(--start-shape "$value")
      ;;
    --limit)
      USER_RUN_ARGS+=(--limit-shapes "$value")
      ;;
    *)
      USER_RUN_ARGS+=("$opt")
      if [[ $# -ge 2 ]]; then
        USER_RUN_ARGS+=("$value")
      fi
      ;;
  esac
}

format_duration() {
  local total="$1"
  local hours=$((total / 3600))
  local minutes=$(((total % 3600) / 60))
  local seconds=$((total % 60))

  if ((hours > 0)); then
    printf "%dh%02dm%02ds" "$hours" "$minutes" "$seconds"
  elif ((minutes > 0)); then
    printf "%dm%02ds" "$minutes" "$seconds"
  else
    printf "%ds" "$seconds"
  fi
}

run_timed() {
  local name="$1"
  shift

  local start_ts end_ts elapsed status
  echo "[TIME] $name start: $(date '+%F %T')"
  start_ts="$(date +%s)"

  set +e
  "$@"
  status="$?"
  set -e

  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))

  if ((status == 0)); then
    echo "[TIME] $name done: $(format_duration "$elapsed") (${elapsed}s)"
  else
    echo "[TIME] $name failed after $(format_duration "$elapsed") (${elapsed}s), exit_code=$status" >&2
  fi
  return "$status"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAGTUNE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$FLAGTUNE_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
CPU_ONLY_ENV=(
  CUDA_VISIBLE_DEVICES=
  MUSA_VISIBLE_DEVICES=
  HIP_VISIBLE_DEVICES=
  ROCR_VISIBLE_DEVICES=
)

PREDICT_MODELS=()
OP="${OP:-mm}"
SUMMARY_MD="${SUMMARY_MD:-}"
SUMMARY_DTYPE="${SUMMARY_DTYPE:-bfloat16}"
OUT_DIR="${OUT_DIR:-mm_xgb_outputs}"
EXPORT_MODEL_DIR="${EXPORT_MODEL_DIR:-}"
SHAPE_CONFIG_DIR="${SHAPE_CONFIG_DIR:-shape-config}"
TOPK_YAML_DIR="${TOPK_YAML_DIR:-}"
TRAIN_MODE="${TRAIN_MODE:-}"
TOP_K="${TOP_K:-10}"
WARMUP="${WARMUP:-1000}"
REP="${REP:-100}"
USER_RUN_ARGS=()
mode="auto"

resolve_output_root() {
  local dir="$1"
  if [[ "$dir" = /* ]]; then
    printf "%s\n" "$dir"
  else
    printf "%s\n" "$FLAGTUNE_DIR/$dir"
  fi
}

safe_path_component() {
  local value="$1"
  value="${value//\//_}"
  value="${value// /_}"
  value="$(printf "%s" "$value" | tr -c '[:alnum:]._=-' '_')"
  printf "%s\n" "${value:-unknown}"
}

infer_train_mode_from_model_dir() {
  local model_dir="$1"
  local base
  base="$(basename "$model_dir")"
  if [[ "$base" =~ ^xgboost_model_(.*)_top[0-9]+$ ]]; then
    printf "%s\n" "${BASH_REMATCH[1]}"
  elif [[ "$base" =~ _(shape[0-9]+_config[0-9]+)_top[0-9]+$ ]]; then
    printf "%s\n" "${BASH_REMATCH[1]}"
  else
    printf "%s\n" "saved_model"
  fi
}

db_output_dir() {
  local model_name="$1"
  local safe_model safe_train_mode
  safe_model="$(safe_path_component "$model_name")"
  safe_train_mode="$(safe_path_component "$TRAIN_MODE")"
  printf "%s\n" "$OUT_DIR/${safe_model}_${safe_train_mode}_top${TOP_K}"
}

list_shape_config_models() {
  local shape_dir="$1"
  local path base

  if [[ ! -d "$shape_dir" ]]; then
    echo "[ERROR] Shape config dir not found: $shape_dir" >&2
    return 1
  fi

  find "$shape_dir" -maxdepth 1 -type f -name "*.txt" | sort | while read -r path; do
    base="$(basename "$path")"
    printf "%s\n" "${base%.txt}"
  done | awk 'NF && !seen[$0]++'
}

while (($#)); do
  arg="$1"
  case "$arg" in
    -h|--help)
      show_help
      exit 0
      ;;
    --run|--latency|--topk-run)
      mode="run"
      shift
      continue
      ;;
  esac

  opt="${arg%%=*}"
  needs_value=0
  dest="global"
  if is_global_value_opt "$opt"; then
    needs_value=1
  elif [[ "$mode" == "run" ]] || is_run_value_opt "$opt" || is_run_flag_opt "$opt"; then
    dest="run"
    if is_run_value_opt "$opt"; then
      needs_value=1
    fi
  else
    echo "Unknown argument '$arg'. Use --run to pass run_topk-specific options explicitly." >&2
    exit 2
  fi

  value=""
  if [[ "$arg" == *=* ]]; then
    value="${arg#*=}"
  elif ((needs_value)); then
    shift
    if (($# == 0)); then
      echo "Missing value for '$arg'" >&2
      exit 2
    fi
    value="$1"
  fi

  if [[ "$dest" == "run" ]]; then
    if ((needs_value)); then
      append_run_opt "$opt" "$value"
    else
      append_run_opt "$opt"
    fi
    shift
    continue
  fi

  case "$opt" in
    --predict-model|--model)
      PREDICT_MODELS+=("$value")
      ;;
    --op)
      OP="$value"
      ;;
    --summary-md)
      SUMMARY_MD="$value"
      ;;
    --summary-dtype)
      SUMMARY_DTYPE="$value"
      ;;
    --out-dir)
      OUT_DIR="$value"
      ;;
    --export-model-dir|--model-dir)
      EXPORT_MODEL_DIR="$value"
      ;;
    --shape-config-dir)
      SHAPE_CONFIG_DIR="$value"
      ;;
    --topk-yaml-dir)
      TOPK_YAML_DIR="$value"
      ;;
    --train-mode)
      TRAIN_MODE="$value"
      ;;
    --top-k)
      TOP_K="$value"
      ;;
    --warmup)
      WARMUP="$value"
      ;;
    --rep)
      REP="$value"
      ;;
  esac

  shift
done

OUT_DIR="$(resolve_output_root "$OUT_DIR")"
SHAPE_CONFIG_DIR="$(resolve_output_root "$SHAPE_CONFIG_DIR")"
if [[ -z "$EXPORT_MODEL_DIR" ]]; then
  echo "[ERROR] --export-model-dir is required." >&2
  exit 1
fi
EXPORT_MODEL_DIR="$(resolve_output_root "$EXPORT_MODEL_DIR")"
if [[ ! -f "$EXPORT_MODEL_DIR/xgboost_ranker.json" || ! -f "$EXPORT_MODEL_DIR/feature_schema.json" ]]; then
  echo "[ERROR] Exported model dir must contain xgboost_ranker.json and feature_schema.json: $EXPORT_MODEL_DIR" >&2
  exit 1
fi
if [[ -z "$TRAIN_MODE" ]]; then
  TRAIN_MODE="$(infer_train_mode_from_model_dir "$EXPORT_MODEL_DIR")"
fi

PREDICT_TARGET_MODELS=()
if ((${#PREDICT_MODELS[@]} == 0)); then
  mapfile -t PREDICT_TARGET_MODELS < <(list_shape_config_models "$SHAPE_CONFIG_DIR")
  if ((${#PREDICT_TARGET_MODELS[@]} == 0)); then
    echo "[ERROR] No --predict-model was passed and no *.txt models were found under $SHAPE_CONFIG_DIR." >&2
    exit 1
  fi
  echo "[INFO] No --predict-model passed; using all ${#PREDICT_TARGET_MODELS[@]} models from $SHAPE_CONFIG_DIR."
else
  mapfile -t PREDICT_TARGET_MODELS < <(printf "%s\n" "${PREDICT_MODELS[@]}" | awk 'NF && !seen[$0]++')
fi

safe_train_mode="$(safe_path_component "$TRAIN_MODE")"
if [[ -z "$TOPK_YAML_DIR" ]]; then
  TOPK_YAML_DIR="$OUT_DIR/predicted_topk_${safe_train_mode}_top${TOP_K}"
else
  TOPK_YAML_DIR="$(resolve_output_root "$TOPK_YAML_DIR")"
fi

mkdir -p "$OUT_DIR" "$TOPK_YAML_DIR"

echo "[INFO] Output root: $OUT_DIR"
echo "[INFO] Exported XGBoost model dir: $EXPORT_MODEL_DIR"
echo "[INFO] Predict models: ${#PREDICT_TARGET_MODELS[@]}"
echo "[INFO] Shape config dir: $SHAPE_CONFIG_DIR"
echo "[INFO] Predicted top-k YAML dir: $TOPK_YAML_DIR"
echo "[INFO] Train mode label: $TRAIN_MODE"
echo "[INFO] Top-k: $TOP_K"
echo "[INFO] Op: $OP"

predict_cmd=(
  "$PYTHON_BIN" -m FlagTune.mm_pipeline.cli.predict_topk
  --model-dir "$EXPORT_MODEL_DIR"
  --shape-config-dir "$SHAPE_CONFIG_DIR"
  --out-dir "$TOPK_YAML_DIR"
  --top-k "$TOP_K"
  --dtype "$SUMMARY_DTYPE"
  --op "$OP"
)
for predict_model in "${PREDICT_TARGET_MODELS[@]}"; do
  predict_cmd+=(--model "$predict_model")
done

run_timed "FlagTune.mm_pipeline.cli.predict_topk [${#PREDICT_TARGET_MODELS[@]} models]" env "${CPU_ONLY_ENV[@]}" "${predict_cmd[@]}"

for idx in "${!PREDICT_TARGET_MODELS[@]}"; do
  model_name="${PREDICT_TARGET_MODELS[$idx]}"
  safe_model="$(safe_path_component "$model_name")"
  run_out_dir="$(db_output_dir "$model_name")"
  mkdir -p "$run_out_dir"

  echo "[INFO] Predict model: $model_name"
  echo "[INFO] Train mode label: $TRAIN_MODE"
  echo "[INFO] Top-k: $TOP_K"
  echo "[INFO] Top-k YAML dir: $TOPK_YAML_DIR"
  echo "[INFO] Output dir: $run_out_dir"

  topk_yaml="$TOPK_YAML_DIR/${safe_model}_predicted_shape_configs_top${TOP_K}.yaml"
  if [[ ! -f "$topk_yaml" ]]; then
    echo "[ERROR] Missing predicted top-k YAML for $model_name: $topk_yaml" >&2
    echo "[ERROR] Existing predicted top-k YAML files under $TOPK_YAML_DIR:" >&2
    find "$TOPK_YAML_DIR" -maxdepth 1 -type f -name "*predicted_shape_configs_top${TOP_K}.yaml" | sort >&2
    exit 1
  fi

  topk_latency_csv="$run_out_dir/predicted_shape_configs_top${TOP_K}_libtuner_latency.csv"
  topk_latency_yaml="$run_out_dir/predicted_shape_configs_top${TOP_K}_libtuner_latency.yaml"
  ga_selected_csv="$run_out_dir/predicted_shape_configs_top${TOP_K}_ga_selected_configs.csv"
  ga_selected_yaml="$run_out_dir/predicted_shape_configs_top${TOP_K}_ga_selected_configs.yaml"
  benchmark_hit_csv="$run_out_dir/top${TOP_K}_benchmark_best_hit_by_shape.csv"
  benchmark_hit_xlsx="$run_out_dir/top${TOP_K}_benchmark_best_hit_by_shape.xlsx"

  run_timed "FlagTune.mm_pipeline.cli.run_topk_latency top${TOP_K} [$model_name]" "$PYTHON_BIN" -m FlagTune.mm_pipeline.cli.run_topk_latency \
    --input "$topk_yaml" \
    --output-csv "$topk_latency_csv" \
    --output-yaml "$topk_latency_yaml" \
    --ga-selected-output-csv "$ga_selected_csv" \
    --ga-selected-output-yaml "$ga_selected_yaml" \
    --benchmark-hit-output-csv "$benchmark_hit_csv" \
    --benchmark-hit-output-xlsx "$benchmark_hit_xlsx" \
    --warmup "$WARMUP" \
    --rep "$REP" \
    "${USER_RUN_ARGS[@]}"

  if [[ ! -f "$topk_latency_csv" || ! -f "$topk_latency_yaml" ]]; then
    echo "[ERROR] Expected top${TOP_K} LibTuner latency outputs were not generated:" >&2
    echo "  csv: $topk_latency_csv" >&2
    echo "  yaml: $topk_latency_yaml" >&2
    exit 1
  fi

  echo "[INFO] top${TOP_K} LibTuner latency csv: $topk_latency_csv"
  echo "[INFO] top${TOP_K} LibTuner latency yaml: $topk_latency_yaml"

  speedup_cmd=(
    "$PYTHON_BIN" -m FlagTune.mm_pipeline.cli.speedup_plot
    --out-dir "$run_out_dir"
    --model "$model_name"
    --op "$OP"
    --summary-dtype "$SUMMARY_DTYPE"
  )
  if [[ -n "$SUMMARY_MD" ]]; then
    speedup_cmd+=(--summary-md "$SUMMARY_MD")
  fi
  run_timed "FlagTune.mm_pipeline.cli.speedup_plot [$model_name]" "${speedup_cmd[@]}"
done
