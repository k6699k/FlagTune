#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  ./run_mm_xgb_then_topk_libtuner_latency.sh [args]
bash run_mm_xgb_then_topk_libtuner_latency.sh \
  --train-db /home/secure/.flaggems/done/Qwen3.5-35B-A3B-p32768d1024_TunedConfig_NVIDIA_H800_triton_3_6.db \
  --predict-model Qwen3.5-35B-A3B-p32768d1024 \
  --train-mode shape50_config50 \
  --top-k 10 \
  --warmup 1000 \
  --rep 100 \
  --parallel 8 \
  --gpus 0,1,2,3,4,5,6,7 \
  --debug-libtuner \
  --show-child-output
The script runs these commands in order:
  1. mm_xgboost_from_benchmark_cache.py --export-only
  2. mm_xgboost_predict_shape_configs.py, one top-k YAML per --predict-model
  3. run_mm_topk_configs_libtuner_latency.py for each generated top-k YAML

By default, outputs are written under:
  ../mm_xgb_outputs/

Training uses one explicit DB file:
  --train-db points to the benchmark DB used to train and export one XGBoost ranker.
  The DB may contain default-stage tables; only expand-stage benchmark rows are used.
  Directories are not scanned for DB files.
  --predict-model can be passed multiple times. If omitted, all models under
  FlagTune/shape-config are predicted. Each model's shapes are read from
  shape-config by default, and all generated YAMLs are written into one folder.

Generated top-k YAML folder:
  ../mm_xgb_outputs/predicted_topk_<train-mode>_top<TOP_K>/

Most arguments are auto-routed by their original Python option names:
  ./run_mm_xgb_then_topk_libtuner_latency.sh --db /path/model_TunedConfig.db --train-mode shape50_config100 --expand-count 480 --top-k 10 --warmup 1000 --rep 100 --parallel 8 --gpus 0,1,2,3,4,5,6,7

If an option name is ambiguous, or you want to be explicit, split args by target:
  ./run_mm_xgb_then_topk_libtuner_latency.sh --xgb --seed 2026 --n-jobs 8 --run --seed 7 --parallel 8

Auto-routed shared option:
  --seed is passed to XGBoost training and run_topk. Use --xgb/--run to pass it to only one.

Global orchestration options:
  --db, --train-db, --predict-model, --model, --op,
  --summary-dtype, --out-dir, --export-model-dir, --shape-config-dir,
  --topk-yaml-dir, --expand-count, --top-k, --warmup, --rep

Model name:
  Use --predict-model, or legacy --model, to choose prediction target names.
  If omitted, every model found under --shape-config-dir is predicted.
  Default all-model discovery only scans:
    FlagTune/shape-config/<model>.txt
  Explicit --predict-model still looks for shape-config files named:
    FlagTune/shape-config/<model>.txt
    FlagTune/shape-config/<model>_<op>.yaml
    FlagTune/shape-config/<model>.yaml

run_mm_topk_configs_libtuner_latency.py examples:
  --parallel 8 --gpus 0,1,2,3,4,5,6,7 --debug-libtuner --show-child-output
  --start-shape 0 --limit-shapes 10 --top-cost-shapes 30 --max-configs-per-shape 5
  --balance-latency-csv ../mm_xgb_outputs/model_mode/speedup_plot_data.csv
EOF
}

is_xgb_value_opt() {
  case "$1" in
    --db|--train-db|--summary-md|--summary-dtype|--out-dir|--benchmark-table|\
    --benchmark-table-like|--kernel-substr|--expand-count|--gemv-expand-count|--target|\
    --train-ratio|--train-mode|--shape-train-ratio|--config-train-ratio|\
    --top-k|\
    --seed|--min-train-rows|--n-estimators|--max-depth|--learning-rate|\
    --subsample|--colsample-bytree|--reg-lambda|--reg-alpha|\
    --min-child-weight|--gamma|--max-bin|--n-jobs|--plot-sort-by|\
    --plot-format|--plot-dpi|--plot-topk-annotate|--export-model-dir)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_xgb_flag_opt() {
  case "$1" in
    --make-plots)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
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
    --db|--train-db|--predict-model|--model|--op|--summary-md|--summary-dtype|\
    --out-dir|--export-model-dir|--shape-config-dir|--topk-yaml-dir|\
    --expand-count|--top-k|--warmup|--rep)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

append_dest() {
  local dest="$1"
  local item="$2"
  case "$dest" in
    xgb)
      USER_XGB_ARGS+=("$item")
      ;;
    run)
      USER_RUN_ARGS+=("$item")
      ;;
    both)
      USER_XGB_ARGS+=("$item")
      USER_RUN_ARGS+=("$item")
      ;;
    *)
      echo "Internal error: unknown destination '$dest'" >&2
      exit 2
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

append_dest_opt() {
  local dest="$1"
  local opt="$2"
  local has_value="$3"
  local value="${4:-}"

  if [[ "$dest" == "run" ]]; then
    if ((has_value)); then
      append_run_opt "$opt" "$value"
    else
      append_run_opt "$opt"
    fi
    return
  fi

  append_dest "$dest" "$opt"
  if ((has_value)); then
    append_dest "$dest" "$value"
  fi
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
PROCESSING_DIR="$FLAGTUNE_DIR/processing"
cd "$PROCESSING_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
CPU_ONLY_ENV=(
  CUDA_VISIBLE_DEVICES=
  MUSA_VISIBLE_DEVICES=
  HIP_VISIBLE_DEVICES=
  ROCR_VISIBLE_DEVICES=
)
DB="${DB:-/home/secure/.flaggems/done}"
TRAIN_DB_INPUTS=()
PREDICT_MODELS=()
OP="${OP:-mm}"
SUMMARY_MD="${SUMMARY_MD:-}"
SUMMARY_DTYPE="${SUMMARY_DTYPE:-bfloat16}"
OUT_DIR="${OUT_DIR:-mm_xgb_outputs}"
EXPORT_MODEL_DIR="${EXPORT_MODEL_DIR:-}"
SHAPE_CONFIG_DIR="${SHAPE_CONFIG_DIR:-shape-config}"
TOPK_YAML_DIR="${TOPK_YAML_DIR:-}"
EXPAND_COUNT="${EXPAND_COUNT:-480}"
TRAIN_MODE="${TRAIN_MODE:-shape100_config100}"
TOP_K="${TOP_K:-10}"
WARMUP="${WARMUP:-1000}"
REP="${REP:-100}"

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

db_output_dir() {
  local model_name="$1"
  local safe_model safe_train_mode
  safe_model="$(safe_path_component "$model_name")"
  safe_train_mode="$(safe_path_component "$TRAIN_MODE")"
  printf "%s\n" "$OUT_DIR/${safe_model}_${safe_train_mode}_top${TOP_K}"
}

list_shape_config_models() {
  local shape_dir="$1"
  local op="$2"
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

USER_XGB_ARGS=()
USER_RUN_ARGS=()
mode="auto"

while (($#)); do
  arg="$1"

  case "$arg" in
    -h|--help)
      show_help
      exit 0
      ;;
    --xgb)
      mode="xgb"
      shift
      continue
      ;;
    --run|--latency|--topk-run)
      mode="run"
      shift
      continue
      ;;
  esac

  opt="${arg%%=*}"
  dest=""
  needs_value=0

  if is_global_value_opt "$opt"; then
    dest="global"
    needs_value=1
  else
    case "$mode" in
      xgb)
        dest="xgb"
        if is_xgb_value_opt "$opt"; then
          needs_value=1
        fi
        ;;
      run)
        dest="run"
        if is_run_value_opt "$opt"; then
          needs_value=1
        fi
        ;;
      auto)
        if is_xgb_value_opt "$opt" && is_run_value_opt "$opt"; then
          dest="both"
          needs_value=1
        elif is_xgb_value_opt "$opt"; then
          dest="xgb"
          needs_value=1
        elif is_run_value_opt "$opt"; then
          dest="run"
          needs_value=1
        elif is_xgb_flag_opt "$opt"; then
          dest="xgb"
        elif is_run_flag_opt "$opt"; then
          dest="run"
        else
          echo "Unknown argument '$arg'. Use --xgb or --run to pass it explicitly." >&2
          exit 2
        fi
        ;;
      *)
        echo "Internal error: unknown mode '$mode'" >&2
        exit 2
        ;;
    esac
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

  if [[ "$opt" == "--train-mode" ]]; then
    TRAIN_MODE="$value"
  fi
  if [[ "$opt" == "--top-k" ]]; then
    TOP_K="$value"
  fi

  managed=0
  case "$opt" in
    --db)
      DB="$value"
      managed=1
      ;;
    --train-db)
      TRAIN_DB_INPUTS+=("$value")
      managed=1
      ;;
    --predict-model)
      PREDICT_MODELS+=("$value")
      managed=1
      ;;
    --model)
      PREDICT_MODELS+=("$value")
      managed=1
      ;;
    --op)
      OP="$value"
      managed=1
      ;;
    --summary-md)
      SUMMARY_MD="$value"
      managed=1
      ;;
    --summary-dtype)
      SUMMARY_DTYPE="$value"
      managed=1
      ;;
    --out-dir)
      OUT_DIR="$value"
      managed=1
      ;;
    --export-model-dir)
      EXPORT_MODEL_DIR="$value"
      managed=1
      ;;
    --shape-config-dir)
      SHAPE_CONFIG_DIR="$value"
      managed=1
      ;;
    --topk-yaml-dir)
      TOPK_YAML_DIR="$value"
      managed=1
      ;;
    --expand-count)
      EXPAND_COUNT="$value"
      managed=1
      ;;
    --top-k)
      TOP_K="$value"
      managed=1
      ;;
    --warmup)
      WARMUP="$value"
      managed=1
      ;;
    --rep)
      REP="$value"
      managed=1
      ;;
  esac

  if ((managed == 0)); then
    if ((needs_value)); then
      append_dest_opt "$dest" "$opt" 1 "$value"
    else
      append_dest_opt "$dest" "$opt" 0
    fi
  fi

  shift
done

OUT_DIR="$(resolve_output_root "$OUT_DIR")"
mkdir -p "$OUT_DIR"

if ((${#TRAIN_DB_INPUTS[@]} == 0)); then
  TRAIN_DB_INPUTS=("$DB")
fi

if ((${#TRAIN_DB_INPUTS[@]} != 1)); then
  echo "[ERROR] Pass exactly one --train-db, or pass one DB file through --db." >&2
  exit 1
fi
TRAIN_DB_FILE="${TRAIN_DB_INPUTS[0]}"
if [[ ! -f "$TRAIN_DB_FILE" ]]; then
  echo "[ERROR] Training DB must be a file, not a directory or missing path: $TRAIN_DB_FILE" >&2
  exit 1
fi

SHAPE_CONFIG_DIR="$(resolve_output_root "$SHAPE_CONFIG_DIR")"

PREDICT_TARGET_MODELS=()
if ((${#PREDICT_MODELS[@]} == 0)); then
  mapfile -t PREDICT_TARGET_MODELS < <(list_shape_config_models "$SHAPE_CONFIG_DIR" "$OP")
  if ((${#PREDICT_TARGET_MODELS[@]} == 0)); then
    echo "[ERROR] No --predict-model was passed and no models were found under $SHAPE_CONFIG_DIR." >&2
    exit 1
  fi
  echo "[INFO] No --predict-model passed; using all ${#PREDICT_TARGET_MODELS[@]} models from $SHAPE_CONFIG_DIR."
else
  mapfile -t PREDICT_TARGET_MODELS < <(printf "%s\n" "${PREDICT_MODELS[@]}" | awk 'NF && !seen[$0]++')
fi

safe_train_mode="$(safe_path_component "$TRAIN_MODE")"
if [[ -z "$EXPORT_MODEL_DIR" ]]; then
  EXPORT_MODEL_DIR="$OUT_DIR/xgboost_model_${safe_train_mode}_top${TOP_K}"
else
  EXPORT_MODEL_DIR="$(resolve_output_root "$EXPORT_MODEL_DIR")"
fi
if [[ -z "$TOPK_YAML_DIR" ]]; then
  TOPK_YAML_DIR="$OUT_DIR/predicted_topk_${safe_train_mode}_top${TOP_K}"
else
  TOPK_YAML_DIR="$(resolve_output_root "$TOPK_YAML_DIR")"
fi

mkdir -p "$EXPORT_MODEL_DIR" "$TOPK_YAML_DIR"

echo "[INFO] Output root: $OUT_DIR"
echo "[INFO] Train DB: $TRAIN_DB_FILE"
echo "[INFO] Predict models: ${#PREDICT_TARGET_MODELS[@]}"
echo "[INFO] Shape config dir: $SHAPE_CONFIG_DIR"
echo "[INFO] XGBoost model export dir: $EXPORT_MODEL_DIR"
echo "[INFO] Predicted top-k YAML dir: $TOPK_YAML_DIR"
echo "[INFO] Top-k: $TOP_K"
echo "[INFO] Op: $OP"

xgb_cmd=(
  "$PYTHON_BIN" mm_xgboost_from_benchmark_cache.py
  --db "$TRAIN_DB_FILE"
  --summary-dtype "$SUMMARY_DTYPE"
  --out-dir "$EXPORT_MODEL_DIR"
  --op "$OP"
  --expand-count "$EXPAND_COUNT"
  --train-mode "$TRAIN_MODE"
  --top-k "$TOP_K"
  --export-model-dir "$EXPORT_MODEL_DIR"
  --export-only
  --train-db "$TRAIN_DB_FILE"
)
if [[ -n "$SUMMARY_MD" ]]; then
  echo "[WARN] --summary-md is ignored in export-only XGBoost training."
fi
xgb_cmd+=("${USER_XGB_ARGS[@]}")

run_timed "mm_xgboost_from_benchmark_cache.py export model" env "${CPU_ONLY_ENV[@]}" "${xgb_cmd[@]}"

predict_cmd=(
  "$PYTHON_BIN" mm_xgboost_predict_shape_configs.py
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

run_timed "mm_xgboost_predict_shape_configs.py [${#PREDICT_TARGET_MODELS[@]} models]" env "${CPU_ONLY_ENV[@]}" "${predict_cmd[@]}"

find "$TOPK_YAML_DIR" -maxdepth 1 -type f \( \
  -name "*_ga_selected_configs.csv" -o \
  -name "*_ga_selected_configs.yaml" -o \
  -name "*_ga_selected_configs.yml" \
\) -exec rm -f {} +

for idx in "${!PREDICT_TARGET_MODELS[@]}"; do
  model_name="${PREDICT_TARGET_MODELS[$idx]}"
  safe_model="$(safe_path_component "$model_name")"
  run_out_dir="$(db_output_dir "$model_name")"

  mkdir -p "$run_out_dir"

  echo "[INFO] Predict model: $model_name"
  echo "[INFO] Train DB: $TRAIN_DB_FILE"
  echo "[INFO] Train mode: $TRAIN_MODE"
  echo "[INFO] Top-k: $TOP_K"
  echo "[INFO] Top-k YAML dir: $TOPK_YAML_DIR"
  echo "[INFO] Output dir: $run_out_dir"

  topk_yaml="$TOPK_YAML_DIR/${safe_model}_predicted_shape_configs_top${TOP_K}.yaml"
  if [[ -f "$topk_yaml" ]]; then
    topk_latency_csv="$run_out_dir/predicted_shape_configs_top${TOP_K}_libtuner_latency.csv"
    topk_latency_yaml="$run_out_dir/predicted_shape_configs_top${TOP_K}_libtuner_latency.yaml"
    ga_selected_csv="$run_out_dir/predicted_shape_configs_top${TOP_K}_ga_selected_configs.csv"
    ga_selected_yaml="$run_out_dir/predicted_shape_configs_top${TOP_K}_ga_selected_configs.yaml"
    benchmark_hit_csv="$run_out_dir/top${TOP_K}_benchmark_best_hit_by_shape.csv"
    benchmark_hit_xlsx="$run_out_dir/top${TOP_K}_benchmark_best_hit_by_shape.xlsx"
    run_timed "run_mm_topk_configs_libtuner_latency.py top${TOP_K} [$model_name]" "$PYTHON_BIN" run_mm_topk_configs_libtuner_latency.py \
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
      echo "[ERROR] Existing latency/config files under $run_out_dir:" >&2
      find "$run_out_dir" -maxdepth 1 -type f \( -name "*latency*" -o -name "predicted_shape_configs*" -o -name "top*_predicted_config_by_shape*" \) | sort >&2
      exit 1
    fi
    echo "[INFO] top${TOP_K} LibTuner latency csv: $topk_latency_csv"
    echo "[INFO] top${TOP_K} LibTuner latency yaml: $topk_latency_yaml"
    speedup_cmd=(
      "$PYTHON_BIN" mm_speedup_plot.py
      --out-dir "$run_out_dir"
      --model "$model_name"
      --op "$OP"
      --summary-dtype "$SUMMARY_DTYPE"
    )
    if [[ -n "$SUMMARY_MD" ]]; then
      speedup_cmd+=(--summary-md "$SUMMARY_MD")
    fi
    run_timed "mm_speedup_plot.py [$model_name]" "${speedup_cmd[@]}"
  else
    echo "[ERROR] Skip top${TOP_K} LibTuner latency for $model_name: missing $topk_yaml" >&2
    echo "[ERROR] Existing predicted top-k YAML files under $TOPK_YAML_DIR:" >&2
    find "$TOPK_YAML_DIR" -maxdepth 1 -type f -name "*predicted_shape_configs_top${TOP_K}.yaml" | sort >&2
    exit 1
  fi
done
