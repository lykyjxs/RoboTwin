#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-.}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${WORKSPACE_ROOT}/third_party/RoboTwin}"
PI0_ROOT="${PI0_ROOT:-${ROBOTWIN_ROOT}/policy/pi0}"
SHARE_ROOT="${SHARE_ROOT:-.}"
OPENPI_DATA_HOME_DIR="${OPENPI_DATA_HOME_DIR:-${SHARE_ROOT}/checkpoints/openpi}"

VARIANT="${VARIANT:-keypose}"
TASK_NAME="${TASK_NAME:-beat_block_hammer}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
EVAL_STEPS="${EVAL_STEPS:-5000 10000 15000 20000}"
TEST_NUM="${TEST_NUM:-100}"
SEED="${SEED:-0}"
PI0_STEP="${PI0_STEP:-50}"
ADAPTIVE_PI0_STEP="${ADAPTIVE_PI0_STEP:-True}"
OUTSIDE_PI0_STEP="${OUTSIDE_PI0_STEP:-50}"
INSIDE_PI0_STEP="${INSIDE_PI0_STEP:-25}"
RUN_ID="${RUN_ID:-leaderboard_demo_clean_seed${SEED}_test${TEST_NUM}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SHARE_ROOT}/beat_block_hammer_leaderboard_compare/${RUN_ID}/${VARIANT}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/summary.csv}"
MASTER_LOG="${MASTER_LOG:-${LOG_ROOT}/run.log}"

EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-python}"
EVAL_CUROBO_SRC="${EVAL_CUROBO_SRC:-./envs/curobo/src}"
EVAL_PYTHONPATH="${ROBOTWIN_ROOT}:${EVAL_CUROBO_SRC}:${PI0_ROOT}/src:${PI0_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID:-0}"
fi

case "${VARIANT}" in
  no_abs)
    TRAIN_CONFIG_NAME="pi0_base_aloha_robotwin_beat_block_hammer_posthit_keystate_stage3_pred_late_xattn_no_keypose_lora"
    CKPT_CONFIG_DIR="pi0_base_aloha_robotwin_beat_block_hammer_posthit_keystate_stage3_pred_late_xattn_lora"
    CKPT_BASE="keystate_stage3"
    MODEL_NAME="beat_block_hammer_50_posthit_stage3_pred_late_xattn_from_stage2best5000_lora_20000_official"
    ;;
  keypose)
    TRAIN_CONFIG_NAME="pi0_base_aloha_robotwin_beat_block_hammer_posthit_keystate_stage3_pred_late_xattn_lora"
    CKPT_CONFIG_DIR="${TRAIN_CONFIG_NAME}"
    CKPT_BASE="keystate_keypose_stage3"
    MODEL_NAME="beat_block_hammer_50_posthit_stage3_pred_late_xattn_keypose_entry_abs_from_stage2_5k_20k_official"
    ;;
  *)
    echo "VARIANT must be one of: no_abs, keypose" >&2
    exit 2
    ;;
esac

SHARED_CKPT_ROOT="${OPENPI_DATA_HOME_DIR}/openpi-assets/checkpoints/${CKPT_BASE}/${CKPT_CONFIG_DIR}/${MODEL_NAME}"
LOCAL_CKPT_ROOT="${PI0_ROOT}/checkpoints/${TRAIN_CONFIG_NAME}/${MODEL_NAME}"

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}" "${LOCAL_CKPT_ROOT}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "========== RoboTwin official Easy eval =========="
echo "VARIANT=${VARIANT}"
echo "RUN_ID=${RUN_ID}"
echo "ROBOTWIN_ROOT=${ROBOTWIN_ROOT}"
echo "TASK_CONFIG=${TASK_CONFIG}"
echo "TEST_NUM=${TEST_NUM}"
echo "SEED=${SEED}"
echo "EVAL_STEPS=${EVAL_STEPS}"
echo "PI0_STEP=${PI0_STEP}"
echo "ADAPTIVE_PI0_STEP=${ADAPTIVE_PI0_STEP}"
echo "OUTSIDE_PI0_STEP=${OUTSIDE_PI0_STEP}"
echo "INSIDE_PI0_STEP=${INSIDE_PI0_STEP}"
echo "TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "SHARED_CKPT_ROOT=${SHARED_CKPT_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SUMMARY_CSV=${SUMMARY_CSV}"
echo "MASTER_LOG=${MASTER_LOG}"

if [[ ! -x "${EVAL_PYTHON_BIN}" ]]; then
  echo "missing python: ${EVAL_PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${ROBOTWIN_ROOT}/script/eval_policy.py" ]]; then
  echo "missing eval_policy.py under ${ROBOTWIN_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${ROBOTWIN_ROOT}/task_config/${TASK_CONFIG}.yml" ]]; then
  echo "missing task config ${TASK_CONFIG}.yml" >&2
  exit 1
fi

normalize_steps() {
  python - <<PY
raw = """${EVAL_STEPS}""".replace(",", " ").split()
print(" ".join(str(int(x)) for x in raw))
PY
}

ensure_checkpoint_link() {
  local step="$1"
  local shared="${SHARED_CKPT_ROOT}/${step}"
  local link="${LOCAL_CKPT_ROOT}/${step}"
  if [[ ! -d "${shared}/params" || ! -d "${shared}/assets" ]]; then
    echo "missing checkpoint params/assets under ${shared}" >&2
    exit 1
  fi
  if [[ -e "${link}" && ! -L "${link}" ]]; then
    echo "local checkpoint path exists and is not a symlink: ${link}" >&2
    exit 1
  fi
  ln -sfn "${shared}" "${link}"
  echo "[OK] ${link} -> ${shared}"
}

append_summary() {
  local step="$1"
  local ckpt_setting="$2"
  python - "${ROBOTWIN_ROOT}" "${TASK_NAME}" "${TASK_CONFIG}" "${ckpt_setting}" "${SUMMARY_CSV}" "${VARIANT}" "${step}" <<'PY'
import csv
import sys
from pathlib import Path

robotwin_root, task_name, task_config, ckpt_setting, summary_csv, variant, step = sys.argv[1:]
eval_root = Path(robotwin_root) / "eval_result" / task_name / "pi0" / task_config / ckpt_setting
runs = sorted([p for p in eval_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
if not runs:
    raise SystemExit(f"no eval run under {eval_root}")
run = runs[0]
episode_log = run / "episode_log.csv"
rows = []
if episode_log.exists():
    with episode_log.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
success = sum(int(r.get("success", 0)) for r in rows)
total = len(rows)
rate = success / total if total else 0.0
out = Path(summary_csv)
out.parent.mkdir(parents=True, exist_ok=True)
fields = ["variant", "step", "success", "total", "success_rate", "task_config", "ckpt_setting", "run_dir", "episode_log", "result_txt"]
write_header = not out.exists()
with out.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        writer.writeheader()
    writer.writerow({
        "variant": variant,
        "step": int(step),
        "success": success,
        "total": total,
        "success_rate": rate,
        "task_config": task_config,
        "ckpt_setting": ckpt_setting,
        "run_dir": str(run),
        "episode_log": str(episode_log),
        "result_txt": str(run / "_result.txt"),
    })
print(f"[SUMMARY] {variant} step={step} success={success}/{total} rate={rate:.3f}")
print(f"[RUN_DIR] {run}")
PY
}

steps="$(normalize_steps)"
for step in ${steps}; do
  ensure_checkpoint_link "${step}"
done

cd "${ROBOTWIN_ROOT}"
for step in ${steps}; do
  ckpt_setting="${MODEL_NAME}_leaderboard_${TASK_CONFIG}_seed${SEED}_test${TEST_NUM}_${VARIANT}_${RUN_ID}_ckpt${step}"
  step_log="${LOG_ROOT}/eval_step${step}.log"
  echo "========== RUN step=${step} ckpt_setting=${ckpt_setting} =========="
  {
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONWARNINGS=ignore::UserWarning PYTHONPATH="${EVAL_PYTHONPATH}" \
      "${EVAL_PYTHON_BIN}" script/eval_policy.py \
        --config policy/pi0/deploy_policy.yml \
        --overrides \
        --task_name "${TASK_NAME}" \
        --task_config "${TASK_CONFIG}" \
        --train_config_name "${TRAIN_CONFIG_NAME}" \
        --model_name "${MODEL_NAME}" \
        --ckpt_setting "${ckpt_setting}" \
        --checkpoint_id "${step}" \
        --pi0_step "${PI0_STEP}" \
        --adaptive_pi0_step "${ADAPTIVE_PI0_STEP}" \
        --outside_pi0_step "${OUTSIDE_PI0_STEP}" \
        --inside_pi0_step "${INSIDE_PI0_STEP}" \
        --test_num "${TEST_NUM}" \
        --seed "${SEED}" \
        --policy_name pi0
  } 2>&1 | tee -a "${step_log}"
  append_summary "${step}" "${ckpt_setting}"
done

echo "========== DONE ${VARIANT} =========="
cat "${SUMMARY_CSV}"
