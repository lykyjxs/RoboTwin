#!/usr/bin/env bash
set -euo pipefail

# Parallel Pi0 finetune checkpoint sweep for RoboTwin stack_bowls_three on multi-GPU H20 jobs.
#
# Goal: compare Pi0 finetune checkpoints 5000 / 10000 / 15000 under the same rollout protocol.
# Each eval job uses one GPU. The script auto-detects CUDA_VISIBLE_DEVICES and runs at most
# one eval process per visible GPU by default. For a 2-GPU job, it runs 2 evals at a time.
#
# Main command in a 2-GPU custom task:
#   CUDA_VISIBLE_DEVICES=0,1 \
#     bash ./run_stack_bowls_pi0_ckpt_sweep_8gpu.sh
#
# Main command in an 8-GPU custom task:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     bash ./run_stack_bowls_pi0_ckpt_sweep_8gpu.sh
#
# Faster seed0-only check:
#   CUDA_VISIBLE_DEVICES=0,1 EVAL_SEEDS="0" \
#     bash ./run_stack_bowls_pi0_ckpt_sweep_8gpu.sh

# ========= Paths =========
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-./third_party/RoboTwin}"
PI0_ROOT="${PI0_ROOT:-${ROBOTWIN_ROOT}/policy/pi0}"
SHARE_ROOT="${SHARE_ROOT:-.}"
OPENPI_DATA_HOME_DIR="${OPENPI_DATA_HOME_DIR:-${SHARE_ROOT}/checkpoints/openpi}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${OPENPI_DATA_HOME_DIR}/openpi-assets/checkpoints/baseline}"

# ========= Pi0 finetune baseline =========
CONFIG_NAME="${CONFIG_NAME:-pi0_base_aloha_robotwin_stack_bowls_three_lora}"
EXP_NAME="${EXP_NAME:-stack_bowls_three_300_pi0_baseline_lora_rerun15000_safe_v2}"
CHECKPOINT_IDS="${CHECKPOINT_IDS:-5000 10000 15000}"
PI0_STEP="${PI0_STEP:-50}"

# ========= Evaluation protocol =========
TASK_NAME="${TASK_NAME:-stack_bowls_three}"
SOURCE_EVAL_TASK_CONFIG="${SOURCE_EVAL_TASK_CONFIG:-demo_clean}"
EVAL_TASK_CONFIG="${EVAL_TASK_CONFIG:-demo_clean_pi0_ckpt_sweep_8gpu_no_video_eval}"
TEST_NUM="${TEST_NUM:-100}"
# Default to 3 seed groups for a stronger checkpoint selection. Use EVAL_SEEDS="0" for a quick 100-episode sweep.
EVAL_SEEDS="${EVAL_SEEDS:-0 1 2}"
# Keep instructions deterministic and identical across checkpoints. Set INSTRUCTION_SEED="" to use RoboTwin's legacy random instruction sampling.
INSTRUCTION_SEED="${INSTRUCTION_SEED:-777}"
MAX_PARALLEL="${MAX_PARALLEL:-}"

# ========= Runtime env =========
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-python}"
EVAL_CUROBO_SRC="${EVAL_CUROBO_SRC:-./envs/curobo/src}"
export PYTHONWARNINGS=ignore::UserWarning
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0"
fi

# ========= Helpers =========
parse_gpus() {
  python - <<PY
visible = "${CUDA_VISIBLE_DEVICES}"
items = [x.strip() for x in visible.split(',') if x.strip()]
print(' '.join(items if items else ['0']))
PY
}
GPU_LIST="$(parse_gpus)"
GPU_COUNT="$(wc -w <<<"${GPU_LIST}" | tr -d ' ')"
if [[ -z "${MAX_PARALLEL}" ]]; then
  MAX_PARALLEL="${GPU_COUNT}"
fi
if [[ "${MAX_PARALLEL}" -lt 1 ]]; then
  MAX_PARALLEL=1
fi

EVAL_PYTHONPATH="${EVAL_CUROBO_SRC}:${PI0_ROOT}/src:${PI0_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"
OUT_ROOT="${ROBOTWIN_ROOT}/eval_result/${TASK_NAME}/pi0/${EVAL_TASK_CONFIG}"
JOB_LOG_DIR="${OUT_ROOT}/_job_logs"
mkdir -p "${JOB_LOG_DIR}"

# ========= Preflight =========
if [[ ! -d "${ROBOTWIN_ROOT}" ]]; then
  echo "[ERROR] Missing ROBOTWIN_ROOT=${ROBOTWIN_ROOT}"
  exit 1
fi
if [[ ! -d "${PI0_ROOT}" ]]; then
  echo "[ERROR] Missing PI0_ROOT=${PI0_ROOT}"
  exit 1
fi
if [[ ! -x "${EVAL_PYTHON_BIN}" ]]; then
  echo "[ERROR] Missing eval Python: ${EVAL_PYTHON_BIN}"
  exit 1
fi
if [[ ! -d "${EVAL_CUROBO_SRC}/curobo" ]]; then
  echo "[ERROR] Missing compatible RoboTwin curobo source: ${EVAL_CUROBO_SRC}"
  exit 1
fi

PYTHONPATH="${EVAL_PYTHONPATH}" "${EVAL_PYTHON_BIN}" - <<PY
from curobo.types.math import Pose as CuroboPose
import openpi.training.config as _config
cfg = _config.get_config("${CONFIG_NAME}")
print("[OK] Eval Python imports OpenPI and compatible Curobo")
print("[OK] Pi0 config:", cfg.name)
PY

for ckpt in ${CHECKPOINT_IDS}; do
  step_dir="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}/${ckpt}"
  if [[ ! -d "${step_dir}/params" || ! -d "${step_dir}/assets" ]]; then
    echo "[ERROR] Missing checkpoint step for eval: ${step_dir}"
    exit 1
  fi
  local_model_dir="${PI0_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
  local_step_link="${local_model_dir}/${ckpt}"
  mkdir -p "${local_model_dir}"
  if [[ -e "${local_step_link}" && ! -L "${local_step_link}" ]]; then
    echo "[ERROR] Local eval checkpoint path exists and is not a symlink: ${local_step_link}"
    exit 1
  fi
  ln -sfn "${step_dir}" "${local_step_link}"
  echo "[OK] Eval checkpoint symlink: ${local_step_link} -> ${step_dir}"
done

python - <<PY
import pathlib, yaml
root = pathlib.Path("${ROBOTWIN_ROOT}")
src = root / "task_config" / "${SOURCE_EVAL_TASK_CONFIG}.yml"
dst = root / "task_config" / "${EVAL_TASK_CONFIG}.yml"
if not src.exists():
    raise FileNotFoundError(src)
with src.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
data["eval_video_log"] = False
with dst.open("w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
print(f"[OK] eval task config written: {dst}, eval_video_log={data['eval_video_log']}")
PY

cat <<EOF
========== Pi0 Checkpoint Sweep ==========
ROBOTWIN_ROOT=${ROBOTWIN_ROOT}
CONFIG_NAME=${CONFIG_NAME}
EXP_NAME=${EXP_NAME}
CHECKPOINT_IDS=${CHECKPOINT_IDS}
PI0_STEP=${PI0_STEP}
TASK_NAME=${TASK_NAME}
EVAL_TASK_CONFIG=${EVAL_TASK_CONFIG}
TEST_NUM=${TEST_NUM}
EVAL_SEEDS=${EVAL_SEEDS}
INSTRUCTION_SEED=${INSTRUCTION_SEED:-<legacy-random>}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
GPU_LIST=${GPU_LIST}
MAX_PARALLEL=${MAX_PARALLEL}
OUT_ROOT=${OUT_ROOT}
==========================================
EOF

run_one() {
  local ckpt="$1"
  local seed="$2"
  local gpu="$3"
  local start_seed=$((100000 * (1 + seed)))
  local ckpt_setting="${EXP_NAME}_ckpt${ckpt}_pi0step${PI0_STEP}_seed${seed}_test${TEST_NUM}_iseed${INSTRUCTION_SEED:-legacy}"
  local log_file="${JOB_LOG_DIR}/ckpt${ckpt}_seed${seed}.log"
  local extra=()
  if [[ -n "${INSTRUCTION_SEED}" ]]; then
    extra+=(--instruction_seed "${INSTRUCTION_SEED}")
  fi
  echo "[START] ckpt=${ckpt} seed=${seed} gpu=${gpu} log=${log_file}"
  (
    cd "${ROBOTWIN_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONWARNINGS=ignore::UserWarning PYTHONPATH="${EVAL_PYTHONPATH}" \
      "${EVAL_PYTHON_BIN}" script/eval_policy.py \
        --config policy/pi0/deploy_policy.yml \
        --overrides \
        --task_name "${TASK_NAME}" \
        --task_config "${EVAL_TASK_CONFIG}" \
        --train_config_name "${CONFIG_NAME}" \
        --model_name "${EXP_NAME}" \
        --ckpt_setting "${ckpt_setting}" \
        --checkpoint_id "${ckpt}" \
        --pi0_step "${PI0_STEP}" \
        --test_num "${TEST_NUM}" \
        --seed "${seed}" \
        --start_seed "${start_seed}" \
        "${extra[@]}" \
        --policy_name pi0
  ) >"${log_file}" 2>&1
  echo "[DONE] ckpt=${ckpt} seed=${seed} gpu=${gpu} log=${log_file}"
}

pids=()
idx=0
for ckpt in ${CHECKPOINT_IDS}; do
  for seed in ${EVAL_SEEDS}; do
    while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "${MAX_PARALLEL}" ]]; do
      sleep 10
    done
    gpu_array=(${GPU_LIST})
    gpu="${gpu_array[$((idx % GPU_COUNT))]}"
    run_one "${ckpt}" "${seed}" "${gpu}" &
    pids+=("$!")
    idx=$((idx + 1))
  done
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done
if [[ "${fail}" -ne 0 ]]; then
  echo "[ERROR] At least one eval job failed. Check logs under: ${JOB_LOG_DIR}"
  exit 1
fi

python - <<PY
from pathlib import Path
import re
from statistics import mean, stdev
out_root = Path("${OUT_ROOT}")
exp_name = "${EXP_NAME}"
ckpts = [int(x) for x in "${CHECKPOINT_IDS}".split()]
seeds = [int(x) for x in "${EVAL_SEEDS}".split()]
test_num = int("${TEST_NUM}")
pi0_step = "${PI0_STEP}"
instruction_seed = "${INSTRUCTION_SEED}" or "legacy"
rows = []
for ckpt in ckpts:
    vals = []
    for seed in seeds:
        setting = f"{exp_name}_ckpt{ckpt}_pi0step{pi0_step}_seed{seed}_test{test_num}_iseed{instruction_seed}"
        root = out_root / setting
        runs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True) if root.exists() else []
        value = None
        run = runs[0] if runs else None
        if run is not None:
            result = run / "_result.txt"
            if result.exists():
                lines = [x.strip() for x in result.read_text(errors="ignore").splitlines() if x.strip()]
                try:
                    value = float(lines[-1])
                except Exception:
                    value = None
        rows.append((ckpt, seed, value, str(run) if run else ""))
        if value is not None:
            vals.append(value)
    if vals:
        avg = mean(vals)
        sd = stdev(vals) if len(vals) > 1 else 0.0
        rows.append((ckpt, "mean", avg, f"std={sd:.6f}, n={len(vals)}"))

summary = out_root / "ckpt_sweep_summary.txt"
with summary.open("w", encoding="utf-8") as f:
    f.write("Pi0 finetune checkpoint sweep\n")
    f.write(f"TEST_NUM={test_num}, PI0_STEP={pi0_step}, EVAL_SEEDS={seeds}, INSTRUCTION_SEED={instruction_seed}\n\n")
    f.write("ckpt\tseed\tsuccess_rate\trun_dir_or_note\n")
    for ckpt, seed, value, note in rows:
        v = "missing" if value is None else f"{value:.6f}"
        f.write(f"{ckpt}\t{seed}\t{v}\t{note}\n")
print(summary.read_text())
PY

echo "========== DONE =========="
echo "Summary: ${OUT_ROOT}/ckpt_sweep_summary.txt"
echo "Job logs: ${JOB_LOG_DIR}"
echo "Results under: ${OUT_ROOT}"
