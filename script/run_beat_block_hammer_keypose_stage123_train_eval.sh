#!/usr/bin/env bash
set -euo pipefail

# Beat-block-hammer post-hit KeyState entry-pose experiment.
#
# Modes:
#   MODE=train  bash script/run_beat_block_hammer_keypose_stage123_train_eval.sh
#   MODE=eval   EVAL_CHECKPOINT_ID=10000 bash script/run_beat_block_hammer_keypose_stage123_train_eval.sh
#   MODE=launch bash script/run_beat_block_hammer_keypose_stage123_train_eval.sh
#
# Protocol:
#   Stage1: 5k steps, checkpoint/phase/type heads.
#   Stage2: 5k steps, adds z_entry_descriptor and keypose_entry_abs heads.
#   Stage3: 20k steps, pred late-xattn fusion, checkpoints at 5k/10k/15k/20k.
#   Eval: RoboTwin rollout eval at Stage3 checkpoints 5k/10k/15k/20k.

MODE="${MODE:-train}"

# ========= Paths =========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PI0_ROOT="${PI0_ROOT:-${ROBOTWIN_ROOT}/policy/pi0}"
SHARE_ROOT="${SHARE_ROOT:-${ROBOTWIN_ROOT}/data}"

LEROBOT_HOME="${LEROBOT_HOME:-${PI0_ROOT}/training_data}"
HF_HOME_DIR="${HF_HOME_DIR:-${SHARE_ROOT}/cache/huggingface}"
HF_DATASETS_CACHE_DIR="${HF_DATASETS_CACHE_DIR:-${HF_HOME_DIR}/datasets}"
OPENPI_DATA_HOME_DIR="${OPENPI_DATA_HOME_DIR:-${SHARE_ROOT}/checkpoints/openpi}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${OPENPI_DATA_HOME_DIR}/openpi-assets/assets}"

STAGE1_CHECKPOINT_BASE_DIR="${STAGE1_CHECKPOINT_BASE_DIR:-${OPENPI_DATA_HOME_DIR}/openpi-assets/checkpoints/keystate_keypose_stage1}"
STAGE2_CHECKPOINT_BASE_DIR="${STAGE2_CHECKPOINT_BASE_DIR:-${OPENPI_DATA_HOME_DIR}/openpi-assets/checkpoints/keystate_keypose_stage2}"
STAGE3_CHECKPOINT_BASE_DIR="${STAGE3_CHECKPOINT_BASE_DIR:-${OPENPI_DATA_HOME_DIR}/openpi-assets/checkpoints/keystate_keypose_stage3}"

# ========= Experiment =========
TASK_NAME="${TASK_NAME:-beat_block_hammer}"
TRAIN_REPO_ID="${TRAIN_REPO_ID:-beat_block_hammer_demo_clean_50_posthit_keystate_stage2_actionexpert}"

STAGE1_CONFIG_NAME="${STAGE1_CONFIG_NAME:-pi0_base_aloha_robotwin_beat_block_hammer_posthit_keystate_lora}"
STAGE2_CONFIG_NAME="${STAGE2_CONFIG_NAME:-pi0_base_aloha_robotwin_beat_block_hammer_posthit_keystate_stage2_lora}"
STAGE3_CONFIG_NAME="${STAGE3_CONFIG_NAME:-pi0_base_aloha_robotwin_beat_block_hammer_posthit_keystate_stage3_pred_late_xattn_lora}"

STAGE1_EXP_NAME="${STAGE1_EXP_NAME:-beat_block_hammer_50_posthit_stage1_keypose_entry_abs_5k_official}"
STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-beat_block_hammer_50_posthit_stage2_keypose_entry_abs_from_stage1_5k_official}"
STAGE3_EXP_NAME="${STAGE3_EXP_NAME:-beat_block_hammer_50_posthit_stage3_pred_late_xattn_keypose_entry_abs_from_stage2_5k_20k_official}"

STAGE1_CKPT_DIR="${STAGE1_CHECKPOINT_BASE_DIR}/${STAGE1_CONFIG_NAME}/${STAGE1_EXP_NAME}"
STAGE2_CKPT_DIR="${STAGE2_CHECKPOINT_BASE_DIR}/${STAGE2_CONFIG_NAME}/${STAGE2_EXP_NAME}"
STAGE3_CKPT_DIR="${STAGE3_CHECKPOINT_BASE_DIR}/${STAGE3_CONFIG_NAME}/${STAGE3_EXP_NAME}"

STAGE1_STEPS="${STAGE1_STEPS:-5000}"
STAGE2_STEPS="${STAGE2_STEPS:-5000}"
STAGE3_STEPS="${STAGE3_STEPS:-20000}"
STAGE1_CHECKPOINT_ID="${STAGE1_CHECKPOINT_ID:-5000}"
STAGE2_CHECKPOINT_ID="${STAGE2_CHECKPOINT_ID:-5000}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-5000 10000 15000 20000}"

SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-2}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"

TRAIN_NORM_STATS_SOURCE="${TRAIN_NORM_STATS_SOURCE:-${ASSETS_BASE_DIR}/${STAGE2_CONFIG_NAME}/${TRAIN_REPO_ID}/norm_stats.json}"

# ========= Eval defaults =========
SOURCE_EVAL_TASK_CONFIG="${SOURCE_EVAL_TASK_CONFIG:-demo_clean}"
EVAL_TASK_CONFIG="${EVAL_TASK_CONFIG:-demo_clean_beat_block_hammer_keypose_stage3_adaptive_no_video_eval}"
EVAL_VIDEO_LOG="${EVAL_VIDEO_LOG:-0}"
EVAL_SEEDS="${EVAL_SEEDS:-0}"
TEST_NUM="${TEST_NUM:-100}"
INSTRUCTION_SEED="${INSTRUCTION_SEED:-777}"
PI0_STEP_FALLBACK="${PI0_STEP_FALLBACK:-50}"
OUTSIDE_PI0_STEP="${OUTSIDE_PI0_STEP:-50}"
INSIDE_PI0_STEP="${INSIDE_PI0_STEP:-25}"
EVAL_CHECKPOINT_ID="${EVAL_CHECKPOINT_ID:-5000}"
WAIT_FOR_CHECKPOINT="${WAIT_FOR_CHECKPOINT:-0}"
WAIT_FOR_STAGE3_FINAL="${WAIT_FOR_STAGE3_FINAL:-0}"
WAIT_SLEEP_SECONDS="${WAIT_SLEEP_SECONDS:-120}"
USE_EVAL_LOCK="${USE_EVAL_LOCK:-0}"
EVAL_LOCK_PATH="${EVAL_LOCK_PATH:-${STAGE3_CKPT_DIR}/eval.lock}"

EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-$(command -v python)}"
EVAL_CUROBO_SRC="${EVAL_CUROBO_SRC:-${ROBOTWIN_ROOT}/envs/curobo/src}"

# ========= Runtime env =========
export HF_LEROBOT_HOME="${LEROBOT_HOME}"
export HF_HOME="${HF_HOME_DIR}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE_DIR}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME_DIR}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONPATH="${PI0_ROOT}/src:${PYTHONPATH:-}"

count_visible_cuda_devices() {
  local visible="$1"
  if [[ -z "${visible}" || "${visible}" == "NoDevFiles" || "${visible}" == "-1" ]]; then
    echo 0
    return
  fi
  python - <<PY
visible = "${visible}"
items = [x.strip() for x in visible.split(',') if x.strip()]
print(len(items))
PY
}

count_nvidia_smi_devices() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
  else
    echo 0
  fi
}

make_cuda_visible_devices() {
  local n="$1"
  if [[ "${n}" -le 0 ]]; then
    echo ""
  else
    python - <<PY
n = int("${n}")
print(','.join(str(i) for i in range(n)))
PY
  fi
}

first_visible_gpu() {
  python - <<PY
visible = "${CUDA_VISIBLE_DEVICES:-0}"
items = [x.strip() for x in visible.split(',') if x.strip()]
print(items[0] if items else '0')
PY
}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_COUNT="$(count_visible_cuda_devices "${CUDA_VISIBLE_DEVICES}")"
else
  GPU_COUNT="$(count_nvidia_smi_devices)"
  if [[ "${GPU_COUNT}" -gt 0 ]]; then
    export CUDA_VISIBLE_DEVICES="$(make_cuda_visible_devices "${GPU_COUNT}")"
  fi
fi
GPU_COUNT="${GPU_COUNT_OVERRIDE:-${GPU_COUNT}}"

if [[ -z "${FSDP_DEVICES:-}" ]]; then
  if [[ "${GPU_COUNT}" -ge 1 ]]; then
    FSDP_DEVICES="${GPU_COUNT}"
  else
    FSDP_DEVICES=1
  fi
fi

if [[ -z "${TRAIN_BATCH_SIZE:-}" ]]; then
  if [[ "${GPU_COUNT}" -ge 4 ]]; then
    TRAIN_BATCH_SIZE=64
  else
    TRAIN_BATCH_SIZE=32
  fi
fi

if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
  PY_CMD=("${PYTHON_BIN}")
elif [[ -x "/tmp/openpi-ks-smoke-venv/bin/python" ]]; then
  PY_CMD=("/tmp/openpi-ks-smoke-venv/bin/python")
elif [[ -x "${PI0_ROOT}/.venv/bin/python" ]]; then
  PY_CMD=("${PI0_ROOT}/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PY_CMD=(uv --project "${PI0_ROOT}" run python)
elif command -v python >/dev/null 2>&1; then
  PY_CMD=(python)
else
  echo "[ERROR] No usable Python found. Set PYTHON_BIN to the OpenPI training environment."
  exit 1
fi

mkdir -p "${HF_DATASETS_CACHE_DIR}" "${ASSETS_BASE_DIR}" \
  "${STAGE1_CHECKPOINT_BASE_DIR}" "${STAGE2_CHECKPOINT_BASE_DIR}" "${STAGE3_CHECKPOINT_BASE_DIR}"

copy_norm_stats() {
  local config_name="$1"
  local repo_id="$2"
  local dst_dir="${ASSETS_BASE_DIR}/${config_name}/${repo_id}"
  local dst="${dst_dir}/norm_stats.json"
  mkdir -p "${dst_dir}"
  if [[ "${TRAIN_NORM_STATS_SOURCE}" != "${dst}" ]]; then
    cp "${TRAIN_NORM_STATS_SOURCE}" "${dst}"
  fi
  echo "[OK] norm stats: ${dst}"
}

preflight_common() {
  if [[ ! -d "${ROBOTWIN_ROOT}" ]]; then
    echo "[ERROR] Missing ROBOTWIN_ROOT=${ROBOTWIN_ROOT}"
    exit 1
  fi
  if [[ ! -d "${PI0_ROOT}" ]]; then
    echo "[ERROR] Missing PI0_ROOT=${PI0_ROOT}"
    exit 1
  fi
  if [[ ! -f "${LEROBOT_HOME}/${TRAIN_REPO_ID}/meta/info.json" ]]; then
    echo "[ERROR] Missing LeRobot repo: ${LEROBOT_HOME}/${TRAIN_REPO_ID}"
    exit 1
  fi
  if [[ ! -f "${TRAIN_NORM_STATS_SOURCE}" ]]; then
    echo "[ERROR] Missing norm stats source: ${TRAIN_NORM_STATS_SOURCE}"
    exit 1
  fi
}

preflight_train_configs() {
  cd "${PI0_ROOT}"
  "${PY_CMD[@]}" - <<PY
import openpi.training.config as _config

stage1 = _config.get_config("${STAGE1_CONFIG_NAME}")
assert stage1.model.use_checkpoint_head, "Stage1 h_entry head is disabled"
assert stage1.model.use_phase_head, "Stage1 phase head is disabled"

stage2 = _config.get_config("${STAGE2_CONFIG_NAME}")
assert stage2.model.use_checkpoint_head, "Stage2 h_entry head is disabled"
assert stage2.model.use_phase_head, "Stage2 phase head is disabled"
assert stage2.model.use_z_entry_descriptor, "Stage2 z head is disabled"
assert stage2.model.use_keypose_entry_abs, "Stage2 keypose head is disabled"
assert stage2.model.keypose_entry_abs_dim == 7, stage2.model.keypose_entry_abs_dim

stage3 = _config.get_config("${STAGE3_CONFIG_NAME}")
assert stage3.model.use_keystate_fusion, "Stage3 fusion is disabled"
assert stage3.model.ks_fusion_source == "pred", stage3.model.ks_fusion_source
assert stage3.model.use_keypose_entry_abs, "Stage3 keypose head is disabled"
assert stage3.model.keypose_entry_abs_dim == 7, stage3.model.keypose_entry_abs_dim
print("[OK] configs:", stage1.name, stage2.name, stage3.name)
PY
}

train_flags() {
  if [[ "${OVERWRITE:-0}" == "1" ]]; then
    printf '%s\n' "--overwrite"
  elif [[ "${RESUME:-0}" == "1" ]]; then
    printf '%s\n' "--resume"
  fi
  if [[ "${WANDB_ENABLED}" != "1" ]]; then
    printf '%s\n' "--no-wandb-enabled"
  fi
}

run_train_stage() {
  local label="$1"
  local config_name="$2"
  local exp_name="$3"
  local checkpoint_base_dir="$4"
  local steps="$5"
  local extra_params_path="${6:-}"
  local extra_missing_regex="${7:-}"
  local ckpt_dir="${checkpoint_base_dir}/${config_name}/${exp_name}"

  if [[ -e "${ckpt_dir}" && "${OVERWRITE:-0}" != "1" && "${RESUME:-0}" != "1" ]]; then
    echo "[ERROR] ${label} checkpoint dir exists: ${ckpt_dir}"
    echo "Set RESUME=1, OVERWRITE=1, or change the experiment name."
    exit 1
  fi

  local flags=()
  while IFS= read -r flag; do
    [[ -n "${flag}" ]] && flags+=("${flag}")
  done < <(train_flags)

  local loader_flags=()
  if [[ -n "${extra_params_path}" ]]; then
    if [[ ! -d "${extra_params_path}" ]]; then
      echo "[ERROR] Missing ${label} init params: ${extra_params_path}"
      exit 1
    fi
    loader_flags+=(--weight-loader.params-path "${extra_params_path}")
  fi
  if [[ -n "${extra_missing_regex}" ]]; then
    loader_flags+=(--weight-loader.missing-regex "${extra_missing_regex}")
  fi

  echo "========== Train ${label} =========="
  echo "config=${config_name}"
  echo "exp=${exp_name}"
  echo "repo=${TRAIN_REPO_ID}"
  echo "steps=${steps}"
  echo "checkpoint_dir=${ckpt_dir}"
  cd "${PI0_ROOT}"
  "${PY_CMD[@]}" scripts/train.py "${config_name}" \
    --project-name openpi-keystate \
    --exp-name "${exp_name}" \
    --assets-base-dir "${ASSETS_BASE_DIR}" \
    --checkpoint-base-dir "${checkpoint_base_dir}" \
    --data.repo-id "${TRAIN_REPO_ID}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    --num-workers "${TRAIN_NUM_WORKERS}" \
    --num-train-steps "${steps}" \
    --log-interval "${LOG_INTERVAL}" \
    --save-interval "${SAVE_INTERVAL}" \
    --keep-period "${KEEP_PERIOD}" \
    --fsdp-devices "${FSDP_DEVICES}" \
    "${loader_flags[@]}" \
    "${flags[@]}"
}

run_train() {
  preflight_common
  copy_norm_stats "${STAGE1_CONFIG_NAME}" "${TRAIN_REPO_ID}"
  copy_norm_stats "${STAGE2_CONFIG_NAME}" "${TRAIN_REPO_ID}"
  copy_norm_stats "${STAGE3_CONFIG_NAME}" "${TRAIN_REPO_ID}"
  preflight_train_configs

  cat <<EOF
========== Beat Block Hammer Keypose Train ==========
PY_CMD=${PY_CMD[*]}
HF_LEROBOT_HOME=${HF_LEROBOT_HOME}
ASSETS_BASE_DIR=${ASSETS_BASE_DIR}
STAGE1_CKPT_DIR=${STAGE1_CKPT_DIR}
STAGE2_CKPT_DIR=${STAGE2_CKPT_DIR}
STAGE3_CKPT_DIR=${STAGE3_CKPT_DIR}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}
GPU_COUNT=${GPU_COUNT}
FSDP_DEVICES=${FSDP_DEVICES}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}
WANDB_ENABLED=${WANDB_ENABLED}
====================================================
EOF

  run_train_stage \
    "Stage1 5k" \
    "${STAGE1_CONFIG_NAME}" \
    "${STAGE1_EXP_NAME}" \
    "${STAGE1_CHECKPOINT_BASE_DIR}" \
    "${STAGE1_STEPS}"

  run_train_stage \
    "Stage2 5k" \
    "${STAGE2_CONFIG_NAME}" \
    "${STAGE2_EXP_NAME}" \
    "${STAGE2_CHECKPOINT_BASE_DIR}" \
    "${STAGE2_STEPS}" \
    "${STAGE1_CKPT_DIR}/${STAGE1_CHECKPOINT_ID}/params" \
    ".*(ks_z_entry_descriptor_head|ks_keypose_entry_abs_head).*"

  run_train_stage \
    "Stage3 20k" \
    "${STAGE3_CONFIG_NAME}" \
    "${STAGE3_EXP_NAME}" \
    "${STAGE3_CHECKPOINT_BASE_DIR}" \
    "${STAGE3_STEPS}" \
    "${STAGE2_CKPT_DIR}/${STAGE2_CHECKPOINT_ID}/params" \
    ".*(ks_keypose_entry_abs_head|ks_type_embed|ks_horizon_embed|ks_phase_proj|ks_z_entry_proj|ks_action_ln|ks_memory_ln|ks_late_xattn|ks_late_xattn_alpha).*"

  echo "========== Train DONE =========="
}

wait_for_step() {
  local step="$1"
  local step_dir="${STAGE3_CKPT_DIR}/${step}"
  while [[ ! -d "${step_dir}/params" || ! -d "${step_dir}/assets" ]]; do
    if [[ "${WAIT_FOR_CHECKPOINT}" != "1" ]]; then
      echo "[ERROR] Missing Stage3 checkpoint: ${step_dir}"
      exit 1
    fi
    echo "[WAIT] missing ${step_dir}/{params,assets}; sleeping ${WAIT_SLEEP_SECONDS}s"
    sleep "${WAIT_SLEEP_SECONDS}"
  done
}

preflight_eval() {
  if [[ ! -x "${EVAL_PYTHON_BIN}" ]]; then
    echo "[ERROR] Missing eval Python: ${EVAL_PYTHON_BIN}"
    exit 1
  fi
  if [[ ! -d "${EVAL_CUROBO_SRC}/curobo" ]]; then
    echo "[ERROR] Missing compatible Curobo source: ${EVAL_CUROBO_SRC}"
    exit 1
  fi

  local eval_pythonpath="${ROBOTWIN_ROOT}:${EVAL_CUROBO_SRC}:${PI0_ROOT}/src:${PI0_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"
  PYTHONPATH="${eval_pythonpath}" "${EVAL_PYTHON_BIN}" - <<PY
from curobo.types.math import Pose as CuroboPose
import openpi.training.config as _config
cfg = _config.get_config("${STAGE3_CONFIG_NAME}")
assert cfg.model.use_keystate_fusion, "Stage3 fusion is disabled"
assert cfg.model.ks_fusion_source == "pred", cfg.model.ks_fusion_source
assert cfg.model.use_checkpoint_head, "Stage3 h_entry head is disabled"
assert cfg.model.use_keypose_entry_abs, "Stage3 keypose head is disabled"
print("[OK] eval config:", cfg.name, "fusion_source=", cfg.model.ks_fusion_source, "keypose_dim=", cfg.model.keypose_entry_abs_dim)
PY
}

prepare_eval_task_config() {
  python - <<PY
import pathlib
import yaml

root = pathlib.Path("${ROBOTWIN_ROOT}")
src = root / "task_config" / "${SOURCE_EVAL_TASK_CONFIG}.yml"
dst = root / "task_config" / "${EVAL_TASK_CONFIG}.yml"
if not src.exists():
    raise FileNotFoundError(src)
with src.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
data["eval_video_log"] = bool(int("${EVAL_VIDEO_LOG}"))
with dst.open("w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
print(f"[OK] eval task config written: {dst}, eval_video_log={data['eval_video_log']}")
PY
}

link_eval_checkpoint() {
  local step="$1"
  local shared_step_dir="${STAGE3_CKPT_DIR}/${step}"
  local local_model_dir="${PI0_ROOT}/checkpoints/${STAGE3_CONFIG_NAME}/${STAGE3_EXP_NAME}"
  local local_step_link="${local_model_dir}/${step}"
  mkdir -p "${local_model_dir}"
  if [[ -e "${local_step_link}" && ! -L "${local_step_link}" ]]; then
    echo "[ERROR] Local eval checkpoint path exists and is not a symlink: ${local_step_link}"
    exit 1
  fi
  ln -sfn "${shared_step_dir}" "${local_step_link}"
  echo "[OK] Eval checkpoint symlink: ${local_step_link} -> ${shared_step_dir}"
}

run_eval_loop() {
  local step="$1"
  local eval_pythonpath="${ROBOTWIN_ROOT}:${EVAL_CUROBO_SRC}:${PI0_ROOT}/src:${PI0_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"
  local first_gpu="${EVAL_GPU_ID:-$(first_visible_gpu)}"

  cd "${ROBOTWIN_ROOT}"
  for seed in ${EVAL_SEEDS}; do
    local ckpt_setting="${STAGE3_EXP_NAME}_ckpt${step}_adaptive_hbin0_${INSIDE_PI0_STEP}_else_${OUTSIDE_PI0_STEP}_seed${seed}_test${TEST_NUM}_iseed${INSTRUCTION_SEED}"
    echo "========== RoboTwin eval checkpoint=${step} seed=${seed} gpu=${first_gpu} =========="
    CUDA_VISIBLE_DEVICES="${first_gpu}" PYTHONWARNINGS=ignore::UserWarning PYTHONPATH="${eval_pythonpath}" \
      "${EVAL_PYTHON_BIN}" script/eval_policy.py \
        --config policy/pi0/deploy_policy.yml \
        --overrides \
        --task_name "${TASK_NAME}" \
        --task_config "${EVAL_TASK_CONFIG}" \
        --train_config_name "${STAGE3_CONFIG_NAME}" \
        --model_name "${STAGE3_EXP_NAME}" \
        --ckpt_setting "${ckpt_setting}" \
        --checkpoint_id "${step}" \
        --pi0_step "${PI0_STEP_FALLBACK}" \
        --adaptive_pi0_step True \
        --outside_pi0_step "${OUTSIDE_PI0_STEP}" \
        --inside_pi0_step "${INSIDE_PI0_STEP}" \
        --test_num "${TEST_NUM}" \
        --seed "${seed}" \
        --instruction_seed "${INSTRUCTION_SEED}" \
        --policy_name pi0
  done
}

run_eval() {
  preflight_common
  if [[ "${WAIT_FOR_STAGE3_FINAL}" == "1" ]]; then
    wait_for_step "${STAGE3_STEPS}"
  fi
  wait_for_step "${EVAL_CHECKPOINT_ID}"
  preflight_eval
  prepare_eval_task_config
  link_eval_checkpoint "${EVAL_CHECKPOINT_ID}"

  if [[ "${USE_EVAL_LOCK}" == "1" && "$(command -v flock || true)" ]]; then
    mkdir -p "$(dirname "${EVAL_LOCK_PATH}")"
    exec 9>"${EVAL_LOCK_PATH}"
    echo "[LOCK] waiting for eval lock: ${EVAL_LOCK_PATH}"
    flock 9
    echo "[LOCK] acquired eval lock: ${EVAL_LOCK_PATH}"
  elif [[ "${USE_EVAL_LOCK}" == "1" ]]; then
    echo "[WARN] USE_EVAL_LOCK=1 but flock is unavailable; running without eval serialization."
  fi

  run_eval_loop "${EVAL_CHECKPOINT_ID}"

  echo "========== Eval DONE =========="
  echo "Stage3 checkpoint: ${STAGE3_CKPT_DIR}/${EVAL_CHECKPOINT_ID}"
  echo "Eval results under: ${ROBOTWIN_ROOT}/eval_result/${TASK_NAME}/pi0/${EVAL_TASK_CONFIG}/"
}

tmux_has_session() {
  tmux has-session -t "$1" 2>/dev/null
}

launch_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[ERROR] tmux is not installed"
    exit 1
  fi

  local script_path
  script_path="$(readlink -f "$0")"
  local log_dir="${LOG_DIR:-${ROBOTWIN_ROOT}/logs/beat_block_hammer_keypose_$(date +%Y%m%d_%H%M%S)}"
  mkdir -p "${log_dir}"

  local train_session="${TRAIN_TMUX_SESSION:-bbh_keypose_stage123_train}"
  if tmux_has_session "${train_session}"; then
    echo "[SKIP] tmux session exists: ${train_session}"
  else
    tmux new-session -d -s "${train_session}" \
      "cd '${ROBOTWIN_ROOT}' && MODE=train bash '${script_path}' 2>&1 | tee -a '${log_dir}/train.log'"
    echo "[OK] launched train tmux: ${train_session}"
  fi

  local visible="${CUDA_VISIBLE_DEVICES:-0}"
  local -a gpu_ids=()
  IFS=',' read -r -a gpu_ids <<< "${visible}"
  if [[ "${#gpu_ids[@]}" -eq 0 || -z "${gpu_ids[0]}" ]]; then
    gpu_ids=("0")
  fi

  local idx=0
  for step in ${CHECKPOINT_STEPS}; do
    local eval_session="${EVAL_TMUX_PREFIX:-bbh_keypose_stage3_eval}_${step}"
    local gpu="${gpu_ids[$((idx % ${#gpu_ids[@]}))]}"
    gpu="$(echo "${gpu}" | xargs)"
    if tmux_has_session "${eval_session}"; then
      echo "[SKIP] tmux session exists: ${eval_session}"
    else
      tmux new-session -d -s "${eval_session}" \
        "cd '${ROBOTWIN_ROOT}' && MODE=eval EVAL_CHECKPOINT_ID='${step}' WAIT_FOR_CHECKPOINT=1 WAIT_FOR_STAGE3_FINAL=1 USE_EVAL_LOCK=1 EVAL_GPU_ID='${gpu}' bash '${script_path}' 2>&1 | tee -a '${log_dir}/eval_${step}.log'"
      echo "[OK] launched eval tmux: ${eval_session} checkpoint=${step} gpu=${gpu}"
    fi
    idx=$((idx + 1))
  done

  echo "========== tmux launch summary =========="
  echo "log_dir=${log_dir}"
  echo "train_session=${train_session}"
  echo "eval_prefix=${EVAL_TMUX_PREFIX:-bbh_keypose_stage3_eval}"
  echo "attach: tmux attach -t ${train_session}"
}

case "${MODE}" in
  train)
    run_train
    ;;
  eval)
    run_eval
    ;;
  launch)
    launch_tmux
    ;;
  *)
    echo "[ERROR] Unknown MODE=${MODE}. Use train, eval, or launch."
    exit 1
    ;;
esac
