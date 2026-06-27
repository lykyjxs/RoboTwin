#!/usr/bin/env bash
set -euo pipefail

# Adaptive Stage3 pred late-xattn rollout evaluation for stack_bowls_three.
#
# Two-mode scheduler requested for KeyState-aware Adaptive Action Chunking:
#   - predicted h_entry bin != 0: outside key window -> execute 50 actions
#   - predicted h_entry bin == 0: inside key window  -> execute 25 actions
#
# This script does not train or overwrite checkpoints. It links the trained Stage3
# checkpoint into RoboTwin's eval layout and runs rollout evaluation.
#
# Main command:
#   CUDA_VISIBLE_DEVICES=0 EVAL_CHECKPOINT_ID=10000 EVAL_SEEDS="0 1 2" \
#     bash ./run_stack_bowls_stage3_pred_adaptive_rollout_eval.sh

# ========= Paths =========
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-./third_party/RoboTwin}"
PI0_ROOT="${PI0_ROOT:-${ROBOTWIN_ROOT}/policy/pi0}"
SHARE_ROOT="${SHARE_ROOT:-.}"
OPENPI_DATA_HOME_DIR="${OPENPI_DATA_HOME_DIR:-${SHARE_ROOT}/checkpoints/openpi}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${OPENPI_DATA_HOME_DIR}/openpi-assets/checkpoints/keystate_stage3}"

# ========= Stage3 checkpoint =========
CONFIG_NAME="${CONFIG_NAME:-pi0_base_aloha_robotwin_stack_bowls_three_keystate_stage3_pred_late_xattn_lora}"
EXP_NAME="${EXP_NAME:-stack_bowls_three_300_stage3_pred_late_xattn_from_stage2_5000_lora}"
EVAL_CHECKPOINT_ID="${EVAL_CHECKPOINT_ID:-10000}"
SHARED_STEP_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}/${EVAL_CHECKPOINT_ID}"

# ========= Adaptive chunk setup =========
# Fixed pi0_step is still passed for fallback, but adaptive mode overrides execution length.
PI0_STEP_FALLBACK="${PI0_STEP_FALLBACK:-50}"
OUTSIDE_PI0_STEP="${OUTSIDE_PI0_STEP:-50}"
INSIDE_PI0_STEP="${INSIDE_PI0_STEP:-25}"

# ========= RoboTwin rollout setup =========
TASK_NAME="${TASK_NAME:-stack_bowls_three}"
SOURCE_EVAL_TASK_CONFIG="${SOURCE_EVAL_TASK_CONFIG:-demo_clean}"
EVAL_TASK_CONFIG="${EVAL_TASK_CONFIG:-demo_clean_stage3_adaptive_no_video_eval}"
EVAL_VIDEO_LOG="${EVAL_VIDEO_LOG:-0}"
EVAL_SEEDS="${EVAL_SEEDS:-0 1 2}"
INSTRUCTION_SEED="${INSTRUCTION_SEED:-}"

# Optional debug/smoke shortcut: if set, only run the first N seed evaluations.
MAX_EVAL_COMBOS="${MAX_EVAL_COMBOS:-0}"

# RoboTwin eval Python must include OpenPI and compatible RoboTwin curobo.
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-python}"
EVAL_CUROBO_SRC="${EVAL_CUROBO_SRC:-./envs/curobo/src}"

# Use one GPU for rollout eval.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID:-0}"
fi
FIRST_GPU="${EVAL_GPU_ID:-$(python - <<PY
visible = "${CUDA_VISIBLE_DEVICES:-0}"
items = [x.strip() for x in visible.split(',') if x.strip()]
print(items[0] if items else '0')
PY
)}"

# ========= Preflight =========
if [[ ! -d "${ROBOTWIN_ROOT}" ]]; then
  echo "[ERROR] Missing ROBOTWIN_ROOT=${ROBOTWIN_ROOT}"
  exit 1
fi
if [[ ! -d "${PI0_ROOT}" ]]; then
  echo "[ERROR] Missing PI0_ROOT=${PI0_ROOT}"
  exit 1
fi
if [[ ! -d "${SHARED_STEP_DIR}/params" || ! -d "${SHARED_STEP_DIR}/assets" ]]; then
  echo "[ERROR] Missing Stage3 checkpoint for eval: ${SHARED_STEP_DIR}"
  echo "Expected params/ and assets/. Set EVAL_CHECKPOINT_ID to an existing step, e.g. 5000, 10000, or 15000."
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

EVAL_PYTHONPATH="${EVAL_CUROBO_SRC}:${PI0_ROOT}/src:${PI0_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"
PYTHONPATH="${EVAL_PYTHONPATH}" "${EVAL_PYTHON_BIN}" - <<PY
from curobo.types.math import Pose as CuroboPose
import openpi.training.config as _config
cfg = _config.get_config("${CONFIG_NAME}")
assert cfg.model.use_keystate_fusion, "Stage3 fusion is not enabled"
assert cfg.model.ks_fusion_source == "pred", cfg.model.ks_fusion_source
assert cfg.model.use_checkpoint_head, "h_entry prediction head is not enabled"
print("[OK] Eval Python imports OpenPI and compatible Curobo")
print("[OK] Stage3 config:", cfg.name, "fusion_source=", cfg.model.ks_fusion_source)
PY

cat <<EOF
========== Stage3 Adaptive Rollout Eval ==========
ROBOTWIN_ROOT=${ROBOTWIN_ROOT}
PI0_ROOT=${PI0_ROOT}
CONFIG_NAME=${CONFIG_NAME}
EXP_NAME=${EXP_NAME}
EVAL_CHECKPOINT_ID=${EVAL_CHECKPOINT_ID}
SHARED_STEP_DIR=${SHARED_STEP_DIR}
TASK_NAME=${TASK_NAME}
EVAL_TASK_CONFIG=${EVAL_TASK_CONFIG}
EVAL_VIDEO_LOG=${EVAL_VIDEO_LOG}
EVAL_SEEDS=${EVAL_SEEDS}
TEST_NUM=${TEST_NUM}
ADAPTIVE_PI0_STEP=True
OUTSIDE_PI0_STEP=${OUTSIDE_PI0_STEP}
INSIDE_PI0_STEP=${INSIDE_PI0_STEP}
PI0_STEP_FALLBACK=${PI0_STEP_FALLBACK}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}
FIRST_GPU=${FIRST_GPU}
EVAL_PYTHON_BIN=${EVAL_PYTHON_BIN}
EVAL_CUROBO_SRC=${EVAL_CUROBO_SRC}
=================================================
EOF

# ========= Make eval task config =========
python - <<PY
import pathlib, yaml
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

# ========= Link shared checkpoint into RoboTwin eval layout =========
# pi_model.py expects policy/pi0/checkpoints/<config>/<model>/<step>/{params,assets}.
LOCAL_MODEL_DIR="${PI0_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
LOCAL_STEP_LINK="${LOCAL_MODEL_DIR}/${EVAL_CHECKPOINT_ID}"
mkdir -p "${LOCAL_MODEL_DIR}"
if [[ -e "${LOCAL_STEP_LINK}" && ! -L "${LOCAL_STEP_LINK}" ]]; then
  echo "[ERROR] Local eval checkpoint path exists and is not a symlink: ${LOCAL_STEP_LINK}"
  echo "Move it away or set a different EXP_NAME."
  exit 1
fi
ln -sfn "${SHARED_STEP_DIR}" "${LOCAL_STEP_LINK}"
echo "[OK] Eval checkpoint symlink: ${LOCAL_STEP_LINK} -> ${SHARED_STEP_DIR}"

# ========= Evaluate =========
cd "${ROBOTWIN_ROOT}"
combo_count=0
for seed in ${EVAL_SEEDS}; do
  if [[ "${MAX_EVAL_COMBOS}" -gt 0 && "${combo_count}" -ge "${MAX_EVAL_COMBOS}" ]]; then
    break
  fi
  CKPT_SETTING="${EXP_NAME}_ckpt${EVAL_CHECKPOINT_ID}_adaptive_hbin0_${INSIDE_PI0_STEP}_else_${OUTSIDE_PI0_STEP}_seed${seed}"
  echo "========== Adaptive eval checkpoint=${EVAL_CHECKPOINT_ID} seed=${seed} =========="
  CUDA_VISIBLE_DEVICES="${FIRST_GPU}" PYTHONWARNINGS=ignore::UserWarning PYTHONPATH="${EVAL_PYTHONPATH}" \
    "${EVAL_PYTHON_BIN}" script/eval_policy.py \
      --config policy/pi0/deploy_policy.yml \
      --overrides \
      --task_name "${TASK_NAME}" \
      --task_config "${EVAL_TASK_CONFIG}" \
      --train_config_name "${CONFIG_NAME}" \
      --model_name "${EXP_NAME}" \
      --ckpt_setting "${CKPT_SETTING}" \
      --checkpoint_id "${EVAL_CHECKPOINT_ID}" \
      --pi0_step "${PI0_STEP_FALLBACK}" \
      --adaptive_pi0_step True \
      --outside_pi0_step "${OUTSIDE_PI0_STEP}" \
      --inside_pi0_step "${INSIDE_PI0_STEP}" \
      --test_num "${TEST_NUM}" \
      --seed "${seed}" \
      --policy_name pi0
  combo_count=$((combo_count + 1))
done

echo "========== DONE =========="
echo "Shared checkpoint: ${SHARED_STEP_DIR}"
echo "Adaptive rule: h_entry_bin == 0 -> ${INSIDE_PI0_STEP}; otherwise -> ${OUTSIDE_PI0_STEP}"
echo "Eval results under: ${ROBOTWIN_ROOT}/eval_result/${TASK_NAME}/pi0/${EVAL_TASK_CONFIG}/"
