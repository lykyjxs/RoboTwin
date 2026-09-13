#!/usr/bin/env bash
set -euo pipefail

# One-click Stage3 pred no-fusion ablation training + held-out test30 validation + adaptive rollout for stack_bowls_three.
#
# Protocol:
#   - TRAIN: original train300 Stage2 LeRobot repo only.
#   - TEST/VAL: fresh val30 Stage2 LeRobot repo only for checkpoint selection / validation.
#   - Norm stats: train300 norm stats reused for both Stage3 train/test assets.
#   - Initialization: Stage2 best checkpoint 5000.
#   - Fusion: disabled (KeyState heads still enabled for adaptive chunking), NOT gt/oracle. This is the non-oracle Stage3 path.
#
# Entry command:
#   bash ./script/run_stack_bowls_stage3_pred_train_test30.sh
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./script/run_stack_bowls_stage3_pred_train_test30.sh
#   NUM_TRAIN_STEPS=15000 SAVE_INTERVAL=5000 bash ./script/run_stack_bowls_stage3_pred_train_test30.sh
#   SKIP_TRAIN=1 bash ./script/run_stack_bowls_stage3_pred_train_test30.sh
#   RESUME=1 bash ./script/run_stack_bowls_stage3_pred_train_test30.sh

# ========= Paths =========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PI0_ROOT="${PI0_ROOT:-${ROBOTWIN_ROOT}/policy/pi0}"
SHARE_ROOT="${SHARE_ROOT:-${ROBOTWIN_ROOT}/data}"

LEROBOT_HOME="${LEROBOT_HOME:-${SHARE_ROOT}/data/lerobot}"
HF_HOME_DIR="${HF_HOME_DIR:-${SHARE_ROOT}/cache/huggingface}"
HF_DATASETS_CACHE_DIR="${HF_DATASETS_CACHE_DIR:-${HF_HOME_DIR}/datasets}"
OPENPI_DATA_HOME_DIR="${OPENPI_DATA_HOME_DIR:-${SHARE_ROOT}/checkpoints/openpi}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${OPENPI_DATA_HOME_DIR}/openpi-assets/assets}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${OPENPI_DATA_HOME_DIR}/openpi-assets/checkpoints/keystate_stage3}"

STAGE2_CONFIG_NAME="${STAGE2_CONFIG_NAME:-pi0_base_aloha_robotwin_stack_bowls_three_keystate_stage2_lora}"
STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-stack_bowls_three_300_stage2_actionexpert_from_stage1_15000_lora}"
STAGE2_CHECKPOINT_ID="${STAGE2_CHECKPOINT_ID:-5000}"
STAGE2_PARAMS_PATH="${STAGE2_PARAMS_PATH:-${OPENPI_DATA_HOME_DIR}/openpi-assets/checkpoints/keystate_stage2/${STAGE2_CONFIG_NAME}/${STAGE2_EXP_NAME}/${STAGE2_CHECKPOINT_ID}/params}"

STAGE3_CONFIG_NAME="${STAGE3_CONFIG_NAME:-pi0_base_aloha_robotwin_stack_bowls_three_keystate_stage3_pred_no_fusion_lora}"
TRAIN_REPO_ID="${TRAIN_REPO_ID:-stack_bowls_three_demo_clean_300_keystate_stage2_actionexpert}"
TEST_REPO_ID="${TEST_REPO_ID:-stack_bowls_three_demo_clean_val30_keystate_stage2_actionexpert}"
EXP_NAME="${EXP_NAME:-stack_bowls_three_300_stage3_pred_no_fusion_from_stage2_5000_lora}"
TRAIN_CKPT_DIR="${CHECKPOINT_BASE_DIR}/${STAGE3_CONFIG_NAME}/${EXP_NAME}"
VAL_EVAL_DIR="${VAL_EVAL_DIR:-${TRAIN_CKPT_DIR}/test30_eval}"

# Norm stats are state/action train300 stats. Stage3 reuses the Stage2 train300 stats.
TRAIN_NORM_STATS_PATH="${TRAIN_NORM_STATS_PATH:-${ASSETS_BASE_DIR}/${STAGE2_CONFIG_NAME}/${TRAIN_REPO_ID}/norm_stats.json}"

# ========= GPU defaults =========
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
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-2}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-15000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"

VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"
# 0 means evaluate all complete test30 batches.
VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-0}"

# ========= Rollout eval defaults =========
TASK_NAME="${TASK_NAME:-stack_bowls_three}"
SOURCE_EVAL_TASK_CONFIG="${SOURCE_EVAL_TASK_CONFIG:-demo_clean}"
EVAL_TASK_CONFIG="${EVAL_TASK_CONFIG:-demo_clean_stage3_no_fusion_adaptive_no_video_eval}"
EVAL_VIDEO_LOG="${EVAL_VIDEO_LOG:-0}"
EVAL_SEEDS="${EVAL_SEEDS:-0}"
TEST_NUM="${TEST_NUM:-100}"
INSTRUCTION_SEED="${INSTRUCTION_SEED:-777}"
PI0_STEP_FALLBACK="${PI0_STEP_FALLBACK:-50}"
OUTSIDE_PI0_STEP="${OUTSIDE_PI0_STEP:-50}"
INSIDE_PI0_STEP="${INSIDE_PI0_STEP:-25}"
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

mkdir -p "${LEROBOT_HOME}" "${HF_DATASETS_CACHE_DIR}" "${ASSETS_BASE_DIR}" "${CHECKPOINT_BASE_DIR}"

if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
  PY_CMD=("${PYTHON_BIN}")
elif command -v uv >/dev/null 2>&1; then
  PY_CMD=(uv --project "${PI0_ROOT}" run python)
elif [[ -x "${PI0_ROOT}/.venv/bin/python" ]]; then
  PY_CMD=("${PI0_ROOT}/.venv/bin/python")
elif command -v python >/dev/null 2>&1; then
  PY_CMD=(python)
else
  echo "[ERROR] No usable Python found. Tried PYTHON_BIN, uv, .venv, python."
  exit 1
fi

cat <<EOF
========== Environment ==========
PI0_ROOT=${PI0_ROOT}
PY_CMD=${PY_CMD[*]}
HF_LEROBOT_HOME=${HF_LEROBOT_HOME}
HF_HOME=${HF_HOME}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
OPENPI_DATA_HOME=${OPENPI_DATA_HOME}
ASSETS_BASE_DIR=${ASSETS_BASE_DIR}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR}
STAGE2_PARAMS_PATH=${STAGE2_PARAMS_PATH}
STAGE3_CONFIG_NAME=${STAGE3_CONFIG_NAME}
TRAIN_REPO_ID=${TRAIN_REPO_ID}
TEST_REPO_ID=${TEST_REPO_ID}
EXP_NAME=${EXP_NAME}
TRAIN_CKPT_DIR=${TRAIN_CKPT_DIR}
TRAIN_NORM_STATS_PATH=${TRAIN_NORM_STATS_PATH}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}
GPU_COUNT=${GPU_COUNT}
FSDP_DEVICES=${FSDP_DEVICES}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS}
SAVE_INTERVAL=${SAVE_INTERVAL}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE}
VAL_NUM_BATCHES=${VAL_NUM_BATCHES}
=================================
EOF

# ========= Preflight =========
if [[ ! -d "${PI0_ROOT}" ]]; then
  echo "[ERROR] Missing PI0_ROOT=${PI0_ROOT}"
  exit 1
fi
if [[ ! -d "${STAGE2_PARAMS_PATH}" ]]; then
  echo "[ERROR] Missing Stage2 selected params: ${STAGE2_PARAMS_PATH}"
  exit 1
fi
if [[ ! -f "${LEROBOT_HOME}/${TRAIN_REPO_ID}/meta/info.json" ]]; then
  echo "[ERROR] Missing train LeRobot repo: ${LEROBOT_HOME}/${TRAIN_REPO_ID}"
  exit 1
fi
if [[ ! -f "${LEROBOT_HOME}/${TEST_REPO_ID}/meta/info.json" ]]; then
  echo "[ERROR] Missing held-out test LeRobot repo: ${LEROBOT_HOME}/${TEST_REPO_ID}"
  exit 1
fi
if [[ ! -f "${TRAIN_NORM_STATS_PATH}" ]]; then
  echo "[ERROR] Missing train norm stats: ${TRAIN_NORM_STATS_PATH}"
  exit 1
fi

cd "${PI0_ROOT}"
"${PY_CMD[@]}" - <<PY
import openpi.training.config as _config
cfg = _config.get_config("${STAGE3_CONFIG_NAME}")
assert cfg.model.use_checkpoint_head, "h_entry prediction head is not enabled"
assert cfg.model.use_phase_head, "phase head is not enabled"
assert cfg.model.use_z_entry_descriptor, "z-entry descriptor head is not enabled"
assert not cfg.model.use_keystate_fusion, "Expected no-fusion ablation"
assert cfg.model.keystate_fusion_mode == "none", cfg.model.keystate_fusion_mode
assert cfg.data.repo_id == "${TRAIN_REPO_ID}", (cfg.data.repo_id, "${TRAIN_REPO_ID}")
print("[OK] Stage3 pred no-fusion config loaded:", cfg.name, "repo_id=", cfg.data.repo_id, "fusion=", cfg.model.use_keystate_fusion)
PY

# ========= Copy train300 norm stats to Stage3 train/test assets =========
echo "========== Copy train300 norm stats to Stage3 assets =========="
mkdir -p "${ASSETS_BASE_DIR}/${STAGE3_CONFIG_NAME}/${TRAIN_REPO_ID}"
mkdir -p "${ASSETS_BASE_DIR}/${STAGE3_CONFIG_NAME}/${TEST_REPO_ID}"
cp "${TRAIN_NORM_STATS_PATH}" "${ASSETS_BASE_DIR}/${STAGE3_CONFIG_NAME}/${TRAIN_REPO_ID}/norm_stats.json"
cp "${TRAIN_NORM_STATS_PATH}" "${ASSETS_BASE_DIR}/${STAGE3_CONFIG_NAME}/${TEST_REPO_ID}/norm_stats.json"
echo "[OK] Stage3 train norm stats: ${ASSETS_BASE_DIR}/${STAGE3_CONFIG_NAME}/${TRAIN_REPO_ID}/norm_stats.json"
echo "[OK] Stage3 test norm stats: ${ASSETS_BASE_DIR}/${STAGE3_CONFIG_NAME}/${TEST_REPO_ID}/norm_stats.json"

# ========= Train Stage3 pred on TRAIN300 only =========
if [[ -e "${TRAIN_CKPT_DIR}" && "${OVERWRITE:-0}" != "1" && "${RESUME:-0}" != "1" && "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "[ERROR] Checkpoint dir already exists: ${TRAIN_CKPT_DIR}"
  echo "Use RESUME=1 to resume, SKIP_TRAIN=1 to only validate, OVERWRITE=1 to intentionally replace, or set EXP_NAME to a new run name."
  exit 1
fi

TRAIN_FLAGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  TRAIN_FLAGS+=(--overwrite)
elif [[ "${RESUME:-0}" == "1" ]]; then
  TRAIN_FLAGS+=(--resume)
fi

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  echo "[SKIP] SKIP_TRAIN=1, skip Stage3 training."
else
  echo "========== Train Stage3 pred no-fusion on TRAIN300 only =========="
  "${PY_CMD[@]}" scripts/train.py "${STAGE3_CONFIG_NAME}" \
    --project-name openpi-keystate \
    --exp-name "${EXP_NAME}" \
    --assets-base-dir "${ASSETS_BASE_DIR}" \
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
    --data.repo-id "${TRAIN_REPO_ID}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    --num-workers "${TRAIN_NUM_WORKERS}" \
    --num-train-steps "${NUM_TRAIN_STEPS}" \
    --log-interval "${LOG_INTERVAL}" \
    --save-interval "${SAVE_INTERVAL}" \
    --keep-period "${KEEP_PERIOD}" \
    --fsdp-devices "${FSDP_DEVICES}" \
    "${TRAIN_FLAGS[@]}"
fi

# ========= Validate Stage3 checkpoints on held-out TEST30 =========
echo "========== Validate Stage3 pred checkpoints on held-out TEST30 =========="
"${PY_CMD[@]}" - <<PY
import csv
import gc
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding

config_name = "${STAGE3_CONFIG_NAME}"
test_repo_id = "${TEST_REPO_ID}"
assets_base_dir = "${ASSETS_BASE_DIR}"
ckpt_dir = Path("${TRAIN_CKPT_DIR}")
out_dir = Path("${VAL_EVAL_DIR}")
val_batch_size = int("${VAL_BATCH_SIZE}")
val_num_workers = int("${VAL_NUM_WORKERS}")
requested_batches = int("${VAL_NUM_BATCHES}")
fsdp_devices = min(int("${FSDP_DEVICES}"), max(1, jax.device_count()))

steps = sorted(int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit())
if not steps:
    raise RuntimeError(f"No checkpoint step dirs found in {ckpt_dir}")
print("checkpoint_steps", steps)
out_dir.mkdir(parents=True, exist_ok=True)
results = []

for step in steps:
    print(f"--- validating step {step} ---", flush=True)
    config = _config.get_config(config_name)
    config.assets_base_dir = assets_base_dir
    config.data.repo_id = test_repo_id
    config.batch_size = val_batch_size
    config.num_workers = val_num_workers
    config.fsdp_devices = fsdp_devices

    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_dataset(data_config, config.model)
    dataset = _data_loader.transform_dataset(dataset, data_config, skip_norm_stats=False)
    num_batches = requested_batches if requested_batches > 0 else len(dataset) // val_batch_size
    if num_batches <= 0:
        raise RuntimeError(f"Validation dataset too small: len={len(dataset)} batch={val_batch_size}")
    print(f"test_frames={len(dataset)} test_batches={num_batches} batch_size={val_batch_size}", flush=True)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=val_batch_size,
        sharding=data_sharding,
        shuffle=False,
        num_batches=num_batches,
        num_workers=val_num_workers,
        seed=0,
    )

    model = config.model.load(_model.restore_params(ckpt_dir / str(step) / "params", dtype=jnp.bfloat16))

    def eval_step(rng, observation, actions):
        flow, ks = model.compute_loss(rng, observation, actions, train=False)
        flow_loss = jnp.mean(flow)
        out = {"flow_loss": flow_loss, **ks}
        out["loss"] = flow_loss + sum(ks.values())
        return out

    peval = jax.jit(eval_step, in_shardings=(replicated_sharding, data_sharding, data_sharding), out_shardings=replicated_sharding)
    sums = {}
    count = 0
    rng = jax.random.key(45678 + step)
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        obs, actions = _model.Observation.from_dict(batch), batch["actions"]
        metrics = jax.device_get(peval(jax.random.fold_in(rng, i), obs, actions))
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + float(np.asarray(v))
        count += 1
    row = {"step": step, "num_batches": count}
    row.update({k: v / count for k, v in sorted(sums.items())})
    print(row, flush=True)
    results.append(row)
    del model
    gc.collect()
    jax.clear_caches()

all_fields = sorted({k for row in results for k in row.keys()})
fields = [k for k in ["step", "num_batches", "loss", "flow_loss", "loss_h_entry", "loss_ph", "loss_type", "loss_z_entry_descriptor"] if k in all_fields]
fields += [k for k in all_fields if k not in fields]
csv_path = out_dir / "test_metrics.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in results:
        writer.writerow(row)

best = min(results, key=lambda r: r.get("loss", float("inf")))
json_path = out_dir / "best_checkpoint.json"
json_path.write_text(json.dumps({"best": best, "all": results}, indent=2))
print(f"wrote {csv_path}")
print(f"wrote {json_path}")
print(f"BEST_STEP={best['step']} BEST_TEST_LOSS={best.get('loss')}")
PY



# ========= Pick checkpoint for rollout =========
if [[ -z "${EVAL_CHECKPOINT_ID:-}" ]]; then
  EVAL_CHECKPOINT_ID="$(${PY_CMD[@]} - <<PY
import json
from pathlib import Path
p = Path("${VAL_EVAL_DIR}") / "best_checkpoint.json"
print(json.loads(p.read_text())["best"]["step"])
PY
)"
fi
SELECTED_STEP_DIR="${TRAIN_CKPT_DIR}/${EVAL_CHECKPOINT_ID}"
if [[ ! -d "${SELECTED_STEP_DIR}/params" || ! -d "${SELECTED_STEP_DIR}/assets" ]]; then
  echo "[ERROR] Missing selected checkpoint step for rollout: ${SELECTED_STEP_DIR}"
  exit 1
fi

python - <<PY
import pathlib, yaml
root = pathlib.Path("${ROBOTWIN_ROOT}")
src = root / "task_config" / "${SOURCE_EVAL_TASK_CONFIG}.yml"
dst = root / "task_config" / "${EVAL_TASK_CONFIG}.yml"
with src.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
data["eval_video_log"] = bool(int("${EVAL_VIDEO_LOG}"))
with dst.open("w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
print(f"[OK] eval task config written: {dst}, eval_video_log={data['eval_video_log']}")
PY

LOCAL_MODEL_DIR="${PI0_ROOT}/checkpoints/${STAGE3_CONFIG_NAME}/${EXP_NAME}"
LOCAL_STEP_LINK="${LOCAL_MODEL_DIR}/${EVAL_CHECKPOINT_ID}"
mkdir -p "${LOCAL_MODEL_DIR}"
if [[ -e "${LOCAL_STEP_LINK}" && ! -L "${LOCAL_STEP_LINK}" ]]; then
  echo "[ERROR] Local eval checkpoint path exists and is not a symlink: ${LOCAL_STEP_LINK}"
  exit 1
fi
ln -sfn "${SELECTED_STEP_DIR}" "${LOCAL_STEP_LINK}"
echo "[OK] Eval checkpoint symlink: ${LOCAL_STEP_LINK} -> ${SELECTED_STEP_DIR}"

EVAL_PYTHONPATH="${ROBOTWIN_ROOT}:${EVAL_CUROBO_SRC}:${PI0_ROOT}/src:${PI0_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"
PYTHONPATH="${EVAL_PYTHONPATH}" "${EVAL_PYTHON_BIN}" - <<PY
from curobo.types.math import Pose as CuroboPose
import openpi.training.config as _config
cfg = _config.get_config("${STAGE3_CONFIG_NAME}")
assert cfg.model.use_checkpoint_head, "h_entry prediction head is not enabled"
assert not cfg.model.use_keystate_fusion, "Expected no-fusion ablation"
assert cfg.model.keystate_fusion_mode == "none", cfg.model.keystate_fusion_mode
print("[OK] Eval Python imports OpenPI/Curobo and config is no-fusion")
PY

cd "${ROBOTWIN_ROOT}"
FIRST_GPU="$(python - <<PY
visible = "${CUDA_VISIBLE_DEVICES:-0}"
items = [x.strip() for x in visible.split(',') if x.strip()]
print(items[0] if items else '0')
PY
)"
for seed in ${EVAL_SEEDS}; do
  CKPT_SETTING="${EXP_NAME}_ckpt${EVAL_CHECKPOINT_ID}_adaptive_hbin0_${INSIDE_PI0_STEP}_else_${OUTSIDE_PI0_STEP}_seed${seed}_test${TEST_NUM}_iseed${INSTRUCTION_SEED}"
  echo "========== No-fusion adaptive eval checkpoint=${EVAL_CHECKPOINT_ID} seed=${seed} =========="
  CUDA_VISIBLE_DEVICES="${FIRST_GPU}" PYTHONWARNINGS=ignore::UserWarning PYTHONPATH="${EVAL_PYTHONPATH}" \
    "${EVAL_PYTHON_BIN}" script/eval_policy.py \
      --config policy/pi0/deploy_policy.yml \
      --overrides \
      --task_name "${TASK_NAME}" \
      --task_config "${EVAL_TASK_CONFIG}" \
      --train_config_name "${STAGE3_CONFIG_NAME}" \
      --model_name "${EXP_NAME}" \
      --ckpt_setting "${CKPT_SETTING}" \
      --checkpoint_id "${EVAL_CHECKPOINT_ID}" \
      --pi0_step "${PI0_STEP_FALLBACK}" \
      --adaptive_pi0_step True \
      --outside_pi0_step "${OUTSIDE_PI0_STEP}" \
      --inside_pi0_step "${INSIDE_PI0_STEP}" \
      --test_num "${TEST_NUM}" \
      --seed "${seed}" \
      --instruction_seed "${INSTRUCTION_SEED}" \
      --policy_name pi0
done

echo "========== DONE =========="
echo "Stage3 config: ${STAGE3_CONFIG_NAME}"
echo "Fusion: disabled (KeyState heads still enabled for adaptive chunking)"
echo "Stage2 init params: ${STAGE2_PARAMS_PATH}"
echo "Stage3 train repo: ${TRAIN_REPO_ID}"
echo "Stage3 held-out test repo: ${TEST_REPO_ID}"
echo "Stage3 checkpoint dir: ${TRAIN_CKPT_DIR}"
echo "Held-out test metrics: ${VAL_EVAL_DIR}/test_metrics.csv"
echo "Best Stage3 checkpoint: ${VAL_EVAL_DIR}/best_checkpoint.json"
