#!/usr/bin/env bash
set -euo pipefail

# Paired 100-episode evaluation for stack_bowls_three:
#   - Ours: Stage3 pred late-xattn checkpoint step=10000, adaptive chunking h_bin0->25 else->50
#   - Pi0 finetune: fixed pi0_step=50 checkpoint step=15000
#
# The script saves all rollout videos, writes per-episode CSV logs, then creates side-by-side
# comparison videos for both disagreement directions:
#   1) Ours success, Pi0 finetune fail
#   2) Ours fail, Pi0 finetune success
#
# Main command:
#   CUDA_VISIBLE_DEVICES=0 bash ./script/run_stack_bowls_stage3_vs_pi0_paired_eval.sh
#
# Common overrides:
#   TEST_NUM=100 START_SEED=100000 INSTRUCTION_SEED=777 CUDA_VISIBLE_DEVICES=0 bash ...
#   MAKE_COMPARISONS_ONLY=1 bash ...  # reuse the latest existing logs/videos

# ========= Paths =========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PI0_ROOT="${PI0_ROOT:-${ROBOTWIN_ROOT}/policy/pi0}"
SHARE_ROOT="${SHARE_ROOT:-${ROBOTWIN_ROOT}/data}"
OPENPI_DATA_HOME_DIR="${OPENPI_DATA_HOME_DIR:-${SHARE_ROOT}/checkpoints/openpi}"

BASELINE_CHECKPOINT_BASE_DIR="${BASELINE_CHECKPOINT_BASE_DIR:-${OPENPI_DATA_HOME_DIR}/openpi-assets/checkpoints/baseline}"
STAGE3_CHECKPOINT_BASE_DIR="${STAGE3_CHECKPOINT_BASE_DIR:-${OPENPI_DATA_HOME_DIR}/openpi-assets/checkpoints/keystate_stage3}"

# ========= Model settings =========
BASELINE_CONFIG_NAME="${BASELINE_CONFIG_NAME:-pi0_base_aloha_robotwin_stack_bowls_three_lora}"
BASELINE_EXP_NAME="${BASELINE_EXP_NAME:-stack_bowls_three_300_pi0_baseline_lora_rerun15000_safe_v2}"
BASELINE_CHECKPOINT_ID="${BASELINE_CHECKPOINT_ID:-15000}"
BASELINE_PI0_STEP="${BASELINE_PI0_STEP:-50}"

OURS_CONFIG_NAME="${OURS_CONFIG_NAME:-pi0_base_aloha_robotwin_stack_bowls_three_keystate_stage3_pred_late_xattn_lora}"
OURS_EXP_NAME="${OURS_EXP_NAME:-stack_bowls_three_300_stage3_pred_late_xattn_from_stage2_5000_lora}"
OURS_CHECKPOINT_ID="${OURS_CHECKPOINT_ID:-10000}"
OURS_PI0_STEP_FALLBACK="${OURS_PI0_STEP_FALLBACK:-50}"
OURS_OUTSIDE_PI0_STEP="${OURS_OUTSIDE_PI0_STEP:-50}"
OURS_INSIDE_PI0_STEP="${OURS_INSIDE_PI0_STEP:-25}"

# ========= Eval settings =========
TASK_NAME="${TASK_NAME:-stack_bowls_three}"
SOURCE_EVAL_TASK_CONFIG="${SOURCE_EVAL_TASK_CONFIG:-demo_clean}"
EVAL_TASK_CONFIG="${EVAL_TASK_CONFIG:-demo_clean_stage3_vs_pi0_paired_video_eval}"
TEST_NUM="${TEST_NUM:-100}"
SEED_GROUP="${SEED_GROUP:-0}"
START_SEED="${START_SEED:-100000}"
INSTRUCTION_SEED="${INSTRUCTION_SEED:-777}"
MAKE_COMPARISONS_ONLY="${MAKE_COMPARISONS_ONLY:-0}"
MAX_COMPARISON_VIDEOS="${MAX_COMPARISON_VIDEOS:-0}"  # 0 means no cap

# ========= Runtime env =========
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-$(command -v python)}"
EVAL_CUROBO_SRC="${EVAL_CUROBO_SRC:-${ROBOTWIN_ROOT}/envs/curobo/src}"
export PYTHONWARNINGS=ignore::UserWarning
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID:-0}"
fi
FIRST_GPU="${EVAL_GPU_ID:-$(python - <<PY
visible = "${CUDA_VISIBLE_DEVICES:-0}"
items = [x.strip() for x in visible.split(',') if x.strip()]
print(items[0] if items else '0')
PY
)}"

EVAL_PYTHONPATH="${EVAL_CUROBO_SRC}:${PI0_ROOT}/src:${PI0_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"
OUT_ROOT="${ROBOTWIN_ROOT}/eval_result/${TASK_NAME}/pi0/${EVAL_TASK_CONFIG}"
PAIR_ROOT="${OUT_ROOT}/paired_comparisons"
BASELINE_CKPT_SETTING="${BASELINE_EXP_NAME}_pi0step${BASELINE_PI0_STEP}_paired100_start${START_SEED}_iseed${INSTRUCTION_SEED}"
OURS_CKPT_SETTING="${OURS_EXP_NAME}_ckpt${OURS_CHECKPOINT_ID}_adaptive_hbin0_${OURS_INSIDE_PI0_STEP}_else_${OURS_OUTSIDE_PI0_STEP}_paired100_start${START_SEED}_iseed${INSTRUCTION_SEED}"

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

BASELINE_SHARED_STEP_DIR="${BASELINE_CHECKPOINT_BASE_DIR}/${BASELINE_CONFIG_NAME}/${BASELINE_EXP_NAME}/${BASELINE_CHECKPOINT_ID}"
OURS_SHARED_STEP_DIR="${STAGE3_CHECKPOINT_BASE_DIR}/${OURS_CONFIG_NAME}/${OURS_EXP_NAME}/${OURS_CHECKPOINT_ID}"
if [[ ! -d "${BASELINE_SHARED_STEP_DIR}/params" || ! -d "${BASELINE_SHARED_STEP_DIR}/assets" ]]; then
  echo "[ERROR] Missing Pi0 finetune checkpoint: ${BASELINE_SHARED_STEP_DIR}"
  exit 1
fi
if [[ ! -d "${OURS_SHARED_STEP_DIR}/params" || ! -d "${OURS_SHARED_STEP_DIR}/assets" ]]; then
  echo "[ERROR] Missing Stage3 checkpoint: ${OURS_SHARED_STEP_DIR}"
  exit 1
fi

PYTHONPATH="${EVAL_PYTHONPATH}" "${EVAL_PYTHON_BIN}" - <<PY
from curobo.types.math import Pose as CuroboPose
import openpi.training.config as _config
base = _config.get_config("${BASELINE_CONFIG_NAME}")
ours = _config.get_config("${OURS_CONFIG_NAME}")
assert ours.model.use_keystate_fusion, "Stage3 fusion is not enabled"
assert ours.model.ks_fusion_source == "pred", ours.model.ks_fusion_source
assert ours.model.use_checkpoint_head, "Stage3 h_entry prediction head is not enabled"
print("[OK] Eval Python imports OpenPI and compatible Curobo")
print("[OK] Baseline config:", base.name)
print("[OK] Ours config:", ours.name, "fusion_source=", ours.model.ks_fusion_source)
PY

cat <<EOF
========== Paired Stage3-vs-Pi0 Eval ==========
ROBOTWIN_ROOT=${ROBOTWIN_ROOT}
TASK_NAME=${TASK_NAME}
EVAL_TASK_CONFIG=${EVAL_TASK_CONFIG}
TEST_NUM=${TEST_NUM}
SEED_GROUP=${SEED_GROUP}
START_SEED=${START_SEED}
INSTRUCTION_SEED=${INSTRUCTION_SEED}
BASELINE=${BASELINE_CONFIG_NAME}/${BASELINE_EXP_NAME}/${BASELINE_CHECKPOINT_ID}, pi0_step=${BASELINE_PI0_STEP}
OURS=${OURS_CONFIG_NAME}/${OURS_EXP_NAME}/${OURS_CHECKPOINT_ID}, adaptive h_bin0->${OURS_INSIDE_PI0_STEP} else->${OURS_OUTSIDE_PI0_STEP}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}
FIRST_GPU=${FIRST_GPU}
OUT_ROOT=${OUT_ROOT}
================================================
EOF

# ========= Prepare eval task config and checkpoint links =========
python - <<PY
import pathlib, yaml
root = pathlib.Path("${ROBOTWIN_ROOT}")
src = root / "task_config" / "${SOURCE_EVAL_TASK_CONFIG}.yml"
dst = root / "task_config" / "${EVAL_TASK_CONFIG}.yml"
if not src.exists():
    raise FileNotFoundError(src)
with src.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
data["eval_video_log"] = True
with dst.open("w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
print(f"[OK] eval task config written: {dst}, eval_video_log={data['eval_video_log']}")
PY

link_checkpoint() {
  local config_name="$1" exp_name="$2" ckpt_id="$3" shared_step_dir="$4"
  local local_model_dir="${PI0_ROOT}/checkpoints/${config_name}/${exp_name}"
  local local_step_link="${local_model_dir}/${ckpt_id}"
  mkdir -p "${local_model_dir}"
  if [[ -e "${local_step_link}" && ! -L "${local_step_link}" ]]; then
    echo "[ERROR] Local eval checkpoint path exists and is not a symlink: ${local_step_link}"
    exit 1
  fi
  ln -sfn "${shared_step_dir}" "${local_step_link}"
  echo "[OK] Eval checkpoint symlink: ${local_step_link} -> ${shared_step_dir}"
}
link_checkpoint "${BASELINE_CONFIG_NAME}" "${BASELINE_EXP_NAME}" "${BASELINE_CHECKPOINT_ID}" "${BASELINE_SHARED_STEP_DIR}"
link_checkpoint "${OURS_CONFIG_NAME}" "${OURS_EXP_NAME}" "${OURS_CHECKPOINT_ID}" "${OURS_SHARED_STEP_DIR}"

latest_run() {
  local ckpt_setting="$1"
  python - <<PY
from pathlib import Path
root=Path("${OUT_ROOT}")/"${ckpt_setting}"
runs=sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p:p.stat().st_mtime, reverse=True) if root.exists() else []
print(runs[0] if runs else "")
PY
}

run_eval() {
  local mode="$1"
  shift
  echo "========== Run ${mode} eval =========="
  cd "${ROBOTWIN_ROOT}"
  CUDA_VISIBLE_DEVICES="${FIRST_GPU}" PYTHONWARNINGS=ignore::UserWarning PYTHONPATH="${EVAL_PYTHONPATH}" \
    "${EVAL_PYTHON_BIN}" script/eval_policy.py "$@"
}

if [[ "${MAKE_COMPARISONS_ONLY}" != "1" ]]; then
  run_eval "Pi0 finetune" \
    --config policy/pi0/deploy_policy.yml \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config "${EVAL_TASK_CONFIG}" \
    --train_config_name "${BASELINE_CONFIG_NAME}" \
    --model_name "${BASELINE_EXP_NAME}" \
    --ckpt_setting "${BASELINE_CKPT_SETTING}" \
    --checkpoint_id "${BASELINE_CHECKPOINT_ID}" \
    --pi0_step "${BASELINE_PI0_STEP}" \
    --test_num "${TEST_NUM}" \
    --seed "${SEED_GROUP}" \
    --start_seed "${START_SEED}" \
    --instruction_seed "${INSTRUCTION_SEED}" \
    --policy_name pi0

  run_eval "Ours Stage3 adaptive" \
    --config policy/pi0/deploy_policy.yml \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config "${EVAL_TASK_CONFIG}" \
    --train_config_name "${OURS_CONFIG_NAME}" \
    --model_name "${OURS_EXP_NAME}" \
    --ckpt_setting "${OURS_CKPT_SETTING}" \
    --checkpoint_id "${OURS_CHECKPOINT_ID}" \
    --pi0_step "${OURS_PI0_STEP_FALLBACK}" \
    --adaptive_pi0_step True \
    --outside_pi0_step "${OURS_OUTSIDE_PI0_STEP}" \
    --inside_pi0_step "${OURS_INSIDE_PI0_STEP}" \
    --test_num "${TEST_NUM}" \
    --seed "${SEED_GROUP}" \
    --start_seed "${START_SEED}" \
    --instruction_seed "${INSTRUCTION_SEED}" \
    --policy_name pi0
fi

BASELINE_RUN="$(latest_run "${BASELINE_CKPT_SETTING}")"
OURS_RUN="$(latest_run "${OURS_CKPT_SETTING}")"
if [[ -z "${BASELINE_RUN}" || -z "${OURS_RUN}" ]]; then
  echo "[ERROR] Missing run dirs. baseline=${BASELINE_RUN} ours=${OURS_RUN}"
  exit 1
fi
if [[ ! -f "${BASELINE_RUN}/episode_log.csv" || ! -f "${OURS_RUN}/episode_log.csv" ]]; then
  echo "[ERROR] Missing episode logs:"
  echo "  ${BASELINE_RUN}/episode_log.csv"
  echo "  ${OURS_RUN}/episode_log.csv"
  exit 1
fi

mkdir -p "${PAIR_ROOT}/ours_success_pi0_fail" "${PAIR_ROOT}/ours_fail_pi0_success"

python - <<PY
import csv
from pathlib import Path
base_run = Path("${BASELINE_RUN}")
ours_run = Path("${OURS_RUN}")
pair_root = Path("${PAIR_ROOT}")
max_videos = int("${MAX_COMPARISON_VIDEOS}")

def read_log(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["actual_seed"] = int(row["actual_seed"])
            row["success"] = int(row["success"])
            rows.append(row)
    return rows

base_rows = read_log(base_run / "episode_log.csv")
ours_rows = read_log(ours_run / "episode_log.csv")
base_by_seed = {r["actual_seed"]: r for r in base_rows}
ours_by_seed = {r["actual_seed"]: r for r in ours_rows}
common = sorted(set(base_by_seed) & set(ours_by_seed))

def write_cases(name, rows):
    out = pair_root / f"{name}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["actual_seed", "instruction", "pi0_success", "ours_success", "pi0_video", "ours_video"])
        writer.writeheader()
        for b, o in rows:
            writer.writerow({
                "actual_seed": b["actual_seed"],
                "instruction": o["instruction"],
                "pi0_success": b["success"],
                "ours_success": o["success"],
                "pi0_video": b["video_path"],
                "ours_video": o["video_path"],
            })
    return out

ours_success_pi0_fail = [(base_by_seed[s], ours_by_seed[s]) for s in common if base_by_seed[s]["success"] == 0 and ours_by_seed[s]["success"] == 1]
ours_fail_pi0_success = [(base_by_seed[s], ours_by_seed[s]) for s in common if base_by_seed[s]["success"] == 1 and ours_by_seed[s]["success"] == 0]
write_cases("ours_success_pi0_fail", ours_success_pi0_fail)
write_cases("ours_fail_pi0_success", ours_fail_pi0_success)

summary = pair_root / "summary.txt"
base_success = sum(r["success"] for r in base_rows)
ours_success = sum(r["success"] for r in ours_rows)
summary.write_text(
    "Paired Stage3-vs-Pi0 evaluation\n"
    f"baseline_run={base_run}\n"
    f"ours_run={ours_run}\n"
    f"common_seed_count={len(common)}\n"
    f"pi0_success={base_success}/{len(base_rows)}={base_success/len(base_rows):.3f}\n"
    f"ours_success={ours_success}/{len(ours_rows)}={ours_success/len(ours_rows):.3f}\n"
    f"ours_success_pi0_fail={len(ours_success_pi0_fail)}\n"
    f"ours_fail_pi0_success={len(ours_fail_pi0_success)}\n",
    encoding="utf-8",
)
print(summary.read_text())

# Emit a shell-consumable manifest for ffmpeg creation.
manifest = pair_root / "make_side_by_side_manifest.tsv"
with manifest.open("w", encoding="utf-8") as f:
    for label, rows in [("ours_success_pi0_fail", ours_success_pi0_fail), ("ours_fail_pi0_success", ours_fail_pi0_success)]:
        limited = rows if max_videos <= 0 else rows[:max_videos]
        for b, o in limited:
            seed = b["actual_seed"]
            if label == "ours_success_pi0_fail":
                title_left = "Pi0 finetune - FAIL"
                title_right = "Ours Stage3 adaptive - SUCCESS"
            else:
                title_left = "Pi0 finetune - SUCCESS"
                title_right = "Ours Stage3 adaptive - FAIL"
            out = pair_root / label / f"seed{seed}_{label}.mp4"
            f.write("\t".join([label, str(seed), b["video_path"], o["video_path"], str(out), title_left, title_right]) + "\n")
print(f"MANIFEST={manifest}")
PY

MANIFEST="${PAIR_ROOT}/make_side_by_side_manifest.tsv"
MADE=0
while IFS=$'\t' read -r label seed pi0_video ours_video out title_left title_right; do
  [[ -z "${label:-}" ]] && continue
  if [[ ! -f "${pi0_video}" || ! -f "${ours_video}" ]]; then
    echo "[WARN] Missing source video for seed=${seed}: pi0=${pi0_video}, ours=${ours_video}"
    continue
  fi
  ffmpeg -y -loglevel error \
    -i "${pi0_video}" \
    -i "${ours_video}" \
    -filter_complex "[0:v]scale=640:-2,setsar=1,pad=640:ih+48:0:48:black,drawtext=text='${title_left}':x=10:y=12:fontsize=24:fontcolor=white[v0];[1:v]scale=640:-2,setsar=1,pad=640:ih+48:0:48:black,drawtext=text='${title_right}':x=10:y=12:fontsize=24:fontcolor=white[v1];[v0][v1]hstack=inputs=2[v]" \
    -map "[v]" -c:v libx264 -pix_fmt yuv420p -shortest "${out}"
  MADE=$((MADE + 1))
  echo "SIDE_BY_SIDE label=${label} seed=${seed} ${out}"
done < "${MANIFEST}"

echo "========== DONE =========="
echo "Baseline run: ${BASELINE_RUN}"
echo "Ours run: ${OURS_RUN}"
echo "Summary: ${PAIR_ROOT}/summary.txt"
echo "Case CSVs: ${PAIR_ROOT}/ours_success_pi0_fail.csv and ${PAIR_ROOT}/ours_fail_pi0_success.csv"
echo "Side-by-side videos under: ${PAIR_ROOT}/"
echo "Created side-by-side videos: ${MADE}"
