# KeyState Stage3 Rollout Evaluation

This document records the rollout/evaluation workflow for the `stack_bowls_three` KeyState Stage3 model and the matched Pi0 fine-tune baseline.

## Final Stage3 model

The deployment model used for the main Stage3 rollout is:

```text
config:     pi0_base_aloha_robotwin_stack_bowls_three_keystate_stage3_pred_late_xattn_lora
exp:        stack_bowls_three_300_stage3_pred_late_xattn_from_stage2_5000_lora
checkpoint: 10000
```

The checkpoint lives in the shared OpenPI checkpoint tree:

```text
./checkpoints/openpi/openpi-assets/checkpoints/keystate_stage3/
  pi0_base_aloha_robotwin_stack_bowls_three_keystate_stage3_pred_late_xattn_lora/
    stack_bowls_three_300_stage3_pred_late_xattn_from_stage2_5000_lora/
      10000/{params,assets}
```

## Adaptive chunk scheduler

The paper method uses a two-mode KeyState-aware action chunk scheduler:

```text
predicted h_entry_bin == 0 -> inside key-state window  -> execute 25 actions
predicted h_entry_bin != 0 -> outside key-state window -> execute 50 actions
```

The selected chunk is executed completely before the next observation/re-prediction. In particular, if the policy starts outside a key window and chooses a 50-action chunk, rollout does not interrupt the chunk mid-execution even if the trajectory enters the key window. This keeps execution smooth while still shortening chunks after the next observation detects `h_entry_bin == 0`.

Implementation files:

```text
policy/pi0/src/openpi/models/pi0.py      # predict_keystate(...) exposes h_entry prediction at inference
policy/pi0/src/openpi/policies/policy.py # returns KeyState auxiliary predictions from Policy.infer(...)
policy/pi0/pi_model.py                   # chooses adaptive_pi0_step from h_entry bin
policy/pi0/deploy_policy.py              # slices the sampled action chunk by the adaptive length
policy/pi0/deploy_policy.yml             # adaptive_pi0_step/outside_pi0_step/inside_pi0_step defaults
```

## Formal Ours evaluation

For the main Stage3 adaptive evaluation, use:

```bash
CUDA_VISIBLE_DEVICES=0 \
EVAL_CHECKPOINT_ID=10000 \
EVAL_SEEDS="0 1 2" \
TEST_NUM=100 \
EVAL_VIDEO_LOG=0 \
bash ./script/run_stack_bowls_stage3_pred_adaptive_rollout_eval.sh
```

Equivalent repo copy:

```bash
CUDA_VISIBLE_DEVICES=0 \
EVAL_CHECKPOINT_ID=10000 \
EVAL_SEEDS="0 1 2" \
TEST_NUM=100 \
EVAL_VIDEO_LOG=0 \
bash script/run_stack_bowls_stage3_pred_adaptive_rollout_eval.sh
```

The existing formal result over 300 rollouts is:

```text
seed0: 56/100 = 56.0%
seed1: 63/100 = 63.0%
seed2: 54/100 = 54.0%
total: 173/300 = 57.7%
mean over seed groups: 57.7 ± 4.7%
```

## Paired qualitative evaluation against Pi0 fine-tune

For qualitative analysis, evaluate both policies on the same 100 expert-feasible RoboTwin seeds and the same instruction sampling seed, save all videos, and then build side-by-side videos for the two disagreement directions.

Script:

```text
script/run_stack_bowls_stage3_vs_pi0_paired_eval.sh
```

Main command:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash script/run_stack_bowls_stage3_vs_pi0_paired_eval.sh
```

Default protocol:

```text
task:             stack_bowls_three
source config:    demo_clean
test_num:         100
seed group:       0
start_seed:       100000
instruction_seed: 777
video:            enabled
```

Default models:

```text
Pi0 fine-tune:
  config:     pi0_base_aloha_robotwin_stack_bowls_three_lora
  exp:        stack_bowls_three_300_pi0_baseline_lora_rerun15000_safe_v2
  checkpoint: 15000
  pi0_step:   50

Ours:
  config:     pi0_base_aloha_robotwin_stack_bowls_three_keystate_stage3_pred_late_xattn_lora
  exp:        stack_bowls_three_300_stage3_pred_late_xattn_from_stage2_5000_lora
  checkpoint: 10000
  adaptive:   h_entry_bin == 0 -> 25, otherwise -> 50
```

Outputs:

```text
eval_result/stack_bowls_three/pi0/demo_clean_stage3_vs_pi0_paired_video_eval/
```

Each run directory contains:

```text
_result.txt
episode_log.csv
episode0.mp4 ... episode99.mp4
```

The paired comparison directory contains:

```text
paired_comparisons/summary.txt
paired_comparisons/ours_success_pi0_fail.csv
paired_comparisons/ours_fail_pi0_success.csv
paired_comparisons/ours_success_pi0_fail/*.mp4
paired_comparisons/ours_fail_pi0_success/*.mp4
```

## Per-episode logging

`script/eval_policy.py` now writes `episode_log.csv` for every rollout run. Columns:

```text
episode_id
actual_seed
instruction
success
steps
video_path
ckpt_setting
```

This is necessary because `_result.txt` only stores aggregate success rate and cannot identify which episodes produced failure/success pairs.

## Seed protocol and reproducibility

RoboTwin uses `--seed` to choose the start of an expert-feasible seed range:

```text
--seed 0 -> default start_seed = 100000
--seed 1 -> default start_seed = 200000
--seed 2 -> default start_seed = 300000
```

`eval_policy.py` also supports explicit `--start_seed`, which is used by paired analysis to make the protocol unambiguous.

Recommended reproducible wording:

```text
We evaluate 100 expert-feasible RoboTwin episodes starting from seed 100000. For paired qualitative analysis, both policies use the same start_seed and instruction_seed.
```

## Pi0 checkpoint sweep

To choose the strongest Pi0 fine-tune checkpoint, sweep:

```text
checkpoint_id = 5000 / 10000 / 15000
pi0_step      = 50
test_num      = 100
```

Multi-GPU script:

```text
script/run_stack_bowls_pi0_ckpt_sweep_multigpu.sh
```

Two-GPU quick seed0 sweep:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
EVAL_SEEDS="0" \
MAX_PARALLEL=2 \
bash script/run_stack_bowls_pi0_ckpt_sweep_multigpu.sh
```

Two-GPU full sweep over seed groups 0/1/2:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MAX_PARALLEL=2 \
bash script/run_stack_bowls_pi0_ckpt_sweep_multigpu.sh
```

The script assigns one rollout process per visible GPU. With two GPUs it runs two eval jobs at a time; with eight GPUs it runs up to eight at a time.

Summary output:

```text
eval_result/stack_bowls_three/pi0/demo_clean_pi0_ckpt_sweep_8gpu_no_video_eval/ckpt_sweep_summary.txt
```

This summary reports each checkpoint/seed success rate plus per-checkpoint mean/std when multiple seed groups are evaluated.
