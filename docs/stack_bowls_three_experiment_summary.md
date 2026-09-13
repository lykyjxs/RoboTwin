# Stack Bowls Three KeyState / Pi0 Experiment Summary

This note summarizes the experiments run so far for **KeyState-aware VLA** on RoboTwin `stack_bowls_three`, including data protocol, checkpoint locations, evaluation scripts, result directories, and current conclusions.

## 1. Evaluation protocol used in the latest comparisons

Unless otherwise stated, the recent Stack Bowls rollout comparisons use:

```text
task:             stack_bowls_three
task config base: demo_clean
test_num:         100 expert-feasible RoboTwin episodes
seed group:       0
start_seed:       100000
instruction_seed: 777
policy:           pi0
```

RoboTwin first filters candidate seeds through expert feasibility checks, then evaluates the policy on the first `test_num` feasible episodes. With `instruction_seed=777`, the language instruction is selected deterministically per actual environment seed.

Important caveat: RoboTwin/Sapien/Curobo rollouts are not perfectly deterministic. Independent runs with the same nominal seed/instruction protocol may differ by a few episodes.

## 2. Main trained checkpoints

### 2.1 Ours: Stage3 pred late cross-attention

Final Stage3 model used in rollout:

```text
config:
  pi0_base_aloha_robotwin_stack_bowls_three_keystate_stage3_pred_late_xattn_lora
experiment:
  stack_bowls_three_300_stage3_pred_late_xattn_from_stage2_5000_lora
checkpoint:
  10000
```

Shared checkpoint path:

```text
./checkpoints/openpi/openpi-assets/checkpoints/keystate_stage3/
  pi0_base_aloha_robotwin_stack_bowls_three_keystate_stage3_pred_late_xattn_lora/
    stack_bowls_three_300_stage3_pred_late_xattn_from_stage2_5000_lora/
      10000/{params,assets}
```

The adaptive deployment rule is:

```text
predicted h_entry_bin == 0 -> inside key-state window  -> execute 25 actions
predicted h_entry_bin != 0 -> outside key-state window -> execute 50 actions
```

The selected chunk is always fully executed before re-observation/re-prediction.

### 2.2 Pi0 fine-tune baseline

Baseline experiment:

```text
config:
  pi0_base_aloha_robotwin_stack_bowls_three_lora
experiment:
  stack_bowls_three_300_pi0_baseline_lora_rerun15000_safe_v2
checkpoints under:
  ./checkpoints/openpi/openpi-assets/checkpoints/baseline/
    pi0_base_aloha_robotwin_stack_bowls_three_lora/
      stack_bowls_three_300_pi0_baseline_lora_rerun15000_safe_v2/
```

Relevant steps evaluated or being evaluated:

```text
5000
10000
15000
```

## 3. Scripts created / used

### 3.1 Ours Stage3 adaptive rollout

Shared script:

```text
./script/run_stack_bowls_stage3_pred_adaptive_rollout_eval.sh
```

Repo copy:

```text
third_party/RoboTwin/script/run_stack_bowls_stage3_pred_adaptive_rollout_eval.sh
```

Typical command:

```bash
CUDA_VISIBLE_DEVICES=0 \
EVAL_CHECKPOINT_ID=10000 \
EVAL_SEEDS="0 1 2" \
TEST_NUM=100 \
EVAL_VIDEO_LOG=0 \
bash ./script/run_stack_bowls_stage3_pred_adaptive_rollout_eval.sh
```

### 3.2 Ours vs Pi0 paired video evaluation

Shared script:

```text
./script/run_stack_bowls_stage3_vs_pi0_paired_eval.sh
```

Repo copy:

```text
third_party/RoboTwin/script/run_stack_bowls_stage3_vs_pi0_paired_eval.sh
```

Purpose:

```text
Run Pi0 fixed50 and Ours adaptive 50/25 on the same 100 episodes,
save videos, write episode_log.csv, and generate side-by-side comparison videos.
```

Output directory:

```text
third_party/RoboTwin/eval_result/stack_bowls_three/pi0/demo_clean_stage3_vs_pi0_paired_video_eval/
```

### 3.3 Pi0 checkpoint sweep

Main multi-GPU checkpoint sweep:

```text
./script/run_stack_bowls_pi0_ckpt_sweep_8gpu.sh
```

Repo copy:

```text
third_party/RoboTwin/script/run_stack_bowls_pi0_ckpt_sweep_multigpu.sh
```

Target:

```text
Pi0 finetune checkpoints 5000 / 10000 / 15000
fixed pi0_step = 50
```

Separate spare scripts:

```text
./script/run_stack_bowls_pi0_ckpt15000_spare_eval.sh
./script/run_stack_bowls_pi0_ckpt10000_spare_eval.sh
```

These use independent output directories to avoid clobbering other custom tasks.

### 3.4 Pi0 fixed chunk-length sweep

Script:

```text
./script/run_stack_bowls_pi0_chunk_sweep_multigpu.sh
```

Target:

```text
Pi0 finetune ckpt15000
pi0_step = 50 / 25 / 10
```

Output directory:

```text
third_party/RoboTwin/eval_result/stack_bowls_three/pi0/demo_clean_pi0_chunk_sweep_multigpu_no_video_eval/
```

## 4. Result locations and current numbers

### 4.1 Formal Ours adaptive no-video evaluation

Output directory:

```text
third_party/RoboTwin/eval_result/stack_bowls_three/pi0/demo_clean_stage3_adaptive_no_video_eval/
```

Results:

| Method | Seed group | Success |
|---|---:|---:|
| Ours adaptive 50/25, Stage3 ckpt10000 | seed0 | 56/100 = 56.0% |
| Ours adaptive 50/25, Stage3 ckpt10000 | seed1 | 63/100 = 63.0% |
| Ours adaptive 50/25, Stage3 ckpt10000 | seed2 | 54/100 = 54.0% |

Aggregate:

```text
total: 173/300 = 57.7%
mean over seed groups: 57.7 ± 4.7%
```

Result files:

```text
.../stack_bowls_three_300_stage3_pred_late_xattn_from_stage2_5000_lora_ckpt10000_adaptive_hbin0_25_else_50_seed0/2026-06-26 07:01:28/_result.txt = 0.56
.../stack_bowls_three_300_stage3_pred_late_xattn_from_stage2_5000_lora_ckpt10000_adaptive_hbin0_25_else_50_seed1/2026-06-26 09:36:21/_result.txt = 0.63
.../stack_bowls_three_300_stage3_pred_late_xattn_from_stage2_5000_lora_ckpt10000_adaptive_hbin0_25_else_50_seed2/2026-06-26 12:04:01/_result.txt = 0.54
```

### 4.2 Paired video evaluation: Pi0 fixed50 vs Ours adaptive

Output directory:

```text
third_party/RoboTwin/eval_result/stack_bowls_three/pi0/demo_clean_stage3_vs_pi0_paired_video_eval/
```

Completed 100-episode results:

| Method | Protocol | Success |
|---|---|---:|
| Pi0 finetune ckpt15000 fixed50 | paired video eval, seed0, instruction_seed777 | 55/100 = 55.0% |
| Ours Stage3 ckpt10000 adaptive 50/25 | paired video eval, seed0, instruction_seed777 | 64/100 = 64.0% |

Result files:

```text
.../stack_bowls_three_300_pi0_baseline_lora_rerun15000_safe_v2_pi0step50_paired100_start100000_iseed777/2026-06-27 16:34:02/_result.txt = 0.55
.../stack_bowls_three_300_stage3_pred_late_xattn_from_stage2_5000_lora_ckpt10000_adaptive_hbin0_25_else_50_paired100_start100000_iseed777/2026-06-28 03:20:49/_result.txt = 0.64
```

Per-episode logs:

```text
.../2026-06-27 16:34:02/episode_log.csv
.../2026-06-28 03:20:49/episode_log.csv
```

Paired comparison files:

```text
.../paired_comparisons/ours_success_pi0_fail.csv
.../paired_comparisons/ours_fail_pi0_success.csv
.../paired_comparisons/ours_success_pi0_fail/*.mp4
.../paired_comparisons/ours_fail_pi0_success/*.mp4
```

Current generated case counts from the paired comparison directory:

```text
Ours success / Pi0 fail: 19 cases, 7 side-by-side videos generated
Ours fail / Pi0 success: 16 cases, 8 side-by-side videos generated
```

Caveat: `paired_comparisons/summary.txt` was generated from an earlier incomplete baseline run with 81 episodes. The correct success rates above should be read from the complete `_result.txt` files and complete `episode_log.csv` files.

### 4.3 Pi0 fixed chunk-length sweep, ckpt15000

Output directory:

```text
third_party/RoboTwin/eval_result/stack_bowls_three/pi0/demo_clean_pi0_chunk_sweep_multigpu_no_video_eval/
```

Protocol:

```text
Pi0 finetune ckpt15000
seed0 / start_seed=100000
test_num=100
instruction_seed=777
video off
```

Results:

| Method | Chunk length | Success |
|---|---:|---:|
| Pi0 ckpt15000 | 50 | 52/100 = 52.0% |
| Pi0 ckpt15000 | 25 | 77/100 = 77.0% |
| Pi0 ckpt15000 | 10 | 66/100 = 66.0% |

Result files:

```text
.../stack_bowls_three_300_pi0_baseline_lora_rerun15000_safe_v2_ckpt15000_pi0step50_seed0_test100_iseed777/2026-06-27 17:04:22/_result.txt = 0.52
.../stack_bowls_three_300_pi0_baseline_lora_rerun15000_safe_v2_ckpt15000_pi0step25_seed0_test100_iseed777/2026-06-28 04:22:32/_result.txt = 0.77
.../stack_bowls_three_300_pi0_baseline_lora_rerun15000_safe_v2_ckpt15000_pi0step10_seed0_test100_iseed777/2026-06-28 15:04:02/_result.txt = 0.66
```

Important note: `chunk_sweep_summary.txt` may show `missing` for pi0_step=10 because an earlier failed run directory was selected. The valid result is the `_result.txt` above.

### 4.4 Pi0 checkpoint sweep with fixed50

Main output directory:

```text
third_party/RoboTwin/eval_result/stack_bowls_three/pi0/demo_clean_pi0_ckpt_sweep_8gpu_no_video_eval/
```

Completed ckpt5000 results:

| Checkpoint | Seed group | Success |
|---:|---:|---:|
| 5000 | seed0 | 55/100 = 55.0% |
| 5000 | seed1 | 42/100 = 42.0% |
| 5000 | seed2 | 41/100 = 41.0% |

Mean over seed groups:

```text
ckpt5000 fixed50 mean = 46.0%
```

Result files:

```text
.../ckpt5000_pi0step50_seed0_test100_iseed777/2026-06-27 07:35:02/_result.txt = 0.55
.../ckpt5000_pi0step50_seed1_test100_iseed777/2026-06-27 18:35:24/_result.txt = 0.42
.../ckpt5000_pi0step50_seed2_test100_iseed777/2026-06-28 06:24:09/_result.txt = 0.41
```

ckpt10000 status:

```text
No completed result in the checked output directory yet.
A spare script exists at ./script/run_stack_bowls_pi0_ckpt10000_spare_eval.sh.
```

ckpt15000 spare output directory:

```text
third_party/RoboTwin/eval_result/stack_bowls_three/pi0/demo_clean_pi0_ckpt15000_spare_no_video_eval/
```

Completed ckpt15000 seed0 result:

```text
ckpt15000 fixed50 seed0 = 56/100 = 56.0%
```

Result file:

```text
.../stack_bowls_three_300_pi0_baseline_lora_rerun15000_safe_v2_ckpt15000_pi0step50_seed0_test100_iseed777/2026-06-28 07:39:11/_result.txt = 0.56
```

## 5. Current interpretation

### 5.1 Ours vs Pi0 fixed50

Under the recent seed0 / instruction_seed777 paired protocol:

```text
Pi0 fixed50:     55.0%
Ours adaptive:   64.0%
```

So Ours improves over Pi0 fixed50 by about:

```text
+9.0 percentage points
```

Under the earlier no-video formal Ours protocol over 3 seed groups, Ours reached:

```text
57.7% over 300 rollouts
```

### 5.2 Strong fixed25 baseline

The fixed chunk-length sweep shows:

```text
Pi0 fixed25: 77.0% on seed0 / instruction_seed777
```

This is stronger than Ours adaptive on the same broad seed/instruction protocol:

```text
Ours adaptive 50/25: 64.0% in paired seed0 eval
Pi0 fixed25:         77.0% in no-video seed0 chunk sweep
```

This means the current method should not be claimed as strictly better than every fixed chunk baseline. A more accurate statement is:

```text
Ours improves over the long-chunk fixed50 baseline, but fixed25 is currently a very strong baseline for stack_bowls_three.
```

This suggests `stack_bowls_three` benefits heavily from frequent re-observation/replanning. For a paper result, fixed25 must be included as a baseline.

### 5.3 Success rate vs efficiency trade-off

The adaptive method was designed to preserve long chunks outside key-state windows and use shorter chunks inside/near key windows. Therefore success rate should be reported together with efficiency metrics such as:

```text
average number of policy queries per episode
average executed chunk count per episode
average episode length / smoothness indicators
```

Without these efficiency metrics, fixed25 appears stronger by success rate alone.

## 6. Known issues / bookkeeping notes

1. Some summary files were generated before all reruns completed and may point to failed or incomplete run directories. Prefer the latest `_result.txt` plus `episode_log.csv` when in doubt.

2. A local eval checkpoint path once became a text file instead of a symlink:

```text
policy/pi0/checkpoints/pi0_base_aloha_robotwin_stack_bowls_three_lora/
  stack_bowls_three_300_pi0_baseline_lora_rerun15000_safe_v2/15000
```

It should be a symlink to:

```text
./checkpoints/openpi/openpi-assets/checkpoints/baseline/
  pi0_base_aloha_robotwin_stack_bowls_three_lora/
    stack_bowls_three_300_pi0_baseline_lora_rerun15000_safe_v2/15000
```

If it becomes a text file again, delete it and recreate the symlink with `ln -sfn`.

3. The Pi0 ckpt10000 fixed50 sweep has not produced a completed result in the checked directories yet. Use:

```bash
CUDA_VISIBLE_DEVICES=0 \
EVAL_SEEDS="0" \
bash ./script/run_stack_bowls_pi0_ckpt10000_spare_eval.sh
```

for a quick seed0 completion.

## 7. Recommended final table structure

For the paper / report, use a table like:

| Method | Checkpoint | Chunk policy | Seed groups | Success |
|---|---:|---|---|---:|
| Pi0 finetune | best of 5k/10k/15k | fixed50 | 0/1/2 | TBD |
| Pi0 finetune | ckpt15000 | fixed50 | seed0 | 52-56% depending run |
| Pi0 finetune | ckpt15000 | fixed25 | seed0 | 77.0% |
| Pi0 finetune | ckpt15000 | fixed10 | seed0 | 66.0% |
| Ours Stage3 | ckpt10000 | adaptive 50/25 | seed0 | 64.0% paired |
| Ours Stage3 | ckpt10000 | adaptive 50/25 | seed0/1/2 | 57.7% total |

The final version should standardize all rows under one exact protocol, ideally:

```text
no-video
instruction_seed=777
seed groups 0/1/2
100 expert-feasible episodes per seed group
```

and include policy-query efficiency metrics.
