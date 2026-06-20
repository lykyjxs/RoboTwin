# KeyState Stage 3 — Late Cross-Attention Fusion

## Scope

Stage3 is the first stage that feeds KeyState information back into action generation. Stage1/Stage2 keep KeyState as auxiliary prediction losses; Stage3 changes the action path itself.

The first implemented variant is **Late KeyState Cross-Attention**:

```text
action_hidden = suffix_out[:, -action_horizon:]
delta = CrossAttention(
    query = action_hidden,
    key   = KeyState memory tokens,
    value = KeyState memory tokens,
)
action_hidden' = action_hidden + alpha * delta
v_t = action_out_proj(action_hidden')
```

This keeps the Pi0 prefix/suffix transformer unchanged and inserts a compact KeyState memory adapter immediately before `action_out_proj`.

## Branch structure

Stage3 is split into two branches:

```text
stage3-common
feature/keystate-stage3-late-xattn
```

`stage3-common` contains the shared Stage3 interface/config surface. `feature/keystate-stage3-late-xattn` implements the first concrete fusion adapter.

## h_entry bucket contract

Stage3 uses the revised Stage1/Stage2 bucket contract:

```text
bin 0: h_entry == 0                 # inside checkpoint window
bin 1: 1 <= h_entry < 4             # very near entry, still before window
bin 2: 4 <= h_entry < 7
bin 3: 7 <= h_entry < 11
bin 4: 11 <= h_entry < 21
bin 5: 21 <= h_entry < 51
bin 6: h_entry >= 51
invalid: h_entry < 0
```

Because `bin 0` already means inside checkpoint window, Stage3 does **not** add a separate `keystate_inside_window` token/head.

## KeyState memory tokens

The adapter builds exactly four KeyState memory tokens:

```text
[type, h_entry_bin, phase, z_entry]
```

Implementation in `policy/pi0/src/openpi/models/pi0.py`:

- `ks_type_embed`: embeds current-or-next checkpoint type.
- `ks_horizon_embed`: embeds revised 7-bin `h_entry` target/prediction.
- `ks_phase_proj`: projects semantic phase vector to action-expert width.
- `ks_z_entry_proj`: projects 64D `z_entry_descriptor` to action-expert width.

The memory tensor shape is:

```text
[B, 4, action_expert_width]
```

For Pi0 base Aloha RobotWin this width is 1024.

## Fusion source modes

Config field:

```python
ks_fusion_source: str = "gt"  # "gt" | "pred" | "mixed"
```

Semantics:

- `gt`: builds memory from ground-truth KeyState labels and z target in `Observation`.
- `pred`: builds memory from the model's own KeyState heads.
- `mixed`: during training, samples per-example between GT and predicted KeyState; during sampling/inference it falls back to `pred`.

The current training config uses `gt` for the first teacher-forced smoke:

```text
pi0_base_aloha_robotwin_keystate_stage3_late_xattn_lora
```

Real rollout experiments should use/check `pred` before claiming deployment performance, because `gt` is oracle fusion.

## Alpha initialization

The residual scale is a learned scalar parameter:

```python
ks_late_xattn_alpha = 1e-3
```

It is intentionally small but nonzero. `0.0` would make the adapter output a no-op and block gradients to the cross-attention parameters at the first step.

## Model changes

Key files:

```text
policy/pi0/src/openpi/models/pi0.py
policy/pi0/src/openpi/training/config.py
```

`pi0.py` adds:

- Stage3 config fields:
  - `use_keystate_fusion`
  - `keystate_fusion_mode`
  - `ks_fusion_source`
  - `ks_xattn_num_heads`
  - `ks_xattn_alpha_init`
  - `ks_xattn_use_layernorm`
  - `ks_mixed_gt_prob`
- `KeyStateLateCrossAttention`, a small native NNX cross-attention module.
- helper methods:
  - `_pool_prefix`
  - `_predict_keystate_features`
  - `_gt_keystate_features`
  - `_select_keystate_features`
  - `_build_keystate_memory_tokens`
  - `_apply_keystate_late_fusion`
- train-time insertion in `compute_loss` before `action_out_proj`.
- sampling-time insertion in `sample_actions.step` before `action_out_proj`.

`config.py` adds:

```text
pi0_base_aloha_robotwin_keystate_stage3_late_xattn_lora
```

This config extends Stage2 with:

```python
use_keystate_fusion=True
keystate_fusion_mode="late_xattn"
ks_fusion_source="gt"
ks_xattn_alpha_init=1e-3
```

It uses the action-expert z-entry dataset:

```text
place_a2b_left_keystate_z_entry_descriptor_actionexpert_oneshot
```

and reuses existing Stage1 norm stats because Stage3 does not change state/action normalization.

## Verification

### Static checks

```bash
python -m py_compile \
  policy/pi0/src/openpi/models/pi0.py \
  policy/pi0/src/openpi/training/config.py

git diff --check
```

### Dummy pred-mode smoke

A CPU dummy-model smoke verified that `ks_fusion_source="pred"` works without GT labels in sampling:

```text
flow (2, 50) True
sample (2, 50, 32) True
```

### Real Stage3 2-step smoke

Command:

```bash
cd third_party/RoboTwin/policy/pi0
HF_LEROBOT_HOME=$PWD/training_data \
OPENPI_DATA_HOME=./checkpoints/openpi \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python scripts/train.py pi0_base_aloha_robotwin_keystate_stage3_late_xattn_lora \
  --project-name openpi-keystate \
  --exp-name stage3_late_xattn_gt_smoke_2step \
  --checkpoint-base-dir ./checkpoints/openpi/openpi-assets/checkpoints/keystate \
  --overwrite \
  --no-wandb-enabled \
  --data.repo-id place_a2b_left_keystate_z_entry_descriptor_actionexpert_oneshot \
  --batch-size 2 \
  --num-workers 0 \
  --num-train-steps 2 \
  --log-interval 1 \
  --save-interval 1000 \
  --no-save-final-checkpoint \
  --fsdp-devices 2
```

The smoke initialized the new Stage3 params, including:

```text
ks_horizon_embed.embedding:       (7, 1024)
ks_late_xattn.attn.query.kernel:  (1024, 8, 128)
ks_late_xattn.attn.key.kernel:    (1024, 8, 128)
ks_late_xattn.attn.value.kernel:  (1024, 8, 128)
ks_late_xattn.attn.out.kernel:    (8, 128, 1024)
ks_late_xattn_alpha:              ()
ks_phase_proj.kernel:             (3, 1024)
ks_z_entry_proj.kernel:           (64, 1024)
```

Losses:

```text
Step 0: flow_loss=0.0438, loss=1.1375, loss_h_entry=0.3383, loss_ph=0.3800, loss_type=0.3755, loss_z_entry_descriptor=0.0000
Step 1: flow_loss=0.0919, loss=0.9131, loss_h_entry=0.2154, loss_ph=0.2546, loss_type=0.2825, loss_z_entry_descriptor=0.0687
```

Conclusion: Stage3 late-xattn code path, config construction, DataLoader, model initialization, base weight loading, forward/backward pass, and loss logging all run successfully for the 2-step GT-fusion smoke.

## Notes and limitations

- `gt` fusion is teacher-forced/oracle and is appropriate for initial adapter learning checks.
- `pred` fusion path is implemented and dummy-smoked, but real rollout evaluation should explicitly train/evaluate with `ks_fusion_source="pred"` or a `mixed -> pred` curriculum.
- This branch implements late fusion only. Layer-wise fusion should branch from `stage3-common`, not from this late-xattn branch.
