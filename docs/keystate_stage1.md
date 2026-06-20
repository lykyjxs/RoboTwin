# KeyState Stage 1 — Pi0 训练端方案、改动与验证记录

本文档记录 **KeyState-aware VLA** 项目 Stage 1 的训练/测评端实现、已完成修改、验证结果与待完成事项。

> 分支：`keystate-stage1-heads`  
> 范围：仅训练/测评端；本轮未修改部署端 `deploy_policy.py` / `deploy_policy.yml` / `pi_model.py`。

---

## 0. Stage 1 方案大纲

Stage 1 的目标是：**在 Pi0 / OpenPI 训练路径中加入 KeyState 监督头，让模型在生成 action chunk 的同时学习当前帧到当前/下一个 checkpoint window 的结构化信息**。

说明：代码字段仍叫 `next_checkpoint_type`，但当前论文语义建议理解为 **current-or-next checkpoint type**：窗口外指向即将进入的窗口，窗口内表示当前所在高风险窗口。

模型预测三类监督：

1. **`next_checkpoint_type`**（历史字段名，当前语义为 current-or-next checkpoint type）
   - 分类任务。
   - 类别：`0=none`，`1=pre_grasp window`，`2=pre_place window`。
   - 窗口外指向 upcoming checkpoint window；窗口内保持 current checkpoint window 类型。
   - 现在 type loss 对 `type=0` 也监督，不再和 horizon 共用 mask。

2. **`h_entry`**
   - horizon-to-window-entry。
   - 原始数据中是整数帧距；训练侧分桶为 horizon bin。
   - 窗口内为 0，无 next/current window 为 -1。

3. **`semantic_phase`**
   - multi-label BCE。
   - `[object_in_hand, lifted, placed_and_released]`。

训练目标：

```text
L = L_flow_action + λ_type L_type + λ_h L_h_entry + λ_phase L_phase
```

当前仍是 **Stage 1 warm-up**：只做监督头训练与评估；Stage 2 latent 和部署端 adaptive execution 尚未实现。

---

## 1. 当前 git 状态与改动范围

本轮主要 commits：

```text
63bc46d Fix KeyState none type supervision
a130a88 Weight near-checkpoint horizon bins
b3b2e3e Add checkpoint window labels
e13169e Propagate h_entry through Pi0 training data path
74fc227 Reserve KeyState latent zone interface
7cb4e82 Update KeyState window visualizer
```

主仓库 submodule pointer 已在：

```text
a30eaad Update RoboTwin KeyState window training changes
```

本轮未提交数据、checkpoint、wandb、processed_data、training_data、eval_outputs 等生成物。

---

## 2. 已修改内容摘要

### 2.1 type loss 修复：监督 terminal `type=0/none`

文件：

```text
policy/pi0/src/openpi/models/pi0.py
```

旧逻辑：

```text
valid = h_ckpt >= 0
loss_type 和 loss_horizon 共用 valid
```

问题：

- `h=-1` 的末段帧不参与 type loss。
- 模型没有被明确教会预测 `0=none`。
- 对后续 adaptive execution 不安全。

新逻辑：

```text
type_valid_b = keystate_type >= 0
h_valid_b    = keystate_h_entry >= 0
```

结果：

- `loss_type` 包含 `type=0/none`。
- `loss_h_entry` 只在有 next/current checkpoint window 时监督。
- `loss_ph` 仍全帧监督。

---

### 2.2 `h_entry` bucket 方案与 near-entry 轻量加权

文件：

```text
policy/pi0/src/openpi/models/pi0.py
policy/pi0/src/openpi/training/config.py
```

新增配置：

```python
horizon_upper_edges: tuple[int, ...] = (1, 4, 7, 11, 21, 51)
horizon_bin0_weight: float = 1.0
horizon_bin1_weight: float = 1.0
```

当前 bucket 语义：

```text
bin 0: h_entry == 0                 # inside checkpoint window
bin 1: 1 <= h_entry < 4             # very near entry, still before window
bin 2: 4 <= h_entry < 7
bin 3: 7 <= h_entry < 11
bin 4: 11 <= h_entry < 21
bin 5: 21 <= h_entry < 51
bin 6: h_entry >= 51
invalid: h_entry < 0, no horizon loss
```

原因：

- `h_entry == 0` 单独保留，直接表示“当前已经在 checkpoint window 内”。
- `h_entry=1/2/3` 合并，避免入口前一两帧样本过稀。
- Stage3 可以直接使用 `h_entry_bin == 0` 表示 inside-window，不需要额外引入 `keystate_inside_window`。

KeyState config 当前设置：

```python
horizon_bin0_weight = 1.0   # bin0 现在可能很大，不再上调
horizon_bin1_weight = 1.10  # 轻量强调 near-entry pre-window 样本
```

建议每次换数据后打印 histogram，再决定是否继续调权重。

加权方式：

```text
weighted_mean = sum(mask * bin_weight[label] * loss) / sum(mask * bin_weight[label])
```

---

### 2.3 `h_ckpt` 统一改为 `h_entry`

相关文件：

```text
policy/pi0/scripts/process_data.py
policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py
policy/pi0/src/openpi/policies/keystate.py
policy/pi0/src/openpi/models/model.py
policy/pi0/src/openpi/models/pi0.py
policy/pi0/src/openpi/training/config.py
policy/pi0/scripts/train.py
```

新数据字段：

```text
/keystate/h_entry
/observations/keystate/h_entry
observation.keystate.h_entry
Observation.keystate_h_entry
```

新 loss log：

```text
loss_h_entry
```

不再使用旧字段：

```text
h_ckpt
keystate_h
loss_h
```

除非是无关的 `action_horizon` / `horizon_upper_edges` 等模型通用命名。

---

### 2.4 KeyState data transform

文件：

```text
policy/pi0/src/openpi/policies/keystate.py
```

当前职责：

```text
keystate.next_checkpoint_type -> keystate_type
keystate.h_entry              -> keystate_h_entry, after bucket_h_entry(...)
keystate.semantic_phase       -> keystate_phase
```

`bucket_h_entry()` 分桶规则：

```text
h_entry == 0       -> bin 0   # inside checkpoint window
1 <= h_entry < 4   -> bin 1   # very near entry, still before window
4 <= h_entry < 7   -> bin 2
7 <= h_entry < 11  -> bin 3
11 <= h_entry < 21 -> bin 4
21 <= h_entry < 51 -> bin 5
h_entry >= 51      -> bin 6
h_entry < 0        -> -1 invalid
```

对应代码配置：

```python
horizon_upper_edges = (1, 4, 7, 11, 21, 51)
```

因此 horizon head 输出维度从旧方案的 6 类变为 7 类；旧 checkpoint 的 `ks_horizon_head` shape 不再兼容，需要按新 bucket 重新训练。

---

### 2.5 Stage 2 latent zone 接口预留

文件：

```text
policy/pi0/src/openpi/models/model.py
policy/pi0/src/openpi/models/pi0.py
```

新增默认关闭字段：

```python
use_z_hat_zone: bool = False
z_hat_zone_dim: int = 64
z_h_interaction: str = "none"
```

新增 Observation pass-through 字段：

```python
keystate_z_hat_zone
```

当前行为：

- 不创建新参数。
- 不实现 z_head。
- 不实现 z loss。
- 默认不影响 Stage 1。
- 如果误启用 `use_z_hat_zone=True` 或 `z_h_interaction != "none"`，会抛 `NotImplementedError`。

设计意图：

```text
z_hat_zone = 未来 checkpoint window 的 latent 表征
h_entry = 当前距离该 window 入口有多远
后续 Stage 2 可让 z 与 h_entry 相互条件化/约束
```

### 2.6 Stage 2 z_entry_descriptor 现状（已实现）

Stage 2 已在后续分支中实现，替代了本节早期预留的 `z_hat_zone` 命名。当前实际使用字段为：

```text
/keystate/z_entry_descriptor
/observations/keystate/z_entry_descriptor
observation.keystate.z_entry_descriptor
Observation.keystate_z_entry_descriptor
```

当前主方案不再使用 prefix-only latent，而是使用 frozen Pi0 action expert hidden 生成 checkpoint-entry key-state latent：

```text
encoder = frozen_pi0_action_expert_demo_t0001_projected_v1
source hidden = suffix_out[:, -action_horizon:] mean over action tokens
source dim = 1024
projection = fixed_random_projection_v1 -> 64D
```

`z_entry_descriptor` 只在 `h_entry > 0` 的窗口前帧监督；`h_entry == 0` 或 `h_entry == -1` 时写 zero vector，并由模型 loss 自动 mask。

Stage2 仍然不改变部署和 action generation 路径；z head 是训练时辅助监督。

---

## 3. 数据管线当前约定

### Stage 0 HDF5

```text
/keystate/next_checkpoint_type
/keystate/h_entry
/keystate/semantic_phase
```

### Pi0 processed HDF5

```text
/observations/keystate/next_checkpoint_type
/observations/keystate/h_entry
/observations/keystate/semantic_phase
```

### LeRobot features

```text
observation.keystate.next_checkpoint_type
observation.keystate.h_entry
observation.keystate.semantic_phase
```

### Model Observation

```python
Observation.keystate_type
Observation.keystate_h_entry
Observation.keystate_phase
Observation.keystate_z_hat_zone  # reserved, optional
```

---

## 4. 已完成验证

### 4.1 Stage 0 labeler / inspector

episode0 window label：

```text
pre_grasp window = [34, 71]
pre_place window = [115, 151]
T = 152
flags = []
```

inspect：

```text
episode0 OK
```

---

### 4.2 processed data 验证

`process_data.py` 重新生成 one-episode processed data 后确认：

```text
observations/keystate keys = ['h_entry', 'next_checkpoint_type', 'semantic_phase']
```

类型：

```text
h_entry: int32
next_checkpoint_type: int8
semantic_phase: uint8 [N,3]
```

---

### 4.3 LeRobot 转换验证

新 repo_id：

```text
place_a2b_left_keystate_window_oneshot
```

LeRobot metadata 含：

```text
observation.keystate.h_entry
observation.keystate.next_checkpoint_type
observation.keystate.semantic_phase
```

DataLoader 检查：

```text
batch keys = ['keystate_h_entry', 'keystate_phase', 'keystate_type']
Observation.keystate_h_entry shape = (2,)
```

---

### 4.4 norm stats

已为新 repo 计算：

```text
assets/pi0_base_aloha_robotwin_keystate_lora/place_a2b_left_keystate_window_oneshot/norm_stats.json
```

---

### 4.5 2-step smoke training

命令使用：

```text
--data.repo-id place_a2b_left_keystate_window_oneshot
--num-train-steps 2
--batch-size 2
--num-workers 0
--fsdp-devices 2
```

结果：

```text
Step 0:
flow_loss=0.0450
loss=0.8995
loss_h_entry=0.1446
loss_ph=0.3438
loss_type=0.3660

Step 1:
flow_loss=0.0777
loss=0.7968
loss_h_entry=0.2660
loss_ph=0.2129
loss_type=0.2403
```

说明：新 `h_entry` 训练链路和 checkpoint 保存都正常。

---

### 4.6 1000-step overfit training

W&B run：

```text
https://wandb.ai/yanko-lan-peking-university/openpi-keystate/runs/5ai6fh3d
```

实验名：

```text
stage1_keystate_window_overfit_1000step
```

checkpoint：

```text
./checkpoints/openpi/openpi-assets/checkpoints/keystate/pi0_base_aloha_robotwin_keystate_lora/stage1_keystate_window_overfit_1000step/1000
```

训练 loss 摘要：

```text
Step 0:
flow_loss=0.0450
loss=0.8995
loss_h_entry=0.1446
loss_ph=0.3438
loss_type=0.3660

Step 500:
flow_loss=0.0246
loss=0.0536
loss_h_entry=0.0212
loss_ph=0.0068
loss_type=0.0010

Step 990:
flow_loss=0.0169
loss=0.0924
loss_h_entry=0.0231
loss_ph=0.0325
loss_type=0.0199
```

结论：`loss_type`、`loss_h_entry`、`loss_ph`、`flow_loss` 均能下降，新的 window/h_entry 监督能被模型拟合。

---

### 4.7 checkpoint 预测评估

评估输出：

```text
policy/pi0/eval_outputs/stage1_keystate_window_overfit_1000step/keystate_window_eval_summary.json
policy/pi0/eval_outputs/stage1_keystate_window_overfit_1000step/keystate_window_eval_rows.jsonl
```

指标：

```json
{
  "num_samples": 151,
  "num_valid_h_entry": 151,
  "type_accuracy_all": 0.9867549668874173,
  "h_entry_bin_accuracy_valid": 0.9668874172185431,
  "h_entry_bin_mae_valid": 0.0728476821192053
}
```

Type confusion matrix：

```text
true_type x pred_type
[
  [0, 0, 0],
  [0, 72, 0],
  [0, 2, 77]
]
```

h_entry bin confusion matrix：

```text
true_h_entry_bin x pred_h_entry_bin
[
  [78, 0, 0, 0, 0, 0],
  [3, 3, 0, 0, 0, 0],
  [0, 0, 10, 0, 0, 0],
  [0, 0, 0, 20, 0, 0],
  [2, 0, 0, 0, 35, 0],
  [0, 0, 0, 0, 0, 0]
]
```

结论：

```text
type accuracy        ≈ 98.7%
h_entry bin accuracy ≈ 96.7%
h_entry bin MAE      ≈ 0.073 bin
```

---

## 5. 当前待完成事项

1. **多 episode 验证**
   - 当前只在 episode0 one-shot 上 overfit。
   - 后续若有要求，需要对更多 episodes 重跑 v3 labeler、process、convert、train/eval。

2. **terminal none 实际数据验证**
   - episode0 processed 后没有 terminal `type=0` 样本，因为 pre_place window 延伸到最后帧。
   - 代码已支持 type=0 监督，但需要带窗口后尾段的 episode 验证实际效果。

3. **bin histogram 与权重调参**
   - 当前新 bucket 下，bin0 表示整个 checkpoint window 内部，样本数可能很大，因此已改为 `horizon_bin0_weight=1.0`。
   - bin1 表示窗口前 `h=1/2/3` 的 near-entry 样本，当前轻量设置为 `1.10`。
   - 换多 episode 数据后先运行 histogram 检查，再比较：`1.0/1.0`、`1.0/1.10`、按 class frequency reweight 等方案。

4. **Stage 2 latent 已在后续分支实现**
   - 早期预留的 `z_hat_zone` 接口仍保留但不作为当前主路径。
   - 当前 Stage2 使用 `z_entry_descriptor`，主 backend 为 `frozen_pi0_action_expert_demo_t0001_projected_v1`。
   - z target 只在 `h_entry > 0` 时监督，窗口内不监督。

5. **部署端尚未修改**
   - 当前没有改 adaptive execution。
   - 后续若进入部署阶段，再考虑用 `h_entry_hat` 控制 long chunk -> short-step mode switch。

---

## 6. 复现命令摘要

### process data

```bash
cd third_party/RoboTwin/policy/pi0
.venv/bin/python scripts/process_data.py place_a2b_left demo_clean 1
```

### convert to LeRobot

```bash
export HF_LEROBOT_HOME=$PWD/training_data
.venv/bin/python examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
  --raw_dir "$PWD/processed_data/place_a2b_left-demo_clean-1" \
  --repo_id place_a2b_left_keystate_window_oneshot
```

### inspect h_entry bucket histogram

```bash
cd third_party/RoboTwin/policy/pi0
.venv/bin/python scripts/inspect_keystate_h_entry_buckets.py \
  processed_data/place_a2b_left-demo_clean-1
```

新 bucket 应打印 7 个有效 bin 加 invalid 计数。对于当前 one-episode，bin0 会覆盖两个 checkpoint window 内的所有帧，因此可能显著大于 near-entry bin1。

### train smoke / overfit

```bash
export HF_LEROBOT_HOME=$PWD/training_data
export OPENPI_DATA_HOME=./checkpoints/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=false

.venv/bin/python scripts/train.py pi0_base_aloha_robotwin_keystate_lora \
  --project-name openpi-keystate \
  --exp-name stage1_keystate_window_overfit_1000step \
  --checkpoint-base-dir ./checkpoints/openpi/openpi-assets/checkpoints/keystate \
  --overwrite \
  --wandb-enabled \
  --data.repo-id place_a2b_left_keystate_window_oneshot \
  --batch-size 2 \
  --num-workers 0 \
  --num-train-steps 1000 \
  --log-interval 10 \
  --save-interval 500 \
  --fsdp-devices 2
```

---

## 7. 注意事项

- 生成物不要提交：
  - `data/`
  - `policy/pi0/processed_data/`
  - `policy/pi0/training_data/`
  - `policy/pi0/eval_outputs/`
  - `policy/pi0/wandb/`
  - checkpoints
- `compute_norm_stats.py` 默认 `num_workers=8` 在当前文件系统上容易卡住；本轮采用单进程等价脚本计算 norm stats。
- checkpoint 保存到 `./checkpoints/openpi/openpi-assets/checkpoints/keystate` 正常；不要写到 `./checkpoints`。
