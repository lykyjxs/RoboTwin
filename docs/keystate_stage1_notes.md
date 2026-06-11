# KeyState Stage 1 — 模型修改 project

> 本文档记录 **KeyState-aware VLA** 项目 Stage 1
> (KeyState Head warm-up,模型修改)的最终 plan、已落盘改动、各文件完成情况、未完成 task。
> 分支:所有 Stage 1 改动在 `keystate-stage1-heads`(从 `keystate-stage0-labeler` 切出)。

---

## 0. 路径与分支(先读)

- **Stage 1 代码基线**:`third_party/RoboTwin/policy/pi0`(纯 JAX/Flax 的 openpi fork,实际训练/eval 都走它)。
  这是**旧版结构**:`Pi0Config` 与 `Pi0` 同在 `pi0.py`(**无独立 pi0_config.py、无 models_pytorch、无 pi05/adarms**)。
  **不要套用** `source code/openpi`(新版,拆了 config、带 PyTorch)的行号/布局。
- **不要修改** `source code/openpi`(仅参考,且自带无关本地改动)。
- 当前分支:`keystate-stage1-heads`,继承 Stage 0 commit `603bcc1`(dense next_checkpoint_type 标签)。

---

## 1. 最终执行的 Plan

### 目标
在 π0(flow-matching VLA)上挂 KeyState 预测头:预测「下一个 control checkpoint 的 type / horizon」+ 辅助「semantic phase」。本轮 = **Stage 1 KeyState Head warm-up**:
- 一次性写齐 scaffold(三头 + 融合 + 配置开关 + 数据管线),
- 但**本轮只实现 type / horizon / phase 头**(horizon loss 默认 CE);**z 头(Stage 2)、KeyState→Action 融合(Stage 3)只保留占位结构,在配置层硬闸 `NotImplementedError`**,
- 让 Stage 1 端到端可跑、能出加权 loss。

### 设计总原则
**开关全关 = 原版 pi0 逐位等价**:所有新模块 `if config.use_*` gate,关闭时不创建任何新参数(freeze filter / FSDP / 权重加载 / loss 数值不变),现有 4 个 robotwin config 不受影响。

### 三个数据语义关键点(读 keystate_labeler.py 源码后确认)
1. **监督用 dense `next_checkpoint_type`**,不是稀疏 `checkpoint_type`(后者全程 0、仅 2 个非零帧)。dense 语义:`≤pre_grasp`→1、`(pre_grasp,pre_place]`→2、`>pre_place`→0。
2. **`h_ckpt=-1` = invalid**(最后一个 checkpoint 之后)。`compute_loss` 现算 `valid=(keystate_h>=0)`,**safe-label `where(valid,label,0)` 再 mask**(避免 -1 进 CE 被当负索引/NaN)。只 mask **type+horizon**;**phase 不 mask**(每帧良定义)。
3. **horizon 对数间隔分桶**:`horizon_upper_edges=(3,6,11,21,51)`,`h<edge` 落该桶(bin0..5)。`horizon_loss_type` 默认 `"ce"`(softmax CE,简单好 debug),`"ordinal"`(CORAL)作可切换 ablation。两模式 head 维度不同:ce=n_bins=6,ordinal=n_bins-1=5。

### 数据流(三跳)
```
Stage0 采集 hdf5  /keystate{checkpoint_type(sparse), next_checkpoint_type(dense), h_ckpt(-1=invalid), semantic_phase}
  ⓪ keystate_labeler.py 已派生 next_checkpoint_type(v2)
  ① process_data.py        → 中间 hdf5 observations/keystate/*
  ② convert_..._lerobot_robotwin.py → LeRobot dataset 注册 observation.keystate.* feature
  ③ data_loader(delta_timestamps 只 window action;keystate 作"当前帧单点"自动返回)
  → KeyStateInputs transform → Observation.keystate_* → compute_loss
```
关键:keystate 是 per-step 标量,只取**当前帧**,**不进** `action_sequence_keys`、不窗口化。

### 头与 loss
- 头挂 **prefix(PaliGemma/VLM)stream**,masked-mean 池化,width=2048。
- `compute_loss` 返回 `(flow_loss[*b,ah], ks_losses_dict)`;`train.py` 用 `has_aux=True`,`total=mean(flow)+sum(ks)`,各分项进 wandb。
- keystate TrainConfig 的 λ 先设 **0.1**(避免 aux 压过 action loss);`Pi0Config` 默认 λ=1.0 不动、只在该 config 覆盖。
- `CheckpointWeightLoader` 加 `missing_regex`,keystate config 设 `".*(lora|ks_).*"` 以容忍新 `ks_*` 头随机初始化。

---

## 2. 已真正落盘修改的文件

**(a) Stage 0 分支 `keystate-stage0-labeler`,已 commit `603bcc1`:**
- `envs/utils/keystate_labeler.py`
- `envs/utils/keystate_inspect.py`

**(b) Stage 1 分支 `keystate-stage1-heads`(全部已落盘,py_compile 通过):**
- `policy/pi0/src/openpi/models/pi0.py`
- `policy/pi0/src/openpi/models/pi0_fast.py`
- `policy/pi0/src/openpi/models/model.py`
- `policy/pi0/scripts/train.py`
- `policy/pi0/scripts/process_data.py` ✅ **已修好(#7)**
- `policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py` ✅ **(#8)**
- `policy/pi0/src/openpi/policies/keystate.py`(新增)✅ **(#9)**
- `policy/pi0/src/openpi/policies/aloha_policy.py`(keystate 透传)✅ **(#9)**
- `policy/pi0/src/openpi/training/weight_loaders.py` ✅ **(#10)**
- `policy/pi0/src/openpi/training/config.py` ✅ **(#11)**

---

## 3. 各文件已完成内容

| 文件 | 完成内容 | 状态 |
|---|---|---|
| `envs/utils/keystate_labeler.py` | h_ckpt 循环里派生 dense `next_checkpoint_type[t]=checkpoint_type[nxt]`,加入 labels,`LABELER_VERSION=2`。`_write_back` 自动写入 /keystate。 | ✅ 已 commit |
| `envs/utils/keystate_inspect.py` | `load_keystate` 读 `next_checkpoint_type`;`check_episode` 加 check 9(valid 帧∈{1,2}、invalid=0、段语义断言)。 | ✅ 已 commit |
| `pi0.py` | `Pi0Config` 加全部 KeyState 字段 + `__post_init__` 硬闸(`use_z_head`/`use_keystate_fusion`→NotImplementedError,`horizon_loss_type` 校验);顶部加 `_softmax_xent`/`_sigmoid_bce`/`_coral_loss`(per-sample [b]);`__init__` 加三头(挂 prefix width 2048,gate)+ z/fusion 占位;`compute_loss` 返回 `(flow_loss, ks_losses)` 并调 `_keystate_losses`(safe-label+valid mask,type/horizon mask、phase 不 mask,horizon ce/ordinal 可切);`embed_suffix` 加可选 `ks_cond=None` 的 Stage3 scaffold(None 时逐位等价)。 | ✅ 落盘,py_compile 通过 |
| `pi0_fast.py` | `compute_loss` 返回注解改 tuple,return 改 `(loss, {})`。 | ✅ 落盘,py_compile 通过 |
| `model.py` | `Observation` 加 `keystate_type/keystate_h/keystate_phase` 可选字段 + `from_dict` 用 `data.get` + `preprocess_observation` 透传;`BaseModel.compute_loss` 注解改 tuple。 | ✅ 落盘,py_compile 通过 |
| `scripts/train.py` | `loss_fn` 解包 `(chunked_loss, ks_losses)`,`total=flow+sum(ks)`,返回 `(total, aux)`;`value_and_grad(has_aux=True)`;`info` 并入 `**aux`。 | ✅ 落盘,py_compile 通过 |
| `scripts/process_data.py` | `load_hdf5` 读 `/keystate` 三字段并返回;`data_transform` 解包 6 值、在 `j!=last`(=qpos/images)分支收集 keystate、写出 `observations/keystate/{next_checkpoint_type(int8),h_ckpt(int32,保留-1),semantic_phase(uint8 [N,3])}`。 | ✅ 已修好 |
| `convert_..._lerobot_robotwin.py` | `create_empty_dataset(has_keystate=)` 注册 `observation.keystate.{next_checkpoint_type(int64,(1,)),h_ckpt(int64,(1,)),semantic_phase(float32,(3,))}`;`has_keystate()` 探测;`load_raw_episode_data` 读 keystate(保留 -1);`populate_dataset` 每帧 add。 | ✅ |
| `policies/keystate.py`(新增) | `KeyStateInputs`:`bucket_horizon`(`h<edge→bin`,与 `Pi0Config.horizon_upper_edges` 完全一致,`h<0` 保留 -1)→ `keystate_h`;`next_checkpoint_type→keystate_type(int32)`、`semantic_phase→keystate_phase(float32)`;squeeze `(1,)`;消费后 pop `keystate`;无 keystate 时 no-op。 | ✅ |
| `policies/aloha_policy.py` | `AlohaInputs` 末尾 `if "keystate" in data: inputs["keystate"]=...` 透传。 | ✅ |
| `training/weight_loaders.py` | `CheckpointWeightLoader.missing_regex` 字段(默认 `.*lora.*` 不变行为);keystate config 覆盖为 `.*(lora|ks_).*`。 | ✅ |
| `training/config.py` | `KeyStateAlohaDataConfig(LeRobotAlohaDataConfig)`:`create` 在父基础上 push `KeyStateInputs`(edges 从 model_config 读);新 `TrainConfig` `pi0_base_aloha_robotwin_keystate_lora`(开 type/phase 头,λ=0.1,repack 加 keystate 子 dict,widen missing_regex)。 | ✅ |

---

## 4. 未完成的 task list

| # | 任务 | 状态 |
|---|---|---|
| 1 | keystate_labeler 派生 next_checkpoint_type | ✅ 完成(已 commit) |
| 2 | pi0_config(本 fork 在 pi0.py 内)加字段 + 硬闸 | ✅ 完成 |
| 3 | model.py Observation + compute_loss 注解 | ✅ 完成 |
| 4 | pi0.py 三头 + compute_loss + 融合 scaffold | ✅ 完成 |
| 5 | pi0_fast.py 兼容垫片 | ✅ 完成 |
| 6 | train.py loss_fn 解包 + has_aux | ✅ 完成 |
| **7** | **process_data.py 写 keystate** | ✅ 完成 |
| 8 | convert_aloha_data_to_lerobot_robotwin.py 注册 keystate feature | ✅ 完成 |
| 9 | 新建 `policies/keystate.py` KeyStateInputs + aloha_policy 透传 | ✅ 完成 |
| 10 | weight_loaders.py 加 `missing_regex` 字段 | ✅ 完成 |
| 11 | config.py 接线 KeyStateAlohaDataConfig + 新 TrainConfig | ✅ 完成 |

> **Stage 1 模型修改全部完成,纯静态验证已过**(全文件 py_compile;`bucket_horizon` 与 `Pi0Config.horizon_upper_edges` 语义逐桶比对一致;`KeyStateInputs` stub 跑通:-1 sentinel 保留、有效 horizon 正确分桶、无 keystate 时 no-op;config.py AST 校验配置名唯一且新 config 存在)。
> **仍需在有数据 + pi0 uv 环境的机器上跑端到端验证(见 §6 Verification 0/1/2/3/4/5/6)** —— 本环境无 hdf5、无 jax/uv,无法实跑训练。

---

## 5. 数据管线 keystate 端到端约定(已实现,供对齐)

三跳 keystate 形状/语义,逐跳保留 `h_ckpt=-1`(invalid)sentinel:

1. **process_data.py** 写中间 hdf5 `observations/keystate/`:`next_checkpoint_type`(int8 [N])、`h_ckpt`(int32 [N],-1 原样)、`semantic_phase`(uint8 [N,3])。keystate 在 `j != last` 分支收集 → 与 **qpos/images 对齐**(观测帧 0..T-2),长度与 qpos 一致。
2. **convert_..._lerobot_robotwin.py** 注册 LeRobot feature:`observation.keystate.next_checkpoint_type`(int64 (1,))、`.h_ckpt`(int64 (1,))、`.semantic_phase`(float32 (3,))。
3. **config.py repack** 把上述映射进子 dict `keystate`;`AlohaInputs` 透传;`KeyStateInputs`(在其后 push)→ `keystate_type`(int32 标量)/`keystate_h`(分桶 index,-1 保留)/`keystate_phase`(float32 [3])→ `Observation.from_dict` → `compute_loss`。

> keystate **不进** `action_sequence_keys` → data_loader 的 `delta_timestamps` 不窗口化它 → LeRobot 自动返回**当前帧单点**。Normalize 用 `strict=False`,keystate 键无 norm_stats → 原样透传不归一化(标签本就不该归一化)。

---

## 6. 下一轮继续顺序

模型修改已全部完成并 commit。下一轮 = **在有数据 + pi0 uv 环境的机器上跑端到端验证**:
1. `git checkout keystate-stage1-heads`,确认 §2(b) 文件齐。
2. 按 §6 Verification 逐项跑(0 labeler / 1 baseline 不回归 / 2 数据管线 / 3 safe-label+mask / 4 训练出 loss / 5 加载兼容 / 6 eval 不挂)。
3. 验证通过后再开 Stage 2(z head / KeyState-JEPA)。

### Verification(plan §Verification)
- **0 labeler**:重跑 `keystate_labeler.py` 后 `keystate_inspect.py` 断言全过(dense type 段语义)。
- **1 baseline 不回归**:`uv run scripts/train.py pi0_base_aloha_robotwin_lora` loss 与改动前一致、无加载报错。
- **2 数据管线**:中间 hdf5 有 `observations/keystate/*`、LeRobot features 含 `observation.keystate.*`;batch 里 `keystate_h` 的 -1 原样保留(没被 clip 成 0)。
- **3 safe-label+mask 关键回归**:全 invalid(h=-1)batch 不产 NaN/不报越界。
- **4 Stage 1 训练**:wandb 出现 `flow_loss/loss_type/loss_h/loss_ph` 且下降。
- **5 加载兼容**:`ks_*` 头被 `missing_regex` 容忍、随机初始化。
- **6 eval 不挂**:`pi_model.py`/`eval.sh` 在 keystate 为 None、融合关时与原版一致。

> ⚠️ **数据相关验证(0/2/3)需要实际采集数据。** 当前 checkout **无 hdf5**(`data/place_a2b_left/demo_clean/data/` 为空,数据在 `.gitignore`),需在有数据的机器上做。

---

## 7. 配置/接口速记(给下一轮对齐)

- `Pi0Config` 新增字段:`use_checkpoint_head/use_phase_head/use_z_head/use_keystate_fusion`(默认 False)、`num_checkpoint_types=3`、`num_phase_classes=3`、`z_dim=64`、`lambda_type/lambda_h/lambda_ph=1.0`、`lambda_z=0.0`、`horizon_upper_edges=(3,6,11,21,51)`、`horizon_loss_type="ce"`。
- `Observation` 新增:`keystate_type`(dense type,[*b] int)、`keystate_h`(桶 index,-1=invalid)、`keystate_phase`([*b,3] float)。
- 新 TrainConfig 名:`pi0_base_aloha_robotwin_keystate_lora`(拷贝 `pi0_base_aloha_robotwin_lora`,开 `use_checkpoint_head/use_phase_head`,λ=0.1,`weight_loader` missing_regex `.*(lora|ks_).*`)。
- 新 transform:`KeyStateInputs(horizon_upper_edges=...)`,在 `AlohaInputs` 之后 push;`AlohaInputs` 加一行 `keystate` 透传;repack 加 `"keystate"` 子 dict 映射 `observation.keystate.{next_checkpoint_type,h_ckpt,semantic_phase}`。
