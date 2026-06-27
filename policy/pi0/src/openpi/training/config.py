"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.keystate as keystate_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=False)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=False)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions", )

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If true, will disable syncing the dataset from the Hugging Face Hub. Allows training on local-only datasets.
    local_files_only: bool = False


class GroupFactory(Protocol):

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=False)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(inputs=[
                    _transforms.InjectDefaultPrompt(self.default_prompt),
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer(model_config.max_token_len), ),
                ], )
            case _model.ModelType.PI0_FAST:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(_tokenizer.FASTTokenizer(model_config.max_token_len), ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=False)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=False)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=False)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=False)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(default=_transforms.Group(inputs=[
        _transforms.RepackTransform({
            "images": {
                "cam_high": "observation.images.top"
            },
            "state": "observation.state",
            "actions": "action",
        })
    ]))
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action", )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=False)
class KeyStateAlohaDataConfig(LeRobotAlohaDataConfig):
    """LeRobotAlohaDataConfig + KeyState (Stage 1) supervision labels.

    Identical to the parent except it (a) repacks the per-frame keystate sub-dict from the
    LeRobot `observation.keystate.*` features, and (b) pushes `KeyStateInputs` after `AlohaInputs`
    to derive the model labels (keystate_type / keystate_h_entry / keystate_phase). keystate stays a
    current-frame singleton: it is NOT added to `action_sequence_keys`, so it is never windowed.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Start from the parent config (AlohaInputs/Outputs, optional delta actions, model transforms).
        base = super().create(assets_dirs, model_config)
        # Bucket edges must match the head: read them off the model config when available.
        horizon_upper_edges = tuple(getattr(model_config, "horizon_upper_edges", (1, 4, 7, 11, 21, 51)))
        data_transforms = base.data_transforms.push(
            inputs=[keystate_policy.KeyStateInputs(horizon_upper_edges=horizon_upper_edges)],
        )
        return dataclasses.replace(base, data_transforms=data_transforms)


@dataclasses.dataclass(frozen=False)
class LeRobotLiberoDataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(inputs=[
            _transforms.RepackTransform({
                "observation/image": "image",
                "observation/wrist_image": "wrist_image",
                "observation/state": "state",
                "actions": "actions",
                "prompt": "prompt",
            })
        ])

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[
                libero_policy.LiberoInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[libero_policy.LiberoOutputs()],
        )
        # Use delta actions (not for gripper)
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=False)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints/"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If true, save a final checkpoint at the end of training. Disable for short overfit
    # smoke tests on filesystems where Orbax/TensorStore checkpoint writes are unreliable.
    save_final_checkpoint: bool = True
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    ###
    ### finetune config for robotwin
    ###
    # pi0_base by lora
    TrainConfig(
        name="pi0_base_aloha_robotwin_lora",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotAlohaDataConfig(
            repo_id="test",  # your datasets repo_id
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        freeze_filter=pi0.Pi0Config(paligemma_variant="gemma_2b_lora",
                                    action_expert_variant="gemma_300m_lora").get_freeze_filter(),
        batch_size=32,  # the total batch_size not pre_gpu batch_size
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30000,
        fsdp_devices=1,  # refer line 359
    ),
    # Stack Bowls baseline config for later fixed-chunk Pi0 comparison. First milestone only uses
    # this for assets/norm-stat plumbing; formal rollout comparison happens after Stage3 pred/mixed fusion.
    TrainConfig(
        name="pi0_base_aloha_robotwin_stack_bowls_three_lora",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotAlohaDataConfig(
            repo_id="stack_bowls_three_demo_clean_300_keystate_stage1",
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        ),
        freeze_filter=pi0.Pi0Config(paligemma_variant="gemma_2b_lora",
                                    action_expert_variant="gemma_300m_lora").get_freeze_filter(),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30000,
        fsdp_devices=1,
    ),
    # Stack Bowls Stage1 smoke config: validates multi-cycle KeyState labels and the training data path only.
    # Do not use Stage1-only rollout as the formal KeyState adaptive-chunking comparison.
    TrainConfig(
        name="pi0_base_aloha_robotwin_stack_bowls_three_keystate_lora",
        model=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_checkpoint_head=True,
            use_phase_head=True,
            lambda_type=0.1,
            lambda_h=0.1,
            lambda_ph=0.1,
            horizon_bin0_weight=1.0,
            horizon_bin1_weight=1.10,
        ),
        data=KeyStateAlohaDataConfig(
            repo_id="stack_bowls_three_demo_clean_300_keystate_stage1",
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                    "keystate": {
                        "next_checkpoint_type": "observation.keystate.next_checkpoint_type",
                        "h_entry": "observation.keystate.h_entry",
                        "semantic_phase": "observation.keystate.semantic_phase",
                    },
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        ),
        freeze_filter=pi0.Pi0Config(paligemma_variant="gemma_2b_lora",
                                    action_expert_variant="gemma_300m_lora").get_freeze_filter(),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "s3://openpi-assets/checkpoints/pi0_base/params",
            missing_regex=".*(lora|ks_).*",
        ),
        num_train_steps=30000,
        fsdp_devices=1,
    ),
    # Stack Bowls Stage2 config: train on train300 with per-window z_entry_descriptor targets,
    # initialized from the selected Stage1 checkpoint (best on held-out val30: step 15000).
    # The fresh val30 set remains held out and should only be used for validation/checkpoint selection.
    TrainConfig(
        name="pi0_base_aloha_robotwin_stack_bowls_three_keystate_stage2_lora",
        model=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_checkpoint_head=True,
            use_phase_head=True,
            use_z_entry_descriptor=True,
            z_entry_descriptor_dim=64,
            lambda_type=0.1,
            lambda_h=0.1,
            lambda_ph=0.1,
            lambda_z_entry_descriptor=0.1,
            horizon_bin0_weight=1.0,
            horizon_bin1_weight=1.10,
        ),
        data=KeyStateAlohaDataConfig(
            repo_id="stack_bowls_three_demo_clean_300_keystate_stage2_actionexpert",
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                    "keystate": {
                        "next_checkpoint_type": "observation.keystate.next_checkpoint_type",
                        "h_entry": "observation.keystate.h_entry",
                        "semantic_phase": "observation.keystate.semantic_phase",
                        "z_entry_descriptor": "observation.keystate.z_entry_descriptor",
                    },
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        ),
        freeze_filter=pi0.Pi0Config(paligemma_variant="gemma_2b_lora",
                                    action_expert_variant="gemma_300m_lora").get_freeze_filter(),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/openpi/openpi-assets/checkpoints/keystate/pi0_base_aloha_robotwin_stack_bowls_three_keystate_lora/stack_bowls_three_300_stage1_lora/15000/params",
            missing_regex=".*ks_z_entry_descriptor_head.*",
        ),
        num_train_steps=30000,
        fsdp_devices=1,
    ),
    # pi0_base by lora + KeyState heads (Stage 1 warm-up): copy of pi0_base_aloha_robotwin_lora with
    # the checkpoint/phase heads turned on, aux lambdas kept small (0.1) so they don't drown the flow
    # loss, and the weight loader widened to tolerate the freshly-initialized ks_* heads.
    TrainConfig(
        name="pi0_base_aloha_robotwin_keystate_lora",
        model=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_checkpoint_head=True,
            use_phase_head=True,
            lambda_type=0.1,
            lambda_h=0.1,
            lambda_ph=0.1,
            horizon_bin0_weight=1.0,
            horizon_bin1_weight=1.10,
        ),
        data=KeyStateAlohaDataConfig(
            repo_id="place_a2b_left_keystate_oneshot",  # 1-episode overfit dataset
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                    "keystate": {
                        "next_checkpoint_type": "observation.keystate.next_checkpoint_type",
                        "h_entry": "observation.keystate.h_entry",
                        "semantic_phase": "observation.keystate.semantic_phase",
                    },
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        freeze_filter=pi0.Pi0Config(paligemma_variant="gemma_2b_lora",
                                    action_expert_variant="gemma_300m_lora").get_freeze_filter(),
        batch_size=32,  # the total batch_size not pre_gpu batch_size
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "s3://openpi-assets/checkpoints/pi0_base/params",
            missing_regex=".*(lora|ks_).*",  # tolerate randomly-initialized LoRA, KeyState heads, and Stage3 ks_* fusion params
        ),
        num_train_steps=30000,
        fsdp_devices=1,  # refer line 359
    ),
    # pi0_base by lora + KeyState heads (Stage 2 bootstrap z-entry descriptor): extends Stage 1 with
    # a deterministic descriptor target for the future/current checkpoint-window entry. This is only a
    # plumbing/training bootstrap and is not the final frozen-Pi0 latent target described in the paper idea.
    TrainConfig(
        name="pi0_base_aloha_robotwin_keystate_stage2_lora",
        model=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_checkpoint_head=True,
            use_phase_head=True,
            use_z_entry_descriptor=True,
            z_entry_descriptor_dim=64,
            lambda_type=0.1,
            lambda_h=0.1,
            lambda_ph=0.1,
            lambda_z_entry_descriptor=0.1,
            horizon_bin0_weight=1.0,
            horizon_bin1_weight=1.10,
        ),
        data=KeyStateAlohaDataConfig(
            repo_id="place_a2b_left_keystate_z_entry_descriptor_bootstrap_oneshot",
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                    "keystate": {
                        "next_checkpoint_type": "observation.keystate.next_checkpoint_type",
                        "h_entry": "observation.keystate.h_entry",
                        "semantic_phase": "observation.keystate.semantic_phase",
                        "z_entry_descriptor": "observation.keystate.z_entry_descriptor",
                    },
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        ),
        freeze_filter=pi0.Pi0Config(paligemma_variant="gemma_2b_lora",
                                    action_expert_variant="gemma_300m_lora").get_freeze_filter(),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "s3://openpi-assets/checkpoints/pi0_base/params",
            missing_regex=".*(lora|ks_).*",
        ),
        num_train_steps=30000,
        fsdp_devices=1,
    ),
    # Stack Bowls Stage3 pred-fusion config: initialize from Stage2 best checkpoint (test30 best: step 5000)
    # and train late KeyState cross-attention using predicted KeyState features, not GT/oracle features.
    TrainConfig(
        name="pi0_base_aloha_robotwin_stack_bowls_three_keystate_stage3_pred_late_xattn_lora",
        model=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_checkpoint_head=True,
            use_phase_head=True,
            use_z_entry_descriptor=True,
            use_keystate_fusion=True,
            keystate_fusion_mode="late_xattn",
            ks_fusion_source="pred",
            z_entry_descriptor_dim=64,
            lambda_type=0.1,
            lambda_h=0.1,
            lambda_ph=0.1,
            lambda_z_entry_descriptor=0.1,
            horizon_bin0_weight=1.0,
            horizon_bin1_weight=1.10,
            ks_xattn_alpha_init=1e-3,
        ),
        data=KeyStateAlohaDataConfig(
            repo_id="stack_bowls_three_demo_clean_300_keystate_stage2_actionexpert",
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                    "keystate": {
                        "next_checkpoint_type": "observation.keystate.next_checkpoint_type",
                        "h_entry": "observation.keystate.h_entry",
                        "semantic_phase": "observation.keystate.semantic_phase",
                        "z_entry_descriptor": "observation.keystate.z_entry_descriptor",
                    },
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        ),
        freeze_filter=pi0.Pi0Config(paligemma_variant="gemma_2b_lora",
                                    action_expert_variant="gemma_300m_lora").get_freeze_filter(),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/openpi/openpi-assets/checkpoints/keystate_stage2/pi0_base_aloha_robotwin_stack_bowls_three_keystate_stage2_lora/stack_bowls_three_300_stage2_actionexpert_from_stage1_15000_lora/5000/params",
            missing_regex=".*(ks_type_embed|ks_horizon_embed|ks_phase_proj|ks_z_entry_proj|ks_action_ln|ks_memory_ln|ks_late_xattn|ks_late_xattn_alpha).*",
        ),
        num_train_steps=15000,
        fsdp_devices=1,
    ),
    # pi0_base by lora + Stage 3 late KeyState cross-attention: starts from the Stage2
    # auxiliary heads and injects [type, h_entry_bin, phase, z_entry] as a compact memory that
    # action tokens cross-attend immediately before action_out_proj.
    TrainConfig(
        name="pi0_base_aloha_robotwin_keystate_stage3_late_xattn_lora",
        model=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_checkpoint_head=True,
            use_phase_head=True,
            use_z_entry_descriptor=True,
            use_keystate_fusion=True,
            keystate_fusion_mode="late_xattn",
            ks_fusion_source="gt",
            z_entry_descriptor_dim=64,
            lambda_type=0.1,
            lambda_h=0.1,
            lambda_ph=0.1,
            lambda_z_entry_descriptor=0.1,
            horizon_bin0_weight=1.0,
            horizon_bin1_weight=1.10,
            ks_xattn_alpha_init=1e-3,
        ),
        data=KeyStateAlohaDataConfig(
            repo_id="place_a2b_left_keystate_z_entry_descriptor_actionexpert_oneshot",
            assets=AssetsConfig(
                assets_dir="./assets/pi0_base_aloha_robotwin_keystate_lora",
                asset_id="place_a2b_left_keystate_window_oneshot",
            ),
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                    "keystate": {
                        "next_checkpoint_type": "observation.keystate.next_checkpoint_type",
                        "h_entry": "observation.keystate.h_entry",
                        "semantic_phase": "observation.keystate.semantic_phase",
                        "z_entry_descriptor": "observation.keystate.z_entry_descriptor",
                    },
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        ),
        freeze_filter=pi0.Pi0Config(paligemma_variant="gemma_2b_lora",
                                    action_expert_variant="gemma_300m_lora").get_freeze_filter(),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "s3://openpi-assets/checkpoints/pi0_base/params",
            missing_regex=".*(lora|ks_).*",
        ),
        num_train_steps=30000,
        fsdp_devices=1,
    ),
    # pi0_fast_base by lora
    TrainConfig(
        name="pi0_fast_aloha_robotwin_lora",
        model=pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora"),
        data=LeRobotAlohaDataConfig(
            repo_id="your_repo_id",  # your datasets repo_id
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        freeze_filter=pi0_fast.Pi0FASTConfig(
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30000,
        fsdp_devices=2,  # refer line 359
    ),
    # pi0_base by full
    TrainConfig(
        name="pi0_base_aloha_robotwin_full",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="your_repo_id",  # your datasets repo_id
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        freeze_filter=pi0.Pi0Config().get_freeze_filter(),
        batch_size=32,  # the total batch_size not pre_gpu batch_size
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30000,
        fsdp_devices=4,  # refer line 359
    ),
    # pi0_fast_base by full
    TrainConfig(
        name="pi0_fast_aloha_robotwin_full",
        model=pi0_fast.Pi0FASTConfig(),
        data=LeRobotAlohaDataConfig(
            repo_id="your_repo_id",  # your datasets repo_id
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        freeze_filter=pi0_fast.Pi0FASTConfig().get_freeze_filter(),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30000,
        fsdp_devices=1,  # refer line 359
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
