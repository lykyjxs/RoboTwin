import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float,
                  max_period: float) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period)**fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


# ---- KeyState loss helpers ----
# All return a per-sample [b] vector (no reduction), so the caller can apply a valid-mask.


def _softmax_xent(logits, labels):
    """Per-sample softmax cross-entropy. logits: [b, k], labels: [b] int. Returns [b]."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(logp, labels[:, None], axis=-1)[:, 0]


def _sigmoid_bce(logits, targets):
    """Per-sample multi-label BCE, summed over labels. logits/targets: [b, c]. Returns [b]."""
    # numerically stable: max(x,0) - x*z + log(1+exp(-|x|))
    per_label = jnp.maximum(logits, 0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    return jnp.sum(per_label, axis=-1)


def _coral_loss(cum_logits, labels, n_bins):
    """CORAL ordinal loss. cum_logits: [b, n_bins-1] cumulative P(bin>j) logits, labels: [b] int.
    Target for rank k is the binary vector 1[k > j] for j=0..n_bins-2. Returns [b]."""
    j = jnp.arange(n_bins - 1)
    targets = (labels[:, None] > j[None, :]).astype(cum_logits.dtype)  # [b, n_bins-1]
    per = jnp.maximum(cum_logits, 0) - cum_logits * targets + jnp.log1p(jnp.exp(-jnp.abs(cum_logits)))
    return jnp.sum(per, axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    # ---- KeyState heads ----
    # All default OFF: when every use_* is False, no new params are created and the model is
    # bit-identical to the original pi0 (freeze filter / FSDP / weight load / loss all unchanged).
    use_checkpoint_head: bool = False  # dense next-checkpoint type classification + horizon binning
    use_phase_head: bool = False  # 3-way multi-label semantic phase (BCE)
    use_z_entry_descriptor: bool = False  # Stage 2 bootstrap descriptor auxiliary prediction
    use_z_head: bool = False  # reserved for future frozen-Pi0 latent / JEPA-style Stage 2 variant
    use_z_hat_zone: bool = False  # reserved old Stage 2 interface name; not used by descriptor bootstrap
    use_keystate_fusion: bool = False  # inject keystate condition into action expert (Stage 3; default off)
    keystate_fusion_mode: str = "none"  # "none" | "late_xattn" | future "layerwise_xattn"
    ks_fusion_source: str = "gt"  # "gt" | "pred" | "mixed"
    ks_memory_dim: int | None = None  # reserved; default Stage3 memory width is action-expert width
    ks_xattn_num_heads: int = 8
    ks_xattn_alpha_init: float = 1e-3
    ks_xattn_use_layernorm: bool = True
    ks_mixed_gt_prob: float = 1.0

    num_checkpoint_types: int = 3  # configurable vocab: {0:none, 1:pre_grasp, 2:pre_place, ...}
    num_phase_classes: int = 3  # {object_in_hand, lifted, placed_and_released}
    z_entry_descriptor_dim: int = 64  # deterministic bootstrap descriptor dim (Stage 2 plumbing target)
    z_dim: int = 64  # latent dim placeholder for future learned/frozen-Pi0 Stage 2 variants
    z_hat_zone_dim: int = 64  # reserved latent size for future checkpoint-window representation
    z_h_interaction: str = "none"  # reserved: how z_hat_zone and h_entry will interact in Stage 2

    lambda_type: float = 1.0
    lambda_h: float = 1.0
    lambda_ph: float = 1.0
    lambda_z_entry_descriptor: float = 0.0
    lambda_z: float = 0.0  # reserved for future learned/frozen-Pi0 z target variants

    # h_entry bins via upper edges, "h < edge -> that bin" semantics:
    #   h==0 -> bin0 (inside checkpoint window)
    #   1<=h<4 -> bin1 (very near entry)  4<=h<7 -> bin2  7<=h<11 -> bin3
    #   11<=h<21 -> bin4  21<=h<51 -> bin5  else -> bin6
    # num_horizon_bins = len(horizon_upper_edges) + 1 = 7
    horizon_upper_edges: tuple[int, ...] = (1, 4, 7, 11, 21, 51)
    horizon_loss_type: str = "ce"  # "ce" (default, simple to debug) | "ordinal" (CORAL, ablation)
    # Extra per-sample weight for near-checkpoint horizon bins. These bins are sparse but important for
    # entering checkpoint windows safely; default 1.0 preserves baseline loss scaling.
    horizon_bin0_weight: float = 1.0
    horizon_bin1_weight: float = 1.0

    def __post_init__(self):
        # Keep the paper/future latent and fusion scaffolds hard-gated. The implemented Stage 2 path in
        # this branch is explicitly a deterministic z_entry_descriptor bootstrap target, not a frozen-Pi0
        # latent or KeyState->Action fusion.
        if self.use_z_head:
            raise NotImplementedError(
                "use_z_head is reserved for a future frozen-Pi0/JEPA z target; use "
                "use_z_entry_descriptor=True for the Stage 2 bootstrap descriptor path.")
        if self.use_z_hat_zone:
            raise NotImplementedError(
                "z_hat_zone is a reserved interface name for future checkpoint-window latents; "
                "use z_entry_descriptor for the Stage 2 bootstrap descriptor path.")
        if self.z_entry_descriptor_dim <= 0:
            raise ValueError(f"z_entry_descriptor_dim must be positive, got {self.z_entry_descriptor_dim}")
        if self.use_z_entry_descriptor and self.lambda_z_entry_descriptor <= 0:
            raise ValueError(
                "use_z_entry_descriptor=True requires lambda_z_entry_descriptor > 0 so the descriptor "
                "head cannot be silently enabled without training signal.")
        if self.z_h_interaction != "none":
            raise NotImplementedError(
                "z_h_interaction is reserved for future z_hat_zone <-> h_entry coupling; keep it 'none'.")
        if self.use_keystate_fusion:
            if self.keystate_fusion_mode == "none":
                raise ValueError("use_keystate_fusion=True requires a concrete keystate_fusion_mode.")
            if self.keystate_fusion_mode != "late_xattn":
                raise NotImplementedError(
                    f"keystate_fusion_mode={self.keystate_fusion_mode!r} is reserved; only 'late_xattn' is planned.")
            if not self.use_checkpoint_head or not self.use_phase_head or not self.use_z_entry_descriptor:
                raise ValueError(
                    "Stage3 KeyState fusion requires checkpoint, phase, and z_entry_descriptor heads to be enabled.")
            raise NotImplementedError(
                "stage3-common defines the KeyState fusion config/data interface only; "
                "implement the concrete adapter on feature/keystate-stage3-late-xattn.")
        elif self.keystate_fusion_mode != "none":
            raise ValueError("keystate_fusion_mode must be 'none' when use_keystate_fusion=False.")
        if self.ks_fusion_source not in ("gt", "pred", "mixed"):
            raise ValueError(f"ks_fusion_source must be 'gt', 'pred', or 'mixed', got {self.ks_fusion_source!r}")
        if not 0.0 <= self.ks_mixed_gt_prob <= 1.0:
            raise ValueError(f"ks_mixed_gt_prob must be in [0, 1], got {self.ks_mixed_gt_prob}")
        if self.ks_xattn_alpha_init == 0.0:
            raise ValueError("ks_xattn_alpha_init must be nonzero so cross-attention parameters receive gradients.")
        if self.ks_xattn_num_heads <= 0:
            raise ValueError(f"ks_xattn_num_heads must be positive, got {self.ks_xattn_num_heads}")
        if self.ks_memory_dim is not None and self.ks_memory_dim <= 0:
            raise ValueError(f"ks_memory_dim must be positive when set, got {self.ks_memory_dim}")
        if self.horizon_loss_type not in ("ce", "ordinal"):
            raise ValueError(f"horizon_loss_type must be 'ce' or 'ordinal', got {self.horizon_loss_type!r}")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                keystate_type=jax.ShapeDtypeStruct([batch_size], jnp.int32)
                if self.use_checkpoint_head or self.use_keystate_fusion else None,
                keystate_h_entry=jax.ShapeDtypeStruct([batch_size], jnp.int32)
                if self.use_checkpoint_head or self.use_keystate_fusion else None,
                keystate_phase=jax.ShapeDtypeStruct([batch_size, self.num_phase_classes], jnp.float32)
                if self.use_phase_head or self.use_keystate_fusion else None,
                keystate_z_entry_descriptor=jax.ShapeDtypeStruct(
                    [batch_size, self.z_entry_descriptor_dim], jnp.float32)
                if self.use_z_entry_descriptor or self.use_keystate_fusion else None,
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter, )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(nnx.Not(action_expert_params_filter), )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter, )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")), )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0(_model.BaseModel):

    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            ))
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            ))
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # ---- KeyState heads (Stage 1) ----
        # Keep the config around so compute_loss can read the use_*/lambda_*/horizon_* knobs.
        self._ks = config
        # Heads read the PaliGemma (prefix / VLM) stream -- "which checkpoint / phase am I in" is a
        # perception+instruction question -- so they project from paligemma width, not action-expert width.
        pg_w = paligemma_config.width
        ae_w = action_expert_config.width
        n_bins = len(config.horizon_upper_edges) + 1
        if config.use_checkpoint_head:
            self.ks_type_head = nnx.Linear(pg_w, config.num_checkpoint_types, rngs=rngs)
            # head output dim depends on loss mode: ce -> n_bins class logits; ordinal(CORAL) -> n_bins-1 cumulative logits
            h_out = n_bins if config.horizon_loss_type == "ce" else n_bins - 1
            self.ks_horizon_head = nnx.Linear(pg_w, h_out, rngs=rngs)
        if config.use_phase_head:
            self.ks_phase_head = nnx.Linear(pg_w, config.num_phase_classes, rngs=rngs)
        # The descriptor branch is Stage 2: predict a z_entry key-state latent target from
        # action-expert hidden, keeping it auxiliary (not fed back into action generation here).
        if config.use_z_entry_descriptor:
            self.ks_z_entry_descriptor_head = nnx.Linear(ae_w, config.z_entry_descriptor_dim, rngs=rngs)
        if config.use_z_head:  # Future frozen-Pi0/JEPA target path; hard-gated in Pi0Config.__post_init__.
            self.ks_z_head = nnx.Linear(pg_w, config.z_dim, rngs=rngs)
        if config.use_keystate_fusion:  # Stage 3 placeholder
            self.ks_type_embed = nnx.Embed(config.num_checkpoint_types, ae_w, rngs=rngs)
            self.ks_horizon_embed = nnx.Embed(n_bins, ae_w, rngs=rngs)
            self.ks_z_proj = nnx.Linear(config.z_dim, ae_w, rngs=rngs)
            self.ks_fuse_proj = nnx.Linear(ae_w, ae_w, rngs=rngs)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(einops.repeat(
                obs.image_masks[name],
                "b -> b s",
                s=image_tokens.shape[1],
            ))
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        ks_cond: at.Float[at.Array, "b emb"] | None = None,
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # KeyState->Action fusion (Stage 3 scaffold, hard-gated off this round): prepend one
        # condition token before the state token. Action tokens stay last so the
        # `suffix_out[:, -action_horizon:]` slice is unaffected. ks_cond is None in Stage 1, so
        # this branch never runs and the suffix layout is bit-identical to the original.
        if ks_cond is not None:
            tokens.append(ks_cond[:, None, :])
            input_mask.append(jnp.ones((ks_cond.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]
        # add a single state token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # mix timestep + action information using an MLP
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(self,
                     rng: at.KeyArrayLike,
                     observation: _model.Observation,
                     actions: _model.Actions,
                     *,
                     train: bool = False) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm([prefix_tokens, suffix_tokens],
                                                         mask=attn_mask,
                                                         positions=positions)
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        ks_losses = self._keystate_losses(prefix_out, prefix_mask, suffix_out, observation)
        return flow_loss, ks_losses

    def _keystate_losses(self, prefix_out, prefix_mask, suffix_out, obs) -> dict[str, at.Array]:
        """KeyState auxiliary losses. Returns {} for baseline / when labels absent,
        so train.py's `sum(ks_losses.values())` is a no-op and behaviour is unchanged."""
        ks_losses: dict[str, at.Array] = {}
        need_ks = self._ks.use_checkpoint_head or self._ks.use_phase_head or self._ks.use_z_entry_descriptor
        if not need_ks:
            return ks_losses
        if obs.keystate_h_entry is None or obs.keystate_type is None:
            if self._ks.use_z_entry_descriptor:
                raise ValueError("Stage2 descriptor training requires keystate_type and keystate_h_entry labels.")
            return ks_losses

        # masked-mean pool over the prefix (VLM) tokens -> [b, pg_w]
        pooled = (prefix_out * prefix_mask[..., None]).sum(1) / jnp.clip(prefix_mask.sum(1, keepdims=True), 1)

        # per-sample masks:
        #   * type is meaningful on every labeled frame, including terminal/no-next-checkpoint
        #     frames where keystate_type == 0 ("none"). This matters for execution: the
        #     model must learn to say "no next checkpoint" instead of freely predicting 1/2.
        #   * horizon/descriptor are meaningful only when a future/current checkpoint target exists.
        type_valid_b = obs.keystate_type >= 0
        h_valid_b = obs.keystate_h_entry >= 0  # [b] bool

        def masked_mean(per_sample, mask):
            mask = mask.astype(jnp.float32)
            return (mask * per_sample).sum() / jnp.clip(mask.sum(), 1.0)

        if self._ks.use_checkpoint_head:
            # Supervise with the DENSE next_checkpoint_type (sparse checkpoint_type marks only 2 frames).
            # safe-label BEFORE the loss: never let an invalid/meaningless label enter CE/ordinal.
            type_label = jnp.where(type_valid_b, obs.keystate_type, 0)
            type_ce = _softmax_xent(self.ks_type_head(pooled), type_label)
            ks_losses["loss_type"] = self._ks.lambda_type * masked_mean(type_ce, type_valid_b)

            h_label = jnp.where(h_valid_b, obs.keystate_h_entry, 0)  # bucket index 0..num_horizon_bins-1; safe 0 for invalid
            h_logits = self.ks_horizon_head(pooled)
            if self._ks.horizon_loss_type == "ce":
                h_loss = _softmax_xent(h_logits, h_label)
            else:  # "ordinal" (CORAL)
                n_bins = len(self._ks.horizon_upper_edges) + 1
                h_loss = _coral_loss(h_logits, h_label, n_bins)
            h_weight = jnp.ones_like(h_loss)
            h_weight = jnp.where(h_label == 0, self._ks.horizon_bin0_weight, h_weight)
            h_weight = jnp.where(h_label == 1, self._ks.horizon_bin1_weight, h_weight)
            h_mask = h_valid_b.astype(jnp.float32) * h_weight
            ks_losses["loss_h_entry"] = self._ks.lambda_h * masked_mean(h_loss, h_mask)

        if self._ks.use_phase_head:
            # phase is well-defined on every frame -> no valid mask, plain batch-mean.
            phase_bce = _sigmoid_bce(self.ks_phase_head(pooled), obs.keystate_phase)
            ks_losses["loss_ph"] = self._ks.lambda_ph * jnp.mean(phase_bce)

        if self._ks.use_z_entry_descriptor:
            if obs.keystate_z_entry_descriptor is None:
                raise ValueError(
                    "use_z_entry_descriptor=True but Observation.keystate_z_entry_descriptor is missing. "
                    "Generate /keystate/z_entry_descriptor, process it, and include it in the Stage2 repack transform.")
            z_target = jax.lax.stop_gradient(obs.keystate_z_entry_descriptor)
            z_pooled = jnp.mean(suffix_out[:, -self.action_horizon:], axis=1)
            z_pred = self.ks_z_entry_descriptor_head(z_pooled)
            per_sample_z = jnp.mean(jnp.square(z_pred - z_target), axis=-1)
            target_nonzero = jnp.linalg.norm(z_target, axis=-1) > 1e-6
            z_valid = (obs.keystate_type > 0) & h_valid_b & target_nonzero
            ks_losses["loss_z_entry_descriptor"] = self._ks.lambda_z_entry_descriptor * masked_mean(per_sample_z,
                                                                                                      z_valid)

        return ks_losses

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t,
                                                                           jnp.broadcast_to(time, batch_size))
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens],
                                                             mask=full_attn_mask,
                                                             positions=positions,
                                                             kv_cache=kv_cache)
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
