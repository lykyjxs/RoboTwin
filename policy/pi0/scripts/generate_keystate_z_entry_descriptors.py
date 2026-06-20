#!/usr/bin/env python3
"""Generate Stage2 z-entry descriptors for KeyState-labeled RoboTwin episodes.

Two descriptor backends are supported:

* ``bootstrap_dct_v1``: deterministic hand-crafted visual descriptor used only to
  validate the Stage2 plumbing/training path.
* ``frozen_pi0_prefix_projected_v1``: frozen Pi0/PaliGemma prefix hidden feature,
  deterministically projected to the requested descriptor dimension.
* ``frozen_pi0_action_expert_demo_t0001_projected_v1``: frozen Pi0 action-expert
  hidden at the checkpoint entry, conditioned on observation, instruction, robot
  state, and demonstration action chunk; projected to the requested descriptor
  dimension.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import h5py
import numpy as np

# Let this script import openpi when launched from the RoboTwin root as
# `policy/pi0/.venv/bin/python policy/pi0/scripts/...`.
PI0_ROOT = Path(__file__).resolve().parents[1]
PI0_SRC = PI0_ROOT / "src"
if str(PI0_SRC) not in sys.path:
    sys.path.insert(0, str(PI0_SRC))

DESCRIPTOR_NAME = "z_entry_descriptor"
BOOTSTRAP_ENCODER = "bootstrap_dct_v1"
FROZEN_PI0_PREFIX_ENCODER = "frozen_pi0_prefix_projected_v1"
FROZEN_PI0_ACTION_EXPERT_ENCODER = "frozen_pi0_action_expert_demo_t0001_projected_v1"
DEFAULT_CAMERAS = ("head_camera", "left_camera", "right_camera")
RAW_TO_ALOHA_CAMERA = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}
TYPE_ENTRY_ATTRS = {
    1: "pre_grasp_window_start",
    2: "pre_place_window_start",
}


def _decode_rgb(frame: np.ndarray | bytes) -> np.ndarray:
    """Decode one raw RoboTwin RGB frame into RGB uint8 HWC."""
    if isinstance(frame, np.ndarray) and frame.ndim == 3:
        image = frame
    else:
        encoded = np.frombuffer(frame, np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("failed to decode compressed camera frame")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _camera_descriptor(image: np.ndarray, *, dct_hw: tuple[int, int] = (16, 16), keep: int = 6) -> np.ndarray:
    """Small deterministic descriptor for one camera view.

    It mixes RGB statistics with low-frequency DCT coefficients. This is deliberately
    lightweight and stable; it is not meant to be a learned representation.
    """
    resized = cv2.resize(image, dct_hw, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    means = resized.mean(axis=(0, 1))
    stds = resized.std(axis=(0, 1))
    mins = resized.min(axis=(0, 1))
    maxs = resized.max(axis=(0, 1))

    gray = cv2.cvtColor((resized * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    dct = cv2.dct(gray)
    dct_low = dct[:keep, :keep].reshape(-1)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))
    grad_stats = np.array([grad_mag.mean(), grad_mag.std(), grad_mag.max()], dtype=np.float32)

    return np.concatenate([means, stds, mins, maxs, dct_low, grad_stats]).astype(np.float32)


def _projection_matrix(in_dim: int, out_dim: int, *, seed_base: int = 1729) -> np.ndarray:
    """Versioned deterministic projection."""
    seed = seed_base + in_dim * 17 + out_dim * 31
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    proj /= np.sqrt(float(in_dim))
    return proj


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm > 1e-6:
        x = x / norm
    return x.astype(np.float32)


def _entry_bootstrap_descriptor(root: h5py.File, entry_idx: int, cameras: tuple[str, ...], z_dim: int) -> np.ndarray:
    per_cam = []
    for camera in cameras:
        dataset_path = f"/observation/{camera}/rgb"
        if dataset_path not in root:
            raise KeyError(f"missing camera dataset {dataset_path}")
        image = _decode_rgb(root[dataset_path][entry_idx])
        per_cam.append(_camera_descriptor(image))
    raw = np.concatenate(per_cam).astype(np.float32)
    descriptor = raw @ _projection_matrix(raw.shape[0], z_dim, seed_base=1729)
    return _l2_normalize(descriptor)


def _state_at(root: h5py.File, frame_idx: int) -> np.ndarray:
    left_arm = np.asarray(root["/joint_action/left_arm"][frame_idx]).reshape(-1)
    right_arm = np.asarray(root["/joint_action/right_arm"][frame_idx]).reshape(-1)
    left_gripper = np.asarray(root["/joint_action/left_gripper"][frame_idx]).reshape(-1)
    right_gripper = np.asarray(root["/joint_action/right_gripper"][frame_idx]).reshape(-1)
    return np.concatenate([left_arm, left_gripper, right_arm, right_gripper]).astype(np.float32)


def _action_chunk_from_states(root: h5py.File, frame_idx: int, horizon: int) -> np.ndarray:
    """Build a raw Aloha-style action chunk from future recorded states.

    This mirrors process_data.py's convention: action at processed index j is the next
    recorded robot state. Near the end of an episode, repeat the last available state
    so the action expert always receives a full horizon.
    """
    total = int(root["/joint_action/left_arm"].shape[0])
    if total < 2:
        raise ValueError("episode is too short to build an action chunk")
    actions = []
    for offset in range(horizon):
        idx = min(frame_idx + 1 + offset, total - 1)
        actions.append(_state_at(root, idx))
    return np.stack(actions, axis=0).astype(np.float32)


def _read_instruction(base_dir: Path, episode: int, instruction_key: str) -> str:
    path = base_dir / "instructions" / f"episode{episode}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing instruction file: {path}")
    with path.open("r") as f:
        data = json.load(f)
    if instruction_key in data:
        value = data[instruction_key]
    elif "instructions" in data:
        value = data["instructions"]
    else:
        keys = ", ".join(sorted(data))
        raise KeyError(f"{path} has no {instruction_key!r} or 'instructions' key; keys={keys}")
    if isinstance(value, list):
        if not value:
            raise ValueError(f"instruction list is empty in {path}")
        value = value[0]
    return str(value)


class FrozenPi0PrefixProjector:
    """Frozen Pi0 prefix feature extractor with deterministic 64-D projection."""

    def __init__(
        self,
        *,
        target_config: str,
        target_params_path: str | None,
        target_missing_regex: str,
        z_dim: int,
        projection_seed_base: int,
    ):
        import jax
        import jax.numpy as jnp
        from flax import nnx

        from openpi import transforms
        from openpi.models import model as _model
        from openpi.models import pi0 as _pi0
        from openpi.models import tokenizer as _tokenizer
        from openpi.policies import aloha_policy
        from openpi.training import config as _config
        from openpi.training import weight_loaders

        train_config = _config.get_config(target_config)
        if target_params_path is not None:
            train_config.weight_loader = weight_loaders.CheckpointWeightLoader(
                target_params_path,
                missing_regex=target_missing_regex,
            )

        model = train_config.model.create(jax.random.key(0))
        graphdef, state = nnx.split(model)
        ref_params = state.to_pure_dict()
        loaded_params = train_config.weight_loader.load(ref_params)
        state.replace_by_pure_dict(loaded_params)
        self._model = nnx.merge(graphdef, state)
        self._model.eval()
        self._model_config = train_config.model
        self._jax = jax
        self._jnp = jnp
        self._model_mod = _model
        self._pi0_mod = _pi0
        self._z_dim = z_dim
        self._projection_seed_base = projection_seed_base
        self.source_dim: int | None = None

        self._input_transform = transforms.compose([
            aloha_policy.AlohaInputs(action_dim=train_config.model.action_dim, adapt_to_pi=False),
            transforms.ResizeImages(224, 224),
            transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer(train_config.model.max_token_len)),
        ])

    def __call__(self, root: h5py.File, entries: list[int], cameras: tuple[str, ...], instruction: str) -> dict[int, np.ndarray]:
        unsupported = [camera for camera in cameras if camera not in RAW_TO_ALOHA_CAMERA]
        if unsupported:
            raise ValueError(
                f"{FROZEN_PI0_PREFIX_ENCODER} only supports default RoboTwin cameras; unsupported={unsupported}")

        transformed = []
        for entry_idx in entries:
            images = {}
            for raw_name in cameras:
                aloha_name = RAW_TO_ALOHA_CAMERA[raw_name]
                image = _decode_rgb(root[f"/observation/{raw_name}/rgb"][entry_idx])
                images[aloha_name] = np.transpose(image, (2, 0, 1))  # AlohaInputs expects CHW.
            transformed.append(self._input_transform({
                "images": images,
                "state": _state_at(root, entry_idx),
                "prompt": instruction,
            }))

        batch = self._jax.tree.map(lambda *xs: self._jnp.asarray(np.stack(xs, axis=0)), *transformed)
        obs = self._model_mod.Observation.from_dict(batch)
        obs = self._model_mod.preprocess_observation(None, obs, train=False)

        prefix_tokens, prefix_mask, prefix_ar_mask = self._model.embed_prefix(obs)
        prefix_attn_mask = self._pi0_mod.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = self._jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = self._model.PaliGemma.llm([prefix_tokens, None],
                                                       mask=prefix_attn_mask,
                                                       positions=positions)
        pooled = (prefix_out * prefix_mask[..., None]).sum(1) / self._jnp.clip(prefix_mask.sum(1, keepdims=True), 1)
        pooled_np = np.asarray(self._jax.device_get(pooled), dtype=np.float32)
        self.source_dim = int(pooled_np.shape[-1])
        projection = _projection_matrix(self.source_dim, self._z_dim, seed_base=self._projection_seed_base)
        projected = pooled_np @ projection
        return {entry: _l2_normalize(projected[i]) for i, entry in enumerate(entries)}


class FrozenPi0ActionExpertProjector(FrozenPi0PrefixProjector):
    """Frozen Pi0 action-expert feature extractor with deterministic 64-D projection."""

    def __call__(self, root: h5py.File, entries: list[int], cameras: tuple[str, ...], instruction: str) -> dict[int, np.ndarray]:
        unsupported = [camera for camera in cameras if camera not in RAW_TO_ALOHA_CAMERA]
        if unsupported:
            raise ValueError(
                f"{FROZEN_PI0_ACTION_EXPERT_ENCODER} only supports default RoboTwin cameras; unsupported={unsupported}")

        transformed = []
        raw_action_chunks = []
        for entry_idx in entries:
            images = {}
            for raw_name in cameras:
                aloha_name = RAW_TO_ALOHA_CAMERA[raw_name]
                image = _decode_rgb(root[f"/observation/{raw_name}/rgb"][entry_idx])
                images[aloha_name] = np.transpose(image, (2, 0, 1))  # AlohaInputs expects CHW.
            raw_action_chunk = _action_chunk_from_states(root, entry_idx, self._model_config.action_horizon)
            raw_action_chunks.append(raw_action_chunk)
            transformed.append(self._input_transform({
                "images": images,
                "state": _state_at(root, entry_idx),
                "actions": raw_action_chunk,
                "prompt": instruction,
            }))

        batch = self._jax.tree.map(lambda *xs: self._jnp.asarray(np.stack(xs, axis=0)), *transformed)
        obs = self._model_mod.Observation.from_dict(batch)
        actions = self._jnp.asarray(np.stack([item["actions"] for item in transformed], axis=0))
        obs = self._model_mod.preprocess_observation(None, obs, train=False)

        timestep = self._jnp.full((len(entries), ), 0.001, dtype=self._jnp.float32)
        prefix_tokens, prefix_mask, prefix_ar_mask = self._model.embed_prefix(obs)
        suffix_tokens, suffix_mask, suffix_ar_mask = self._model.embed_suffix(obs, actions, timestep)
        input_mask = self._jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = self._jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = self._pi0_mod.make_attn_mask(input_mask, ar_mask)
        positions = self._jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self._model.PaliGemma.llm([prefix_tokens, suffix_tokens],
                                                       mask=attn_mask,
                                                       positions=positions)
        action_hidden = suffix_out[:, -self._model_config.action_horizon:]
        pooled = self._jnp.mean(action_hidden, axis=1)
        pooled_np = np.asarray(self._jax.device_get(pooled), dtype=np.float32)
        self.source_dim = int(pooled_np.shape[-1])
        projection = _projection_matrix(self.source_dim, self._z_dim, seed_base=self._projection_seed_base)
        projected = pooled_np @ projection
        return {entry: _l2_normalize(projected[i]) for i, entry in enumerate(entries)}


def _entry_from_attrs_or_labels(ks: h5py.Group, next_type: np.ndarray, h_entry: np.ndarray, type_id: int) -> int:
    attr_name = TYPE_ENTRY_ATTRS.get(type_id)
    if attr_name is not None and attr_name in ks.attrs:
        entry = int(ks.attrs[attr_name])
        if entry >= 0:
            return entry

    candidates = np.nonzero((next_type == type_id) & (h_entry == 0))[0]
    if candidates.size:
        return int(candidates[0])
    return -1


def build_descriptors_for_episode(
    hdf5_path: Path,
    *,
    episode: int,
    base_dir: Path,
    z_dim: int,
    cameras: tuple[str, ...],
    encoder: str,
    frozen_projector: FrozenPi0PrefixProjector | None,
    instruction_key: str,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    with h5py.File(hdf5_path, "r" if dry_run else "a") as root:
        if "keystate" not in root:
            raise KeyError(f"{hdf5_path} has no /keystate group; run Stage0 labeler first")
        ks = root["keystate"]
        if "next_checkpoint_type" not in ks or "h_entry" not in ks:
            raise KeyError(f"{hdf5_path} is missing Stage0 next_checkpoint_type/h_entry labels")
        if DESCRIPTOR_NAME in ks and not overwrite and not dry_run:
            raise FileExistsError(f"{hdf5_path} already has /keystate/{DESCRIPTOR_NAME}; pass --overwrite")

        next_type = ks["next_checkpoint_type"][()].astype(np.int32)
        h_entry = ks["h_entry"][()].astype(np.int32)
        if next_type.shape[0] != h_entry.shape[0]:
            raise ValueError(f"label length mismatch in {hdf5_path}: type={next_type.shape}, h_entry={h_entry.shape}")
        T = int(next_type.shape[0])

        entry_by_type = {}
        for type_id in sorted(int(x) for x in np.unique(next_type) if int(x) > 0):
            entry = _entry_from_attrs_or_labels(ks, next_type, h_entry, type_id)
            if entry < 0 or entry >= T:
                raise ValueError(f"cannot resolve valid entry frame for checkpoint type {type_id} in {hdf5_path}")
            entry_by_type[type_id] = entry

        if encoder == BOOTSTRAP_ENCODER:
            descriptor_by_type = {
                type_id: _entry_bootstrap_descriptor(root, entry, cameras, z_dim)
                for type_id, entry in entry_by_type.items()
            }
            source_dim = None
        elif encoder in (FROZEN_PI0_PREFIX_ENCODER, FROZEN_PI0_ACTION_EXPERT_ENCODER):
            if frozen_projector is None:
                raise ValueError("frozen_projector is required for frozen Pi0 encoder")
            instruction = _read_instruction(base_dir, episode, instruction_key)
            descriptor_by_entry = frozen_projector(root, list(entry_by_type.values()), cameras, instruction)
            descriptor_by_type = {type_id: descriptor_by_entry[entry] for type_id, entry in entry_by_type.items()}
            source_dim = frozen_projector.source_dim
        else:
            raise ValueError(f"unknown encoder {encoder!r}")

        descriptors = np.zeros((T, z_dim), dtype=np.float32)
        # z_entry_descriptor means "what will the upcoming checkpoint-window entry look like".
        # Once h_entry reaches 0, the robot is already inside that window, so the z-entry target is
        # no longer a future-entry prediction target. Keep the descriptor zero there; the model loss
        # masks zero targets via target_nonzero.
        z_valid_mask = (next_type > 0) & (h_entry > 0)
        for type_id, descriptor in descriptor_by_type.items():
            descriptors[(next_type == type_id) & z_valid_mask] = descriptor

        summary = {
            "path": str(hdf5_path),
            "T": T,
            "z_dim": z_dim,
            "entries": entry_by_type,
            "valid_frames": int(z_valid_mask.sum()),
            "encoder": encoder,
            "source_dim": source_dim,
            "dry_run": dry_run,
        }
        if dry_run:
            return summary

        if DESCRIPTOR_NAME in ks:
            del ks[DESCRIPTOR_NAME]
        ks.create_dataset(DESCRIPTOR_NAME, data=descriptors, dtype="float32")
        ks.attrs[f"{DESCRIPTOR_NAME}_dim"] = int(z_dim)
        ks.attrs[f"{DESCRIPTOR_NAME}_encoder"] = encoder
        ks.attrs[f"{DESCRIPTOR_NAME}_source"] = "entry"
        ks.attrs[f"{DESCRIPTOR_NAME}_normalized"] = True
        ks.attrs[f"{DESCRIPTOR_NAME}_version"] = 3 if encoder == FROZEN_PI0_ACTION_EXPERT_ENCODER else (2 if encoder == FROZEN_PI0_PREFIX_ENCODER else 1)
        ks.attrs[f"{DESCRIPTOR_NAME}_supervision"] = "pre_entry_only_h_entry_gt_0"
        ks.attrs[f"{DESCRIPTOR_NAME}_entries"] = json.dumps(entry_by_type)
        if source_dim is not None:
            ks.attrs[f"{DESCRIPTOR_NAME}_source_dim"] = int(source_dim)
            ks.attrs[f"{DESCRIPTOR_NAME}_projection"] = "fixed_random_projection_v1"
            if encoder == FROZEN_PI0_ACTION_EXPERT_ENCODER:
                ks.attrs[f"{DESCRIPTOR_NAME}_hidden_source"] = "action_expert_suffix_out_action_tokens_mean"
                ks.attrs[f"{DESCRIPTOR_NAME}_action_context"] = "demo_action_chunk"
                ks.attrs[f"{DESCRIPTOR_NAME}_timestep"] = 0.001
        return summary


def _episodes_from_args(data_dir: Path, all_episodes: bool, episode: int | None) -> list[int]:
    if all_episodes:
        episodes = []
        for path in data_dir.glob("episode*.hdf5"):
            stem = path.stem
            suffix = stem[len("episode"):]
            if suffix.isdigit():
                episodes.append(int(suffix))
        return sorted(episodes)
    if episode is None:
        raise ValueError("pass --all or --episode")
    return [episode]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage2 z-entry descriptors")
    parser.add_argument("--task", default="place_a2b_left")
    parser.add_argument("--config", default="demo_clean")
    parser.add_argument("--data-root", default="data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="process every episode found")
    group.add_argument("--episode", type=int, help="process a single episode index")
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--source", choices=("entry",), default="entry")
    parser.add_argument("--encoder", choices=(BOOTSTRAP_ENCODER, FROZEN_PI0_PREFIX_ENCODER, FROZEN_PI0_ACTION_EXPERT_ENCODER), default=BOOTSTRAP_ENCODER)
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--instruction-key", default="seen")
    parser.add_argument("--target-config", default="pi0_base_aloha_robotwin_lora")
    parser.add_argument("--target-params-path", default=None)
    parser.add_argument("--target-missing-regex", default=".*lora.*")
    parser.add_argument("--projection-seed-base", type=int, default=8675309)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.z_dim <= 0:
        raise ValueError("--z-dim must be positive")

    base_dir = Path(args.data_root) / args.task / args.config
    data_dir = base_dir / "data"
    episodes = _episodes_from_args(data_dir, args.all, args.episode)
    if not episodes:
        raise FileNotFoundError(f"no episodes found in {data_dir}")

    frozen_projector = None
    if args.encoder == FROZEN_PI0_PREFIX_ENCODER:
        frozen_projector = FrozenPi0PrefixProjector(
            target_config=args.target_config,
            target_params_path=args.target_params_path,
            target_missing_regex=args.target_missing_regex,
            z_dim=args.z_dim,
            projection_seed_base=args.projection_seed_base,
        )
    elif args.encoder == FROZEN_PI0_ACTION_EXPERT_ENCODER:
        frozen_projector = FrozenPi0ActionExpertProjector(
            target_config=args.target_config,
            target_params_path=args.target_params_path,
            target_missing_regex=args.target_missing_regex,
            z_dim=args.z_dim,
            projection_seed_base=args.projection_seed_base,
        )

    ok = 0
    for ep in episodes:
        hdf5_path = data_dir / f"episode{ep}.hdf5"
        if not hdf5_path.exists():
            print(f"[skip] episode{ep}: missing {hdf5_path}")
            continue
        summary = build_descriptors_for_episode(
            hdf5_path,
            episode=ep,
            base_dir=base_dir,
            z_dim=args.z_dim,
            cameras=tuple(args.cameras),
            encoder=args.encoder,
            frozen_projector=frozen_projector,
            instruction_key=args.instruction_key,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        action = "dry-run" if args.dry_run else "wrote"
        source = f" source_dim={summary['source_dim']}" if summary.get("source_dim") else ""
        print(
            f"[ok] {action} episode{ep}: T={summary['T']} valid={summary['valid_frames']} "
            f"entries={summary['entries']} z_dim={summary['z_dim']} encoder={summary['encoder']}{source}"
        )
        ok += 1
    print(f"summary: ok={ok} total_requested={len(episodes)}")


if __name__ == "__main__":
    main()
