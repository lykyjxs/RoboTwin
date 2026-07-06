#!/usr/bin/env python3
"""Offline validation for Pi0 KeyState auxiliary heads.

This evaluates teacher-forced validation batches from a LeRobot repo and reports:
type / h-entry / phase classification metrics, optional z-entry descriptor metrics,
optional absolute keypose metrics, and the teacher-forced flow loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

PI0_ROOT = Path(__file__).resolve().parents[1]
PI0_SRC = PI0_ROOT / "src"
if str(PI0_SRC) not in sys.path:
    sys.path.insert(0, str(PI0_SRC))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Pi0 KeyState heads on a LeRobot validation repo.")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint step directory containing params/ and assets/.")
    parser.add_argument("--repo-id", required=True, help="Local LeRobot validation repo id.")
    parser.add_argument(
        "--norm-asset-id",
        default="beat_block_hammer_demo_clean_50_posthit_keystate_stage2_actionexpert",
        help="Asset id for norm stats inside <checkpoint-dir>/assets.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=0, help="0 means one full drop-last pass over the dataset.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--time-mode", choices=("fixed", "beta"), default="fixed")
    parser.add_argument("--fixed-time", type=float, default=0.001)
    parser.add_argument("--hf-lerobot-home", default="./data/lerobot")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class MetricAccumulator:
    def __init__(self, *, num_types: int, num_h_bins: int, num_phase_classes: int):
        self.num_types = num_types
        self.num_h_bins = num_h_bins
        self.num_phase_classes = num_phase_classes
        self.frames = 0
        self.flow_sum = 0.0
        self.flow_count = 0

        self.type_total = 0
        self.type_correct = 0
        self.type_confusion = np.zeros((num_types, num_types), dtype=np.int64)

        self.h_total = 0
        self.h_correct = 0
        self.h_within1 = 0
        self.h_abs_bin_error_sum = 0.0
        self.h_confusion = np.zeros((num_h_bins, num_h_bins), dtype=np.int64)

        self.phase_label_total = 0
        self.phase_label_correct = 0
        self.phase_sample_total = 0
        self.phase_sample_exact = 0
        self.phase_tp = 0
        self.phase_fp = 0
        self.phase_fn = 0

        self.z_count = 0
        self.z_mse_sum = 0.0
        self.z_cos_sum = 0.0

        self.pose_count = 0
        self.pose_mse_sum = 0.0
        self.pose_pos_l2_sum = 0.0
        self.pose_pos_l2_squares_sum = 0.0
        self.pose_pos_l2_values: list[float] = []
        self.pose_quat_count = 0
        self.pose_quat_angle_sum = 0.0
        self.pose_quat_angle_values: list[float] = []

    def update(self, batch: dict[str, np.ndarray]) -> None:
        type_t = np.asarray(batch["type_target"]).astype(np.int64)
        type_p = np.asarray(batch["type_pred"]).astype(np.int64)
        h_t = np.asarray(batch["h_target"]).astype(np.int64)
        h_p = np.asarray(batch["h_pred"]).astype(np.int64)
        phase_t = np.asarray(batch["phase_target"]).astype(np.float32)
        phase_prob = np.asarray(batch["phase_prob"]).astype(np.float32)
        flow = np.asarray(batch["flow_loss"]).astype(np.float64)

        self.frames += int(type_t.shape[0])
        self.flow_sum += float(flow.sum())
        self.flow_count += int(flow.size)

        type_valid = type_t >= 0
        if type_valid.any():
            tgt = np.clip(type_t[type_valid], 0, self.num_types - 1)
            pred = np.clip(type_p[type_valid], 0, self.num_types - 1)
            self.type_total += int(tgt.size)
            self.type_correct += int((tgt == pred).sum())
            for t, p in zip(tgt, pred, strict=False):
                self.type_confusion[t, p] += 1

        h_valid = h_t >= 0
        if h_valid.any():
            tgt = np.clip(h_t[h_valid], 0, self.num_h_bins - 1)
            pred = np.clip(h_p[h_valid], 0, self.num_h_bins - 1)
            abs_err = np.abs(tgt - pred)
            self.h_total += int(tgt.size)
            self.h_correct += int((tgt == pred).sum())
            self.h_within1 += int((abs_err <= 1).sum())
            self.h_abs_bin_error_sum += float(abs_err.sum())
            for t, p in zip(tgt, pred, strict=False):
                self.h_confusion[t, p] += 1

        phase_pred = phase_prob >= 0.5
        phase_true = phase_t >= 0.5
        self.phase_label_total += int(phase_true.size)
        self.phase_label_correct += int((phase_pred == phase_true).sum())
        sample_exact = np.all(phase_pred == phase_true, axis=-1)
        self.phase_sample_total += int(sample_exact.size)
        self.phase_sample_exact += int(sample_exact.sum())
        self.phase_tp += int((phase_pred & phase_true).sum())
        self.phase_fp += int((phase_pred & ~phase_true).sum())
        self.phase_fn += int((~phase_pred & phase_true).sum())

        z_pred = batch.get("z_pred")
        z_target = batch.get("z_target")
        if z_pred is not None and z_target is not None:
            z_pred = np.asarray(z_pred, dtype=np.float32)
            z_target = np.asarray(z_target, dtype=np.float32)
            z_valid = (type_t > 0) & h_valid & (np.linalg.norm(z_target, axis=-1) > 1e-6)
            if z_valid.any():
                zp = z_pred[z_valid]
                zt = z_target[z_valid]
                per = np.mean(np.square(zp - zt), axis=-1)
                denom = np.maximum(np.linalg.norm(zp, axis=-1) * np.linalg.norm(zt, axis=-1), 1e-8)
                cos = np.sum(zp * zt, axis=-1) / denom
                self.z_count += int(per.size)
                self.z_mse_sum += float(per.sum())
                self.z_cos_sum += float(cos.sum())

        pose_pred = batch.get("keypose_pred")
        pose_target = batch.get("keypose_target")
        if pose_pred is not None and pose_target is not None:
            pose_pred = np.asarray(pose_pred, dtype=np.float32)
            pose_target = np.asarray(pose_target, dtype=np.float32)
            pose_valid = (type_t > 0) & h_valid & (np.linalg.norm(pose_target, axis=-1) > 1e-6)
            if pose_valid.any():
                pp = pose_pred[pose_valid]
                pt = pose_target[pose_valid]
                per = np.mean(np.square(pp - pt), axis=-1)
                pos_l2 = np.linalg.norm(pp[:, :3] - pt[:, :3], axis=-1)
                self.pose_count += int(per.size)
                self.pose_mse_sum += float(per.sum())
                self.pose_pos_l2_sum += float(pos_l2.sum())
                self.pose_pos_l2_squares_sum += float(np.square(pos_l2).sum())
                self.pose_pos_l2_values.extend(float(x) for x in pos_l2)

                q_pred = pp[:, 3:7]
                q_target = pt[:, 3:7]
                q_pred_norm = np.linalg.norm(q_pred, axis=-1)
                q_target_norm = np.linalg.norm(q_target, axis=-1)
                q_valid = (q_pred_norm > 1e-6) & (q_target_norm > 1e-6)
                if q_valid.any():
                    qp = q_pred[q_valid] / q_pred_norm[q_valid, None]
                    qt = q_target[q_valid] / q_target_norm[q_valid, None]
                    dots = np.clip(np.abs(np.sum(qp * qt, axis=-1)), 0.0, 1.0)
                    angles = np.degrees(2.0 * np.arccos(dots))
                    self.pose_quat_count += int(angles.size)
                    self.pose_quat_angle_sum += float(angles.sum())
                    self.pose_quat_angle_values.extend(float(x) for x in angles)

    def result(self) -> dict[str, Any]:
        z_mse = _safe_div(self.z_mse_sum, self.z_count)
        pose_mse = _safe_div(self.pose_mse_sum, self.pose_count)
        precision = _safe_div(self.phase_tp, self.phase_tp + self.phase_fp)
        recall = _safe_div(self.phase_tp, self.phase_tp + self.phase_fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        pos_values = np.asarray(self.pose_pos_l2_values, dtype=np.float64)
        quat_values = np.asarray(self.pose_quat_angle_values, dtype=np.float64)
        return {
            "frames": self.frames,
            "flow_mse": _safe_div(self.flow_sum, self.flow_count),
            "type": {
                "count": self.type_total,
                "accuracy": _safe_div(self.type_correct, self.type_total),
                "confusion_target_x_pred": self.type_confusion.tolist(),
            },
            "h_entry_bin": {
                "count": self.h_total,
                "accuracy": _safe_div(self.h_correct, self.h_total),
                "within_1_bin_accuracy": _safe_div(self.h_within1, self.h_total),
                "mean_abs_bin_error": _safe_div(self.h_abs_bin_error_sum, self.h_total),
                "confusion_target_x_pred": self.h_confusion.tolist(),
            },
            "phase": {
                "label_accuracy": _safe_div(self.phase_label_correct, self.phase_label_total),
                "sample_exact_accuracy": _safe_div(self.phase_sample_exact, self.phase_sample_total),
                "micro_precision": precision,
                "micro_recall": recall,
                "micro_f1": f1,
            },
            "z_entry_descriptor": {
                "count": self.z_count,
                "mse": z_mse,
                "rmse": math.sqrt(z_mse) if self.z_count else 0.0,
                "cosine_mean": _safe_div(self.z_cos_sum, self.z_count),
            },
            "keypose_entry_abs": {
                "count": self.pose_count,
                "mse_7d": pose_mse,
                "rmse_7d": math.sqrt(pose_mse) if self.pose_count else 0.0,
                "position_l2_mean": _safe_div(self.pose_pos_l2_sum, self.pose_count),
                "position_l2_rmse": math.sqrt(_safe_div(self.pose_pos_l2_squares_sum, self.pose_count))
                if self.pose_count
                else 0.0,
                "position_l2_median": float(np.median(pos_values)) if pos_values.size else 0.0,
                "quat_angle_deg_count": self.pose_quat_count,
                "quat_angle_deg_mean": _safe_div(self.pose_quat_angle_sum, self.pose_quat_count),
                "quat_angle_deg_median": float(np.median(quat_values)) if quat_values.size else 0.0,
            },
        }


def main() -> None:
    args = _parse_args()
    os.environ["HF_LEROBOT_HOME"] = args.hf_lerobot_home
    os.environ.pop("LEROBOT_HOME", None)

    import jax
    import jax.numpy as jnp
    from flax import nnx

    from openpi.models import model as _model
    from openpi.models import pi0 as _pi0
    from openpi.training import config as _config
    from openpi.training import data_loader as _data_loader

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    train_config = _config.get_config(args.config_name)
    train_config.batch_size = args.batch_size
    train_config.num_workers = args.num_workers
    train_config.seed = args.eval_seed
    train_config.data.repo_id = args.repo_id
    train_config.data.assets.assets_dir = str(checkpoint_dir / "assets")
    train_config.data.assets.asset_id = args.norm_asset_id

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    base_dataset = _data_loader.create_dataset(data_config, train_config.model)
    dataset_len = len(base_dataset)
    dataset = _data_loader.transform_dataset(base_dataset, data_config)
    full_batches = dataset_len // args.batch_size
    num_batches = full_batches if args.num_batches == 0 else args.num_batches
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=args.batch_size,
        shuffle=False,
        num_batches=num_batches,
        num_workers=args.num_workers,
        seed=args.eval_seed,
    )

    print(f"[load] config={args.config_name}")
    print(f"[load] checkpoint={checkpoint_dir}")
    print(f"[data] repo={args.repo_id} dataset_len={dataset_len} batches={num_batches} batch_size={args.batch_size}")

    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = train_config.model.load(params)
    model.eval()
    graphdef, state = nnx.split(model)

    use_phase = bool(getattr(train_config.model, "use_phase_head", False))
    use_z = bool(getattr(train_config.model, "use_z_entry_descriptor", False))
    use_keypose = bool(getattr(train_config.model, "use_keypose_entry_abs", False))
    num_h_bins = len(getattr(train_config.model, "horizon_upper_edges", (1, 4, 7, 11, 21, 51))) + 1
    num_types = int(getattr(train_config.model, "num_checkpoint_types", 3))
    num_phase_classes = int(getattr(train_config.model, "num_phase_classes", 3))
    horizon_loss_type = str(getattr(train_config.model, "horizon_loss_type", "ce"))

    def eval_batch(state, rng, observation, actions):
        module = nnx.merge(graphdef, state)
        observation = _model.preprocess_observation(None, observation, train=False)
        noise_rng, time_rng, ks_rng = jax.random.split(rng, 3)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        if args.time_mode == "fixed":
            time = jnp.full(batch_shape, args.fixed_time, dtype=actions.dtype)
        else:
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1.0 - time[..., None, None]) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = module.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = module.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = module.PaliGemma.llm([prefix_tokens, suffix_tokens],
                                                           mask=attn_mask,
                                                           positions=positions)
        prefix_pooled = module._pool_prefix(prefix_out, prefix_mask)
        action_hidden = suffix_out[:, -module.action_horizon:]

        type_logits = module.ks_type_head(prefix_pooled)
        h_logits = module.ks_horizon_head(prefix_pooled)
        if horizon_loss_type == "ce":
            h_pred = jnp.argmax(h_logits, axis=-1).astype(jnp.int32)
        else:
            h_pred = jnp.sum(jax.nn.sigmoid(h_logits) > 0.5, axis=-1).astype(jnp.int32)
        if use_phase:
            phase_prob = jax.nn.sigmoid(module.ks_phase_head(prefix_pooled))
        else:
            phase_prob = jnp.zeros((actions.shape[0], num_phase_classes), dtype=jnp.float32)

        flow_hidden = module._apply_keystate_late_fusion(observation, prefix_pooled, action_hidden, ks_rng, train=False)
        v_t = module.action_out_proj(flow_hidden)
        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=(-2, -1))

        out = {
            "type_target": observation.keystate_type,
            "type_pred": jnp.argmax(type_logits, axis=-1).astype(jnp.int32),
            "h_target": observation.keystate_h_entry,
            "h_pred": h_pred,
            "phase_target": observation.keystate_phase,
            "phase_prob": phase_prob,
            "flow_loss": flow_loss,
        }
        if use_z:
            out["z_target"] = observation.keystate_z_entry_descriptor
            out["z_pred"] = module.ks_z_entry_descriptor_head(jnp.mean(action_hidden, axis=1))
        if use_keypose:
            out["keypose_target"] = observation.keystate_keypose_entry_abs
            out["keypose_pred"] = module.ks_keypose_entry_abs_head(jnp.mean(action_hidden, axis=1))
        return out

    eval_batch_jit = jax.jit(eval_batch)
    acc = MetricAccumulator(num_types=num_types, num_h_bins=num_h_bins, num_phase_classes=num_phase_classes)
    base_rng = jax.random.key(args.eval_seed)

    for batch_idx, batch in enumerate(loader):
        obs = _model.Observation.from_dict(batch)
        rng = jax.random.fold_in(base_rng, batch_idx)
        out = eval_batch_jit(state, rng, obs, batch["actions"])
        out_np = jax.tree.map(lambda x: np.asarray(jax.device_get(x)) if x is not None else None, out)
        acc.update(out_np)
        if batch_idx == 0 or (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == num_batches:
            print(f"[eval] batch {batch_idx + 1}/{num_batches}")

    result = {
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "repo_id": args.repo_id,
        "norm_asset_id": args.norm_asset_id,
        "dataset_len": dataset_len,
        "drop_last_eval_frames": num_batches * args.batch_size,
        "batch_size": args.batch_size,
        "num_batches": num_batches,
        "eval_seed": args.eval_seed,
        "time_mode": args.time_mode,
        "fixed_time": args.fixed_time,
        "use_phase_head": use_phase,
        "use_z_entry_descriptor": use_z,
        "use_keypose_entry_abs": use_keypose,
        "metrics": acc.result(),
    }
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[write] {output_json}")

    if args.output_csv:
        metrics = result["metrics"]
        row = {
            "config_name": args.config_name,
            "checkpoint_dir": str(checkpoint_dir),
            "repo_id": args.repo_id,
            "time_mode": args.time_mode,
            "fixed_time": args.fixed_time,
            "frames": metrics["frames"],
            "flow_mse": metrics["flow_mse"],
            "type_acc": metrics["type"]["accuracy"],
            "h_acc": metrics["h_entry_bin"]["accuracy"],
            "h_within1": metrics["h_entry_bin"]["within_1_bin_accuracy"],
            "h_mae": metrics["h_entry_bin"]["mean_abs_bin_error"],
            "phase_label_acc": metrics["phase"]["label_accuracy"],
            "phase_exact_acc": metrics["phase"]["sample_exact_accuracy"],
            "phase_micro_f1": metrics["phase"]["micro_f1"],
            "z_count": metrics["z_entry_descriptor"]["count"],
            "z_mse": metrics["z_entry_descriptor"]["mse"],
            "z_cos": metrics["z_entry_descriptor"]["cosine_mean"],
            "keypose_count": metrics["keypose_entry_abs"]["count"],
            "keypose_mse_7d": metrics["keypose_entry_abs"]["mse_7d"],
            "keypose_pos_l2_mean": metrics["keypose_entry_abs"]["position_l2_mean"],
            "keypose_pos_l2_median": metrics["keypose_entry_abs"]["position_l2_median"],
            "keypose_quat_angle_deg_mean": metrics["keypose_entry_abs"]["quat_angle_deg_mean"],
        }
        csv_path = Path(args.output_csv).resolve()
        _append_csv(csv_path, row)
        print(f"[write] {csv_path}")


if __name__ == "__main__":
    main()
