"""
KeyState Stage 0 offline labeler.

Reads the per-frame substep tags that were passively logged during data collection
(see envs/_base_task.py: move(stage_tag=...) -> get_obs() -> /keystate_substep/*),
and derives the 5 keystates by looking up a FIXED semantic mapping table -- no
human/VLM observation, no thresholds for the phase boundaries.

For place_a2b_left, play_once() always emits this fixed sequence of Actions:
    move(grasp_actor)          -> [move(pre_grasp), move(grasp), close]   stage="grasp"
    move(move_by_displacement) -> [move(lift)]                           stage="lift"
    move(place_actor)          -> [move(place_pre), move(place), open]   stage="place"

Fixed mapping (stage, sub_index, action_type) -> keystate:
    (grasp, 1, move)          -> end frame = pre-grasp checkpoint
    (grasp, 3, gripper_close) -> from its end: object-in-hand = 1
    (lift,  1, move)          -> from its end: lifted = 1
    (place, 1, move)          -> end frame = pre-place checkpoint
    (place, 3, gripper_open)  -> from its end: placed-and-released = 1

Output: writes a /keystate group back into each episode hdf5 (idempotent) plus a
JSON sidecar at <data_root>/<task>/<config>/keystate/episode{N}.json.

Usage:
    python -m envs.utils.keystate_labeler --task place_a2b_left --config demo_clean --all
    python -m envs.utils.keystate_labeler --task place_a2b_left --config demo_clean --episode 0
"""
import argparse
import json
import os

import h5py
import numpy as np

LABELER_VERSION = 2  # v2: add dense next_checkpoint_type (Stage 1 supervision target)

# Inverse of Base_Task.KEYSTATE_*_CODES (kept in sync with envs/_base_task.py).
STAGE_DECODE = {0: None, 1: "grasp", 2: "lift", 3: "place"}
ACTION_DECODE = {0: None, 1: "move", 2: "gripper_close", 3: "gripper_open"}
ARM_DECODE = {0: None, 1: "left", 2: "right"}

DT = 15.0 / 250.0  # save_freq / physics_freq ~ 0.06s effective
LIFT_THRESHOLD = 0.03  # m, cross-check for `lifted` against object z


def _read_episode(hdf5_path):
    """Read the fields the labeler needs. Returns a dict, or raises with a clear msg."""
    with h5py.File(hdf5_path, "r") as f:
        if "keystate_substep" not in f:
            raise KeyError(
                f"{hdf5_path} has no /keystate_substep group. It was collected before the "
                f"embedding hook was added -- re-collect this episode.")
        ss = f["keystate_substep"]
        data = {
            "stage_code": ss["stage_code"][()].astype(np.int64),
            "sub_index": ss["sub_index"][()].astype(np.int64),
            "action_code": ss["action_code"][()].astype(np.int64),
            "arm_code": ss["arm_code"][()].astype(np.int64),
            "left_endpose": f["/endpose/left_endpose"][()],
            "right_endpose": f["/endpose/right_endpose"][()],
            "left_gripper": np.atleast_1d(f["/endpose/left_gripper"][()]).astype(np.float64),
            "right_gripper": np.atleast_1d(f["/endpose/right_gripper"][()]).astype(np.float64),
        }
        if "object_pose" in f:
            data["object_pose"] = f["/object_pose/object"][()]
            data["target_object_pose"] = f["/object_pose/target_object"][()]
        else:
            data["object_pose"] = None
            data["target_object_pose"] = None
    return data


def _segment_end_frame(stage_codes, action_codes, sub_indices, stage_name, sub_index, action_name):
    """Last frame index whose substep tag matches (stage, sub_index, action). -1 if none.

    A single Action spans several saved frames; we want the LAST frame of that span,
    i.e. the moment that Action finished.
    """
    stage_c = next(k for k, v in STAGE_DECODE.items() if v == stage_name)
    action_c = next(k for k, v in ACTION_DECODE.items() if v == action_name)
    mask = (stage_codes == stage_c) & (action_codes == action_c) & (sub_indices == sub_index)
    idx = np.nonzero(mask)[0]
    return int(idx[-1]) if idx.size else -1


def _segment_first_frame(stage_codes, action_codes, sub_indices, stage_name, sub_index, action_name):
    stage_c = next(k for k, v in STAGE_DECODE.items() if v == stage_name)
    action_c = next(k for k, v in ACTION_DECODE.items() if v == action_name)
    mask = (stage_codes == stage_c) & (action_codes == action_c) & (sub_indices == sub_index)
    idx = np.nonzero(mask)[0]
    return int(idx[0]) if idx.size else -1


def label_episode(data):
    """Apply the fixed mapping table. Returns (labels_dict, flags_list)."""
    sc, ac, si = data["stage_code"], data["action_code"], data["sub_index"]
    T = sc.shape[0]
    flags = []

    # active arm: from the substep arm_code (the arm that actually executed tagged actions)
    arm_codes_active = data["arm_code"][data["arm_code"] != 0]
    if arm_codes_active.size:
        arm_code = int(np.bincount(arm_codes_active).argmax())
        arm = ARM_DECODE[arm_code]
    else:
        arm = None
        flags.append("no_arm_in_substep")

    # checkpoints (single frames) = end of the first move in grasp / place
    pre_grasp_idx = _segment_end_frame(sc, ac, si, "grasp", 1, "move")
    pre_place_idx = _segment_end_frame(sc, ac, si, "place", 1, "move")
    if pre_grasp_idx < 0:
        flags.append("no_pre_grasp")
    if pre_place_idx < 0:
        flags.append("no_pre_place")

    # phase boundaries (multi-label, 1 from a frame onward)
    close_end = _segment_end_frame(sc, ac, si, "grasp", 3, "gripper_close")
    lift_end = _segment_end_frame(sc, ac, si, "lift", 1, "move")
    open_end = _segment_end_frame(sc, ac, si, "place", 3, "gripper_open")
    if close_end < 0:
        flags.append("no_grasp_close")
    if lift_end < 0:
        flags.append("no_lift")
    if open_end < 0:
        flags.append("no_place_open")

    object_in_hand = np.zeros(T, dtype=np.uint8)
    lifted = np.zeros(T, dtype=np.uint8)
    placed_and_released = np.zeros(T, dtype=np.uint8)

    # object-in-hand: from end of grasp-close until end of place-open (release)
    if close_end >= 0:
        end = open_end if open_end >= 0 else T - 1
        object_in_hand[close_end:end + 1] = 1
    # lifted: from end of lift move onward (until release)
    if lift_end >= 0:
        end = open_end if open_end >= 0 else T - 1
        lifted[lift_end:end + 1] = 1
    # placed-and-released: from end of place-open onward
    if open_end >= 0:
        placed_and_released[open_end:] = 1

    # --- cross-check `lifted` against object z, if object_pose available ---
    if data["object_pose"] is not None and T >= 1:
        obj_z = data["object_pose"][:, 2]
        resting_z = float(np.median(obj_z[:min(3, T)]))
        z_lifted = (obj_z - resting_z) > LIFT_THRESHOLD
        # if the script says lifted but z never rises (or vice versa), flag for spot-check
        if lift_end >= 0 and not bool(z_lifted[lift_end:open_end if open_end >= 0 else T].any()):
            flags.append("lifted_z_mismatch")
    else:
        resting_z = float("nan")

    # --- derived supervision fields ---
    checkpoint_type = np.zeros(T, dtype=np.int8)
    if pre_grasp_idx >= 0:
        checkpoint_type[pre_grasp_idx] = 1
    if pre_place_idx >= 0:
        checkpoint_type[pre_place_idx] = 2

    ckpts = sorted([c for c in (pre_grasp_idx, pre_place_idx) if c >= 0])
    h_ckpt = np.full(T, -1, dtype=np.int32)
    # dense next-checkpoint type: type of the nearest *future* checkpoint at each t.
    # This is what the model is supervised on (sparse `checkpoint_type` above is only the
    # 2 event frames; the model needs "looking forward from t, what's the next checkpoint").
    # Strictly co-derived with h_ckpt so the two are always consistent (same `nxt`).
    # 0 on frames past the last checkpoint (h_ckpt == -1); those are masked out in training.
    next_checkpoint_type = np.zeros(T, dtype=np.int8)
    for t in range(T):
        nxt = next((c for c in ckpts if c >= t), None)
        if nxt is not None:
            h_ckpt[t] = nxt - t
            next_checkpoint_type[t] = checkpoint_type[nxt]

    semantic_phase = np.stack([object_in_hand, lifted, placed_and_released], axis=1).astype(np.uint8)

    # ordering sanity (non-fatal)
    if pre_grasp_idx >= 0 and pre_place_idx >= 0 and not (pre_grasp_idx < pre_place_idx):
        flags.append("checkpoint_order_violation")

    labels = {
        "checkpoint_type": checkpoint_type,
        "next_checkpoint_type": next_checkpoint_type,
        "h_ckpt": h_ckpt,
        "object_in_hand": object_in_hand,
        "lifted": lifted,
        "placed_and_released": placed_and_released,
        "semantic_phase": semantic_phase,
        "_attrs": {
            "arm": arm if arm is not None else "unknown",
            "pre_grasp_idx": pre_grasp_idx,
            "pre_place_idx": pre_place_idx,
            "grasp_close_end": close_end,
            "lift_end": lift_end,
            "place_open_end": open_end,
            "resting_z": resting_z,
            "lift_threshold": LIFT_THRESHOLD,
            "T": int(T),
            "labeler_version": LABELER_VERSION,
            "flags": flags,
        },
    }
    return labels, flags


def _write_back(hdf5_path, labels):
    with h5py.File(hdf5_path, "a") as f:
        if "keystate" in f:
            del f["keystate"]
        grp = f.create_group("keystate")
        for k, v in labels.items():
            if k == "_attrs":
                continue
            grp.create_dataset(k, data=v)
        for ak, av in labels["_attrs"].items():
            if isinstance(av, list):
                grp.attrs[ak] = json.dumps(av)
            else:
                grp.attrs[ak] = av


def _write_sidecar(sidecar_path, ep_idx, labels):
    os.makedirs(os.path.dirname(sidecar_path), exist_ok=True)
    out = dict(labels["_attrs"])
    out["episode"] = ep_idx
    with open(sidecar_path, "w") as f:
        json.dump(out, f, indent=2)


def process(task, config, data_root, episodes):
    base = os.path.join(data_root, task, config)
    data_dir = os.path.join(base, "data")
    sidecar_dir = os.path.join(base, "keystate")

    summary = {"ok": 0, "flagged": 0, "errors": 0, "flagged_episodes": {}}
    for ep in episodes:
        hdf5_path = os.path.join(data_dir, f"episode{ep}.hdf5")
        if not os.path.isfile(hdf5_path):
            print(f"[skip] episode{ep}: missing {hdf5_path}")
            summary["errors"] += 1
            continue
        try:
            data = _read_episode(hdf5_path)
            labels, flags = label_episode(data)
            _write_back(hdf5_path, labels)
            _write_sidecar(os.path.join(sidecar_dir, f"episode{ep}.json"), ep, labels)
            a = labels["_attrs"]
            note = f" FLAGS={flags}" if flags else ""
            print(f"[ok]   episode{ep}: arm={a['arm']} pre_grasp={a['pre_grasp_idx']} "
                  f"pre_place={a['pre_place_idx']} T={a['T']}{note}")
            summary["ok"] += 1
            if flags:
                summary["flagged"] += 1
                summary["flagged_episodes"][ep] = flags
        except Exception as e:
            print(f"[err]  episode{ep}: {e}")
            summary["errors"] += 1

    print("\n===== SUMMARY =====")
    print(f"ok={summary['ok']} flagged={summary['flagged']} errors={summary['errors']}")
    if summary["flagged_episodes"]:
        print("flagged episodes:")
        for ep, fl in sorted(summary["flagged_episodes"].items()):
            print(f"  episode{ep}: {fl}")
    return summary


def main():
    p = argparse.ArgumentParser(description="KeyState Stage 0 offline labeler")
    p.add_argument("--task", default="place_a2b_left")
    p.add_argument("--config", default="demo_clean")
    p.add_argument("--data-root", default="data")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="label every episode found")
    g.add_argument("--episode", type=int, help="label a single episode index")
    args = p.parse_args()

    if args.all:
        data_dir = os.path.join(args.data_root, args.task, args.config, "data")
        eps = sorted(
            int(fn[len("episode"):-len(".hdf5")])
            for fn in os.listdir(data_dir)
            if fn.startswith("episode") and fn.endswith(".hdf5"))
    else:
        eps = [args.episode]

    process(args.task, args.config, args.data_root, eps)


if __name__ == "__main__":
    main()
