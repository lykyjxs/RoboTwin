"""
KeyState Stage 0 offline labeler.

Reads the per-frame substep tags that were passively logged during data collection
(see envs/_base_task.py: move(stage_tag=...) -> get_obs() -> /keystate_substep/*),
and derives checkpoint-window labels by looking up a FIXED semantic mapping table --
no human/VLM observation, no thresholds for the checkpoint windows.

Canonical Stage 1 supervision fields:
    next_checkpoint_type  int8  (T,)    0=none / 1=pre_grasp_window / 2=pre_place_window
    h_entry               int32 (T,)    distance to checkpoint-window entry; 0 inside window; -1=no next/current window
    semantic_phase        uint8 (T,3)   object_in_hand / lifted / placed_and_released

Supported task mappings:
    place_a2b_left:       single pick/place cycle (backward-compatible with v3 labels)
    stack_bowls_three:    three repeated pick/place cycles, sharing the same dense fields

Usage:
    python envs/utils/keystate_labeler.py --task place_a2b_left --config demo_clean --all
    python envs/utils/keystate_labeler.py --task stack_bowls_three --config demo_clean --episode 0
"""
import argparse
import json
import os

import h5py
import numpy as np

LABELER_VERSION = 4  # v4: task dispatch + multi-cycle windows; v3 fields remain unchanged

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
            "object_poses": {},
        }
        if "object_pose" in f:
            for name, ds in f["object_pose"].items():
                data["object_poses"][name] = ds[()]
            data["object_pose"] = data["object_poses"].get("object")
            data["target_object_pose"] = data["object_poses"].get("target_object")
        else:
            data["object_pose"] = None
            data["target_object_pose"] = None
    return data


def _code_for(mapping, name):
    return next(k for k, v in mapping.items() if v == name)


def _segment_end_frame(stage_codes, action_codes, sub_indices, stage_name, sub_index, action_name):
    """Last frame index whose substep tag matches (stage, sub_index, action). -1 if none."""
    idx = _segment_end_frames(stage_codes, action_codes, sub_indices, stage_name, sub_index, action_name)
    return int(idx[-1]) if idx.size else -1


def _segment_start_frames(stage_codes, action_codes, sub_indices, stage_name, sub_index, action_name):
    """Start frame of every contiguous span matching (stage, sub_index, action)."""
    stage_c = _code_for(STAGE_DECODE, stage_name)
    action_c = _code_for(ACTION_DECODE, action_name)
    mask = (stage_codes == stage_c) & (action_codes == action_c) & (sub_indices == sub_index)
    idx = np.nonzero(mask)[0]
    if not idx.size:
        return np.array([], dtype=np.int64)

    breaks = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.concatenate([idx[:1], idx[breaks + 1]])
    return starts.astype(np.int64)


def _segment_end_frames(stage_codes, action_codes, sub_indices, stage_name, sub_index, action_name):
    """End frame of every contiguous span matching (stage, sub_index, action)."""
    stage_c = _code_for(STAGE_DECODE, stage_name)
    action_c = _code_for(ACTION_DECODE, action_name)
    mask = (stage_codes == stage_c) & (action_codes == action_c) & (sub_indices == sub_index)
    idx = np.nonzero(mask)[0]
    if not idx.size:
        return np.array([], dtype=np.int64)

    breaks = np.nonzero(np.diff(idx) > 1)[0]
    ends = np.concatenate([idx[breaks], idx[-1:]])
    return ends.astype(np.int64)


def _active_arm(data):
    arm_codes_active = data["arm_code"][data["arm_code"] != 0]
    if arm_codes_active.size:
        arm_code = int(np.bincount(arm_codes_active).argmax())
        return ARM_DECODE[arm_code], []
    return None, ["no_arm_in_substep"]


def _window_json(windows):
    return [{
        "cycle": int(w.get("cycle", -1)),
        "name": str(w["name"]),
        "type": int(w["type"]),
        "start": int(w["start"]),
        "entry": int(w.get("entry", w["start"])),
        "end": int(w["end"]),
        "actor": str(w.get("actor", "")),
    } for w in windows]


def build_labels_from_ordered_windows(T, windows, object_in_hand, lifted, placed_and_released, flags):
    """Build dense current-or-next checkpoint labels from time-ordered windows."""
    next_checkpoint_type = np.zeros(T, dtype=np.int8)
    h_entry = np.full(T, -1, dtype=np.int32)
    cycle_id = np.full(T, -1, dtype=np.int16)
    window_id = np.full(T, -1, dtype=np.int16)

    valid_windows = []
    for i, w in enumerate(windows):
        start = int(w["start"])
        end = int(w["end"])
        if start < 0 or end < 0:
            flags.append(f"{w['name']}_cycle{w.get('cycle', -1)}_missing_boundary")
            continue
        if end < start:
            flags.append(f"{w['name']}_cycle{w.get('cycle', -1)}_end_before_start")
            continue
        if start >= T:
            flags.append(f"{w['name']}_cycle{w.get('cycle', -1)}_start_oob")
            continue
        if end >= T:
            flags.append(f"{w['name']}_cycle{w.get('cycle', -1)}_end_oob")
            end = T - 1
            w = dict(w, end=end)
        valid_windows.append((i, w))

    valid_windows.sort(key=lambda item: int(item[1]["start"]))
    for (prev_i, prev), (cur_i, cur) in zip(valid_windows, valid_windows[1:]):
        if int(prev["end"]) >= int(cur["start"]):
            flags.append(f"window_overlap_{prev['name']}{prev.get('cycle', -1)}_{cur['name']}{cur.get('cycle', -1)}")
        if int(prev["start"]) >= int(cur["start"]):
            flags.append("window_order_violation")

    for ordinal, (original_id, w) in enumerate(valid_windows):
        start = int(w["start"])
        end = int(w["end"])
        prev_end = int(valid_windows[ordinal - 1][1]["end"]) if ordinal > 0 else -1
        pre_start = max(prev_end + 1, 0)
        if pre_start < start:
            pre = np.arange(pre_start, start)
            next_checkpoint_type[pre] = int(w["type"])
            h_entry[pre] = start - pre
            cycle_id[pre] = int(w.get("cycle", -1))
            window_id[pre] = original_id

        win = np.arange(start, end + 1)
        next_checkpoint_type[win] = int(w["type"])
        h_entry[win] = 0
        cycle_id[win] = int(w.get("cycle", -1))
        window_id[win] = original_id

    semantic_phase = np.stack([object_in_hand, lifted, placed_and_released], axis=1).astype(np.uint8)
    return {
        "next_checkpoint_type": next_checkpoint_type,
        "h_entry": h_entry,
        "object_in_hand": object_in_hand,
        "lifted": lifted,
        "placed_and_released": placed_and_released,
        "semantic_phase": semantic_phase,
        "cycle_id": cycle_id,
        "window_id": window_id,
    }


def label_single_cycle_pick_place(data, task=None):
    """Backward-compatible single pick/place mapping used by place_a2b_left."""
    sc, ac, si = data["stage_code"], data["action_code"], data["sub_index"]
    T = sc.shape[0]
    flags = []
    arm, arm_flags = _active_arm(data)
    flags.extend(arm_flags)

    pre_grasp_idx = _segment_end_frame(sc, ac, si, "grasp", 1, "move")
    pre_place_idx = _segment_end_frame(sc, ac, si, "place", 1, "move")
    close_end = _segment_end_frame(sc, ac, si, "grasp", 3, "gripper_close")
    lift_end = _segment_end_frame(sc, ac, si, "lift", 1, "move")
    open_end = _segment_end_frame(sc, ac, si, "place", 3, "gripper_open")

    if pre_grasp_idx < 0:
        flags.append("no_pre_grasp")
    if pre_place_idx < 0:
        flags.append("no_pre_place")
    if close_end < 0:
        flags.append("no_grasp_close")
    if lift_end < 0:
        flags.append("no_lift")
    if open_end < 0:
        flags.append("no_place_open")

    object_in_hand = np.zeros(T, dtype=np.uint8)
    lifted = np.zeros(T, dtype=np.uint8)
    placed_and_released = np.zeros(T, dtype=np.uint8)

    if close_end >= 0:
        end = open_end if open_end >= 0 else T - 1
        object_in_hand[close_end:end + 1] = 1
    if lift_end >= 0:
        end = open_end if open_end >= 0 else T - 1
        lifted[lift_end:end + 1] = 1
    if open_end >= 0:
        placed_and_released[open_end:] = 1

    resting_z = float("nan")
    if data["object_pose"] is not None and T >= 1:
        obj_z = data["object_pose"][:, 2]
        resting_z = float(np.median(obj_z[:min(3, T)]))
        z_lifted = (obj_z - resting_z) > LIFT_THRESHOLD
        if lift_end >= 0 and not bool(z_lifted[lift_end:open_end if open_end >= 0 else T].any()):
            flags.append("lifted_z_mismatch")

    pre_grasp_window_start = pre_grasp_idx
    pre_grasp_window_end = close_end
    pre_place_window_start = pre_place_idx
    pre_place_window_end = open_end
    if pre_grasp_window_start >= 0 and pre_grasp_window_end >= 0 and pre_place_window_start >= 0:
        if pre_grasp_window_end >= pre_place_window_start:
            flags.append("pre_grasp_window_overlaps_pre_place")
            pre_grasp_window_end = pre_place_window_start - 1

    windows = [
        {"name": "pre_grasp", "type": 1, "cycle": 0, "start": pre_grasp_window_start, "entry": pre_grasp_window_start, "end": pre_grasp_window_end, "actor": "object"},
        {"name": "pre_place", "type": 2, "cycle": 0, "start": pre_place_window_start, "entry": pre_place_window_start, "end": pre_place_window_end, "actor": "object"},
    ]
    labels = build_labels_from_ordered_windows(T, windows, object_in_hand, lifted, placed_and_released, flags)

    if pre_grasp_idx >= 0 and pre_place_idx >= 0 and not (pre_grasp_idx < pre_place_idx):
        flags.append("checkpoint_order_violation")

    labels["_attrs"] = {
        "task": task or "single_cycle_pick_place",
        "arm": arm if arm is not None else "unknown",
        "pre_grasp_idx": pre_grasp_idx,
        "pre_grasp_window_start": pre_grasp_window_start,
        "pre_grasp_window_end": pre_grasp_window_end,
        "pre_place_idx": pre_place_idx,
        "pre_place_window_start": pre_place_window_start,
        "pre_place_window_end": pre_place_window_end,
        "grasp_close_end": close_end,
        "lift_end": lift_end,
        "place_open_end": open_end,
        "num_cycles": 1,
        "checkpoint_windows_json": _window_json(windows),
        "resting_z": resting_z,
        "lift_threshold": LIFT_THRESHOLD,
        "T": int(T),
        "labeler_version": LABELER_VERSION,
        "flags": flags,
    }
    return labels, flags


def label_multi_cycle_stack_bowls(data):
    """Task-specific mapping for stack_bowls_three's three repeated bowl transfers."""
    sc, ac, si = data["stage_code"], data["action_code"], data["sub_index"]
    T = sc.shape[0]
    flags = []
    arm, arm_flags = _active_arm(data)
    flags.extend(arm_flags)

    pre_grasp = _segment_end_frames(sc, ac, si, "grasp", 1, "move")
    close_end = _segment_end_frames(sc, ac, si, "grasp", 3, "gripper_close")
    lift_start = _segment_start_frames(sc, ac, si, "lift", 1, "move")
    lift_end = _segment_end_frames(sc, ac, si, "lift", 1, "move")
    pre_place = _segment_end_frames(sc, ac, si, "place", 1, "move")
    open_start = _segment_start_frames(sc, ac, si, "place", 3, "gripper_open")
    open_end = _segment_end_frames(sc, ac, si, "place", 3, "gripper_open")

    expected = 3
    counts = {
        "pre_grasp": len(pre_grasp),
        "grasp_close": len(close_end),
        "lift": len(lift_end),
        "lift_start": len(lift_start),
        "pre_place": len(pre_place),
        "place_open": len(open_end),
        "place_open_start": len(open_start),
    }
    for name, count in counts.items():
        if count != expected:
            flags.append(f"expected_3_{name}_got_{count}")

    cycles = min(expected, len(pre_grasp), len(close_end), len(lift_start), len(lift_end), len(pre_place), len(open_start), len(open_end))
    actors = ["bowl1", "bowl2", "bowl3"]
    windows = []
    for cycle in range(cycles):
        windows.append({
            "name": "pre_grasp",
            "type": 1,
            "cycle": cycle,
            "start": int(pre_grasp[cycle]),
            "entry": int(pre_grasp[cycle]),
            "end": int(close_end[cycle]),
            "actor": actors[cycle],
        })
        windows.append({
            "name": "pre_place",
            "type": 2,
            "cycle": cycle,
            "start": int(pre_place[cycle]),
            "entry": int(pre_place[cycle]),
            "end": int(open_end[cycle]),
            "actor": actors[cycle],
        })

    expected_names = ["pre_grasp", "pre_place", "pre_grasp", "pre_place", "pre_grasp", "pre_place"]
    if len(windows) == 6:
        starts = [int(w["start"]) for w in windows]
        names = [w["name"] for w in windows]
        if names != expected_names or any(a >= b for a, b in zip(starts, starts[1:])):
            flags.append("stack_bowls_window_order_violation")
    else:
        flags.append(f"stack_bowls_incomplete_windows_{len(windows)}")

    object_in_hand = np.zeros(T, dtype=np.uint8)
    lifted = np.zeros(T, dtype=np.uint8)
    placed_and_released = np.zeros(T, dtype=np.uint8)
    for cycle in range(cycles):
        release_start = int(open_start[cycle])
        release_end = int(open_end[cycle])
        next_pre_grasp = int(pre_grasp[cycle + 1]) if cycle + 1 < cycles else T
        if close_end[cycle] >= 0:
            object_in_hand[int(close_end[cycle]):release_end + 1] = 1
        if lift_start[cycle] >= 0:
            lifted[int(lift_start[cycle]):release_end + 1] = 1
        placed_and_released[release_start:next_pre_grasp] = 1

    resting_z = float("nan")
    missing_poses = [name for name in actors if name not in data["object_poses"]]
    if missing_poses:
        flags.append("missing_object_pose_" + ",".join(missing_poses))
    else:
        rests = []
        for cycle, actor in enumerate(actors[:cycles]):
            obj_z = data["object_poses"][actor][:, 2]
            rest = float(np.median(obj_z[:min(3, T)]))
            rests.append(rest)
            if cycle < len(lift_end) and cycle < len(open_end):
                z_lifted = (obj_z - rest) > LIFT_THRESHOLD
                if not bool(z_lifted[int(lift_end[cycle]):int(open_end[cycle])].any()):
                    flags.append(f"{actor}_lifted_z_mismatch")
        if rests:
            resting_z = float(np.mean(rests))

    labels = build_labels_from_ordered_windows(T, windows, object_in_hand, lifted, placed_and_released, flags)
    labels["_attrs"] = {
        "task": "stack_bowls_three",
        "arm": arm if arm is not None else "unknown",
        "num_cycles": int(cycles),
        "expected_cycles": expected,
        "checkpoint_windows_json": _window_json(windows),
        "stack_bowls_counts": counts,
        "pre_grasp_indices": [int(x) for x in pre_grasp],
        "grasp_close_ends": [int(x) for x in close_end],
        "lift_starts": [int(x) for x in lift_start],
        "lift_ends": [int(x) for x in lift_end],
        "pre_place_indices": [int(x) for x in pre_place],
        "place_open_starts": [int(x) for x in open_start],
        "place_open_ends": [int(x) for x in open_end],
        "resting_z": resting_z,
        "lift_threshold": LIFT_THRESHOLD,
        "T": int(T),
        "labeler_version": LABELER_VERSION,
        "flags": flags,
    }
    return labels, flags


def label_episode(data, task="place_a2b_left"):
    if task == "stack_bowls_three":
        return label_multi_cycle_stack_bowls(data)
    if task == "place_a2b_left":
        return label_single_cycle_pick_place(data, task=task)
    raise ValueError(
        f"No KeyState Stage0 mapping registered for task={task!r}. "
        "Add a task-specific labeler instead of silently reusing the wrong mapping.")


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
            if isinstance(av, (list, dict)):
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
            labels, flags = label_episode(data, task=task)
            _write_back(hdf5_path, labels)
            _write_sidecar(os.path.join(sidecar_dir, f"episode{ep}.json"), ep, labels)
            a = labels["_attrs"]
            note = f" FLAGS={flags}" if flags else ""
            if task == "stack_bowls_three":
                windows = a.get("checkpoint_windows_json", [])
                win_s = ", ".join(f"c{w['cycle']}:{w['name']}=[{w['start']},{w['end']}]" for w in windows)
                print(f"[ok]   episode{ep}: cycles={a['num_cycles']} {win_s} T={a['T']}{note}")
            else:
                print(f"[ok]   episode{ep}: arm={a['arm']} pre_grasp=[{a['pre_grasp_window_start']},"
                      f"{a['pre_grasp_window_end']}] pre_place=[{a['pre_place_window_start']},"
                      f"{a['pre_place_window_end']}] T={a['T']}{note}")
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
