"""
KeyState Stage 0 self-inspection tool (read-only).

Checks the checkpoint-window labels produced by keystate_labeler.py. The original
single-cycle checks are kept for place_a2b_left; stack_bowls_three additionally
checks the three grasp/place cycles encoded in checkpoint_windows_json.

Usage:
    python envs/utils/keystate_inspect.py --task place_a2b_left --config demo_clean
    python envs/utils/keystate_inspect.py --task stack_bowls_three --config demo_clean --episode 0
"""
import argparse
import json
import os

import h5py
import numpy as np


TYPE_NAMES = {0: "none", 1: "pre_grasp", 2: "pre_place"}


def _maybe_json(v, default=None):
    if default is None:
        default = []
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return default
    return v if v is not None else default


def load_keystate(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        if "keystate" not in f:
            raise KeyError(f"{hdf5_path} has no /keystate group (run keystate_labeler first)")
        ks = f["keystate"]
        required = ["next_checkpoint_type", "h_entry", "semantic_phase"]
        missing = [k for k in required if k not in ks]
        if missing:
            raise KeyError(f"{hdf5_path} /keystate missing v3 fields: {missing}")
        a = dict(ks.attrs)
        sem = ks["semantic_phase"][()]
        d = {
            "next_checkpoint_type": ks["next_checkpoint_type"][()],
            "h_entry": ks["h_entry"][()],
            "object_in_hand": ks["object_in_hand"][()] if "object_in_hand" in ks else sem[:, 0],
            "lifted": ks["lifted"][()] if "lifted" in ks else sem[:, 1],
            "placed_and_released": ks["placed_and_released"][()] if "placed_and_released" in ks else sem[:, 2],
            "semantic_phase": sem,
            "cycle_id": ks["cycle_id"][()] if "cycle_id" in ks else None,
            "window_id": ks["window_id"][()] if "window_id" in ks else None,
            "attrs": a,
            "windows": _maybe_json(a.get("checkpoint_windows_json", "[]"), []),
        }
        d["T_endpose"] = f["/endpose/left_endpose"].shape[0]
        d["T_rgb"] = f["/observation/head_camera/rgb"].shape[0] if "observation" in f else None
        d["object_poses"] = {}
        if "object_pose" in f:
            for name, ds in f["object_pose"].items():
                d["object_poses"][name] = ds[()]
        d["obj_z"] = d["object_poses"].get("object", None)
        if d["obj_z"] is not None:
            d["obj_z"] = d["obj_z"][:, 2]
    return d


def _attr_int(attrs, key):
    return int(attrs[key]) if key in attrs else -1


def _check_common(d, problems):
    lengths = {
        "next_checkpoint_type": len(d["next_checkpoint_type"]),
        "h_entry": len(d["h_entry"]),
        "object_in_hand": len(d["object_in_hand"]),
        "lifted": len(d["lifted"]),
        "placed_and_released": len(d["placed_and_released"]),
        "semantic_phase": len(d["semantic_phase"]),
        "endpose": d["T_endpose"],
    }
    if d["T_rgb"] is not None:
        lengths["rgb"] = d["T_rgb"]
    if d["cycle_id"] is not None:
        lengths["cycle_id"] = len(d["cycle_id"])
    if d["window_id"] is not None:
        lengths["window_id"] = len(d["window_id"])
    if len(set(lengths.values())) != 1:
        problems.append(f"FRAME MISALIGN: {lengths}")

    if d["semantic_phase"].ndim != 2 or d["semantic_phase"].shape[1] != 3:
        problems.append(f"semantic_phase shape is {d['semantic_phase'].shape}, expected [T,3]")

    nct = d["next_checkpoint_type"]
    h = d["h_entry"]
    if np.any(~np.isin(nct, [0, 1, 2])):
        problems.append("next_checkpoint_type contains values outside {0,1,2}")
    if np.any((nct == 0) & (h >= 0)):
        problems.append("type=0 frames with non-negative h_entry")
    if np.any((nct > 0) & (h < 0)):
        problems.append("type>0 frames with negative h_entry")


def _check_windows_dense(d, windows, problems):
    nct = d["next_checkpoint_type"]
    h = d["h_entry"]
    T = len(nct)
    valid = [w for w in windows if int(w.get("start", -1)) >= 0 and int(w.get("end", -1)) >= 0]
    valid = sorted(valid, key=lambda w: int(w["start"]))

    for prev, cur in zip(valid, valid[1:]):
        if int(prev["end"]) >= int(cur["start"]):
            problems.append(f"window overlap: {prev} -> {cur}")

    for i, w in enumerate(valid):
        start = int(w["start"])
        end = int(w["end"])
        typ = int(w["type"])
        if start > end:
            problems.append(f"window start>end: {w}")
            continue
        if end >= T:
            problems.append(f"window out of range T={T}: {w}")
            continue
        win = np.arange(start, end + 1)
        if np.any(nct[win] != typ) or np.any(h[win] != 0):
            problems.append(f"{w['name']} cycle{w.get('cycle', -1)} window not labeled type={typ},h_entry=0 throughout")

        prev_end = int(valid[i - 1]["end"]) if i > 0 else -1
        pre = np.arange(prev_end + 1, start)
        if pre.size and (np.any(nct[pre] != typ) or np.any(h[pre] != (start - pre))):
            problems.append(f"frames before {w['name']} cycle{w.get('cycle', -1)} do not point to its entry")

    if valid:
        tail_start = int(valid[-1]["end"]) + 1
        if tail_start < T:
            tail = np.arange(tail_start, T)
            if np.any(nct[tail] != 0) or np.any(h[tail] != -1):
                problems.append("terminal frames after final checkpoint window are not type=0,h_entry=-1")


def _check_single_cycle(ep, d, scene_info):
    a = d["attrs"]
    problems = []
    _check_common(d, problems)

    T = int(a["T"])
    pg = _attr_int(a, "pre_grasp_idx")
    pp = _attr_int(a, "pre_place_idx")
    pg_s = _attr_int(a, "pre_grasp_window_start")
    pg_e = _attr_int(a, "pre_grasp_window_end")
    pp_s = _attr_int(a, "pre_place_window_start")
    pp_e = _attr_int(a, "pre_place_window_end")
    flags = _maybe_json(a.get("flags", "[]"), [])

    if pg >= 0 and pp >= 0 and not (pg < pp):
        problems.append(f"pre_grasp({pg}) >= pre_place({pp})")
    if pg_s >= 0 and pg_e >= 0 and not (pg_s <= pg_e):
        problems.append(f"pre_grasp_window start({pg_s}) > end({pg_e})")
    if pp_s >= 0 and pp_e >= 0 and not (pp_s <= pp_e):
        problems.append(f"pre_place_window start({pp_s}) > end({pp_e})")
    if pg_e >= 0 and pp_s >= 0 and not (pg_e < pp_s):
        problems.append(f"pre_grasp_window overlaps pre_place_window ({pg_e} >= {pp_s})")

    oih = np.nonzero(d["object_in_hand"])[0]
    lif = np.nonzero(d["lifted"])[0]
    rel = np.nonzero(d["placed_and_released"])[0]
    oih_start = int(oih.min()) if oih.size else -1
    lif_start = int(lif.min()) if lif.size else -1
    rel_start = int(rel.min()) if rel.size else -1
    if pg_s >= 0 and oih_start >= 0 and not (pg_s <= oih_start):
        problems.append(f"pre_grasp_window_start({pg_s}) after in-hand start({oih_start})")
    if pp_s >= 0 and rel_start >= 0 and not (pp_s <= rel_start):
        problems.append(f"pre_place_window_start({pp_s}) after released start({rel_start})")
    if oih.size and lif.size:
        overlap = bool((d["object_in_hand"].astype(bool) & d["lifted"].astype(bool)).any())
        if not overlap:
            problems.append("lifted does not overlap object_in_hand")

    windows = d["windows"] or [
        {"name": "pre_grasp", "type": 1, "cycle": 0, "start": pg_s, "end": pg_e},
        {"name": "pre_place", "type": 2, "cycle": 0, "start": pp_s, "end": pp_e},
    ]
    _check_windows_dense(d, windows, problems)

    z_note = ""
    if d["obj_z"] is not None:
        rise = float(d["obj_z"].max() - a.get("resting_z", d["obj_z"].min()))
        z_note = f"z_rise={rise:.3f}"
        if lif.size and rise < a.get("lift_threshold", 0.03):
            problems.append(f"lifted set but object z barely rose ({rise:.3f})")

    arm = a.get("arm", "?")
    scene_arm = scene_info.get(f"episode_{ep}", {}).get("info", {}).get("{a}")
    arm_match = "" if (scene_arm is None or scene_arm == arm) else f"ARM MISMATCH(scene={scene_arm})"
    if arm_match:
        problems.append(arm_match)

    return {
        "ep": ep,
        "arm": arm,
        "T": T,
        "pg_s": pg_s,
        "pg_e": pg_e,
        "pp_s": pp_s,
        "pp_e": pp_e,
        "oih": oih_start,
        "lif": lif_start,
        "rel": rel_start,
        "flags": flags,
        "z_note": z_note,
        "problems": problems,
    }


def _check_stack_bowls(ep, d):
    a = d["attrs"]
    problems = []
    _check_common(d, problems)
    windows = d["windows"]
    flags = _maybe_json(a.get("flags", "[]"), [])

    for name in ["bowl1", "bowl2", "bowl3"]:
        if name not in d["object_poses"]:
            problems.append(f"missing /object_pose/{name}")

    if len(windows) != 6:
        problems.append(f"checkpoint_windows_json has {len(windows)} windows, expected 6")
    else:
        names = [w.get("name") for w in windows]
        cycles = [int(w.get("cycle", -1)) for w in windows]
        expected_names = ["pre_grasp", "pre_place", "pre_grasp", "pre_place", "pre_grasp", "pre_place"]
        expected_cycles = [0, 0, 1, 1, 2, 2]
        if names != expected_names or cycles != expected_cycles:
            problems.append(f"unexpected window sequence names={names} cycles={cycles}")
        starts = [int(w["start"]) for w in windows]
        if any(a0 >= b0 for a0, b0 in zip(starts, starts[1:])):
            problems.append(f"window starts are not strictly increasing: {starts}")

        pre_grasp_count = sum(1 for w in windows if w.get("name") == "pre_grasp")
        pre_place_count = sum(1 for w in windows if w.get("name") == "pre_place")
        if pre_grasp_count != 3 or pre_place_count != 3:
            problems.append(f"expected 3 grasp/3 place windows, got {pre_grasp_count}/{pre_place_count}")

    _check_windows_dense(d, windows, problems)

    nct = d["next_checkpoint_type"]
    h = d["h_entry"]
    oih = np.nonzero(d["object_in_hand"])[0]
    lif = np.nonzero(d["lifted"])[0]
    rel = np.nonzero(d["placed_and_released"])[0]
    if oih.size and lif.size and not bool((d["object_in_hand"].astype(bool) & d["lifted"].astype(bool)).any()):
        problems.append("lifted does not overlap object_in_hand")

    z_notes = []
    for name in ["bowl1", "bowl2", "bowl3"]:
        if name in d["object_poses"]:
            z = d["object_poses"][name][:, 2]
            z_notes.append(f"{name}:{float(z.max() - z.min()):.3f}")

    return {
        "ep": ep,
        "arm": a.get("arm", "?"),
        "T": int(a.get("T", len(nct))),
        "windows": windows,
        "oih": int(oih.min()) if oih.size else -1,
        "lif": int(lif.min()) if lif.size else -1,
        "rel": int(rel.min()) if rel.size else -1,
        "flags": flags,
        "z_note": ",".join(z_notes),
        "problems": problems,
        "tail_none": int(np.sum((nct == 0) & (h == -1))),
    }


def check_episode(ep, hdf5_path, scene_info, task):
    d = load_keystate(hdf5_path)
    if task == "stack_bowls_three":
        return _check_stack_bowls(ep, d)
    return _check_single_cycle(ep, d, scene_info)


def _print_single_row(r):
    status = "OK" if not r["problems"] else "!! " + "; ".join(r["problems"])
    zr = r["z_note"].replace("z_rise=", "") if r["z_note"] else "-"
    print(f"{r['ep']:>3} {r['arm']:>5} {r['T']:>4} [{r['pg_s']:>3},{r['pg_e']:<3}]      "
          f"[{r['pp_s']:>3},{r['pp_e']:<3}]      {r['oih']:>8} {r['lif']:>7} {r['rel']:>8} "
          f"{zr:>9}  {status}")


def _print_stack_row(r):
    status = "OK" if not r["problems"] else "!! " + "; ".join(r["problems"])
    win_text = " ".join(f"c{w.get('cycle')}:{w.get('name')}[{w.get('start')},{w.get('end')}]" for w in r["windows"])
    print(f"{r['ep']:>3} {r['T']:>4} {win_text} z_rise={r['z_note'] or '-'} tail_none={r['tail_none']}  {status}")


def main():
    p = argparse.ArgumentParser(description="KeyState Stage 0 self-inspection")
    p.add_argument("--task", default="place_a2b_left")
    p.add_argument("--config", default="demo_clean")
    p.add_argument("--data-root", default="data")
    p.add_argument("--episode", type=int, help="inspect a single episode verbosely")
    args = p.parse_args()

    base = os.path.join(args.data_root, args.task, args.config)
    data_dir = os.path.join(base, "data")
    scene_info_path = os.path.join(base, "scene_info.json")
    scene_info = json.load(open(scene_info_path)) if os.path.isfile(scene_info_path) else {}

    if args.episode is not None:
        eps = [args.episode]
    else:
        eps = sorted(
            int(fn[len("episode"):-len(".hdf5")])
            for fn in os.listdir(data_dir)
            if fn.startswith("episode") and fn.endswith(".hdf5"))

    if args.task == "stack_bowls_three":
        print(f"{'ep':>3} {'T':>4} windows  status")
        print("-" * 140)
    else:
        print(f"{'ep':>3} {'arm':>5} {'T':>4} {'pg_win':>13} {'pp_win':>13} "
              f"{'in-hand':>8} {'lifted':>7} {'released':>8} {'z_rise':>9}  status")
        print("-" * 105)

    bad = []
    for ep in eps:
        hdf5_path = os.path.join(data_dir, f"episode{ep}.hdf5")
        try:
            r = check_episode(ep, hdf5_path, scene_info, args.task)
        except Exception as e:
            print(f"{ep:>3}  ERROR: {e}")
            bad.append(ep)
            continue
        if args.task == "stack_bowls_three":
            _print_stack_row(r)
        else:
            _print_single_row(r)
        if r["problems"]:
            bad.append(ep)

    print("-" * (140 if args.task == "stack_bowls_three" else 105))
    print(f"Total {len(eps)} episodes. Clean: {len(eps) - len(bad)}. Problematic: {len(bad)}"
          + (f"  -> {bad}" if bad else "  ✅ all good"))

    if args.episode is not None:
        d = load_keystate(os.path.join(data_dir, f"episode{args.episode}.hdf5"))
        print(f"\n=== per-step arrays for episode{args.episode} ===")
        nct, h = d["next_checkpoint_type"], d["h_entry"]
        oih, lif, rel = d["object_in_hand"], d["lifted"], d["placed_and_released"]
        cyc = d["cycle_id"] if d["cycle_id"] is not None else np.full(len(nct), -1)
        wid = d["window_id"] if d["window_id"] is not None else np.full(len(nct), -1)
        for t in range(len(nct)):
            mark = ""
            if nct[t] in (1, 2) and h[t] == 0:
                mark = f"  <-- {TYPE_NAMES[int(nct[t])].upper()} WINDOW"
            elif nct[t] == 0:
                mark = "  <-- NONE"
            print(f"  t={t:>3}: cycle={int(cyc[t]):>2} window={int(wid[t]):>2} "
                  f"type={TYPE_NAMES.get(int(nct[t]), int(nct[t]))} h_entry={int(h[t]):>3} "
                  f"oih={int(oih[t])} lif={int(lif[t])} rel={int(rel[t])}{mark}")


if __name__ == "__main__":
    main()
