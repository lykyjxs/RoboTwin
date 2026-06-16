"""
KeyState Stage 0 self-inspection tool (read-only).

For manually checking the quality of the checkpoint-window labels produced by
keystate_labeler.py. Prints a per-episode table and a summary, and runs sanity
assertions. Also cross-checks the detected arm against scene_info.json's {a} field
(the arm the demo script actually used).

Usage:
    python envs/utils/keystate_inspect.py --task place_a2b_left --config demo_clean
    python envs/utils/keystate_inspect.py --task place_a2b_left --config demo_clean --episode 0
"""
import argparse
import json
import os

import h5py
import numpy as np


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
        d = {
            "next_checkpoint_type": ks["next_checkpoint_type"][()],
            "h_entry": ks["h_entry"][()],
            "object_in_hand": ks["object_in_hand"][()] if "object_in_hand" in ks else ks["semantic_phase"][:, 0],
            "lifted": ks["lifted"][()] if "lifted" in ks else ks["semantic_phase"][:, 1],
            "placed_and_released": ks["placed_and_released"][()]
            if "placed_and_released" in ks else ks["semantic_phase"][:, 2],
            "semantic_phase": ks["semantic_phase"][()],
            "attrs": a,
        }
        # raw signals for alignment checks
        d["T_endpose"] = f["/endpose/left_endpose"].shape[0]
        d["T_rgb"] = f["/observation/head_camera/rgb"].shape[0] if "observation" in f else None
        d["obj_z"] = f["/object_pose/object"][()][:, 2] if "object_pose" in f else None
    return d


def _attr_int(attrs, key):
    return int(attrs[key]) if key in attrs else -1


def check_episode(ep, hdf5_path, scene_info):
    d = load_keystate(hdf5_path)
    a = d["attrs"]
    T = int(a["T"])
    pg = _attr_int(a, "pre_grasp_idx")
    pp = _attr_int(a, "pre_place_idx")
    pg_s = _attr_int(a, "pre_grasp_window_start")
    pg_e = _attr_int(a, "pre_grasp_window_end")
    pp_s = _attr_int(a, "pre_place_window_start")
    pp_e = _attr_int(a, "pre_place_window_end")
    flags = a.get("flags", "[]")
    if isinstance(flags, str):
        flags = json.loads(flags)

    problems = []

    # 1. frame alignment: every per-step array must have the same length T
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
    if len(set(lengths.values())) != 1:
        problems.append(f"FRAME MISALIGN: {lengths}")

    # 2. ordering: pre_grasp < pre_place and window start/end are sane.
    if pg >= 0 and pp >= 0 and not (pg < pp):
        problems.append(f"pre_grasp({pg}) >= pre_place({pp})")
    if pg_s >= 0 and pg_e >= 0 and not (pg_s <= pg_e):
        problems.append(f"pre_grasp_window start({pg_s}) > end({pg_e})")
    if pp_s >= 0 and pp_e >= 0 and not (pp_s <= pp_e):
        problems.append(f"pre_place_window start({pp_s}) > end({pp_e})")
    if pg_e >= 0 and pp_s >= 0 and not (pg_e < pp_s):
        problems.append(f"pre_grasp_window overlaps pre_place_window ({pg_e} >= {pp_s})")

    # 3. phase starts are after or at the corresponding window entry.
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

    # 4. lifted should overlap object_in_hand (you can't lift what you're not holding)
    if oih.size and lif.size:
        overlap = bool((d["object_in_hand"].astype(bool) & d["lifted"].astype(bool)).any())
        if not overlap:
            problems.append("lifted does not overlap object_in_hand")

    nct = d["next_checkpoint_type"]
    h = d["h_entry"]

    # 5. h_entry sanity and dense type consistency.
    if pg_s >= 0:
        if h[pg_s] != 0 or nct[pg_s] != 1:
            problems.append(f"pre_grasp entry labels wrong: type={nct[pg_s]} h_entry={h[pg_s]}")
    if pg_s >= 0 and pg_e >= 0:
        window = np.arange(pg_s, pg_e + 1)
        if np.any(nct[window] != 1) or np.any(h[window] != 0):
            problems.append("pre_grasp window not labeled type=1,h_entry=0 throughout")
    if pp_s >= 0:
        if h[pp_s] != 0 or nct[pp_s] != 2:
            problems.append(f"pre_place entry labels wrong: type={nct[pp_s]} h_entry={h[pp_s]}")
    if pp_s >= 0 and pp_e >= 0:
        window = np.arange(pp_s, pp_e + 1)
        if np.any(nct[window] != 2) or np.any(h[window] != 0):
            problems.append("pre_place window not labeled type=2,h_entry=0 throughout")
    if pg_s >= 0:
        before = np.arange(0, pg_s)
        if before.size and (np.any(nct[before] != 1) or np.any(h[before] != (pg_s - before))):
            problems.append("frames before pre_grasp do not point to pre_grasp window entry")
    if pg_e >= 0 and pp_s >= 0:
        mid = np.arange(pg_e + 1, pp_s)
        if mid.size and (np.any(nct[mid] != 2) or np.any(h[mid] != (pp_s - mid))):
            problems.append("frames between windows do not point to pre_place window entry")
    if pp_e >= 0 and pp_e + 1 < T:
        tail = np.arange(pp_e + 1, T)
        if np.any(nct[tail] != 0) or np.any(h[tail] != -1):
            problems.append("terminal frames after pre_place window are not type=0,h_entry=-1")

    # 6. object actually rises (cross-check lifted against object z)
    z_note = ""
    if d["obj_z"] is not None:
        rise = float(d["obj_z"].max() - a.get("resting_z", d["obj_z"].min()))
        z_note = f"z_rise={rise:.3f}"
        if lif.size and rise < a.get("lift_threshold", 0.03):
            problems.append(f"lifted set but object z barely rose ({rise:.3f})")

    # 7. arm cross-check vs scene_info {a}
    arm = a.get("arm", "?")
    scene_arm = scene_info.get(f"episode_{ep}", {}).get("info", {}).get("{a}")
    arm_match = "" if (scene_arm is None or scene_arm == arm) else f"ARM MISMATCH(scene={scene_arm})"
    if arm_match:
        problems.append(arm_match)

    return {
        "ep": ep,
        "arm": arm,
        "T": T,
        "pg": pg,
        "pp": pp,
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

    print(f"{'ep':>3} {'arm':>5} {'T':>4} {'pg_win':>13} {'pp_win':>13} "
          f"{'in-hand':>8} {'lifted':>7} {'released':>8} {'z_rise':>9}  status")
    print("-" * 105)
    bad = []
    for ep in eps:
        hdf5_path = os.path.join(data_dir, f"episode{ep}.hdf5")
        try:
            r = check_episode(ep, hdf5_path, scene_info)
        except Exception as e:
            print(f"{ep:>3}  ERROR: {e}")
            bad.append(ep)
            continue
        status = "OK" if not r["problems"] else "!! " + "; ".join(r["problems"])
        zr = r["z_note"].replace("z_rise=", "") if r["z_note"] else "-"
        print(f"{r['ep']:>3} {r['arm']:>5} {r['T']:>4} [{r['pg_s']:>3},{r['pg_e']:<3}]      "
              f"[{r['pp_s']:>3},{r['pp_e']:<3}]      {r['oih']:>8} {r['lif']:>7} {r['rel']:>8} "
              f"{zr:>9}  {status}")
        if r["problems"]:
            bad.append(ep)

    print("-" * 105)
    print(f"Total {len(eps)} episodes. Clean: {len(eps) - len(bad)}. Problematic: {len(bad)}"
          + (f"  -> {bad}" if bad else "  ✅ all good"))

    if args.episode is not None:
        # verbose dump of the per-step arrays for the single episode
        ep = args.episode
        d = load_keystate(os.path.join(data_dir, f"episode{ep}.hdf5"))
        print(f"\n=== per-step arrays for episode{ep} (frame: type/h_entry/oih/lif/rel) ===")
        nct, h = d["next_checkpoint_type"], d["h_entry"]
        oih, lif, rel = d["object_in_hand"], d["lifted"], d["placed_and_released"]
        for t in range(len(nct)):
            mark = ""
            if nct[t] == 1 and h[t] == 0:
                mark = "  <-- PRE-GRASP WINDOW"
            elif nct[t] == 2 and h[t] == 0:
                mark = "  <-- PRE-PLACE WINDOW"
            elif nct[t] == 0:
                mark = "  <-- NONE"
            print(f"  t={t:>3}: type={nct[t]} h_entry={h[t]:>3} oih={oih[t]} lif={lif[t]} rel={rel[t]}{mark}")


if __name__ == "__main__":
    main()
