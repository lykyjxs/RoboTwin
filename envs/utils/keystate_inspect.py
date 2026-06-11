"""
KeyState Stage 0 self-inspection tool (read-only).

For manually checking the quality of the keystate labels produced by
keystate_labeler.py. Prints a per-episode table and a summary, and runs
sanity assertions. Also cross-checks the detected arm against scene_info.json's
{a} field (the arm the demo script actually used).

Usage:
    python envs/utils/keystate_inspect.py --task place_a2b_left --config demo_clean
    python envs/utils/keystate_inspect.py --task place_a2b_left --config demo_clean --episode 0   # one episode, verbose
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
        a = dict(ks.attrs)
        d = {
            "checkpoint_type": ks["checkpoint_type"][()],
            # dense next_checkpoint_type (labeler v2+); fall back gracefully if labeling is stale
            "next_checkpoint_type": ks["next_checkpoint_type"][()] if "next_checkpoint_type" in ks else None,
            "h_ckpt": ks["h_ckpt"][()],
            "object_in_hand": ks["object_in_hand"][()],
            "lifted": ks["lifted"][()],
            "placed_and_released": ks["placed_and_released"][()],
            "semantic_phase": ks["semantic_phase"][()],
            "attrs": a,
        }
        # raw signals for alignment checks
        d["T_endpose"] = f["/endpose/left_endpose"].shape[0]
        d["T_rgb"] = f["/observation/head_camera/rgb"].shape[0] if "observation" in f else None
        d["obj_z"] = f["/object_pose/object"][()][:, 2] if "object_pose" in f else None
    return d


def check_episode(ep, hdf5_path, scene_info):
    d = load_keystate(hdf5_path)
    a = d["attrs"]
    T = int(a["T"])
    pg = int(a["pre_grasp_idx"])
    pp = int(a["pre_place_idx"])
    flags = a.get("flags", "[]")
    if isinstance(flags, str):
        flags = json.loads(flags)

    problems = []

    # 1. frame alignment: every per-step array must have the same length T
    lengths = {
        "checkpoint_type": len(d["checkpoint_type"]),
        "h_ckpt": len(d["h_ckpt"]),
        "object_in_hand": len(d["object_in_hand"]),
        "lifted": len(d["lifted"]),
        "placed_and_released": len(d["placed_and_released"]),
        "endpose": d["T_endpose"],
    }
    if d["T_rgb"] is not None:
        lengths["rgb"] = d["T_rgb"]
    if len(set(lengths.values())) != 1:
        problems.append(f"FRAME MISALIGN: {lengths}")

    # 2. ordering: pre_grasp < pre_place
    if pg >= 0 and pp >= 0 and not (pg < pp):
        problems.append(f"pre_grasp({pg}) >= pre_place({pp})")

    # 3. phase starts are after the corresponding checkpoint
    oih = np.nonzero(d["object_in_hand"])[0]
    lif = np.nonzero(d["lifted"])[0]
    rel = np.nonzero(d["placed_and_released"])[0]
    oih_start = int(oih.min()) if oih.size else -1
    lif_start = int(lif.min()) if lif.size else -1
    rel_start = int(rel.min()) if rel.size else -1
    if pg >= 0 and oih_start >= 0 and not (pg < oih_start):
        problems.append(f"pre_grasp({pg}) not before in-hand start({oih_start})")
    if pp >= 0 and rel_start >= 0 and not (pp < rel_start):
        problems.append(f"pre_place({pp}) not before released start({rel_start})")

    # 4. lifted should overlap object_in_hand (you can't lift what you're not holding)
    if oih.size and lif.size:
        overlap = bool((d["object_in_hand"].astype(bool) & d["lifted"].astype(bool)).any())
        if not overlap:
            problems.append("lifted does not overlap object_in_hand")

    # 5. h_ckpt sanity: 0 at each checkpoint frame
    if pg >= 0 and d["h_ckpt"][pg] != 0:
        problems.append(f"h_ckpt at pre_grasp != 0 (={d['h_ckpt'][pg]})")
    if pp >= 0 and d["h_ckpt"][pp] != 0:
        problems.append(f"h_ckpt at pre_place != 0 (={d['h_ckpt'][pp]})")

    # 6. checkpoint_type codes correct
    if pg >= 0 and d["checkpoint_type"][pg] != 1:
        problems.append("checkpoint_type at pre_grasp != 1")
    if pp >= 0 and d["checkpoint_type"][pp] != 2:
        problems.append("checkpoint_type at pre_place != 2")

    # 9. dense next_checkpoint_type consistency (labeler v2+):
    #    valid frames (h_ckpt>=0) must have type in {1,2}; invalid frames (h_ckpt<0) must have type 0.
    #    Also the dense type must equal the type of the nearest future checkpoint (== checkpoint_type[nxt]).
    nct = d["next_checkpoint_type"]
    if nct is not None:
        h = d["h_ckpt"]
        valid = h >= 0
        if not np.all(np.isin(nct[valid], (1, 2))):
            problems.append(f"next_checkpoint_type on valid frames not in {{1,2}} (got {set(nct[valid].tolist())})")
        if np.any(nct[~valid] != 0):
            problems.append("next_checkpoint_type on invalid (h<0) frames != 0")
        # past pre_grasp but before pre_place -> next target is pre_place -> type must be 2 (not 1)
        if pg >= 0 and pp >= 0:
            mid = np.arange(T)
            mid_mask = (mid > pg) & (mid <= pp)
            if mid_mask.any() and np.any(nct[mid_mask] != 2):
                problems.append("next_checkpoint_type between pre_grasp and pre_place != 2")
            before_mask = (mid <= pg)
            if before_mask.any() and np.any(nct[before_mask] != 1):
                problems.append("next_checkpoint_type up to pre_grasp != 1")
    else:
        problems.append("next_checkpoint_type missing (stale labeler < v2; re-run keystate_labeler)")

    # 7. object actually rises (cross-check lifted against object z)
    z_note = ""
    if d["obj_z"] is not None:
        rise = float(d["obj_z"].max() - a.get("resting_z", d["obj_z"].min()))
        z_note = f"z_rise={rise:.3f}"
        if lif.size and rise < a.get("lift_threshold", 0.03):
            problems.append(f"lifted set but object z barely rose ({rise:.3f})")

    # 8. arm cross-check vs scene_info {a}
    arm = a.get("arm", "?")
    scene_arm = scene_info.get(f"episode_{ep}", {}).get("info", {}).get("{a}")
    arm_match = "" if (scene_arm is None or scene_arm == arm) else f"ARM MISMATCH(scene={scene_arm})"
    if arm_match:
        problems.append(arm_match)

    return {
        "ep": ep, "arm": arm, "T": T, "pg": pg, "pp": pp,
        "oih": oih_start, "lif": lif_start, "rel": rel_start,
        "flags": flags, "z_note": z_note, "problems": problems,
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

    print(f"{'ep':>3} {'arm':>5} {'T':>4} {'pre_grasp':>9} {'pre_place':>9} "
          f"{'in-hand':>8} {'lifted':>7} {'released':>8} {'z_rise':>9}  status")
    print("-" * 90)
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
        print(f"{r['ep']:>3} {r['arm']:>5} {r['T']:>4} {r['pg']:>9} {r['pp']:>9} "
              f"{r['oih']:>8} {r['lif']:>7} {r['rel']:>8} {zr:>9}  {status}")
        if r["problems"]:
            bad.append(ep)

    print("-" * 90)
    print(f"Total {len(eps)} episodes. Clean: {len(eps) - len(bad)}. Problematic: {len(bad)}"
          + (f"  -> {bad}" if bad else "  ✅ all good"))

    if args.episode is not None:
        # verbose dump of the per-step arrays for the single episode
        ep = args.episode
        d = load_keystate(os.path.join(data_dir, f"episode{ep}.hdf5"))
        print(f"\n=== per-step arrays for episode{ep} (frame: ct/h/oih/lif/rel) ===")
        ct, h = d["checkpoint_type"], d["h_ckpt"]
        oih, lif, rel = d["object_in_hand"], d["lifted"], d["placed_and_released"]
        for t in range(len(ct)):
            mark = ""
            if ct[t] == 1:
                mark = "  <-- PRE-GRASP"
            elif ct[t] == 2:
                mark = "  <-- PRE-PLACE"
            print(f"  t={t:>3}: ct={ct[t]} h={h[t]:>3} oih={oih[t]} lif={lif[t]} rel={rel[t]}{mark}")


if __name__ == "__main__":
    main()
