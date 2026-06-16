"""
KeyState Stage 0 visualizer (read-only on the hdf5; only writes new png/mp4 artifacts).

Two outputs per episode, for human spot-checking the checkpoint-window labels produced by
`envs/utils/keystate_labeler.py`:

  1. Curve plot (keystate/plots/episode{N}.png):
       active-arm gripper value, object z, EE speed, `next_checkpoint_type`, and `h_entry`,
       with shaded pre_grasp/pre_place checkpoint windows plus semantic phases.
  2. Annotated mp4 (keystate/video_annot/episode{N}.mp4):
       head_camera frames with a text banner of current type/h_entry/phases and a border
       while inside a checkpoint window. Frame t of the video == label t (1:1).

Usage:
    python envs/utils/keystate_visualize.py --task place_a2b_left --config demo_clean --all
    python envs/utils/keystate_visualize.py --task place_a2b_left --config demo_clean --episode 0 --no-video
"""
import argparse
import json
import os

import cv2
import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Load the two lightweight sibling utils directly by file path, so this script never
# triggers envs/utils/__init__.py (which chain-imports sapien + asset-dependent modules).
import importlib.util as _ilu  # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(modname, filename):
    spec = _ilu.spec_from_file_location(modname, os.path.join(_here, filename))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


images_to_video = _load_sibling("_ks_i2v", "images_to_video.py").images_to_video
parse_img_array = _load_sibling("_ks_parse", "parse_hdf5.py").parse_img_array

DT = 15.0 / 250.0
TYPE_NAMES = {0: "none", 1: "pre_grasp", 2: "pre_place"}
TYPE_COLORS = {1: "tab:green", 2: "tab:orange"}
TYPE_BGR = {1: (0, 255, 0), 2: (0, 165, 255)}


def _load(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        if "keystate" not in f:
            raise KeyError(f"{hdf5_path} has no /keystate group -- run keystate_labeler first.")
        ks = f["keystate"]
        required = ["next_checkpoint_type", "h_entry", "semantic_phase"]
        missing = [k for k in required if k not in ks]
        if missing:
            raise KeyError(f"{hdf5_path} /keystate missing v3 fields: {missing}")
        attrs = dict(ks.attrs)
        sem = ks["semantic_phase"][()]
        d = {
            "next_checkpoint_type": ks["next_checkpoint_type"][()],
            "h_entry": ks["h_entry"][()],
            "object_in_hand": ks["object_in_hand"][()] if "object_in_hand" in ks else sem[:, 0],
            "lifted": ks["lifted"][()] if "lifted" in ks else sem[:, 1],
            "placed_and_released": ks["placed_and_released"][()] if "placed_and_released" in ks else sem[:, 2],
            "semantic_phase": sem,
            "attrs": attrs,
            "left_endpose": f["/endpose/left_endpose"][()],
            "right_endpose": f["/endpose/right_endpose"][()],
            "left_gripper": np.atleast_1d(f["/endpose/left_gripper"][()]).astype(float),
            "right_gripper": np.atleast_1d(f["/endpose/right_gripper"][()]).astype(float),
        }
        d["object_pose"] = f["/object_pose/object"][()] if "object_pose" in f else None
        # head camera rgb (JPEG-encoded), decoded lazily only when video requested
        d["_rgb_raw"] = f["/observation/head_camera/rgb"][()] if "observation" in f else None
    return d


def _ee_speed(endpose):
    p = endpose[:, 0:3]
    sp = np.zeros(p.shape[0])
    sp[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) / DT
    return sp


def _span(ax, mask, color, label, alpha=0.15):
    """Shade contiguous regions where mask==1."""
    on = np.asarray(mask).astype(bool)
    if not on.any():
        return
    idx = np.nonzero(on)[0]
    start = idx[0]
    prev = idx[0]
    first = True
    for i in idx[1:]:
        if i != prev + 1:
            ax.axvspan(start, prev, color=color, alpha=alpha, label=label if first else None)
            first = False
            start = i
        prev = i
    ax.axvspan(start, prev, color=color, alpha=alpha, label=label if first else None)


def _window_mask(d, typ):
    return (d["next_checkpoint_type"] == typ) & (d["h_entry"] == 0)


def _shade_windows(ax, d):
    _span(ax, _window_mask(d, 1), TYPE_COLORS[1], "pre_grasp window", alpha=0.20)
    _span(ax, _window_mask(d, 2), TYPE_COLORS[2], "pre_place window", alpha=0.20)


def plot_curves(d, out_path, ep):
    a = d["attrs"]
    arm = a.get("arm", "left")
    g = d[f"{arm}_gripper"] if arm in ("left", "right") else d["left_gripper"]
    ee = d[f"{arm}_endpose"] if arm in ("left", "right") else d["left_endpose"]
    ee_sp = _ee_speed(ee)
    T = len(g)
    x = np.arange(T)

    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)

    ax = axes[0]
    ax.plot(x, g, "b-", label=f"{arm} gripper")
    ax.axhline(0.8, color="gray", ls="--", lw=0.8)
    ax.axhline(0.2, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("gripper")
    _shade_windows(ax, d)
    _span(ax, d["object_in_hand"], "tab:blue", "in-hand", alpha=0.10)
    _span(ax, d["lifted"], "tab:green", "lifted", alpha=0.08)
    _span(ax, d["placed_and_released"], "tab:red", "released", alpha=0.10)
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[1]
    if d["object_pose"] is not None:
        oz = d["object_pose"][:, 2]
        ax.plot(x, oz, "m-", label="object z")
        rz = a.get("resting_z", float("nan"))
        if rz == rz:  # not nan
            ax.axhline(rz, color="gray", ls=":", lw=0.8)
            ax.axhline(rz + a.get("lift_threshold", 0.03), color="green", ls=":", lw=0.8)
    _shade_windows(ax, d)
    ax.set_ylabel("object z (m)")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[2]
    ax.plot(x, ee_sp, "c-", label="EE speed (m/s)")
    _shade_windows(ax, d)
    ax.set_ylabel("EE speed")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[3]
    ax.step(x, d["next_checkpoint_type"], where="post", color="k", label="next_checkpoint_type")
    _shade_windows(ax, d)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["none", "pre_grasp", "pre_place"])
    ax.set_ylabel("type")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[4]
    h_plot = d["h_entry"].astype(float)
    h_plot[h_plot < 0] = np.nan
    ax.plot(x, h_plot, "r-", label="h_entry")
    _shade_windows(ax, d)
    ax.set_ylabel("h_entry")
    ax.set_xlabel("frame")
    ax.legend(loc="upper right", fontsize=7)

    pg_s = int(a.get("pre_grasp_window_start", -1))
    pg_e = int(a.get("pre_grasp_window_end", -1))
    pp_s = int(a.get("pre_place_window_start", -1))
    pp_e = int(a.get("pre_place_window_end", -1))
    for axx in axes:
        for pos, color, ls in [(pg_s, "green", "-"), (pg_e, "green", ":"), (pp_s, "orange", "-"), (pp_e, "orange", ":")]:
            if pos >= 0:
                axx.axvline(pos, color=color, lw=1.2, ls=ls)

    flags = a.get("flags", "[]")
    title = (f"episode{ep} arm={arm}  pre_grasp=[{pg_s},{pg_e}] green  "
             f"pre_place=[{pp_s},{pp_e}] orange")
    if flags and flags != "[]":
        title += f"  FLAGS={flags}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def annotate_video(d, out_path, ep):
    if d["_rgb_raw"] is None:
        print(f"[warn] episode{ep}: no head_camera rgb, skip video")
        return
    frames = parse_img_array(d["_rgb_raw"])  # (T,H,W,3) BGR
    nct, h_entry = d["next_checkpoint_type"], d["h_entry"]
    oih, lif, rel = d["object_in_hand"], d["lifted"], d["placed_and_released"]
    T = len(frames)

    out = []
    for t in range(T):
        img = frames[t].copy()
        h, w = img.shape[:2]
        typ = int(nct[t]) if t < len(nct) else 0
        h_val = int(h_entry[t]) if t < len(h_entry) else -1
        phases = []
        if t < len(oih) and oih[t]:
            phases.append("in-hand")
        if t < len(lif) and lif[t]:
            phases.append("lifted")
        if t < len(rel) and rel[t]:
            phases.append("released")
        phase_text = "|".join(phases) if phases else "-"
        banner = f"t={t} type={TYPE_NAMES.get(typ, typ)} h_entry={h_val} phase={phase_text}"
        cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
        cv2.putText(img, banner, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if typ in TYPE_BGR and h_val == 0:
            color = TYPE_BGR[typ]
            cv2.rectangle(img, (1, 1), (w - 2, h - 2), color, 4)
            cv2.putText(img, f"{TYPE_NAMES[typ].upper()} WINDOW", (w // 2 - 120, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        out.append(img)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # frames are BGR (cv2.imdecode); images_to_video expects RGB when is_rgb=True,
    # so pass is_rgb=False to keep colors correct.
    images_to_video(np.stack(out, axis=0), out_path, fps=30.0, is_rgb=False)


def main():
    p = argparse.ArgumentParser(description="KeyState Stage 0 visualizer")
    p.add_argument("--task", default="place_a2b_left")
    p.add_argument("--config", default="demo_clean")
    p.add_argument("--data-root", default="data")
    p.add_argument("--no-video", action="store_true", help="only produce curve plots")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--episode", type=int)
    args = p.parse_args()

    base = os.path.join(args.data_root, args.task, args.config)
    data_dir = os.path.join(base, "data")
    if args.all:
        eps = sorted(
            int(fn[len("episode"):-len(".hdf5")])
            for fn in os.listdir(data_dir)
            if fn.startswith("episode") and fn.endswith(".hdf5"))
    else:
        eps = [args.episode]

    for ep in eps:
        hdf5_path = os.path.join(data_dir, f"episode{ep}.hdf5")
        try:
            d = _load(hdf5_path)
            plot_curves(d, os.path.join(base, "keystate", "plots", f"episode{ep}.png"), ep)
            if not args.no_video:
                annotate_video(d, os.path.join(base, "keystate", "video_annot", f"episode{ep}.mp4"), ep)
            print(f"[ok] episode{ep} visualized")
        except Exception as e:
            print(f"[err] episode{ep}: {e}")


if __name__ == "__main__":
    main()
