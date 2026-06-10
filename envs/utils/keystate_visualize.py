"""
KeyState Stage 0 visualizer (read-only on the hdf5; only writes new png/mp4 artifacts).

Two outputs per episode, for human spot-checking the labels produced by
envs/utils/keystate_labeler.py:

  1. Curve plot (keystate/plots/episode{N}.png):
       active-arm gripper value, object z, EE speed, object z-speed on a shared frame
       axis, with a GREEN vline at pre_grasp_idx, ORANGE vline at pre_place_idx, and
       shaded spans for object_in_hand / lifted / placed_and_released.
  2. Annotated mp4 (keystate/video_annot/episode{N}.mp4):
       head_camera frames with a text banner of currently-active phases and a highlight
       box on the two checkpoint frames. Frame t of the video == label t (1:1).

Usage:
    python -m envs.utils.keystate_visualize --task place_a2b_left --config demo_clean --all
    python -m envs.utils.keystate_visualize --task place_a2b_left --config demo_clean --episode 0 --no-video
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


def _load(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        if "keystate" not in f:
            raise KeyError(f"{hdf5_path} has no /keystate group -- run keystate_labeler first.")
        ks = f["keystate"]
        attrs = dict(ks.attrs)
        d = {
            "checkpoint_type": ks["checkpoint_type"][()],
            "object_in_hand": ks["object_in_hand"][()],
            "lifted": ks["lifted"][()],
            "placed_and_released": ks["placed_and_released"][()],
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


def _span(ax, mask, color, label):
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
            ax.axvspan(start, prev, color=color, alpha=0.15, label=label if first else None)
            first = False
            start = i
        prev = i
    ax.axvspan(start, prev, color=color, alpha=0.15, label=label if first else None)


def plot_curves(d, out_path, ep):
    a = d["attrs"]
    arm = a.get("arm", "left")
    g = d[f"{arm}_gripper"] if arm in ("left", "right") else d["left_gripper"]
    ee = d[f"{arm}_endpose"] if arm in ("left", "right") else d["left_endpose"]
    ee_sp = _ee_speed(ee)
    T = len(g)
    x = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    ax.plot(x, g, "b-", label=f"{arm} gripper")
    ax.axhline(0.8, color="gray", ls="--", lw=0.8)
    ax.axhline(0.2, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("gripper")
    _span(ax, d["object_in_hand"], "tab:blue", "in-hand")
    _span(ax, d["lifted"], "tab:green", "lifted")
    _span(ax, d["placed_and_released"], "tab:red", "released")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[1]
    if d["object_pose"] is not None:
        oz = d["object_pose"][:, 2]
        ax.plot(x, oz, "m-", label="object z")
        rz = a.get("resting_z", float("nan"))
        if rz == rz:  # not nan
            ax.axhline(rz, color="gray", ls=":", lw=0.8)
            ax.axhline(rz + a.get("lift_threshold", 0.03), color="green", ls=":", lw=0.8)
    ax.set_ylabel("object z (m)")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[2]
    ax.plot(x, ee_sp, "c-", label="EE speed (m/s)")
    ax.set_ylabel("EE speed")
    ax.set_xlabel("frame")
    ax.legend(loc="upper right", fontsize=7)

    pg = int(a.get("pre_grasp_idx", -1))
    pp = int(a.get("pre_place_idx", -1))
    for axx in axes:
        if pg >= 0:
            axx.axvline(pg, color="green", lw=1.6)
        if pp >= 0:
            axx.axvline(pp, color="orange", lw=1.6)

    flags = a.get("flags", "[]")
    title = f"episode{ep}  arm={arm}  pre_grasp={pg}(green)  pre_place={pp}(orange)"
    if flags and flags != "[]":
        title += f"  FLAGS={flags}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def annotate_video(d, out_path, ep):
    if d["_rgb_raw"] is None:
        print(f"[warn] episode{ep}: no head_camera rgb, skip video")
        return
    frames = parse_img_array(d["_rgb_raw"])  # (T,H,W,3) BGR
    a = d["attrs"]
    pg = int(a.get("pre_grasp_idx", -1))
    pp = int(a.get("pre_place_idx", -1))
    oih, lif, rel = d["object_in_hand"], d["lifted"], d["placed_and_released"]
    T = len(frames)

    out = []
    for t in range(T):
        img = frames[t].copy()
        h, w = img.shape[:2]
        phases = []
        if t < len(oih) and oih[t]:
            phases.append("in-hand")
        if t < len(lif) and lif[t]:
            phases.append("lifted")
        if t < len(rel) and rel[t]:
            phases.append("released")
        banner = " ".join(phases) if phases else "-"
        cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
        cv2.putText(img, f"t={t} {banner}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if t == pg:
            cv2.rectangle(img, (1, 1), (w - 2, h - 2), (0, 255, 0), 4)
            cv2.putText(img, "PRE-GRASP", (w // 2 - 60, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
        if t == pp:
            cv2.rectangle(img, (1, 1), (w - 2, h - 2), (0, 165, 255), 4)
            cv2.putText(img, "PRE-PLACE", (w // 2 - 60, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 165, 255), 2, cv2.LINE_AA)
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
