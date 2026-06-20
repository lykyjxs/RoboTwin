"""Inspect h_entry bucket histograms for KeyState datasets.

This is a lightweight sanity-check helper for the Stage1/Stage2 horizon head. It reads raw
RoboTwin HDF5 (`/keystate/h_entry`) or processed Pi0 HDF5
(`/observations/keystate/h_entry`) files and prints counts under the current bucket scheme:

  bin 0: h == 0
  bin 1: 1 <= h < 4
  bin 2: 4 <= h < 7
  bin 3: 7 <= h < 11
  bin 4: 11 <= h < 21
  bin 5: 21 <= h < 51
  bin 6: h >= 51
  invalid: h < 0
"""

from __future__ import annotations

import argparse
import pathlib

import h5py
import numpy as np


DEFAULT_UPPER_EDGES = (1, 4, 7, 11, 21, 51)
RAW_DATASET = "/keystate/h_entry"
PROCESSED_DATASET = "/observations/keystate/h_entry"


def bucket_h_entry(h_entry: np.ndarray, upper_edges: tuple[int, ...]) -> np.ndarray:
    h_entry = np.asarray(h_entry)
    bins = np.searchsorted(np.asarray(upper_edges), h_entry, side="right").astype(np.int32)
    return np.where(h_entry < 0, np.int32(-1), bins)


def iter_hdf5_paths(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.hdf5")))
            files.extend(sorted(path.rglob("*.h5")))
        else:
            files.append(path)
    return files


def pick_dataset(root: h5py.File, requested: str | None) -> str:
    if requested is not None:
        if requested not in root:
            raise KeyError(f"requested dataset {requested!r} not found")
        return requested
    if PROCESSED_DATASET in root:
        return PROCESSED_DATASET
    if RAW_DATASET in root:
        return RAW_DATASET
    raise KeyError(f"neither {PROCESSED_DATASET!r} nor {RAW_DATASET!r} found")


def histogram_for_file(path: pathlib.Path, dataset: str | None, upper_edges: tuple[int, ...]) -> np.ndarray:
    with h5py.File(path, "r") as root:
        ds = pick_dataset(root, dataset)
        h_entry = np.asarray(root[ds][()]).reshape(-1)
    bins = bucket_h_entry(h_entry, upper_edges)
    counts = np.zeros(len(upper_edges) + 2, dtype=np.int64)  # bins + invalid at the end.
    for bin_idx in range(len(upper_edges) + 1):
        counts[bin_idx] = int(np.sum(bins == bin_idx))
    counts[-1] = int(np.sum(bins < 0))
    return counts


def format_counts(counts: np.ndarray, upper_edges: tuple[int, ...]) -> str:
    labels = [
        "bin 0 (h == 0)",
        "bin 1 (1 <= h < 4)",
        "bin 2 (4 <= h < 7)",
        "bin 3 (7 <= h < 11)",
        "bin 4 (11 <= h < 21)",
        "bin 5 (21 <= h < 51)",
        "bin 6 (h >= 51)",
    ]
    if upper_edges != DEFAULT_UPPER_EDGES:
        labels = [f"bin {i}" for i in range(len(upper_edges) + 1)]
    labels.append("invalid (h < 0)")
    return "\n".join(f"  {label}: {int(count)}" for label, count in zip(labels, counts, strict=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print KeyState h_entry bucket histograms.")
    parser.add_argument("paths", nargs="+", type=pathlib.Path, help="HDF5 file(s) or directories to inspect.")
    parser.add_argument(
        "--dataset",
        default=None,
        help=f"Dataset path to read. Defaults to {PROCESSED_DATASET}, then {RAW_DATASET}.",
    )
    parser.add_argument(
        "--upper-edges",
        default=",".join(str(x) for x in DEFAULT_UPPER_EDGES),
        help="Comma-separated bucket upper edges. Default: 1,4,7,11,21,51.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upper_edges = tuple(int(x) for x in args.upper_edges.split(",") if x)
    paths = iter_hdf5_paths(args.paths)
    if not paths:
        raise FileNotFoundError("no HDF5 files found")

    total = np.zeros(len(upper_edges) + 2, dtype=np.int64)
    for path in paths:
        counts = histogram_for_file(path, args.dataset, upper_edges)
        total += counts
        print(path)
        print(format_counts(counts, upper_edges))

    if len(paths) > 1:
        print("TOTAL")
        print(format_counts(total, upper_edges))


if __name__ == "__main__":
    main()
