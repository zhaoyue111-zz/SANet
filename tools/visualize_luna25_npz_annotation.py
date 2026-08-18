#!/usr/bin/env python3
"""Visualize luna25_npz annotation.csv boxes on npz ``image`` slices.

For each nodule: mid-slice +/- 2 (5 axial slices), saved as one horizontal PNG.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_NPZ_DIR = os.path.join(REPO_ROOT, "luna25_npz")
DEFAULT_CSV = os.path.join(DEFAULT_NPZ_DIR, "annotation.csv")
DEFAULT_OUT_DIR = os.path.join(DEFAULT_NPZ_DIR, "vis_anno")


def resolve_npz_path(pid: str, npz_dir: str, repo_root: str) -> str:
    candidates = [
        pid,
        os.path.join(repo_root, pid),
        os.path.join(npz_dir, os.path.basename(pid)),
        os.path.join(npz_dir, pid),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("Cannot resolve npz for pid=%s" % pid)


def _clamp_z(z: int, z_size: int) -> int:
    return int(min(max(z, 0), z_size - 1))


def draw_nodule_row(image_zyx: np.ndarray, row: pd.Series, out_path: str, offset: int = 2,
                    box_pad: int = 2):
    """Save mid-2..mid+2 slices in one horizontal PNG with the nodule bbox.

    ``box_pad`` expands the drawn rectangle outward so thin boxes do not cover
    tiny lesions.
    """
    z_size, y_size, x_size = image_zyx.shape
    z0, z1 = int(row["bbox_min_z"]), int(row["bbox_max_z"])
    y0, y1 = int(row["bbox_min_y"]), int(row["bbox_max_y"])
    x0, x1 = int(row["bbox_min_x"]), int(row["bbox_max_x"])
    z_mid = _clamp_z(int(round(0.5 * (z0 + z1))), z_size)
    zs = [_clamp_z(z_mid + d, z_size) for d in range(-offset, offset + 1)]

    # Expand box for display only (annotation CSV unchanged).
    y0_d = max(0, y0 - box_pad)
    x0_d = max(0, x0 - box_pad)
    y1_d = min(y_size - 1, y1 + box_pad)
    x1_d = min(x_size - 1, x1 + box_pad)

    n = len(zs)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    if n == 1:
        axes = [axes]

    for ax, z in zip(axes, zs):
        ax.imshow(image_zyx[z], cmap="gray", vmin=0.0, vmax=1.0)
        if z0 <= z <= z1:
            rect = patches.Rectangle(
                (x0_d, y0_d),
                max(x1_d - x0_d, 1),
                max(y1_d - y0_d, 1),
                linewidth=1.5,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)
        title = "z=%d" % z
        if z == z_mid:
            title += " (mid)"
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        "bbox z[%d,%d] y[%d,%d] x[%d,%d] (draw pad=%d)" % (z0, z1, y0, y1, x0, x1, box_pad),
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def draw_case(image_zyx: np.ndarray, boxes: pd.DataFrame, out_prefix: str, box_pad: int = 2):
    """One PNG per nodule: 5 axial slices centered on the lesion, horizontal layout."""
    for local_i, (_, row) in enumerate(boxes.iterrows()):
        out_path = "%s_nodule%d.png" % (out_prefix, local_i)
        draw_nodule_row(image_zyx, row, out_path, offset=2, box_pad=box_pad)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="annotation.csv path")
    parser.add_argument("--npz-dir", default=DEFAULT_NPZ_DIR, help="Directory containing npz files")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for PNGs")
    parser.add_argument("--repo-root", default=REPO_ROOT, help="Repo root for resolving relative pid")
    parser.add_argument("--max-cases", type=int, default=0, help="Limit number of cases (0=all)")
    parser.add_argument("--box-pad", type=int, default=2,
                        help="Expand drawn bbox outward by N pixels (display only)")
    parser.add_argument("--pid", default=None, help="Visualize only this pid/basename")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = [
        "pid", "bbox_min_z", "bbox_min_y", "bbox_min_x",
        "bbox_max_z", "bbox_max_y", "bbox_max_x",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("CSV missing columns: %s" % missing)

    if args.pid:
        base = os.path.basename(args.pid)
        df = df[df["pid"].astype(str).str.contains(base, regex=False)]
        if df.empty:
            raise ValueError("No rows matched --pid %s" % args.pid)

    os.makedirs(args.out_dir, exist_ok=True)
    pids = list(dict.fromkeys(df["pid"].astype(str).tolist()))
    if args.max_cases > 0:
        pids = pids[: args.max_cases]

    for i, pid in enumerate(pids, 1):
        npz_path = resolve_npz_path(pid, args.npz_dir, args.repo_root)
        image = np.asarray(np.load(npz_path, allow_pickle=True)["image"], dtype=np.float32)
        boxes = df[df["pid"].astype(str) == pid]
        stem = os.path.splitext(os.path.basename(npz_path))[0]
        out_prefix = os.path.join(args.out_dir, stem)
        draw_case(image, boxes, out_prefix, box_pad=args.box_pad)
        print("[%d/%d] %s -> %d PNG(s) under %s_nodule*.png"
              % (i, len(pids), stem, len(boxes), out_prefix))

    print("Done. PNGs in %s" % args.out_dir)


if __name__ == "__main__":
    main()
