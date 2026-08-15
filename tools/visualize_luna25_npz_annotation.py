#!/usr/bin/env python3
"""Visualize original-resolution LUNA25 annotation.csv boxes.

The CSV boxes are on the original full CT grid (ZYX). ``image_original`` is
padded back to that grid using ``mask_size_original`` before plotting.
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
import SimpleITK as sitk
from scipy import ndimage


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_NPZ_DIR = os.path.join(REPO_ROOT, "luna25_npz")
DEFAULT_CSV = os.path.join(DEFAULT_NPZ_DIR, "annotation.csv")
DEFAULT_OUT_DIR = os.path.join(DEFAULT_NPZ_DIR, "vis_anno")
DEFAULT_MASK_DIR = os.path.join(REPO_ROOT, "data", "luna25", "output")


def normalize_pid(pid) -> str:
    text = str(pid).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def build_linear_pid_map(npz_dir: str, mask_dir: str) -> dict[str, str]:
    """Map row-level pid values to NPZs using original-mask component counts."""
    mapping = {}
    next_pid = 0
    npz_files = sorted(
        os.path.join(npz_dir, name)
        for name in os.listdir(npz_dir)
        if name.endswith(".npz")
    )
    for npz_path in npz_files:
        d = dict(np.load(npz_path, allow_pickle=True))
        series_uid = str(np.asarray(d.get("seriesUID", "")).item())
        mask_path = os.path.join(mask_dir, "%s_nodule_mask.nii.gz" % series_uid)
        if not os.path.isfile(mask_path):
            raise FileNotFoundError("Missing nodule mask: %s" % mask_path)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        _, n_components = ndimage.label(np.asarray(mask) > 0)
        for _ in range(int(n_components)):
            mapping[str(next_pid)] = npz_path
            next_pid += 1
    return mapping


def resolve_npz_path(pid: str, npz_dir: str, repo_root: str,
                     linear_pid_map: dict[str, str] | None = None) -> str:
    pid_text = normalize_pid(pid)
    if linear_pid_map is not None and pid_text in linear_pid_map:
        return linear_pid_map[pid_text]
    npz_files = sorted(
        os.path.join(npz_dir, name)
        for name in os.listdir(npz_dir)
        if name.endswith(".npz")
    )
    if pid_text.isdigit():
        index = int(pid_text)
        if 0 <= index < len(npz_files):
            return npz_files[index]
        raise FileNotFoundError("Numeric pid=%s is outside NPZ range [0,%d)"
                                % (pid_text, len(npz_files)))

    candidates = [
        pid_text,
        os.path.join(repo_root, pid_text),
        os.path.join(npz_dir, os.path.basename(pid_text)),
        os.path.join(npz_dir, pid_text),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("Cannot resolve npz for pid=%s" % pid)


def _clamp_z(z: int, z_size: int) -> int:
    return int(min(max(z, 0), z_size - 1))


def restore_original_hu(d: dict) -> np.ndarray:
    """Pad image_original back to the original full-resolution ZYX grid."""
    full_shape = tuple(np.asarray(d["image_shape_original"][0], dtype=int))
    crop = np.asarray(d["image_original"])
    box = np.asarray(d["mask_size_original"], dtype=int)
    expected_crop_shape = tuple((box[:, 1] - box[:, 0]).tolist())
    if tuple(crop.shape) != expected_crop_shape:
        raise ValueError(
            "image_original shape %s != mask_size_original extent %s"
            % (crop.shape, expected_crop_shape)
        )
    full = np.full(full_shape, -1024, dtype=crop.dtype)
    z0, z1 = box[0]
    y0, y1 = box[1]
    x0, x1 = box[2]
    full[z0:z1, y0:y1, x0:x1] = crop
    return full


def draw_nodule_row(image_zyx: np.ndarray, row: pd.Series, out_path: str, offset: int = 2,
                    box_pad: int = 2, window_min: float = -1000,
                    window_max: float = 400):
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
        ax.imshow(image_zyx[z], cmap="gray", vmin=window_min, vmax=window_max)
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


def draw_case(image_zyx: np.ndarray, boxes: pd.DataFrame, out_prefix: str, box_pad: int = 2,
              window_min: float = -1000, window_max: float = 400):
    """One PNG per nodule: 5 axial slices centered on the lesion, horizontal layout."""
    for local_i, (_, row) in enumerate(boxes.iterrows()):
        out_path = "%s_nodule%d.png" % (out_prefix, local_i)
        draw_nodule_row(
            image_zyx, row, out_path, offset=2, box_pad=box_pad,
            window_min=window_min, window_max=window_max,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="annotation.csv path")
    parser.add_argument("--npz-dir", default=DEFAULT_NPZ_DIR, help="Directory containing npz files")
    parser.add_argument("--mask-dir", default=DEFAULT_MASK_DIR,
                        help="Directory of original-resolution nodule masks used to map linear pid")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for PNGs")
    parser.add_argument("--repo-root", default=REPO_ROOT, help="Repo root for resolving relative pid")
    parser.add_argument("--max-cases", type=int, default=0, help="Limit number of cases (0=all)")
    parser.add_argument("--case-index", type=int, default=None,
                        help="Visualize only one sorted NPZ case index (0-based)")
    parser.add_argument("--box-pad", type=int, default=2,
                        help="Expand drawn bbox outward by N pixels (display only)")
    parser.add_argument("--window-min", type=float, default=-1000,
                        help="HU display window lower bound")
    parser.add_argument("--window-max", type=float, default=400,
                        help="HU display window upper bound")
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

    df["_pid_key"] = df["pid"].map(normalize_pid)
    numeric_pid = bool(df["_pid_key"].map(str.isdigit).all())
    linear_pid_map = build_linear_pid_map(args.npz_dir, args.mask_dir) if numeric_pid else None

    if args.pid:
        pid_key = normalize_pid(args.pid)
        if pid_key.isdigit():
            df = df[df["_pid_key"] == pid_key]
        else:
            base = os.path.basename(pid_key)
            df = df[(df["_pid_key"] == pid_key) | df["_pid_key"].str.contains(base, regex=False)]
        if df.empty:
            raise ValueError("No rows matched --pid %s" % args.pid)

    os.makedirs(args.out_dir, exist_ok=True)
    df["_npz_path"] = df["_pid_key"].map(
        lambda pid: resolve_npz_path(pid, args.npz_dir, args.repo_root, linear_pid_map)
    )
    npz_paths = list(dict.fromkeys(df["_npz_path"].tolist()))
    if args.case_index is not None:
        if args.case_index < 0 or args.case_index >= len(npz_paths):
            raise ValueError("--case-index %d is outside range [0,%d)"
                             % (args.case_index, len(npz_paths)))
        npz_paths = [npz_paths[args.case_index]]
    if args.max_cases > 0:
        npz_paths = npz_paths[: args.max_cases]

    for i, npz_path in enumerate(npz_paths, 1):
        d = dict(np.load(npz_path, allow_pickle=True))
        image = restore_original_hu(d)
        boxes = df[df["_npz_path"] == npz_path]
        stem = os.path.splitext(os.path.basename(npz_path))[0]
        out_prefix = os.path.join(args.out_dir, stem)
        draw_case(
            image, boxes, out_prefix, box_pad=args.box_pad,
            window_min=args.window_min, window_max=args.window_max,
        )
        pid_values = boxes["_pid_key"].tolist()
        pid_range = pid_values[0] if len(pid_values) == 1 else "%s..%s" % (
            pid_values[0], pid_values[-1]
        )
        print("[%d/%d] pid=%s %s -> %d PNG(s) under %s_nodule*.png"
              % (i, len(npz_paths), pid_range, stem, len(boxes), out_prefix))

    print("Done. PNGs in %s" % args.out_dir)


if __name__ == "__main__":
    main()
