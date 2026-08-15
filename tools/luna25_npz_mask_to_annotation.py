#!/usr/bin/env python3
"""Convert LUNA25 nodule masks into original-resolution bbox annotations.

Reads:
  - luna25_npz/*.npz  (1mm cropped working volumes)
  - data/luna25/output/{UID}_nodule_mask.nii.gz  (nodule mask on original CT grid)

Writes:
  - luna25_npz/annotation.csv

Coordinates are inclusive pixel indices on the original CT grid (ZYX).
``pid`` is a stable zero-based index assigned to sorted NPZ files.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_NPZ_DIR = os.path.join(REPO_ROOT, "luna25_npz")
DEFAULT_MASK_DIR = os.path.join(REPO_ROOT, "data", "luna25", "output")
DEFAULT_OUT_CSV = os.path.join(DEFAULT_NPZ_DIR, "annotation.csv")


def _as_scalar_str(value) -> str:
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else str(value)
    return str(value)


def validate_original_mask(nodule_zyx: np.ndarray, d: dict) -> np.ndarray:
    """Validate that a nodule mask is already on the original full CT grid."""
    full_shape = tuple(np.asarray(d["image_shape_original"][0], dtype=int))
    if tuple(nodule_zyx.shape) != full_shape:
        raise ValueError(
            "Nodule mask shape %s != original CT shape %s"
            % (nodule_zyx.shape, full_shape)
        )
    return np.asarray(nodule_zyx)


def extract_bboxes(mask_zyx: np.ndarray, min_voxels: int = 1) -> List[Tuple[int, int, int, int, int, int, int]]:
    """Return list of (voxel_count, z0,y0,x0,z1,y1,x1) with inclusive corners."""
    binary = np.asarray(mask_zyx > 0)
    if not binary.any():
        return []
    labeled, n_comp = ndimage.label(binary)
    boxes = []
    for label_id in range(1, n_comp + 1):
        zz, yy, xx = np.where(labeled == label_id)
        n_vox = int(zz.size)
        if n_vox < min_voxels:
            continue
        boxes.append((
            n_vox,
            int(zz.min()), int(yy.min()), int(xx.min()),
            int(zz.max()), int(yy.max()), int(xx.max()),
        ))
    return boxes


def process_one(npz_path: str, mask_dir: str, pid_start: int) -> List[dict]:
    d = dict(np.load(npz_path, allow_pickle=True))
    uid = _as_scalar_str(d.get("seriesUID", os.path.splitext(os.path.basename(npz_path))[0]))
    mask_path = os.path.join(mask_dir, "%s_nodule_mask.nii.gz" % uid)
    if not os.path.isfile(mask_path):
        raise FileNotFoundError("Missing nodule mask: %s" % mask_path)

    nodule = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    original_mask = validate_original_mask(nodule, d)

    rows = []
    for local_i, (n_vox, z0, y0, x0, z1, y1, x1) in enumerate(
            extract_bboxes(original_mask, min_voxels=1)):
        rows.append({
            "pid": pid_start + local_i,
            "bbox_min_z": z0,
            "bbox_min_y": y0,
            "bbox_min_x": x0,
            "bbox_max_z": z1,
            "bbox_max_y": y1,
            "bbox_max_x": x1,
            "voxel_count": n_vox,
            "seriesUID": uid,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", default=DEFAULT_NPZ_DIR, help="Directory of *_buffer / UID.npz files")
    parser.add_argument("--mask-dir", default=DEFAULT_MASK_DIR, help="Directory of {UID}_nodule_mask.nii.gz")
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="Output annotation CSV path")
    parser.add_argument("--repo-root", default=REPO_ROOT,
                        help="Retained for command-line compatibility; not used for numeric pid")
    parser.add_argument("--keep-meta", action="store_true",
                        help="Keep helper columns voxel_count/seriesUID in CSV")
    args = parser.parse_args()

    npz_files = sorted(
        os.path.join(args.npz_dir, name)
        for name in os.listdir(args.npz_dir)
        if name.endswith(".npz")
    )
    if not npz_files:
        raise FileNotFoundError("No .npz files under %s" % args.npz_dir)

    all_rows = []
    skipped = []
    next_pid = 0
    for npz_index, npz_path in enumerate(npz_files):
        try:
            rows = process_one(npz_path, args.mask_dir, next_pid)
            all_rows.extend(rows)
            pid_end = next_pid + len(rows) - 1
            print("[%d/%d] pid=%d..%d %s -> %d nodule(s)"
                  % (npz_index + 1, len(npz_files), next_pid, pid_end,
                     os.path.basename(npz_path), len(rows)))
            next_pid += len(rows)
        except Exception as exc:
            skipped.append((npz_path, str(exc)))
            print("[%d/%d] SKIP %s: %s"
                  % (npz_index + 1, len(npz_files), os.path.basename(npz_path), exc))

    if not all_rows:
        raise RuntimeError("No annotations produced. Skipped=%d" % len(skipped))

    df = pd.DataFrame(all_rows)
    out_cols = [
        "pid", "bbox_min_z", "bbox_min_y", "bbox_min_x",
        "bbox_max_z", "bbox_max_y", "bbox_max_x",
    ]
    if args.keep_meta:
        out_cols += ["voxel_count", "seriesUID"]
    df = df[out_cols]
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print("Wrote %d boxes from %d npz (skipped %d) -> %s"
          % (len(df), len(npz_files), len(skipped), args.out_csv))
    if skipped:
        print("Skipped examples:")
        for path, err in skipped[:10]:
            print("  %s: %s" % (path, err))


if __name__ == "__main__":
    main()
