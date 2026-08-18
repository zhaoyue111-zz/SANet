#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize selected REAL RCNN after-NMS boxes as 3D box boundaries in NIfTI.

Input CSV should be produced by test_instance.py, e.g.:
    00291_rcnn_after_nms.csv

Required columns:
    detection_id, x, y, z, dx, dy, dz

Coordinate convention:
    x, y, z = box center in voxel coordinates
    dx       = box size along x
    dy       = box size along y
    dz       = box size along z

Unlike the old visualization based on results.csv:
- no probability matching
- no single diameter
- no cube approximation
- box geometry uses the true RCNN-regressed x/y/z sizes
- only the six box boundary faces are painted, not the solid interior

Each selected detection is assigned mask label 1, 2, 3, ...
in the same order given to --detection-ids.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize selected RCNN after-NMS detections using the real "
            "x,y,z,dx,dy,dz box geometry as NIfTI box boundaries."
        )
    )

    parser.add_argument(
        "--boxes-csv",
        type=Path,
        required=True,
        help=(
            "Path to <pid>_rcnn_after_nms.csv with columns "
            "detection_id,x,y,z,dx,dy,dz."
        ),
    )

    parser.add_argument(
        "--pid",
        required=True,
        help="Case id used to locate the CT volume, e.g. 00291.",
    )

    parser.add_argument(
        "--detection-ids",
        type=int,
        nargs="+",
        required=True,
        help=(
            "detection_id values to visualize, in paint order. "
            "Example: --detection-ids 0 3 7"
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Dataset root, e.g. /mnt/afs2/data/LUNA16. "
            "Looks for full/<pid>_zoom.npy or <pid>_zoom.npy."
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output boundary mask .nii.gz. "
            "Default: <out-dir>/<pid>_rcnn_boxes_boundary.nii.gz"
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("rcnn_box_mask_nii"),
        help="Output directory when --out is omitted.",
    )

    parser.add_argument(
        "--boundary-thickness",
        type=int,
        default=1,
        help=(
            "Boundary thickness in voxels. Default: 1. "
            "Use 2 or 3 if the boundary is hard to see in ITK-SNAP."
        ),
    )

    parser.add_argument(
        "--min-box-size",
        type=float,
        default=1.0,
        help="Minimum box size per axis in voxels. Default: 1.0.",
    )

    parser.add_argument(
        "--save-ct",
        action="store_true",
        help="Also save the CT volume as .nii.gz next to the mask.",
    )

    return parser.parse_args()


def normalize_pid(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def safe_pid(value) -> str:
    pid = normalize_pid(value)
    return pid.zfill(5) if pid.isdigit() else pid


def pid_candidates(pid: str) -> list[str]:
    raw = normalize_pid(pid)
    cands = [raw, safe_pid(raw)]

    if raw.isdigit():
        cands.extend([
            str(int(raw)),
            raw.zfill(5),
            raw.zfill(6),
        ])

    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)

    return out


def find_volume_path(data_dir: Path, pid: str) -> Path:
    for name in pid_candidates(pid):
        candidates = [
            data_dir / "full" / ("%s_zoom.npy" % name),
            data_dir / ("%s_zoom.npy" % name),
        ]

        for path in candidates:
            if path.exists():
                return path

    tried = []
    for name in pid_candidates(pid):
        tried.append(str(data_dir / "full" / ("%s_zoom.npy" % name)))
        tried.append(str(data_dir / ("%s_zoom.npy" % name)))

    raise FileNotFoundError(
        "CT npy not found for pid=%s under %s. Tried:\n  %s"
        % (pid, data_dir, "\n  ".join(tried))
    )


def load_volume(path: Path) -> np.ndarray:
    volume = np.load(path)

    if volume.ndim == 4:
        volume = volume[0]

    if volume.ndim != 3:
        raise ValueError(
            "Expected 3D CT volume in %s, got shape=%s"
            % (path, volume.shape)
        )

    return np.asarray(volume)


def load_spacing(data_dir: Path, pid: str) -> tuple[float, float, float]:
    meta_path = data_dir / "meta.csv"

    if not meta_path.exists():
        return (1.0, 1.0, 1.0)

    meta = pd.read_csv(meta_path)

    if (
        "sanet_pid" not in meta.columns
        or "processed_spacing_xyz" not in meta.columns
    ):
        return (1.0, 1.0, 1.0)

    targets = set(pid_candidates(pid))

    matched = meta[
        meta["sanet_pid"].map(normalize_pid).isin(targets)
    ]

    if matched.empty:
        return (1.0, 1.0, 1.0)

    raw = str(matched.iloc[0]["processed_spacing_xyz"])
    parts = [
        float(x.strip())
        for x in raw.split(",")
        if x.strip()
    ]

    if len(parts) != 3:
        return (1.0, 1.0, 1.0)

    # SimpleITK spacing order is x,y,z
    return tuple(parts)


def select_rows(
    df: pd.DataFrame,
    detection_ids: list[int],
) -> pd.DataFrame:

    required = {
        "detection_id",
        "x",
        "y",
        "z",
        "dx",
        "dy",
        "dz",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "RCNN after-NMS CSV missing columns: %s"
            % ", ".join(sorted(missing))
        )

    work = df.copy()
    work["detection_id"] = (
        pd.to_numeric(work["detection_id"], errors="raise")
        .astype(int)
    )

    duplicated = work["detection_id"].duplicated(keep=False)
    if duplicated.any():
        dup_ids = sorted(
            work.loc[duplicated, "detection_id"].unique().tolist()
        )
        raise ValueError(
            "detection_id is not unique in CSV. Duplicated ids: %s"
            % dup_ids
        )

    selected = []

    for label, detection_id in enumerate(detection_ids, start=1):
        matched = work[
            work["detection_id"] == int(detection_id)
        ]

        if matched.empty:
            available = sorted(
                work["detection_id"].astype(int).tolist()
            )
            raise ValueError(
                "detection_id=%d not found. Available detection_ids: %s"
                % (detection_id, available)
            )

        row = matched.iloc[0].copy()
        row["label"] = label
        selected.append(row)

    return pd.DataFrame(selected)


def real_box_bounds(
    row: pd.Series,
    volume_shape: tuple[int, int, int],
    min_box_size: float,
) -> tuple[int, int, int, int, int, int]:
    """
    Convert the TRUE RCNN center-size box to integer voxel bounds.

    Input:
        x,y,z,dx,dy,dz

    Floating boundaries:
        xmin = x - dx/2
        xmax = x + dx/2
        ...

    Integer mask range:
        lower = floor(min)
        upper = ceil(max)

    Returned order:
        z0,z1,y0,y1,x0,x1
    """
    depth, height, width = volume_shape

    x = float(row["x"])
    y = float(row["y"])
    z = float(row["z"])

    dx = max(float(row["dx"]), min_box_size)
    dy = max(float(row["dy"]), min_box_size)
    dz = max(float(row["dz"]), min_box_size)

    xmin = x - dx / 2.0
    xmax = x + dx / 2.0

    ymin = y - dy / 2.0
    ymax = y + dy / 2.0

    zmin = z - dz / 2.0
    zmax = z + dz / 2.0

    x0 = int(np.floor(xmin))
    x1 = int(np.ceil(xmax))

    y0 = int(np.floor(ymin))
    y1 = int(np.ceil(ymax))

    z0 = int(np.floor(zmin))
    z1 = int(np.ceil(zmax))

    x0 = max(0, min(x0, width - 1))
    x1 = max(0, min(x1, width - 1))

    y0 = max(0, min(y0, height - 1))
    y1 = max(0, min(y1, height - 1))

    z0 = max(0, min(z0, depth - 1))
    z1 = max(0, min(z1, depth - 1))

    # Ensure a valid inclusive range.
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if z1 < z0:
        z0, z1 = z1, z0

    return z0, z1, y0, y1, x0, x1


def paint_box_boundary(
    mask: np.ndarray,
    bounds: tuple[int, int, int, int, int, int],
    label: int,
    thickness: int,
) -> None:
    """
    Paint the six faces of an axis-aligned 3D box.

    mask indexing:
        mask[z, y, x]

    Bounds are inclusive:
        z0,z1,y0,y1,x0,x1
    """
    z0, z1, y0, y1, x0, x1 = bounds

    t = max(1, int(thickness))

    # Limit thickness so that slicing remains valid.
    tz = min(t, z1 - z0 + 1)
    ty = min(t, y1 - y0 + 1)
    tx = min(t, x1 - x0 + 1)

    value = np.uint16(label)

    # z-min / z-max faces
    mask[
        z0 : z0 + tz,
        y0 : y1 + 1,
        x0 : x1 + 1,
    ] = value

    mask[
        z1 - tz + 1 : z1 + 1,
        y0 : y1 + 1,
        x0 : x1 + 1,
    ] = value

    # y-min / y-max faces
    mask[
        z0 : z1 + 1,
        y0 : y0 + ty,
        x0 : x1 + 1,
    ] = value

    mask[
        z0 : z1 + 1,
        y1 - ty + 1 : y1 + 1,
        x0 : x1 + 1,
    ] = value

    # x-min / x-max faces
    mask[
        z0 : z1 + 1,
        y0 : y1 + 1,
        x0 : x0 + tx,
    ] = value

    mask[
        z0 : z1 + 1,
        y0 : y1 + 1,
        x1 - tx + 1 : x1 + 1,
    ] = value


def write_nii(
    path: Path,
    volume: np.ndarray,
    spacing_xyz: tuple[float, float, float],
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    image = sitk.GetImageFromArray(
        np.ascontiguousarray(volume)
    )

    image.SetSpacing(
        tuple(float(x) for x in spacing_xyz)
    )

    sitk.WriteImage(
        image,
        str(path),
        useCompression=True,
    )


def normalize_ct_to_uint8(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume)

    if (
        arr.dtype == np.uint8
        and arr.min() >= 0
        and arr.max() <= 255
    ):
        return arr

    finite = np.isfinite(arr)

    if not finite.any():
        return np.zeros(
            arr.shape,
            dtype=np.uint8,
        )

    valid = arr[finite]

    low, high = np.percentile(
        valid,
        [1, 99],
    )

    if high <= low:
        low = float(valid.min())
        high = float(valid.max())

    if high <= low:
        return np.zeros(
            arr.shape,
            dtype=np.uint8,
        )

    out = (
        np.clip(
            (arr - low) / (high - low),
            0.0,
            1.0,
        )
        * 255.0
    )

    return out.astype(np.uint8)


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.boxes_csv)

    rows = select_rows(
        df,
        args.detection_ids,
    )

    volume_path = find_volume_path(
        args.data_dir,
        args.pid,
    )

    volume = load_volume(volume_path)

    spacing = load_spacing(
        args.data_dir,
        args.pid,
    )

    # uint16 allows many different detection labels safely.
    mask = np.zeros(
        volume.shape,
        dtype=np.uint16,
    )

    print(
        "CT: %s  shape=%s  spacing_xyz=%s"
        % (
            volume_path,
            volume.shape,
            spacing,
        )
    )

    print(
        "Boxes CSV: %s"
        % args.boxes_csv
    )

    print(
        "Selected detection_ids: %s"
        % args.detection_ids
    )

    print("")

    for _, row in rows.iterrows():

        label = int(row["label"])
        detection_id = int(row["detection_id"])

        bounds = real_box_bounds(
            row,
            volume.shape,
            args.min_box_size,
        )

        paint_box_boundary(
            mask,
            bounds,
            label,
            args.boundary_thickness,
        )

        z0, z1, y0, y1, x0, x1 = bounds

        x = float(row["x"])
        y = float(row["y"])
        z = float(row["z"])

        dx = float(row["dx"])
        dy = float(row["dy"])
        dz = float(row["dz"])

        print(
            "label=%d detection_id=%d"
            % (
                label,
                detection_id,
            )
        )

        if "rcnn_score" in row.index:
            print(
                "  rcnn_score=%.8f"
                % float(row["rcnn_score"])
            )

        print(
            "  center xyz = (%.4f, %.4f, %.4f)"
            % (
                x,
                y,
                z,
            )
        )

        print(
            "  size   xyz = (dx=%.4f, dy=%.4f, dz=%.4f)"
            % (
                dx,
                dy,
                dz,
            )
        )

        print(
            "  float bounds:"
        )

        print(
            "    x=[%.4f, %.4f]"
            % (
                x - dx / 2.0,
                x + dx / 2.0,
            )
        )

        print(
            "    y=[%.4f, %.4f]"
            % (
                y - dy / 2.0,
                y + dy / 2.0,
            )
        )

        print(
            "    z=[%.4f, %.4f]"
            % (
                z - dz / 2.0,
                z + dz / 2.0,
            )
        )

        print(
            "  painted voxel bounds:"
        )

        print(
            "    x=[%d, %d], y=[%d, %d], z=[%d, %d]"
            % (
                x0,
                x1,
                y0,
                y1,
                z0,
                z1,
            )
        )

        print("")

    out_mask = args.out

    if out_mask is None:
        args.out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        out_mask = (
            args.out_dir
            / (
                "%s_rcnn_boxes_boundary.nii.gz"
                % safe_pid(args.pid)
            )
        )

    write_nii(
        out_mask,
        mask,
        spacing,
    )

    labels = sorted(
        int(x)
        for x in np.unique(mask)
        if x > 0
    )

    print(
        "Saved boundary mask: %s"
        % out_mask
    )

    print(
        "Mask shape=%s labels=%s"
        % (
            mask.shape,
            labels,
        )
    )

    print("")
    print("Label -> detection_id mapping:")

    for _, row in rows.iterrows():
        print(
            "  label %d -> detection_id %d"
            % (
                int(row["label"]),
                int(row["detection_id"]),
            )
        )

    if args.save_ct:

        if out_mask.name.endswith(
            "_rcnn_boxes_boundary.nii.gz"
        ):
            prefix = out_mask.name[
                : -len(
                    "_rcnn_boxes_boundary.nii.gz"
                )
            ]

            out_ct = out_mask.with_name(
                prefix + "_ct.nii.gz"
            )

        else:
            out_ct = out_mask.with_name(
                "%s_ct.nii.gz"
                % safe_pid(args.pid)
            )

        write_nii(
            out_ct,
            normalize_ct_to_uint8(volume),
            spacing,
        )

        print(
            "Saved CT: %s  shape=%s"
            % (
                out_ct,
                volume.shape,
            )
        )


if __name__ == "__main__":
    main()

'''
python tools/visualize_rcnn_after_nms_boxes_nii.py \
  --boxes-csv /mnt/afs2/code/SANet/test_instance_output/00291_rcnn_after_nms.csv \
  --pid 00291 \
  --detection-ids 0 11 41 \
  --data-dir /mnt/afs2/data/LUNA16 \
  --out-dir /mnt/afs2/code/SANet/test_instance_output \
  --save-ct
'''