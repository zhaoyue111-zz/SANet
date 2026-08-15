#!/usr/bin/env python3
"""Visualize cropped-original-resolution annotations in luna25_organized.

The organized annotation CSV stores bbox corners in the local coordinate
system of ``image_original``.  For each series this script draws five axial
slices around each lesion, matching the layout of
``visualize_luna25_npz_annotation.py``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORGANIZED_DIR = REPO_ROOT / "luna25_organized"
DEFAULT_CSV = DEFAULT_ORGANIZED_DIR / "annotation.csv"
DEFAULT_OUT_DIR = DEFAULT_ORGANIZED_DIR / "vis_anno"


def normalize_pid(value) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def resolve_npz_path(row: pd.Series, organized_dir: Path) -> Path:
    patient_id = str(row["patient_id"])
    study_uid = str(row["studyInstanceUID"])
    series_uid = str(row["seriesInstanceUID"])
    path = organized_dir / patient_id / study_uid / series_uid / f"{series_uid}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot resolve organized NPZ: {path}")
    return path


def _clamp_z(z: int, z_size: int) -> int:
    return int(min(max(z, 0), z_size - 1))


def validate_bbox(row: pd.Series, image_shape: tuple[int, int, int]) -> None:
    minimum = np.array([int(row[f"bbox_min_{axis}"]) for axis in "zyx"])
    maximum = np.array([int(row[f"bbox_max_{axis}"]) for axis in "zyx"])
    shape = np.asarray(image_shape, dtype=int)
    if np.any(minimum < 0) or np.any(maximum >= shape) or np.any(minimum > maximum):
        raise ValueError(
            f"pid={row['pid']} bbox {minimum.tolist()}..{maximum.tolist()} "
            f"is invalid for image_original shape {image_shape}"
        )


def draw_nodule_row(image_zyx: np.ndarray, row: pd.Series, out_path: Path,
                    offset: int = 2, box_pad: int = 2,
                    window_min: float = -1000, window_max: float = 400) -> None:
    """Save mid-2..mid+2 axial slices with one crop-local bbox."""
    z_size, y_size, x_size = image_zyx.shape
    z0, z1 = int(row["bbox_min_z"]), int(row["bbox_max_z"])
    y0, y1 = int(row["bbox_min_y"]), int(row["bbox_max_y"])
    x0, x1 = int(row["bbox_min_x"]), int(row["bbox_max_x"])
    z_mid = _clamp_z(int(round(0.5 * (z0 + z1))), z_size)
    zs = [_clamp_z(z_mid + delta, z_size) for delta in range(-offset, offset + 1)]

    y0_draw = max(0, y0 - box_pad)
    x0_draw = max(0, x0 - box_pad)
    y1_draw = min(y_size - 1, y1 + box_pad)
    x1_draw = min(x_size - 1, x1 + box_pad)

    fig, axes = plt.subplots(1, len(zs), figsize=(3.2 * len(zs), 3.4))
    if len(zs) == 1:
        axes = [axes]
    for ax, z in zip(axes, zs):
        ax.imshow(image_zyx[z], cmap="gray", vmin=window_min, vmax=window_max)
        if z0 <= z <= z1:
            ax.add_patch(patches.Rectangle(
                (x0_draw, y0_draw),
                max(x1_draw - x0_draw, 1),
                max(y1_draw - y0_draw, 1),
                linewidth=1.5,
                edgecolor="lime",
                facecolor="none",
            ))
        title = f"z={z}"
        if z == z_mid:
            title += " (mid)"
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"bbox z[{z0},{z1}] y[{y0},{y1}] x[{x0},{x1}] (draw pad={box_pad})",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def draw_case(image_zyx: np.ndarray, boxes: pd.DataFrame, out_prefix: Path,
              box_pad: int, window_min: float, window_max: float) -> None:
    for local_index, (_, row) in enumerate(boxes.iterrows()):
        draw_nodule_row(
            image_zyx,
            row,
            Path(f"{out_prefix}_nodule{local_index}.png"),
            box_pad=box_pad,
            window_min=window_min,
            window_max=window_max,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--organized-dir", type=Path, default=DEFAULT_ORGANIZED_DIR)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-cases", type=int, default=0,
                        help="Limit series cases (0=all)")
    parser.add_argument("--case-index", type=int, default=None,
                        help="Visualize one sorted series case, 0-based")
    parser.add_argument("--box-pad", type=int, default=2,
                        help="Expand drawn bbox by N pixels for display")
    parser.add_argument("--window-min", type=float, default=-1000,
                        help="HU display window lower bound")
    parser.add_argument("--window-max", type=float, default=400,
                        help="HU display window upper bound")
    parser.add_argument("--pid", default=None,
                        help="Visualize only one linear annotation pid")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = [
        "pid", "patient_id", "studyInstanceUID", "seriesInstanceUID",
        "bbox_min_z", "bbox_min_y", "bbox_min_x",
        "bbox_max_z", "bbox_max_y", "bbox_max_x",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["_pid_key"] = df["pid"].map(normalize_pid)
    if args.pid is not None:
        pid_key = normalize_pid(args.pid)
        df = df[df["_pid_key"] == pid_key]
        if df.empty:
            raise ValueError(f"No rows matched --pid {args.pid}")

    group_columns = ["patient_id", "studyInstanceUID", "seriesInstanceUID"]
    groups = list(df.groupby(group_columns, sort=False))
    if args.case_index is not None:
        if args.case_index < 0 or args.case_index >= len(groups):
            raise ValueError(f"--case-index {args.case_index} outside [0,{len(groups)})")
        groups = [groups[args.case_index]]
    elif args.max_cases > 0:
        groups = groups[:args.max_cases]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index, (_, boxes) in enumerate(groups, 1):
        first_row = boxes.iloc[0]
        npz_path = resolve_npz_path(first_row, args.organized_dir)
        with np.load(npz_path, allow_pickle=True) as data:
            image = np.asarray(data["image_original"], dtype=np.float32)
        for _, row in boxes.iterrows():
            validate_bbox(row, tuple(image.shape))

        series_uid = str(first_row["seriesInstanceUID"])
        out_prefix = args.out_dir / series_uid
        draw_case(image, boxes, out_prefix, args.box_pad, args.window_min, args.window_max)
        pid_values = boxes["_pid_key"].tolist()
        pid_label = pid_values[0] if len(pid_values) == 1 else f"{pid_values[0]}..{pid_values[-1]}"
        print(f"[{index}/{len(groups)}] pid={pid_label} {series_uid} -> "
              f"{len(boxes)} PNG(s) under {out_prefix}_nodule*.png")

    print(f"Done. PNGs in {args.out_dir}")


if __name__ == "__main__":
    main()
