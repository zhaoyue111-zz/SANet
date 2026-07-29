"""Visualize original Histopathology BMP annotations without slice-order ambiguity."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


def numeric_key(path: Path):
    match = re.search(r"\d+", path.stem)
    return int(match.group(0)) if match else path.stem


def visualize_slice(
    bmp_dir: str | Path,
    csv_path: str | Path,
    case_id: int,
    slice_number: int,
    output: str | Path | None = None,
) -> Path:
    case_name = f"{case_id:04d}"
    case_dir = Path(bmp_dir) / case_name
    slice_paths = sorted(case_dir.glob("*.bmp"), key=numeric_key)
    by_number = {numeric_key(path): path for path in slice_paths}
    if slice_number not in by_number:
        raise FileNotFoundError(
            f"Slice {slice_number}.bmp not found under {case_dir}"
        )

    image_path = by_number[slice_number]
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read {image_path}")

    annotation_name = f"{case_name}_{slice_number}.bmp"
    annotations = pd.read_csv(csv_path)
    rows = annotations[annotations["image"].astype(str) == annotation_name]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray", origin="upper")
    for _, row in rows.iterrows():
        ax.add_patch(
            Rectangle(
                (row.x_min, row.y_min),
                row.x_max - row.x_min + 1,
                row.y_max - row.y_min + 1,
                fill=False,
                edgecolor="lime",
                linewidth=2,
            )
        )
    ax.set_title(
        f"Original case {case_name}, slice file {image_path.name}, "
        f"annotations={len(rows)}"
    )
    ax.axis("off")

    output_path = (
        Path(output)
        if output
        else Path("visualizations")
        / f"histopathology_original_{case_name}_{slice_number}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Image: {image_path}")
    print(f"Annotation key: {annotation_name}, rows={len(rows)}")
    print(f"Saved: {output_path}")
    return output_path


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bmp-dir",
        default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/Histopathology/BMP_3D",
    )
    parser.add_argument(
        "--csv",
        default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/Histopathology/all_anno_3D.csv",
    )
    parser.add_argument("--case-id", type=int, default=61)
    parser.add_argument("--slice", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    visualize_slice(
        args.bmp_dir,
        args.csv,
        args.case_id,
        args.slice,
        args.output,
    )
