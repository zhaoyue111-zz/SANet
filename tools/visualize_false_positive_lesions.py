#!/usr/bin/env python3
"""Visualize false-positive detections as five-slice row montages.

The script scans a SANet result directory that contains per-dataset FROC
outputs. For each false positive it saves one PNG containing z-2, z-1, z,
z+1, z+2 crops arranged in a single row.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

'''
遍历 /data/医保大赛/code/SANet/test_output/res/14_pretrained_rcnn20 下每个
  数据集，优先读取 FROC/res_froc_eval/FPs.csv 定位假阳病灶，然后从 data/
  <dataset>/full/<pid>_zoom.npy 取中心层 z 及上下两层 z-2..z+2，横向拼成一张
  PNG；每个假阳病灶保存一张图，并生成索引 CSV。
'''

DEFAULT_RESULT_DIR = Path("/mnt/afs2/code/SANet/test_output/res/14_pretrained_rcnn20_th05_nms01")
DEFAULT_DATA_ROOT = Path("/mnt/afs2/data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save five-slice row images for false-positive lesions."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="SANet result directory containing dataset subdirectories.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing SANet-ready datasets.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <result-dir>/false_positive_visualizations.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to process. Defaults to all datasets in result-dir.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=80,
        help="Square crop size around the FP center in pixels.",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.0,
        help="Only visualize false positives with probability >= this value.",
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=None,
        help="Optional maximum number of false positives per dataset, sorted by probability.",
    )
    parser.add_argument(
        "--no-marker",
        action="store_true",
        help="Do not draw the FP lesion box.",
    )
    parser.add_argument(
        "--box-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the FP diameter when drawing the box.",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=8.0,
        help="Minimum drawn box side length in pixels.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if a referenced preprocessed volume is missing.",
    )
    return parser.parse_args()


def read_false_positives(dataset_result_dir: Path) -> pd.DataFrame:
    fp_path = dataset_result_dir / "FROC" / "res_froc_eval_diamter05" / "FPs.csv"
    if fp_path.exists():
        fps = pd.read_csv(fp_path, dtype={"seriesuid": str})
        return fps.rename(
            columns={
                "seriesuid": "pid",
                "coordX": "center_x",
                "coordY": "center_y",
                "coordZ": "center_z",
                "radius": "diameter",
            }
        )

    results_path = dataset_result_dir / "FROC" / "results.csv"
    annotations_path = dataset_result_dir / "FROC" / "annotations.csv"
    if not results_path.exists() or not annotations_path.exists():
        raise FileNotFoundError(
            "missing FPs.csv and fallback files under %s" % dataset_result_dir
        )

    preds = pd.read_csv(results_path, dtype={"pid": str})
    annos = pd.read_csv(annotations_path, dtype={"pid": str})
    return fallback_fp_match(preds, annos)


def fallback_fp_match(preds: pd.DataFrame, annos: pd.DataFrame) -> pd.DataFrame:
    """Fallback FP matcher compatible with the LUNA-style center criterion."""
    rows = []
    for pid, pred_group in preds.groupby("pid"):
        gt_group = annos[annos["pid"] == pid]
        gt_centers = gt_group[["center_x", "center_y", "center_z"]].to_numpy(float)
        gt_radii = gt_group["diameter"].to_numpy(float) / 2.0

        for _, pred in pred_group.iterrows():
            if len(gt_centers) == 0:
                rows.append(pred)
                continue

            center = pred[["center_x", "center_y", "center_z"]].to_numpy(float)
            distances = np.sqrt(np.sum((gt_centers - center) ** 2, axis=1))
            if not np.any(distances <= gt_radii):
                rows.append(pred)

    if not rows:
        return pd.DataFrame(columns=preds.columns)
    return pd.DataFrame(rows)


def load_volume(data_root: Path, dataset_name: str, pid: str) -> np.ndarray:
    image_path = data_root / dataset_name / "full" / ("%s_zoom.npy" % pid)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    volume = np.load(image_path)
    if volume.ndim == 4:
        volume = volume[0]
    if volume.ndim != 3:
        raise ValueError("expected 3D volume in %s, got %s" % (image_path, volume.shape))
    return volume


def normalize_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(slice_2d, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    valid = arr[finite]
    low, high = np.percentile(valid, [1, 99])
    if high <= low:
        low, high = float(valid.min()), float(valid.max())
    if high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def crop_around_center(slice_2d: np.ndarray, x: float, y: float, crop_size: int) -> np.ndarray:
    h, w = slice_2d.shape
    half = crop_size // 2
    cx = int(round(x))
    cy = int(round(y))

    out = np.zeros((crop_size, crop_size), dtype=slice_2d.dtype)
    want_x0 = cx - half
    want_y0 = cy - half
    want_x1 = want_x0 + crop_size
    want_y1 = want_y0 + crop_size

    src_x0 = max(want_x0, 0)
    src_y0 = max(want_y0, 0)
    src_x1 = min(want_x1, w)
    src_y1 = min(want_y1, h)

    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out

    dst_x0 = src_x0 - want_x0
    dst_y0 = src_y0 - want_y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = slice_2d[src_y0:src_y1, src_x0:src_x1]
    return out


def draw_centered_lesion_box(
        draw: ImageDraw.ImageDraw,
        crop_size: int,
        diameter: float,
        box_scale: float,
        min_box_size: float,
) -> None:
    side = max(float(diameter) * float(box_scale), float(min_box_size))
    side = min(side, crop_size - 2)
    cx = crop_size / 2.0
    cy = crop_size / 2.0
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = int(round(cx + side / 2.0))
    y1 = int(round(cy + side / 2.0))
    draw.rectangle((x0, y0, x1, y1), outline=(255, 64, 64), width=2)


def draw_slice_lesion_box(
        draw: ImageDraw.ImageDraw,
        x: float,
        y: float,
        image_width: int,
        image_height: int,
        diameter: float,
        box_scale: float,
        min_box_size: float,
) -> None:
    side = max(float(diameter) * float(box_scale), float(min_box_size))
    x0 = int(round(float(x) - side / 2.0))
    y0 = int(round(float(y) - side / 2.0))
    x1 = int(round(float(x) + side / 2.0))
    y1 = int(round(float(y) + side / 2.0))
    x0 = max(0, min(x0, image_width - 1))
    y0 = max(0, min(y0, image_height - 1))
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    draw.rectangle((x0, y0, x1, y1), outline=(255, 64, 64), width=2)


def make_montage(
        volume: np.ndarray,
        row: pd.Series,
        crop_size: int,
        draw_box: bool,
        box_scale: float,
        min_box_size: float,
) -> Image.Image:
    x = float(row["center_x"])
    y = float(row["center_y"])
    z = int(round(float(row["center_z"])))
    diameter = float(row["diameter"])
    z_count = volume.shape[0]
    z_indices = [min(max(z + offset, 0), z_count - 1) for offset in [-2, -1, 0, 1, 2]]

    label_height = 22
    slice_h, slice_w = volume[z_indices[0]].shape
    top_width = slice_w * len(z_indices)
    bottom_width = crop_size * len(z_indices)
    montage_width = max(top_width, bottom_width)
    montage_height = slice_h + crop_size + label_height * 2
    top_x0 = (montage_width - top_width) // 2
    bottom_x0 = (montage_width - bottom_width) // 2
    bottom_y0 = slice_h + label_height
    montage = Image.new("RGB", (montage_width, montage_height), "black")
    montage_draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default()

    for col, z_idx in enumerate(z_indices):
        full_img = Image.fromarray(normalize_to_uint8(volume[z_idx]), mode="L").convert("RGB")
        if draw_box:
            draw_slice_lesion_box(
                ImageDraw.Draw(full_img),
                x=x,
                y=y,
                image_width=slice_w,
                image_height=slice_h,
                diameter=diameter,
                box_scale=box_scale,
                min_box_size=min_box_size,
            )
        full_x = top_x0 + col * slice_w
        montage.paste(full_img, (full_x, 0))
        montage_draw.text((full_x + 4, slice_h + 4), "z=%d" % z_idx, fill=(230, 230, 230), font=font)

        crop = crop_around_center(volume[z_idx], x, y, crop_size)
        crop_img = Image.fromarray(normalize_to_uint8(crop), mode="L").convert("RGB")
        if draw_box:
            draw_centered_lesion_box(
                ImageDraw.Draw(crop_img),
                crop_size=crop_size,
                diameter=diameter,
                box_scale=box_scale,
                min_box_size=min_box_size,
            )
        crop_x = bottom_x0 + col * crop_size
        montage.paste(crop_img, (crop_x, bottom_y0))
        montage_draw.text(
            (crop_x + 4, bottom_y0 + crop_size + 4),
            "z=%d zoom" % z_idx,
            fill=(230, 230, 230),
            font=font,
        )

    return montage


def safe_pid(value: object) -> str:
    pid = str(value)
    if pid.endswith(".0"):
        pid = pid[:-2]
    return pid.zfill(5) if pid.isdigit() else pid


def discover_datasets(result_dir: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return sorted(
        p.name for p in result_dir.iterdir() if p.is_dir() and (p / "FROC").is_dir()
    )


def process_dataset(
        result_dir: Path,
        data_root: Path,
        out_dir: Path,
        dataset_name: str,
        crop_size: int,
        min_prob: float,
        max_per_dataset: int | None,
        draw_box: bool,
        box_scale: float,
        min_box_size: float,
        strict: bool,
) -> list[dict[str, object]]:
    dataset_result_dir = result_dir / dataset_name
    fps = read_false_positives(dataset_result_dir)
    if fps.empty:
        return []

    fps["pid"] = fps["pid"].map(safe_pid)
    fps = fps[fps["probability"].astype(float) >= min_prob].copy()
    fps = fps.sort_values("probability", ascending=False)
    if max_per_dataset is not None:
        fps = fps.head(max_per_dataset)

    dataset_out_dir = out_dir / dataset_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    volume_cache: dict[str, np.ndarray] = {}
    records = []
    missing = set()
    for index, (_, row) in enumerate(fps.iterrows(), start=1):
        pid = row["pid"]
        if pid in missing:
            continue
        if pid not in volume_cache:
            try:
                volume_cache[pid] = load_volume(data_root, dataset_name, pid)
            except FileNotFoundError as exc:
                if strict:
                    raise
                print("warning: skip missing volume %s" % exc)
                missing.add(pid)
                continue

        image = make_montage(
            volume=volume_cache[pid],
            row=row,
            crop_size=crop_size,
            draw_box=draw_box,
            box_scale=box_scale,
            min_box_size=min_box_size,
        )
        stem = "%04d_%s_prob%.4f_x%.1f_y%.1f_z%.1f" % (
            index,
            pid,
            float(row["probability"]),
            float(row["center_x"]),
            float(row["center_y"]),
            float(row["center_z"]),
        )
        out_path = dataset_out_dir / (stem.replace("/", "_") + ".png")
        image.save(out_path)
        records.append(
            {
                "dataset": dataset_name,
                "pid": pid,
                "center_x": float(row["center_x"]),
                "center_y": float(row["center_y"]),
                "center_z": float(row["center_z"]),
                "diameter_or_radius": float(row["diameter"]),
                "probability": float(row["probability"]),
                "image_path": str(out_path),
            }
        )

    return records


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    data_root = args.data_root.resolve()
    out_dir = (args.out_dir or (result_dir / "false_positive_visualizations")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for dataset_name in discover_datasets(result_dir, args.datasets):
        records = process_dataset(
            result_dir=result_dir,
            data_root=data_root,
            out_dir=out_dir,
            dataset_name=dataset_name,
            crop_size=args.crop_size,
            min_prob=args.min_prob,
            max_per_dataset=args.max_per_dataset,
            draw_box=not args.no_marker,
            box_scale=args.box_scale,
            min_box_size=args.min_box_size,
            strict=args.strict,
        )

        print("%s: saved %d false-positive montage(s)" % (dataset_name, len(records)))
        all_records.extend(records)

    summary_path = out_dir / "false_positive_visualizations.csv"
    with open(summary_path, "w", newline="") as f:
        fieldnames = [
            "dataset",
            "pid",
            "center_x",
            "center_y",
            "center_z",
            "diameter_or_radius",
            "probability",
            "image_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print("summary: %s" % summary_path)


if __name__ == "__main__":
    main()

'''
python tools/visualize_false_positive_lesions.py \
    --result-dir test_output/res/14_pretrained_rcnn20 \
    --data-root data \
    --max-per-dataset 100

python tools/visualize_false_positive_lesions.py \
    --result-dir test_output/res/14_pretrained_rcnn20 \
    --data-root data \
    --min-prob 0.9
'''
