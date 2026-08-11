#!/usr/bin/env python3
"""Visualize and summarize PN9 lesions recovered by ensemble scoring."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_RCNN_DIR = Path("test_output/res/48_pretrained_hardsamples_rcnn40_PN11_v3_lt1002")
DEFAULT_ENSEMBLE_DIR = Path("test_output/res/48_pretrained_hardsamples_rcnn40_PN11_v3_lt1002_ensemble")
DEFAULT_DATA_DIR = Path("data/PN9")
DEFAULT_OUT_DIR = Path("test_output/analysis/pn9_recovered_by_ensemble")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze PN9 FNs recovered by RPN+RCNN weighted scoring."
    )
    parser.add_argument("--rcnn-dir", type=Path, default=DEFAULT_RCNN_DIR)
    parser.add_argument("--ensemble-dir", type=Path, default=DEFAULT_ENSEMBLE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--eval-subdir",
        default="res_froc_eval_diamter05",
        help="FROC result subdir containing FNs.csv.",
    )
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--min-box-size", type=float, default=8.0)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Optional per-case top-K CAD marks to keep before analysis. Default: disabled.",
    )
    parser.add_argument(
        "--strict-volumes",
        action="store_true",
        help="Fail when a recovered lesion CT volume is missing.",
    )
    return parser.parse_args()


def normalize_pid(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5) if text.isdigit() else text


def read_fns(result_dir: Path, eval_subdir: str) -> pd.DataFrame:
    path = result_dir / "PN9" / "FROC" / eval_subdir / "FNs.csv"
    df = pd.read_csv(path, dtype={"seriesuid": str})
    df["pid"] = df["seriesuid"].map(normalize_pid)
    for col in ["coordX", "coordY", "coordZ", "radius"]:
        df[col] = df[col].astype(float)
    return df


def fn_key(row: pd.Series) -> tuple[object, ...]:
    return (
        normalize_pid(row["seriesuid"]),
        round(float(row["coordX"]), 3),
        round(float(row["coordY"]), 3),
        round(float(row["coordZ"]), 3),
    )


def load_annotations(result_dir: Path) -> pd.DataFrame:
    path = result_dir / "PN9" / "FROC" / "annotations_froc_eval.csv"
    df = pd.read_csv(path, dtype={"pid": str})
    df["pid"] = df["pid"].map(normalize_pid)
    for col in ["center_x", "center_y", "center_z", "diameter"]:
        df[col] = df[col].astype(float)
    return df


def load_results(result_dir: Path) -> pd.DataFrame:
    path = result_dir / "PN9" / "FROC" / "results_froc_eval.csv"
    df = pd.read_csv(path, dtype={"pid": str})
    df["pid"] = df["pid"].map(normalize_pid)
    for col in ["center_x", "center_y", "center_z", "diameter", "probability"]:
        df[col] = df[col].astype(float)
    return df


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
    out[dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = (
        slice_2d[src_y0:src_y1, src_x0:src_x1]
    )
    return out


def draw_box(draw: ImageDraw.ImageDraw, center_x: float, center_y: float, side: float, width: int, height: int) -> None:
    half = side / 2.0
    x0 = max(0, min(int(round(center_x - half)), width - 1))
    y0 = max(0, min(int(round(center_y - half)), height - 1))
    x1 = max(0, min(int(round(center_x + half)), width - 1))
    y1 = max(0, min(int(round(center_y + half)), height - 1))
    draw.rectangle((x0, y0, x1, y1), outline=(255, 64, 64), width=2)


def make_montage(volume: np.ndarray, gt: pd.Series, crop_size: int, min_box_size: float) -> Image.Image:
    x = float(gt["center_x"])
    y = float(gt["center_y"])
    z = int(round(float(gt["center_z"])))
    diameter = max(float(gt["diameter"]), min_box_size)
    z_indices = [min(max(z + offset, 0), volume.shape[0] - 1) for offset in [-2, -1, 0, 1, 2]]

    full_h, full_w = volume[z_indices[0]].shape
    label_h = 22
    montage = Image.new("RGB", (max(full_w * 5, crop_size * 5), full_h + crop_size + label_h * 2), "black")
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(montage)
    top_x0 = (montage.width - full_w * 5) // 2
    crop_x0 = (montage.width - crop_size * 5) // 2
    crop_y0 = full_h + label_h

    for col, z_idx in enumerate(z_indices):
        full = Image.fromarray(normalize_to_uint8(volume[z_idx]), mode="L").convert("RGB")
        draw_box(ImageDraw.Draw(full), x, y, diameter, full_w, full_h)
        full_x = top_x0 + col * full_w
        montage.paste(full, (full_x, 0))
        draw.text((full_x + 4, full_h + 4), "z=%d" % z_idx, fill=(230, 230, 230), font=font)

        crop = crop_around_center(volume[z_idx], x, y, crop_size)
        crop_img = Image.fromarray(normalize_to_uint8(crop), mode="L").convert("RGB")
        draw_box(
            ImageDraw.Draw(crop_img),
            crop_size / 2.0,
            crop_size / 2.0,
            diameter,
            crop_size,
            crop_size,
        )
        crop_x = crop_x0 + col * crop_size
        montage.paste(crop_img, (crop_x, crop_y0))
        draw.text((crop_x + 4, crop_y0 + crop_size + 4), "z=%d crop" % z_idx, fill=(230, 230, 230), font=font)

    return montage


def candidate_stats(results: pd.DataFrame, gt: pd.Series, max_candidates: int) -> dict[str, float | int | str]:
    case = results[results["pid"] == gt["pid"]].copy()
    if max_candidates > 0 and len(case) > max_candidates:
        case = case.sort_values("probability", ascending=False).head(max_candidates).copy()
    radius = float(gt["diameter"]) / 2.0
    if case.empty:
        return {
            "num_candidates": 0,
            "num_hits": 0,
            "best_hit_prob": math.nan,
            "best_hit_dist": math.nan,
            "nearest_dist": math.nan,
            "nearest_prob": math.nan,
            "nearest_margin": math.nan,
        }
    dx = case["center_x"] - float(gt["center_x"])
    dy = case["center_y"] - float(gt["center_y"])
    dz = case["center_z"] - float(gt["center_z"])
    case["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)
    hits = case[case["dist"] < radius].sort_values("probability", ascending=False)
    nearest = case.sort_values("dist").iloc[0]
    out = {
        "num_candidates": int(len(case)),
        "num_hits": int(len(hits)),
        "best_hit_prob": math.nan,
        "best_hit_dist": math.nan,
        "nearest_dist": float(nearest["dist"]),
        "nearest_prob": float(nearest["probability"]),
        "nearest_margin": float(nearest["dist"] - radius),
    }
    if not hits.empty:
        best = hits.iloc[0]
        out["best_hit_prob"] = float(best["probability"])
        out["best_hit_dist"] = float(best["dist"])
    return out


def lesion_texture_stats(volume: np.ndarray, gt: pd.Series) -> dict[str, float]:
    x = int(round(float(gt["center_x"])))
    y = int(round(float(gt["center_y"])))
    z = int(round(float(gt["center_z"])))
    radius = max(2, int(round(float(gt["diameter"]) / 2.0)))
    z0, z1 = max(0, z - 1), min(volume.shape[0], z + 2)
    y0, y1 = max(0, y - radius), min(volume.shape[1], y + radius + 1)
    x0, x1 = max(0, x - radius), min(volume.shape[2], x + radius + 1)
    patch = np.asarray(volume[z0:z1, y0:y1, x0:x1], dtype=np.float32)
    return {
        "patch_mean": float(np.mean(patch)),
        "patch_std": float(np.std(patch)),
        "patch_p95_minus_p05": float(np.percentile(patch, 95) - np.percentile(patch, 5)),
    }


def missing_texture_stats() -> dict[str, float]:
    return {
        "patch_mean": math.nan,
        "patch_std": math.nan,
        "patch_p95_minus_p05": math.nan,
    }


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rcnn_fns = read_fns(args.rcnn_dir, args.eval_subdir)
    ensemble_fns = read_fns(args.ensemble_dir, args.eval_subdir)
    ensemble_keys = {fn_key(row) for _, row in ensemble_fns.iterrows()}
    recovered_fns = [row for _, row in rcnn_fns.iterrows() if fn_key(row) not in ensemble_keys]

    annotations = load_annotations(args.rcnn_dir)
    rcnn_results = load_results(args.rcnn_dir)
    ensemble_results = load_results(args.ensemble_dir)

    records = []
    volume_cache: dict[str, np.ndarray] = {}
    for i, fn in enumerate(recovered_fns, start=1):
        pid = normalize_pid(fn["seriesuid"])
        matches = annotations[
            (annotations["pid"] == pid)
            & np.isclose(annotations["center_x"], float(fn["coordX"]))
            & np.isclose(annotations["center_y"], float(fn["coordY"]))
            & np.isclose(annotations["center_z"], float(fn["coordZ"]))
        ]
        if matches.empty:
            raise ValueError("No annotation match for recovered FN %s" % (fn_key(fn),))
        gt = matches.iloc[0].copy()
        image_path = ""
        volume_available = False
        volume_path = args.data_dir / "full" / ("%s_zoom.npy" % pid)
        if pid not in volume_cache and volume_path.exists():
            volume = np.load(volume_path)
            if volume.ndim == 4:
                volume = volume[0]
            volume_cache[pid] = volume
        if pid in volume_cache:
            volume_available = True
            image = make_montage(volume_cache[pid], gt, args.crop_size, args.min_box_size)
            image_path = image_dir / ("%02d_%s_nodule%s_d%.1f_z%.1f.png" % (
                i,
                pid,
                int(gt["nodule_id"]),
                float(gt["diameter"]),
                float(gt["center_z"]),
            ))
            image.save(image_path)
        elif args.strict_volumes:
            raise FileNotFoundError(volume_path)

        rcnn = candidate_stats(rcnn_results, gt, args.max_candidates)
        ensemble = candidate_stats(ensemble_results, gt, args.max_candidates)
        texture = lesion_texture_stats(volume_cache[pid], gt) if volume_available else missing_texture_stats()
        record = {
            "pid": pid,
            "nodule_id": int(gt["nodule_id"]),
            "center_x": float(gt["center_x"]),
            "center_y": float(gt["center_y"]),
            "center_z": float(gt["center_z"]),
            "diameter": float(gt["diameter"]),
            "radius": float(gt["diameter"]) / 2.0,
            "volume_available": volume_available,
            "image_path": str(image_path),
        }
        for prefix, values in [("rcnn", rcnn), ("ensemble", ensemble)]:
            for key, value in values.items():
                record["%s_%s" % (prefix, key)] = value
        for key, value in texture.items():
            record[key] = value
        record["nearest_margin_change"] = float(ensemble["nearest_margin"]) - float(rcnn["nearest_margin"])
        record["nearest_prob_change"] = float(ensemble["nearest_prob"]) - float(rcnn["nearest_prob"])
        records.append(record)

    summary_path = out_dir / "recovered_lesions_summary.csv"
    with open(summary_path, "w", newline="") as f:
        fieldnames = list(records[0].keys()) if records else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("recovered lesions: %d" % len(records))
    print("images: %s" % image_dir)
    print("summary: %s" % summary_path)
    for record in records:
        print(
            "{pid} nodule={nodule_id} d={diameter:.1f} "
            "rcnn_hits={rcnn_num_hits} ens_hits={ensemble_num_hits} "
            "rcnn_nearest_margin={rcnn_nearest_margin:.3f} "
            "ens_nearest_margin={ensemble_nearest_margin:.3f} "
            "ens_best_hit_prob={ensemble_best_hit_prob:.6f}".format(**record)
        )


if __name__ == "__main__":
    main()
