#!/usr/bin/env python3
"""Export CT and filled GT/before-FP/after-FP masks as NIfTI files."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


DEFAULT_DATA_DIR = "/data/医保大赛/code/SANet/data/LNDB"
DEFAULT_BEFORE_CSV = "/data/医保大赛/code/SANet/train_hard_examples/hard_fps.csv"
DEFAULT_GT_CANDIDATES = (
    "/data/医保大赛/code/SANet/data/LNDB/split/all_anno.csv",
    "/data/医保大赛/code/SANet/data/LNDB/split/train_anno.csv",
    "/data/医保大赛/code/SANet/data/LNDB/split/val_anno.csv",
    "/data/医保大赛/code/SANet/data/LNDB/split/test_anno.csv",
    "/data/医保大赛/code/SANet/train_hard_examples/per_dataset/LNDB/FROC/annotations.csv",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Write ITK-SNAP-friendly .nii.gz files per case: one CT image and "
            "three independent filled masks for GT, FP before deduplication, "
            "and FP after deduplication."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(DEFAULT_DATA_DIR),
        help="SANet-ready dataset directory containing full/<pid>_zoom.npy.",
    )
    parser.add_argument(
        "--before-csv",
        type=Path,
        default=Path(DEFAULT_BEFORE_CSV),
        help="FP CSV before deduplication.",
    )
    parser.add_argument(
        "--after-csv",
        type=Path,
        required=True,
        help="FP CSV after deduplication.",
    )
    parser.add_argument(
        "--gt-csv",
        type=Path,
        default=None,
        help="GT annotation CSV. If omitted, common LNDB annotation paths are tried.",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        default=Path("fp_dedup_nii"),
        help="Output directory.",
    )
    parser.add_argument(
        "--dataset",
        default="LNDB",
        help="Dataset name in FP CSV dataset column. Empty means all datasets.",
    )
    parser.add_argument(
        "--pid",
        default="",
        help="Case pid to export. Empty means all pids selected by --dataset.",
    )
    parser.add_argument(
        "--max-pids",
        type=int,
        default=20,
        help="Maximum pids to export when --pid is empty. Use 0 for no limit.",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.0,
        help="Only export FP boxes with probability >= this value.",
    )
    parser.add_argument(
        "--box-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to diameter when filling boxes.",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=3.0,
        help="Minimum box side length in voxels.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately when a selected CT volume is missing.",
    )
    return parser.parse_args()


def normalize_pid(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def safe_pid(value):
    pid = normalize_pid(value)
    return pid.zfill(5) if pid.isdigit() else pid


def validate_fp_columns(df, path):
    required = {
        "dataset",
        "pid",
        "center_x",
        "center_y",
        "center_z",
        "diameter",
        "probability",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("%s is missing columns: %s" % (path, ", ".join(sorted(missing))))


def validate_gt_columns(df, path):
    center_columns = {"pid", "center_x", "center_y", "center_z", "diameter"}
    corner_columns = {"pid", "zmin", "zmax", "ymin", "ymax", "xmin", "xmax"}
    if center_columns.issubset(df.columns) or corner_columns.issubset(df.columns):
        return
    missing_center = center_columns.difference(df.columns)
    missing_corner = corner_columns.difference(df.columns)
    raise ValueError(
        "%s needs either center columns missing %s or corner columns missing %s"
        % (path, sorted(missing_center), sorted(missing_corner))
    )


def find_gt_csv(user_path):
    if user_path is not None:
        return user_path
    for path in DEFAULT_GT_CANDIDATES:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None


def read_fp_csv(path, dataset, pid, min_prob):
    df = pd.read_csv(path)
    validate_fp_columns(df, path)
    if dataset:
        df = df[df["dataset"].astype(str) == dataset]
    if pid:
        target = normalize_pid(pid)
        df = df[df["pid"].map(normalize_pid) == target]
    df = df[df["probability"].astype(float) >= min_prob].copy()
    df["pid_norm"] = df["pid"].map(normalize_pid)
    return df


def read_gt_csv(path, pid):
    if path is None:
        return pd.DataFrame(columns=["pid", "center_x", "center_y", "center_z", "diameter", "pid_norm"])

    df = pd.read_csv(path)
    validate_gt_columns(df, path)
    if not {"center_x", "center_y", "center_z", "diameter"}.issubset(df.columns):
        df = df.copy()
        df["center_x"] = (df["xmin"].astype(float) + df["xmax"].astype(float)) / 2.0
        df["center_y"] = (df["ymin"].astype(float) + df["ymax"].astype(float)) / 2.0
        df["center_z"] = (df["zmin"].astype(float) + df["zmax"].astype(float)) / 2.0
        dx = df["xmax"].astype(float) - df["xmin"].astype(float) + 1.0
        dy = df["ymax"].astype(float) - df["ymin"].astype(float) + 1.0
        dz = df["zmax"].astype(float) - df["zmin"].astype(float) + 1.0
        df["diameter"] = pd.concat([dx, dy, dz], axis=1).max(axis=1)
    if pid:
        target = normalize_pid(pid)
        df = df[df["pid"].map(normalize_pid) == target]
    df = df.copy()
    df["pid_norm"] = df["pid"].map(normalize_pid)
    return df


def load_volume(data_dir, pid):
    image_path = data_dir / "full" / ("%s_zoom.npy" % safe_pid(pid))
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    volume = np.load(image_path)
    if volume.ndim == 4:
        volume = volume[0]
    if volume.ndim != 3:
        raise ValueError("expected 3D volume in %s, got %s" % (image_path, volume.shape))
    return np.asarray(volume)


def load_spacing(data_dir, pid):
    meta_path = data_dir / "meta.csv"
    if not meta_path.exists():
        return (1.0, 1.0, 1.0)
    meta = pd.read_csv(meta_path)
    if "sanet_pid" not in meta.columns or "processed_spacing_xyz" not in meta.columns:
        return (1.0, 1.0, 1.0)
    matched = meta[meta["sanet_pid"].map(normalize_pid) == normalize_pid(pid)]
    if matched.empty:
        return (1.0, 1.0, 1.0)
    raw = str(matched.iloc[0]["processed_spacing_xyz"])
    parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(parts) != 3:
        return (1.0, 1.0, 1.0)
    return tuple(parts)


def normalize_ct_to_uint8(volume):
    arr = np.asarray(volume)
    if arr.dtype == np.uint8 and arr.min() >= 0 and arr.max() <= 255:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    valid = arr[finite]
    low, high = np.percentile(valid, [1, 99])
    if high <= low:
        low, high = float(valid.min()), float(valid.max())
    if high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - low) / (high - low), 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def box_limits(row, shape, box_scale, min_box_size):
    depth, height, width = shape
    side = max(float(row["diameter"]) * box_scale, min_box_size)
    half = side / 2.0
    x = float(row["center_x"])
    y = float(row["center_y"])
    z = float(row["center_z"])

    x0 = max(0, min(int(np.floor(x - half)), width - 1))
    x1 = max(0, min(int(np.ceil(x + half)), width - 1))
    y0 = max(0, min(int(np.floor(y - half)), height - 1))
    y1 = max(0, min(int(np.ceil(y + half)), height - 1))
    z0 = max(0, min(int(np.floor(z - half)), depth - 1))
    z1 = max(0, min(int(np.ceil(z + half)), depth - 1))
    return z0, z1, y0, y1, x0, x1


def fill_box(mask, row, value, box_scale, min_box_size):
    z0, z1, y0, y1, x0, x1 = box_limits(row, mask.shape, box_scale, min_box_size)
    mask[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = value


def make_mask(rows, shape, box_scale, min_box_size):
    mask = np.zeros(shape, dtype=np.uint8)
    for _, row in rows.iterrows():
        fill_box(mask, row, 1, box_scale, min_box_size)
    return mask


def write_volume(path, volume, spacing_xyz, pixel_type=None):
    image = sitk.GetImageFromArray(volume)
    if pixel_type is not None:
        image = sitk.Cast(image, pixel_type)
    image.SetSpacing(tuple(float(x) for x in spacing_xyz))
    sitk.WriteImage(image, str(path), useCompression=True)


def export_case(args, pid, before, after, gt):
    volume = load_volume(args.data_dir, pid)
    ct = normalize_ct_to_uint8(volume)
    spacing_xyz = load_spacing(args.data_dir, pid)

    before_rows = before[before["pid_norm"] == normalize_pid(pid)]
    after_rows = after[after["pid_norm"] == normalize_pid(pid)]
    gt_rows = gt[gt["pid_norm"] == normalize_pid(pid)]

    masks = {
        "gt_mask": make_mask(gt_rows, ct.shape, args.box_scale, args.min_box_size),
        "fp_before_mask": make_mask(before_rows, ct.shape, args.box_scale, args.min_box_size),
        "fp_after_mask": make_mask(after_rows, ct.shape, args.box_scale, args.min_box_size),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ct_path = args.out_dir / ("%s_ct.nii.gz" % safe_pid(pid))
    write_volume(ct_path, ct, spacing_xyz, sitk.sitkUInt8)

    out = {"ct": ct_path}
    for name, mask in masks.items():
        if ct.shape != mask.shape:
            raise ValueError("CT shape %s != mask shape %s" % (ct.shape, mask.shape))
        path = args.out_dir / ("%s_%s.nii.gz" % (safe_pid(pid), name))
        write_volume(path, mask, spacing_xyz, sitk.sitkUInt8)
        out[name] = path

    return {
        "pid": safe_pid(pid),
        "ct_shape_zyx": "x".join(str(v) for v in ct.shape),
        "nifti_size_xyz": "x".join(str(v) for v in ct.shape[::-1]),
        "gt_count": len(gt_rows),
        "before_count": len(before_rows),
        "after_count": len(after_rows),
        "ct": str(out["ct"]),
        "gt_mask": str(out["gt_mask"]),
        "fp_before_mask": str(out["fp_before_mask"]),
        "fp_after_mask": str(out["fp_after_mask"]),
    }


def main():
    args = parse_args()
    dataset = args.dataset.strip()
    pid = args.pid.strip()
    gt_path = find_gt_csv(args.gt_csv)

    before = read_fp_csv(args.before_csv, dataset, pid, args.min_prob)
    after = read_fp_csv(args.after_csv, dataset, pid, args.min_prob)
    gt = read_gt_csv(gt_path, pid)

    pids = sorted(set(before["pid_norm"]).union(set(after["pid_norm"])).union(set(gt["pid_norm"])))
    if args.max_pids and args.max_pids > 0:
        pids = pids[:args.max_pids]

    records = []
    for current_pid in pids:
        try:
            record = export_case(args, current_pid, before, after, gt)
        except FileNotFoundError as exc:
            if args.strict:
                raise
            print("warning: skip missing volume %s" % exc)
            continue
        records.append(record)
        print(
            "saved pid=%s shape=%s gt=%d before=%d after=%d"
            % (
                record["pid"],
                record["ct_shape_zyx"],
                record["gt_count"],
                record["before_count"],
                record["after_count"],
            )
        )

    summary_path = args.out_dir / "nii_export_index.csv"
    pd.DataFrame(records).to_csv(summary_path, index=False)
    print("gt csv: %s" % (gt_path if gt_path is not None else "not found"))
    print("summary: %s" % summary_path)


if __name__ == "__main__":
    main()

'''
python tools/export_dedup_fp_nii.py \
    --data-dir /data/医保大赛/code/SANet/data/LNDB \
    --before-csv /data/医保大赛/code/SANet/train_hard_examples/hard_fps.csv \
    --after-csv /data/医保大赛/code/SANet/train_hard_examples/hard_fps_dedup_lndb_75.csv \
    --dataset LNDB \
    --pid 75 \
    --out-dir fp_dedup_nii
    
默认 GT 会优先读取：/data/医保大赛/code/SANet/data/LNDB/split/all_anno.csv
也可以手动指定：--gt-csv /path/to/all_anno.csv
'''
