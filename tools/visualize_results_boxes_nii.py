#!/usr/bin/env python3
"""Draw solid box masks from results.csv rows into a NIfTI volume.

For one pid, pick rows by the given probability list (in order) and fill each
box with label 1, 2, 3, ... The output mask has the same shape as the CT
``*_zoom.npy`` so it can be overlaid in ITK-SNAP.
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
            "Visualize selected results.csv detections for one pid as a solid "
            "box mask NIfTI (labels 1..K by probability order)."
        )
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        required=True,
        help="Path to results.csv with columns pid,center_x,center_y,center_z,diameter,probability.",
    )
    parser.add_argument(
        "--pid",
        required=True,
        help="Case id to visualize.",
    )
    parser.add_argument(
        "--probabilities",
        type=float,
        nargs="+",
        required=True,
        help="Probability values to select, in paint order (label 1, 2, 3, ...).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Dataset root, e.g. /data/医保大赛/code/SANet/data/LUNA16 (looks under full/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output mask .nii.gz path. Default: <out-dir>/<pid>_boxes_mask.nii.gz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("box_mask_nii"),
        help="Output directory when --out is omitted.",
    )
    parser.add_argument(
        "--prob-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance when matching probability values.",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=1.0,
        help="Minimum cube side length in voxels.",
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
        cands.extend([str(int(raw)), raw.zfill(5), raw.zfill(6)])
    # preserve order, drop duplicates
    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def find_volume_path(data_dir: Path, pid: str) -> Path:
    for name in pid_candidates(pid):
        for rel in (Path("full") / ("%s_zoom.npy" % name), Path("%s_zoom.npy" % name)):
            path = data_dir / rel
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
        raise ValueError("expected 3D volume in %s, got %s" % (path, volume.shape))
    return np.asarray(volume)


def load_spacing(data_dir: Path, pid: str) -> tuple[float, float, float]:
    meta_path = data_dir / "meta.csv"
    if not meta_path.exists():
        return (1.0, 1.0, 1.0)
    meta = pd.read_csv(meta_path)
    if "sanet_pid" not in meta.columns or "processed_spacing_xyz" not in meta.columns:
        return (1.0, 1.0, 1.0)
    targets = set(pid_candidates(pid))
    matched = meta[meta["sanet_pid"].map(normalize_pid).isin(targets)]
    if matched.empty:
        return (1.0, 1.0, 1.0)
    raw = str(matched.iloc[0]["processed_spacing_xyz"])
    parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(parts) != 3:
        return (1.0, 1.0, 1.0)
    return tuple(parts)


def select_rows(df: pd.DataFrame, pid: str, probabilities: list[float], atol: float) -> pd.DataFrame:
    required = {"pid", "center_x", "center_y", "center_z", "diameter", "probability"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("results.csv missing columns: %s" % ", ".join(sorted(missing)))

    targets = set(pid_candidates(pid))
    case = df[df["pid"].map(normalize_pid).isin(targets)].copy()
    if case.empty:
        raise ValueError("No rows for pid=%s in results.csv" % pid)

    case["probability"] = case["probability"].astype(float)
    used = set()
    selected = []
    for i, prob in enumerate(probabilities):
        candidates = case.loc[~case.index.isin(used)].copy()
        if candidates.empty:
            raise ValueError("Not enough unused rows for probability #%d (%s)" % (i + 1, prob))
        diffs = (candidates["probability"] - float(prob)).abs()
        best_pos = int(diffs.to_numpy().argmin())
        best_idx = candidates.index.to_numpy()[best_pos]
        best_diff = float(diffs.loc[best_idx])
        if best_diff > atol:
            raise ValueError(
                "No row for pid=%s matching probability=%s within atol=%s "
                "(closest=%s, diff=%s)"
                % (pid, prob, atol, float(case.loc[best_idx, "probability"]), best_diff)
            )
        used.add(best_idx)
        row = case.loc[best_idx].copy()
        row["label"] = i + 1
        row["query_probability"] = float(prob)
        selected.append(row)

    return pd.DataFrame(selected)


def fill_cube(mask: np.ndarray, row: pd.Series, label: int, min_box_size: float) -> tuple[int, int, int, int, int, int]:
    depth, height, width = mask.shape
    side = max(float(row["diameter"]), min_box_size)
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
    mask[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = np.uint8(label)
    return z0, z1, y0, y1, x0, x1


def write_nii(path: Path, volume: np.ndarray, spacing_xyz: tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.ascontiguousarray(volume))
    image.SetSpacing(tuple(float(x) for x in spacing_xyz))
    sitk.WriteImage(image, str(path), useCompression=True)


def normalize_ct_to_uint8(volume: np.ndarray) -> np.ndarray:
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


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results_csv)
    rows = select_rows(df, args.pid, args.probabilities, args.prob_atol)

    volume_path = find_volume_path(args.data_dir, args.pid)
    volume = load_volume(volume_path)
    spacing = load_spacing(args.data_dir, args.pid)

    mask = np.zeros(volume.shape, dtype=np.uint8)
    print("CT: %s  shape=%s" % (volume_path, volume.shape))
    for _, row in rows.iterrows():
        z0, z1, y0, y1, x0, x1 = fill_cube(mask, row, int(row["label"]), args.min_box_size)
        print(
            "label=%d query_p=%.8f csv_p=%.8f center=(%.3f,%.3f,%.3f) diameter=%.3f "
            "box=[z%d:%d, y%d:%d, x%d:%d]"
            % (
                int(row["label"]),
                float(row["query_probability"]),
                float(row["probability"]),
                float(row["center_x"]),
                float(row["center_y"]),
                float(row["center_z"]),
                float(row["diameter"]),
                z0,
                z1,
                y0,
                y1,
                x0,
                x1,
            )
        )

    out_mask = args.out
    if out_mask is None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_mask = args.out_dir / ("%s_boxes_mask.nii.gz" % safe_pid(args.pid))
    write_nii(out_mask, mask, spacing)
    print("Saved mask: %s  shape=%s  labels=%s" % (out_mask, mask.shape, sorted(int(x) for x in np.unique(mask) if x > 0)))

    if args.save_ct:
        if out_mask.name.endswith("_boxes_mask.nii.gz"):
            out_ct = out_mask.with_name(out_mask.name[: -len("_boxes_mask.nii.gz")] + "_ct.nii.gz")
        else:
            out_ct = out_mask.with_name("%s_ct.nii.gz" % safe_pid(args.pid))
        write_nii(out_ct, normalize_ct_to_uint8(volume), spacing)
        print("Saved CT:   %s  shape=%s" % (out_ct, volume.shape))


if __name__ == "__main__":
    main()
