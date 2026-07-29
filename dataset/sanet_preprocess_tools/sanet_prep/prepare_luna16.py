#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert LUNA16 NIfTI (.nii/.nii.gz) to SANet/PN9-style npy format.

Supported input layout:
    LUNA16_NII_ROOT/
      subset0_nii/
        <seriesuid>.nii.gz
        ...
      subset1_nii/
        <seriesuid>.nii.gz
      ...
      subset9_nii/
        <seriesuid>.nii.gz
      annotations.csv

Output:
    out_dir/
      full/
        000001_zoom.npy
        ...
      split/
        train.txt
        val.txt
        test.txt
        all_anno.csv
        train_anno.csv      # train + val annotations, matching original SANet behavior
        val_anno.csv
        test_anno.csv
      pid_map.csv
      meta.csv
      dataset_summary.json
      sanet_config_example.json

Target image format:
    float32 npy, shape [1, D, H, W], intensity [0, 255],
    CT resampled to 1mm x 1mm x 1mm.

Target annotation format:
    pid,zmin,zmax,ymin,ymax,xmin,xmax,nodule_id
    Coordinates are voxel indices after 1mm resampling, array order z,y,x.

Important:
    LUNA16 annotations.csv uses physical/world coordinates in mm:
        seriesuid,coordX,coordY,coordZ,diameter_mm

    This script assumes your NIfTI files preserve the same physical space
    as original LUNA16 images. If the NIfTI conversion reset origin/direction,
    annotations will not align. Run with --check-annotation-overlap to print
    sanity warnings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


SANET_ANNO_COLUMNS = ["pid", "zmin", "zmax", "ymin", "ymax", "xmin", "xmax", "nodule_id"]


@dataclass
class CaseMeta:
    seriesuid: str
    sanet_pid: int
    subset: int
    image_path: str
    original_shape_zyx: Tuple[int, int, int]
    processed_shape_zyx: Tuple[int, int, int]
    original_spacing_xyz: Tuple[float, float, float]
    processed_spacing_xyz: Tuple[float, float, float]
    original_origin_xyz: Tuple[float, float, float]
    processed_origin_xyz: Tuple[float, float, float]
    original_direction: str
    processed_direction: str
    original_min: float
    original_max: float
    processed_min: float
    processed_max: float
    n_bboxes: int
    n_bbox_clipped_or_invalid: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "seriesuid": self.seriesuid,
            "sanet_pid": self.sanet_pid,
            "sanet_filename_stem": pid_to_filename(self.sanet_pid),
            "subset": self.subset,
            "image_path": self.image_path,
            "original_shape_zyx": "x".join(map(str, self.original_shape_zyx)),
            "processed_shape_zyx": "x".join(map(str, self.processed_shape_zyx)),
            "original_spacing_xyz": ",".join(f"{v:.6g}" for v in self.original_spacing_xyz),
            "processed_spacing_xyz": ",".join(f"{v:.6g}" for v in self.processed_spacing_xyz),
            "original_origin_xyz": ",".join(f"{v:.6g}" for v in self.original_origin_xyz),
            "processed_origin_xyz": ",".join(f"{v:.6g}" for v in self.processed_origin_xyz),
            "original_direction": self.original_direction,
            "processed_direction": self.processed_direction,
            "original_min": self.original_min,
            "original_max": self.original_max,
            "processed_min": self.processed_min,
            "processed_max": self.processed_max,
            "n_bboxes": self.n_bboxes,
            "n_bbox_clipped_or_invalid": self.n_bbox_clipped_or_invalid,
        }


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def natural_key(s: str) -> List[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def pid_to_filename(pid: int, width: int = 6) -> str:
    return f"{int(pid):0{width}d}"


def strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def normalize_id(x: str) -> str:
    """
    Normalize LUNA16 seriesuid or image filename.

    Important:
    LUNA16 seriesuid contains many dots, e.g.
    1.3.6.1.4.1.14519.5.2.1.6279.6001.100225...

    So DO NOT use Path(x).stem directly, because it will incorrectly truncate
    the UID at the last dot.
    """
    s = str(x).strip()
    name = Path(s).name

    # Only remove known medical-image suffixes.
    # Do not treat the final UID segment as a file extension.
    known_suffixes = [
        ".nii.gz",
        ".nii",
        ".mhd",
        ".raw",
        ".npy",
        ".npz",
    ]

    for suf in known_suffixes:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break

    # Remove optional image suffixes only after removing file extension.
    for tail in ["_ct", "_CT", "_image", "_img", "-image", "_scan"]:
        if name.endswith(tail):
            name = name[: -len(tail)]
            break

    return name


def build_numeric_pid_map(ids: Sequence[str], start: int = 1) -> Dict[str, int]:
    ids = sorted(set(map(str, ids)), key=natural_key)
    return {oid: i + start for i, oid in enumerate(ids)}


def find_luna_nii_files(luna_root: Path) -> Dict[str, Path]:
    files = []
    for ext in ("*.nii.gz", "*.nii"):
        files.extend(luna_root.rglob(ext))
    files = sorted(files, key=lambda p: natural_key(str(p)))
    if not files:
        raise FileNotFoundError(f"No .nii/.nii.gz files found under {luna_root}")

    out: Dict[str, Path] = {}
    for p in files:
        # Skip possible masks/labels if mixed in the same root. Lung masks are
        # stored as subset*_nii/lungseg/<seriesuid>.nii.gz, so check path parts too.
        low_name = p.name.lower()
        low_parts = {part.lower() for part in p.parts}
        if any(t in low_name for t in ["mask", "label", "seg", "tumor", "nodule_mask"]):
            continue
        if any(part in low_parts for part in ["lungseg", "mask", "masks", "label", "labels", "seg"]):
            continue
        sid = normalize_id(p.name)
        if sid in out:
            raise RuntimeError(f"Duplicate seriesuid {sid}: {out[sid]} and {p}")
        out[sid] = p
    if not out:
        raise FileNotFoundError(f"No image .nii/.nii.gz files found under {luna_root}")
    return out


def resolve_lung_mask_roots(lung_mask_root: Path) -> List[Path]:
    if lung_mask_root.exists():
        return [lung_mask_root]
    raw = str(lung_mask_root)
    if "subsetx_nii" in raw:
        roots = sorted(Path(raw.replace("subsetx_nii", "subset0_nii")).parent.parent.glob("subset*_nii/lungseg"))
        roots = [r for r in roots if r.exists()]
        if roots:
            return roots
    raise FileNotFoundError(f"No lung mask root found: {lung_mask_root}")


def collect_lung_masks_by_id(lung_mask_root: Path) -> Dict[str, Path]:
    masks: Dict[str, Path] = {}
    roots = resolve_lung_mask_roots(lung_mask_root)
    for root in roots:
        files = []
        for ext in ("*.nii.gz", "*.nii"):
            files.extend(root.rglob(ext))
        for path in sorted(files, key=lambda p: natural_key(str(p))):
            if "lungseg" not in {part.lower() for part in path.parts}:
                continue
            sid = normalize_id(path.name)
            if sid in masks and masks[sid] != path:
                raise RuntimeError(f"Duplicate lung mask for {sid}: {masks[sid]} and {path}")
            masks[sid] = path
    if not masks:
        raise FileNotFoundError(f"No lung mask files found under {lung_mask_root}")
    return masks


def infer_subset_from_path(path: Path) -> int:
    # Handles subset0, subset0_nii, subset_0, subset-0.
    for part in path.parts[::-1]:
        m = re.search(r"subset[_-]?(\d+)", part.lower())
        if m:
            return int(m.group(1))
    return -1


def read_luna_annotations(anno_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(anno_csv)
    required = ["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"annotations.csv missing columns {missing}. Available: {list(df.columns)}")
    df = df[required].copy()
    df["seriesuid"] = df["seriesuid"].astype(str).map(normalize_id)
    for c in ["coordX", "coordY", "coordZ", "diameter_mm"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    return df


def image_stats(img: sitk.Image) -> Tuple[Tuple[int, int, int], Tuple[float, float, float], Tuple[float, float, float], str, float, float]:
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    shape_zyx = tuple(int(v) for v in arr.shape)
    spacing_xyz = tuple(float(v) for v in img.GetSpacing())
    origin_xyz = tuple(float(v) for v in img.GetOrigin())
    direction = ",".join(f"{v:.6g}" for v in img.GetDirection())
    return shape_zyx, spacing_xyz, origin_xyz, direction, float(np.min(arr)), float(np.max(arr))


def resample_image(
    img: sitk.Image,
    out_spacing_xyz: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    default_value: float = -1200.0,
) -> sitk.Image:
    in_spacing = np.array(img.GetSpacing(), dtype=np.float64)  # x,y,z
    in_size = np.array(img.GetSize(), dtype=np.int64)          # x,y,z
    out_spacing = np.array(out_spacing_xyz, dtype=np.float64)
    out_size = np.maximum(1, np.round(in_size * in_spacing / out_spacing).astype(np.int64))

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(x) for x in out_spacing))
    resampler.SetSize([int(x) for x in out_size])
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(default_value))
    return resampler.Execute(img)


def resample_to_reference(
    img: sitk.Image,
    ref_img: sitk.Image,
    is_label: bool = False,
    default_value: float = 0.0,
) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(default_value))
    return resampler.Execute(img)


def crop_to_mask_bbox(img: sitk.Image, mask_img: sitk.Image) -> sitk.Image:
    mask_ref = resample_to_reference(mask_img, img, is_label=True, default_value=0)
    coords = np.where(sitk.GetArrayFromImage(mask_ref) > 0)
    if len(coords[0]) == 0:
        raise ValueError("Lung mask is empty after resampling to CT reference.")
    z, y, x = coords
    index_xyz = [int(x.min()), int(y.min()), int(z.min())]
    size_xyz = [
        int(x.max() - x.min() + 1),
        int(y.max() - y.min() + 1),
        int(z.max() - z.min() + 1),
    ]
    return sitk.RegionOfInterest(img, size_xyz, index_xyz)


def sanet_window_from_hu(arr_zyx: np.ndarray, hu_min: float = -1200.0, hu_max: float = 600.0) -> np.ndarray:
    arr = arr_zyx.astype(np.float32, copy=False)
    arr = np.clip(arr, hu_min, hu_max)
    arr = (arr - hu_min) / (hu_max - hu_min) * 255.0
    return arr.astype(np.float32, copy=False)


def image_to_sanet_array(
    img: sitk.Image,
    hu_min: float = -1200.0,
    hu_max: float = 600.0,
    assume_already_0_255: bool = False,
    auto_skip_window_if_0_255: bool = False,
) -> np.ndarray:
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # z,y,x
    if assume_already_0_255:
        if arr.min() >= 0 and arr.max() <= 1.0 + 1e-6:
            out = np.clip(arr * 255.0, 0, 255).astype(np.float32)
        else:
            out = np.clip(arr, 0, 255).astype(np.float32)
    elif auto_skip_window_if_0_255 and arr.min() >= 0 and arr.max() <= 255:
        if arr.max() <= 1.0 + 1e-6:
            out = (arr * 255.0).astype(np.float32)
        else:
            out = arr.astype(np.float32)
    else:
        out = sanet_window_from_hu(arr, hu_min=hu_min, hu_max=hu_max)
    return out[None, ...]  # [1,D,H,W]


def physical_center_diameter_to_bbox_zyx(
    img_1mm: sitk.Image,
    center_xyz_mm: Tuple[float, float, float],
    diameter_mm: float,
) -> Tuple[int, int, int, int, int, int]:
    """Convert LUNA physical center + diameter to processed-image voxel bbox.

    The bbox is the axis-aligned cube enclosing a sphere of radius diameter/2.
    Uses SimpleITK physical/index transforms, so image origin/direction are respected.
    Returns zmin,zmax,ymin,ymax,xmin,xmax.
    """
    r = float(diameter_mm) / 2.0
    cx, cy, cz = map(float, center_xyz_mm)

    corners_xyz = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                p = (cx + sx * r, cy + sy * r, cz + sz * r)
                idx = img_1mm.TransformPhysicalPointToContinuousIndex(p)  # x,y,z
                corners_xyz.append(idx)

    corners_xyz = np.asarray(corners_xyz, dtype=np.float64)
    mins_xyz = np.floor(corners_xyz.min(axis=0)).astype(int)
    maxs_xyz = np.ceil(corners_xyz.max(axis=0)).astype(int)

    size_xyz = np.asarray(img_1mm.GetSize(), dtype=int)
    mins_xyz_clipped = np.maximum(mins_xyz, 0)
    maxs_xyz_clipped = np.minimum(maxs_xyz, size_xyz - 1)

    xmin, ymin, zmin = mins_xyz_clipped.tolist()
    xmax, ymax, zmax = maxs_xyz_clipped.tolist()
    return int(zmin), int(zmax), int(ymin), int(ymax), int(xmin), int(xmax)


def center_inside_image(img: sitk.Image, center_xyz_mm: Tuple[float, float, float]) -> bool:
    idx = np.asarray(img.TransformPhysicalPointToContinuousIndex(tuple(map(float, center_xyz_mm))), dtype=float)
    size = np.asarray(img.GetSize(), dtype=float)
    return bool(np.all(idx >= 0) and np.all(idx <= size - 1))


def clip_bbox_to_shape(
    bbox: Tuple[int, int, int, int, int, int],
    shape_zyx: Tuple[int, int, int],
) -> Optional[Tuple[int, int, int, int, int, int]]:
    zmin, zmax, ymin, ymax, xmin, xmax = [int(round(v)) for v in bbox]
    D, H, W = shape_zyx
    zmin, zmax = max(0, zmin), min(D - 1, zmax)
    ymin, ymax = max(0, ymin), min(H - 1, ymax)
    xmin, xmax = max(0, xmin), min(W - 1, xmax)
    if zmax < zmin or ymax < ymin or xmax < xmin:
        return None
    return zmin, zmax, ymin, ymax, xmin, xmax


def save_sanet_npy(arr_1zyx: np.ndarray, out_path: Path) -> None:
    arr = np.asarray(arr_1zyx, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[0] != 1:
        raise ValueError(f"SANet array must have shape [1,D,H,W], got {arr.shape}")
    ensure_dir(out_path.parent)
    np.save(str(out_path), arr)


def write_list_file(path: Path, pids: Iterable[int], width: int) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for pid in pids:
            f.write(pid_to_filename(pid, width=width) + "\n")


def write_annotation_csv(path: Path, rows: Iterable[Dict[str, int]]) -> None:
    ensure_dir(path.parent)
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SANET_ANNO_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: int(row[c]) for c in SANET_ANNO_COLUMNS})


def write_pid_map(path: Path, pid_map: Dict[str, int], subset_by_id: Dict[str, int], width: int) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seriesuid", "sanet_pid", "sanet_filename_stem", "subset"])
        writer.writeheader()
        for sid, pid in sorted(pid_map.items(), key=lambda kv: kv[1]):
            writer.writerow({
                "seriesuid": sid,
                "sanet_pid": pid,
                "sanet_filename_stem": pid_to_filename(pid, width),
                "subset": subset_by_id.get(sid, -1),
            })


def write_meta_csv(path: Path, metas: Sequence[CaseMeta]) -> None:
    ensure_dir(path.parent)
    pd.DataFrame([m.as_dict() for m in metas]).to_csv(path, index=False)


def write_summary(path: Path, metas: Sequence[CaseMeta], split: Dict[str, List[int]], ann_rows: Sequence[Dict[str, int]]) -> None:
    ensure_dir(path.parent)
    summary = {
        "dataset_name": "LUNA16_NIfTI",
        "num_cases": len(metas),
        "num_boxes": len(ann_rows),
        "split_counts": {k: len(v) for k, v in split.items()},
        "split_note": "LUNA16 is organized as subset0~subset9 for 10-fold use. Default here: subset0_nii-subset7_nii train, subset8_nii val, subset9_nii test unless changed by arguments.",
        "npy_format": "float32 [1,D,H,W], intensity [0,255]",
        "preprocessing": "resample to 1mm isotropic; HU clip [-1200,600] and linear map to [0,255]",
        "annotation_csv": SANET_ANNO_COLUMNS,
        "warning": "annotations.csv is in physical coordinates. NIfTI origin/direction/spacing must match original LUNA16 space.",
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def read_split_ids_from_dir(split_dir: Path) -> Dict[str, List[str]]:
    aliases = {
        "train": ["train.txt", "training.txt", "train_ids.txt", "train_list.txt"],
        "val": ["val.txt", "valid.txt", "validation.txt", "val_ids.txt", "valid_ids.txt"],
        "test": ["test.txt", "testing.txt", "test_ids.txt", "test_list.txt"],
    }
    out: Dict[str, List[str]] = {}
    for name, cands in aliases.items():
        found = None
        for c in cands:
            p = split_dir / c
            if p.exists():
                found = p
                break
        if found is None:
            continue
        ids = []
        with found.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ids.append(normalize_id(line.split()[0].split(",")[0]))
        out[name] = ids
    if "train" not in out or "test" not in out:
        raise RuntimeError(f"Split dir {split_dir} must contain at least train.txt and test.txt")
    out.setdefault("val", [])
    return out


def make_split(
    series_ids: Sequence[str],
    pid_map: Dict[str, int],
    subset_by_id: Dict[str, int],
    split_dir: Optional[Path],
    split_mode: str,
    test_fold: int,
    val_fold: int,
    allow_no_subset_fallback: bool,
    fallback_ratio: Tuple[float, float, float],
    seed: int,
) -> Dict[str, List[int]]:
    if split_dir is not None:
        raw_split = read_split_ids_from_dir(split_dir)
        out = {"train": [], "val": [], "test": []}
        known = set(pid_map)
        for k in ["train", "val", "test"]:
            for sid in raw_split.get(k, []):
                if sid not in known:
                    raise KeyError(f"{sid} from {split_dir}/{k}.txt not found in LUNA image ids")
                out[k].append(pid_map[sid])
        return {k: sorted(set(v)) for k, v in out.items()}

    if any(subset_by_id.get(sid, -1) < 0 for sid in series_ids):
        if not allow_no_subset_fallback:
            raise RuntimeError(
                "Some .nii.gz files are not under subset0_nii~subset9_nii folders. "
                "Pass --split-dir, or use --allow-no-subset-fallback for deterministic ratio split."
            )
        rng = np.random.default_rng(seed)
        ids = np.array(sorted(series_ids, key=natural_key), dtype=object)
        rng.shuffle(ids)
        n = len(ids)
        tr, va, te = fallback_ratio
        n_train = int(round(n * tr / (tr + va + te)))
        n_val = int(round(n * va / (tr + va + te)))
        split_ids = {
            "train": list(ids[:n_train]),
            "val": list(ids[n_train:n_train + n_val]),
            "test": list(ids[n_train + n_val:]),
        }
        return {k: sorted(pid_map[sid] for sid in v) for k, v in split_ids.items()}

    if split_mode == "fixed":
        split_ids = {"train": [], "val": [], "test": []}
        for sid in series_ids:
            s = subset_by_id[sid]
            # 7:1:2 split for LUNA16 ten subsets:
            # subset0-6 -> train
            # subset7   -> val
            # subset8-9 -> test
            if s in range(0, 7):
                split_ids["train"].append(sid)
            elif s == 7:
                split_ids["val"].append(sid)
            elif s in [8, 9]:
                split_ids["test"].append(sid)
            else:
                raise ValueError(f"Unexpected subset number {s} for {sid}")
    elif split_mode == "fold":
        if test_fold == val_fold:
            raise ValueError("--test-fold and --val-fold must be different")
        split_ids = {"train": [], "val": [], "test": []}
        for sid in series_ids:
            s = subset_by_id[sid]
            if s == test_fold:
                split_ids["test"].append(sid)
            elif s == val_fold:
                split_ids["val"].append(sid)
            else:
                split_ids["train"].append(sid)
    else:
        raise ValueError("--split-mode must be fixed or fold")

    return {k: sorted(pid_map[sid] for sid in v) for k, v in split_ids.items()}


def split_annotations(all_rows: Sequence[Dict[str, int]], split: Dict[str, List[int]]) -> Dict[str, List[Dict[str, int]]]:
    pid_to_split: Dict[int, str] = {}
    for sname, pids in split.items():
        for pid in pids:
            pid_to_split[int(pid)] = sname

    out = {"train": [], "val": [], "test": []}
    for row in all_rows:
        s = pid_to_split.get(int(row["pid"]))
        if s is not None:
            out[s].append(row)
    return out


def save_common_outputs(
    out_dir: Path,
    pid_map: Dict[str, int],
    subset_by_id: Dict[str, int],
    split: Dict[str, List[int]],
    ann_rows: Sequence[Dict[str, int]],
    metas: Sequence[CaseMeta],
    pid_width: int,
) -> None:
    split_dir = ensure_dir(out_dir / "split")
    ensure_dir(out_dir / "full")

    write_pid_map(out_dir / "pid_map.csv", pid_map, subset_by_id, width=pid_width)
    write_meta_csv(out_dir / "meta.csv", metas)

    for s in ["train", "val", "test"]:
        write_list_file(split_dir / f"{s}.txt", split.get(s, []), width=pid_width)

    ann_by_split = split_annotations(ann_rows, split)
    write_annotation_csv(split_dir / "all_anno.csv", ann_rows)
    # SANet original config usually reads train_anno for train and val; keep train+val together.
    write_annotation_csv(split_dir / "train_anno.csv", ann_by_split["train"] + ann_by_split["val"])
    write_annotation_csv(split_dir / "val_anno.csv", ann_by_split["val"])
    write_annotation_csv(split_dir / "test_anno.csv", ann_by_split["test"])

    write_summary(out_dir / "dataset_summary.json", metas, split, ann_rows)

    cfg = {
        "preprocessed_data_dir": str(out_dir / "full"),
        "train_set_list": str(split_dir / "train.txt"),
        "val_set_list": str(split_dir / "val.txt"),
        "test_set_name": str(split_dir / "test.txt"),
        "train_anno": str(split_dir / "train_anno.csv"),
        "test_anno": str(split_dir / "test_anno.csv"),
        "roi_names": ["nodule"],
        "num_class": 2,
        "crop_size": [128, 128, 128],
        "bbox_border": 8,
        "pad_value": 170,
    }
    with (out_dir / "sanet_config_example.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def run(args: argparse.Namespace) -> None:
    luna_root = Path(args.luna_root)
    out_dir = Path(args.out_dir)
    full_dir = ensure_dir(out_dir / "full")

    img_by_id = find_luna_nii_files(luna_root)
    series_ids = sorted(img_by_id.keys(), key=natural_key)
    subset_by_id = {sid: infer_subset_from_path(p) for sid, p in img_by_id.items()}
    pid_map = build_numeric_pid_map(series_ids, start=1)
    try:
        lung_masks_by_id = collect_lung_masks_by_id(Path(args.lung_mask_root))
    except FileNotFoundError:
        if not args.allow_missing_lung_masks:
            raise
        lung_masks_by_id = {}

    missing_lung_masks = sorted(set(series_ids) - set(lung_masks_by_id), key=natural_key)
    if missing_lung_masks and not args.allow_missing_lung_masks:
        missing_by_subset: Dict[int, int] = {}
        for sid in missing_lung_masks:
            subset = subset_by_id.get(sid, -1)
            missing_by_subset[subset] = missing_by_subset.get(subset, 0) + 1
        raise RuntimeError(
            f"{len(missing_lung_masks)} image seriesuid(s) do not have lung masks. "
            f"Missing by subset: {missing_by_subset}. "
            f"Examples: {missing_lung_masks[:5]}. "
            "Pass --allow-missing-lung-masks to keep these CTs uncropped, or add the missing lungseg files."
        )

    ann_df = read_luna_annotations(Path(args.annotations_csv))
    ann_by_sid = {sid: g.copy() for sid, g in ann_df.groupby("seriesuid")}

    missing_images = sorted(set(ann_by_sid) - set(img_by_id), key=natural_key)
    if missing_images:
        raise RuntimeError(
            f"{len(missing_images)} annotation seriesuid(s) do not have .nii/.nii.gz image files. "
            f"Examples: {missing_images[:5]}\n"
            f"Tip: check whether your NIfTI filenames exactly match annotations.csv seriesuid."
        )

    all_rows: List[Dict[str, int]] = []
    metas: List[CaseMeta] = []
    bad_center_cases = []
    missing_lung_mask_cases = []

    for sid in tqdm(series_ids, desc="LUNA16 NIfTI -> SANet/PN9 npy"):
        img_path = img_by_id[sid]
        sanet_pid = pid_map[sid]
        img = sitk.ReadImage(str(img_path))
        orig_shape, orig_spacing, orig_origin, orig_direction, orig_min, orig_max = image_stats(img)
        lung_mask_path = lung_masks_by_id.get(sid)
        if lung_mask_path is None:
            missing_lung_mask_cases.append((sid, img_path))
            img_cropped = img
        else:
            lung_mask = sitk.ReadImage(str(lung_mask_path))
            img_cropped = crop_to_mask_bbox(img, lung_mask)

        img_1mm = resample_image(
            img_cropped,
            out_spacing_xyz=tuple(args.target_spacing),
            default_value=args.pad_hu,
        )

        arr = image_to_sanet_array(
            img_1mm,
            hu_min=args.hu_min,
            hu_max=args.hu_max,
            assume_already_0_255=args.assume_already_0_255,
            auto_skip_window_if_0_255=args.auto_skip_window_if_0_255,
        )
        proc_shape = tuple(int(v) for v in arr.shape[1:])
        save_sanet_npy(arr, full_dir / f"{pid_to_filename(sanet_pid, args.pid_width)}_zoom.npy")

        n_boxes = 0
        n_bad = 0
        case_ann = ann_by_sid.get(sid)
        if case_ann is not None:
            nodule_id = 0
            for _, r in case_ann.iterrows():
                diameter = float(r["diameter_mm"])
                if diameter < args.min_diameter_mm:
                    continue
                center = (float(r["coordX"]), float(r["coordY"]), float(r["coordZ"]))

                if args.check_annotation_overlap and not center_inside_image(img_1mm, center):
                    n_bad += 1
                    bad_center_cases.append((sid, center, img_path))

                bbox = physical_center_diameter_to_bbox_zyx(img_1mm, center, diameter)
                bbox = clip_bbox_to_shape(bbox, proc_shape)
                if bbox is None:
                    n_bad += 1
                    continue
                zmin, zmax, ymin, ymax, xmin, xmax = bbox
                all_rows.append({
                    "pid": sanet_pid,
                    "zmin": zmin,
                    "zmax": zmax,
                    "ymin": ymin,
                    "ymax": ymax,
                    "xmin": xmin,
                    "xmax": xmax,
                    "nodule_id": nodule_id,
                })
                nodule_id += 1
                n_boxes += 1

        metas.append(CaseMeta(
            seriesuid=sid,
            sanet_pid=sanet_pid,
            subset=subset_by_id.get(sid, -1),
            image_path=str(img_path),
            original_shape_zyx=orig_shape,
            processed_shape_zyx=proc_shape,
            original_spacing_xyz=orig_spacing,
            processed_spacing_xyz=tuple(float(x) for x in args.target_spacing),
            original_origin_xyz=orig_origin,
            processed_origin_xyz=tuple(float(v) for v in img_1mm.GetOrigin()),
            original_direction=orig_direction,
            processed_direction=",".join(f"{v:.6g}" for v in img_1mm.GetDirection()),
            original_min=orig_min,
            original_max=orig_max,
            processed_min=float(arr.min()),
            processed_max=float(arr.max()),
            n_bboxes=n_boxes,
            n_bbox_clipped_or_invalid=n_bad,
        ))

    split = make_split(
        series_ids=series_ids,
        pid_map=pid_map,
        subset_by_id=subset_by_id,
        split_dir=Path(args.split_dir) if args.split_dir else None,
        split_mode=args.split_mode,
        test_fold=args.test_fold,
        val_fold=args.val_fold,
        allow_no_subset_fallback=args.allow_no_subset_fallback,
        fallback_ratio=tuple(args.fallback_ratio),
        seed=args.seed,
    )

    if args.positive_only_train:
        positive_pids = {int(r["pid"]) for r in all_rows}
        split["train"] = [p for p in split["train"] if p in positive_pids]
        split["val"] = [p for p in split["val"] if p in positive_pids]

    save_common_outputs(out_dir, pid_map, subset_by_id, split, all_rows, metas, pid_width=args.pid_width)

    if missing_lung_mask_cases:
        warn_path = out_dir / "missing_lung_mask_warnings.csv"
        with warn_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["seriesuid", "image_path", "note"])
            for sid, p in missing_lung_mask_cases:
                writer.writerow([sid, str(p), "lung mask missing; CT was not cropped"])
        print(f"WARNING: {len(missing_lung_mask_cases)} cases had no lung mask and were kept uncropped.")
        print(f"Saved missing lung mask table: {warn_path}")

    if bad_center_cases:
        warn_path = out_dir / "annotation_center_outside_image_warnings.csv"
        with warn_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["seriesuid", "coordX", "coordY", "coordZ", "image_path"])
            for sid, center, p in bad_center_cases:
                writer.writerow([sid, center[0], center[1], center[2], str(p)])
        print(f"WARNING: {len(bad_center_cases)} annotation centers appear outside image physical space.")
        print(f"Saved warning table: {warn_path}")
        print("This often means your NIfTI conversion reset origin/direction. Use original LUNA16 MHD/RAW or fix NIfTI affine/origin.")

    print(f"Done. Output: {out_dir}")
    print(f"Cases: {len(series_ids)}")
    print(f"Boxes: {len(all_rows)}")
    print(f"Split counts: { {k: len(v) for k, v in split.items()} }")
    print(f"Check outputs under: {out_dir / 'split'}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare LUNA16 NIfTI images to SANet/PN9-style npy and bbox CSV.")
    p.add_argument("--luna-root", default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/LUNA16/raw", help="LUNA16 NIfTI root directory containing subset0_nii~subset9_nii folders.")
    p.add_argument("--annotations-csv", default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/LUNA16/raw/annotations.csv", help="LUNA16 annotations.csv path.")
    p.add_argument("--lung-mask-root", default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/LUNA16/raw", help="Lung mask root. Recursively collects subset*_nii/lungseg masks; a literal subsetx_nii/lungseg path is also expanded.")
    p.add_argument("--out-dir", default="../../data/LUNA16", help="Output SANet-ready directory.")
    p.add_argument("--split-dir", default=None, help="Optional directory containing train.txt/val.txt/test.txt using LUNA seriesuid.")
    p.add_argument("--split-mode", choices=["fixed", "fold"], default="fixed",
                   help="fixed: subset0-7 train, subset8 val, subset9 test. fold: set --test-fold and --val-fold.")
    p.add_argument("--test-fold", type=int, default=9, help="For --split-mode fold: subset used as test.")
    p.add_argument("--val-fold", type=int, default=8, help="For --split-mode fold: subset used as val.")
    p.add_argument("--allow-no-subset-fallback", action="store_true",
                   help="If files are not under subset folders, use deterministic fallback ratio.")
    p.add_argument("--fallback-ratio", nargs=3, type=float, default=[7, 1, 2],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--pid-width", type=int, default=5)

    p.add_argument("--target-spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0],
                   metavar=("SX", "SY", "SZ"), help="Output spacing x y z. Default 1 1 1.")
    p.add_argument("--hu-min", type=float, default=-1200.0)
    p.add_argument("--hu-max", type=float, default=600.0)
    p.add_argument("--pad-hu", type=float, default=-1200.0,
                   help="Default HU value for regions outside image during resampling.")
    p.add_argument("--assume-already-0-255", action="store_true",
                   help="Use only if your NIfTI images are already [0,255], not normal LUNA16 HU.")
    p.add_argument("--auto-skip-window-if-0-255", action="store_true",
                   help="Skip HU window if the loaded image min/max already lie in [0,255].")
    p.add_argument("--min-diameter-mm", type=float, default=0.0,
                   help="Filter annotations smaller than this diameter. Default keeps all LUNA annotations.")
    p.add_argument("--positive-only-train", action="store_true",
                   help="Drop negative cases from train/val lists. Usually not needed for full LUNA16, but useful for SANet original reader assumptions.")
    p.add_argument("--check-annotation-overlap", action="store_true",
                   help="Warn if annotation physical centers fall outside image physical space. Strongly recommended for converted NIfTI.")
    p.add_argument("--allow-missing-lung-masks", action="store_true",
                   help="Allow cases without lung masks to be processed without lung-crop and write missing_lung_mask_warnings.csv.")
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())

'''
python3 -m sanet_prep.prepare_luna16 --allow-no-subset-fallback --check-annotation-overlap
'''
