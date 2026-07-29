"""Common utilities for converting lung CT datasets to SANet/PN9-style npy format.

Target image format:
    {pid}_zoom.npy containing float32 array with shape [1, D, H, W], values in [0, 255].

Target annotation format:
    CSV columns: pid,zmin,zmax,ymin,ymax,xmin,xmax
    Coordinates are voxel indices in the processed 1 mm isotropic image, order z,y,x.

The scripts intentionally output numeric pids and a pid_map.csv so that the original SANet
code path `int(fn)` can still work without modifying bbox_reader.py.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SimpleITK is required. Install it with `pip install SimpleITK`."
    ) from exc


SANET_ANNO_COLUMNS = ["pid", "zmin", "zmax", "ymin", "ymax", "xmin", "xmax", "nodule_id"]


@dataclass
class CaseMeta:
    original_id: str
    sanet_pid: int
    image_path: str
    original_shape_zyx: Tuple[int, int, int]
    processed_shape_zyx: Tuple[int, int, int]
    original_spacing_xyz: Tuple[float, float, float]
    processed_spacing_xyz: Tuple[float, float, float]
    original_min: float
    original_max: float
    processed_min: float
    processed_max: float
    n_bboxes: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "original_id": self.original_id,
            "sanet_pid": self.sanet_pid,
            "image_path": self.image_path,
            "original_shape_zyx": "x".join(map(str, self.original_shape_zyx)),
            "processed_shape_zyx": "x".join(map(str, self.processed_shape_zyx)),
            "original_spacing_xyz": ",".join(f"{v:.6g}" for v in self.original_spacing_xyz),
            "processed_spacing_xyz": ",".join(f"{v:.6g}" for v in self.processed_spacing_xyz),
            "original_min": self.original_min,
            "original_max": self.original_max,
            "processed_min": self.processed_min,
            "processed_max": self.processed_max,
            "n_bboxes": self.n_bboxes,
        }


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def natural_key(s: str) -> List[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def extract_first_int(text: str) -> Optional[int]:
    m = re.search(r"\d+", str(text))
    return int(m.group(0)) if m else None


def normalize_original_id(path_or_id: Path | str) -> str:
    """Return a stable original id from a path or string.

    Examples:
        LNDb-0001.mhd -> LNDb-0001
        0001/ -> 0001
        ID_CT.nii.gz -> ID
    """
    s = str(path_or_id)
    name = Path(s).name
    for suffix in [".nii.gz", ".mhd", ".raw", ".npy", ".npz", ".csv", ".gz", ".nii"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    for tail in ["_CT", "_ct", "_image", "_img", "_scan", "_tumor", "_mask", "-mask", "_label", "_seg"]:
        if name.endswith(tail):
            name = name[: -len(tail)]
    return name


def build_numeric_pid_map(original_ids: Sequence[str], start: int = 1) -> Dict[str, int]:
    ids = sorted(set(map(str, original_ids)), key=natural_key)
    return {oid: i + start for i, oid in enumerate(ids)}


def pid_to_filename(pid: int, width: int = 6) -> str:
    return f"{int(pid):0{width}d}"


def read_image(path: Path | str) -> "sitk.Image":
    path = Path(path)
    if path.is_dir():
        return read_dicom_series(path)
    return sitk.ReadImage(str(path))


def read_dicom_series(dicom_dir: Path | str) -> "sitk.Image":
    dicom_dir = Path(dicom_dir)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found under {dicom_dir}")
    # Pick the first series by default. If multiple series exist, users should split folders beforehand.
    file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(file_names)
    return reader.Execute()


def image_stats(img: "sitk.Image") -> Tuple[Tuple[int, int, int], Tuple[float, float, float], float, float]:
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    shape_zyx = tuple(int(v) for v in arr.shape)
    spacing_xyz = tuple(float(v) for v in img.GetSpacing())
    return shape_zyx, spacing_xyz, float(np.min(arr)), float(np.max(arr))


def maybe_convert_to_hu_from_dicom_like(img: "sitk.Image") -> "sitk.Image":
    """SimpleITK usually applies DICOM rescale slope/intercept during ReadImage.

    This function is a placeholder for explicit pipelines and returns the input unchanged.
    Keeping it avoids a common mistake: reapplying slope/intercept twice.
    """
    return img


def resample_image(
    img: "sitk.Image",
    out_spacing_xyz: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    is_label: bool = False,
    default_value: float = 0.0,
) -> "sitk.Image":
    """Resample a 3D SimpleITK image to target spacing.

    SimpleITK uses x,y,z ordering for spacing and size. `GetArrayFromImage` uses z,y,x.
    """
    in_spacing = np.array(img.GetSpacing(), dtype=np.float64)
    in_size = np.array(img.GetSize(), dtype=np.int64)
    out_spacing = np.array(out_spacing_xyz, dtype=np.float64)
    out_size = np.maximum(1, np.round(in_size * in_spacing / out_spacing).astype(np.int64))

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(x) for x in out_spacing))
    resampler.SetSize([int(x) for x in out_size])
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(img)


def sanet_window_from_hu(arr_zyx: np.ndarray, hu_min: float = -1200.0, hu_max: float = 600.0) -> np.ndarray:
    arr = arr_zyx.astype(np.float32, copy=False)
    arr = np.clip(arr, hu_min, hu_max)
    arr = (arr - hu_min) / (hu_max - hu_min) * 255.0
    return arr.astype(np.float32, copy=False)


def image_to_sanet_array(
    img: "sitk.Image",
    assume_already_0_255: bool = False,
    auto_skip_window_if_0_255: bool = False,
    hu_min: float = -1200.0,
    hu_max: float = 600.0,
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
        out = sanet_window_from_hu(arr, hu_min=hu_min, hu_max=hu_max)  # hu截断
        # Keep original HU values without windowing
        # out = arr.astype(np.float32)
    return out[None, ...]  # [1,D,H,W]


def physical_center_radius_to_bbox_zyx(
    img: "sitk.Image",
    center_xyz_mm: Tuple[float, float, float],
    radius_mm: Tuple[float, float, float] | float,
) -> Tuple[int, int, int, int, int, int]:
    """Convert physical center/radius to processed-image voxel bbox in z,y,x order.

    Handles image direction through SimpleITK physical/index transforms.
    For non-axis-aligned direction matrices, it transforms the 8 physical corners and takes the index bounds.
    """
    if isinstance(radius_mm, (float, int)):
        rx = ry = rz = float(radius_mm)
    else:
        rx, ry, rz = map(float, radius_mm)
    cx, cy, cz = map(float, center_xyz_mm)
    corners = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                p = (cx + sx * rx, cy + sy * ry, cz + sz * rz)
                idx_xyz = img.TransformPhysicalPointToContinuousIndex(p)
                corners.append(idx_xyz)
    corners = np.array(corners, dtype=np.float64)  # x,y,z index order
    mins_xyz = np.floor(corners.min(axis=0)).astype(int)
    maxs_xyz = np.ceil(corners.max(axis=0)).astype(int)
    size_xyz = np.array(img.GetSize(), dtype=int)
    mins_xyz = np.maximum(mins_xyz, 0)
    maxs_xyz = np.minimum(maxs_xyz, size_xyz - 1)
    xmin, ymin, zmin = mins_xyz.tolist()
    xmax, ymax, zmax = maxs_xyz.tolist()
    return int(zmin), int(zmax), int(ymin), int(ymax), int(xmin), int(xmax)


def bbox_from_mask_array(mask_zyx: np.ndarray, min_voxels: int = 1, morph_close: bool = True) -> List[Tuple[int, int, int, int, int, int]]:
    """Return connected-component bboxes from mask array in z,y,x order.
    
    Args:
        mask_zyx: 3D mask array (z, y, x)
        min_voxels: Minimum voxel count to keep a component
        morph_close: If True, apply morphological closing to connect broken regions
    """
    mask = np.asarray(mask_zyx)
    mask = mask > 0
    if not mask.any():
        return []
    try:
        from scipy import ndimage as ndi
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scipy is required for connected component extraction from masks.") from exc
    
    # Apply morphological closing to connect regions broken by resampling # 形态学闭运算：连接因重采样断裂的区域
    if morph_close:
        # Use a 3x3x3 cube structuring element to connect nearby regions
        structure = ndi.generate_binary_structure(3, 1)  # 6-connectivity
        mask = ndi.binary_closing(mask, structure=structure, iterations=2)
    
    lab, n = ndi.label(mask)
    bboxes: List[Tuple[int, int, int, int, int, int]] = []
    for i in range(1, n + 1):
        coords = np.where(lab == i)
        if len(coords[0]) < min_voxels:
            continue
        z, y, x = coords
        bboxes.append((int(z.min()), int(z.max()), int(y.min()), int(y.max()), int(x.min()), int(x.max())))
    return bboxes


def bbox_from_label_image(mask_img: "sitk.Image", min_voxels: int = 1, morph_close: bool = False) -> List[Tuple[int, int, int, int, int, int]]:
    arr = sitk.GetArrayFromImage(mask_img)
    return bbox_from_mask_array(arr, min_voxels=min_voxels, morph_close=morph_close)


def clip_bbox_to_shape(
    bbox: Tuple[int, int, int, int, int, int], shape_zyx: Tuple[int, int, int]
) -> Optional[Tuple[int, int, int, int, int, int]]:
    zmin, zmax, ymin, ymax, xmin, xmax = [int(round(v)) for v in bbox]
    D, H, W = shape_zyx
    zmin, zmax = max(0, zmin), min(D - 1, zmax)
    ymin, ymax = max(0, ymin), min(H - 1, ymax)
    xmin, xmax = max(0, xmin), min(W - 1, xmax)
    if zmax < zmin or ymax < ymin or xmax < xmin:
        return None
    return zmin, zmax, ymin, ymax, xmin, xmax


def save_sanet_npy(arr_1zyx: np.ndarray, out_path: Path | str) -> None:
    arr = np.asarray(arr_1zyx, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[0] != 1:
        raise ValueError(f"SANet array must have shape [1,D,H,W], got {arr.shape}")
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    np.save(str(out_path), arr)


def write_list_file(path: Path | str, pids: Iterable[int], width: int = 6) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for pid in pids:
            f.write(pid_to_filename(int(pid), width=width) + "\n")


def write_annotation_csv(path: Path | str, rows: Iterable[Dict[str, int]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SANET_ANNO_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: int(row[c]) for c in SANET_ANNO_COLUMNS})


def write_pid_map(path: Path | str, pid_map: Dict[str, int], width: int = 6) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["original_id", "sanet_pid", "sanet_filename_stem"])
        writer.writeheader()
        for oid, pid in sorted(pid_map.items(), key=lambda kv: kv[1]):
            writer.writerow({"original_id": oid, "sanet_pid": pid, "sanet_filename_stem": pid_to_filename(pid, width)})


def write_meta_csv(path: Path | str, metas: Sequence[CaseMeta]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    pd.DataFrame([m.as_dict() for m in metas]).to_csv(path, index=False)


def write_dataset_summary(path: Path | str, dataset_name: str, metas: Sequence[CaseMeta], split: Dict[str, List[int]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    summary = {
        "dataset_name": dataset_name,
        "num_cases": len(metas),
        "num_boxes": int(sum(m.n_bboxes for m in metas)),
        "split_counts": {k: len(v) for k, v in split.items()},
        "npy_format": "float32 [1,D,H,W], intensity [0,255]",
        "spacing": "1mm x 1mm x 1mm",
        "annotation_csv": SANET_ANNO_COLUMNS,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def find_files(root: Path | str, exts: Sequence[str], recursive: bool = True) -> List[Path]:
    root = Path(root)
    files: List[Path] = []
    pattern_iter = root.rglob("*") if recursive else root.glob("*")
    exts_l = tuple(e.lower() for e in exts)
    for p in pattern_iter:
        if p.is_file() and str(p).lower().endswith(exts_l):
            files.append(p)
    return sorted(files, key=lambda p: natural_key(str(p)))


def read_split_ids_from_dir(split_dir: Path | str) -> Dict[str, List[str]]:
    """Read train/val/test txt files from a split directory if present.

    Lines may contain ids with or without suffixes; comments beginning with # are ignored.
    Returns original ids as strings.
    """
    split_dir = Path(split_dir)
    out: Dict[str, List[str]] = {}
    aliases = {
        "train": ["train.txt", "training.txt", "train_ids.txt", "train_list.txt"],
        "val": ["val.txt", "valid.txt", "validation.txt", "val_ids.txt", "valid_ids.txt"],
        "test": ["test.txt", "testing.txt", "test_ids.txt", "test_list.txt"],
    }
    for name, cand in aliases.items():
        found = None
        for fn in cand:
            p = split_dir / fn
            if p.exists():
                found = p
                break
        if found is None:
            continue
        ids: List[str] = []
        with found.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # use first token to support lines like "pid label"
                token = line.split()[0].split(",")[0]
                ids.append(normalize_original_id(token))
        out[name] = ids
    return out


def derive_split_from_ids(
    original_ids: Sequence[str],
    split_dir: Optional[Path | str],
    pid_map: Dict[str, int],
    ratio: Optional[Tuple[float, float, float]] = None,
    seed: int = 2026,
    allow_fallback: bool = False,
) -> Dict[str, List[int]]:
    """Convert original split files to numeric pid split.

    If split_dir is absent or incomplete and allow_fallback=True, use a deterministic ratio split.
    Otherwise raise, because silently inventing splits hurts reproducibility.
    """
    all_ids = sorted(set(map(str, original_ids)), key=natural_key)
    split_ids: Dict[str, List[str]] = {}
    if split_dir is not None:
        split_ids = read_split_ids_from_dir(split_dir)

    if "train" in split_ids and "test" in split_ids:
        # If no val exists, keep val empty. Some original datasets are train/test only.
        split_ids.setdefault("val", [])
    elif allow_fallback and ratio is not None:
        rng = np.random.default_rng(seed)
        shuffled = np.array(all_ids, dtype=object)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(n * ratio[0] / sum(ratio)))
        n_val = int(round(n * ratio[1] / sum(ratio)))
        split_ids = {
            "train": list(shuffled[:n_train]),
            "val": list(shuffled[n_train : n_train + n_val]),
            "test": list(shuffled[n_train + n_val :]),
        }
    else:
        raise RuntimeError(
            "Could not find original split files. Pass --split-dir containing train.txt/test.txt "
            "or set --allow-fallback-split with an explicit ratio."
        )

    known = set(pid_map)
    numeric_split: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for k in ["train", "val", "test"]:
        for oid in split_ids.get(k, []):
            oid_norm = normalize_original_id(oid)
            if oid_norm not in known:
                # Try numeric-only matching, useful for 0001 vs 1.mhd.
                oi = extract_first_int(oid_norm)
                matches = [x for x in known if extract_first_int(x) == oi] if oi is not None else []
                if len(matches) == 1:
                    oid_norm = matches[0]
                else:
                    raise KeyError(f"Split id {oid!r} not found in image ids. Example known ids: {list(sorted(known))[:5]}")
            numeric_split[k].append(int(pid_map[oid_norm]))

    # Remove accidental duplicates while preserving order.
    seen = set()
    for k in ["train", "val", "test"]:
        clean = []
        for pid in numeric_split[k]:
            if pid not in seen:
                clean.append(pid)
                seen.add(pid)
        numeric_split[k] = clean

    return numeric_split


def split_annotations(all_rows: Sequence[Dict[str, int]], split: Dict[str, List[int]]) -> Dict[str, List[Dict[str, int]]]:
    pid_to_split = {}
    for sname, pids in split.items():
        for pid in pids:
            pid_to_split[int(pid)] = sname
    out = {"train": [], "val": [], "test": []}
    for row in all_rows:
        s = pid_to_split.get(int(row["pid"]))
        if s:
            out[s].append(row)
    return out


def save_common_outputs(
    out_dir: Path | str,
    dataset_name: str,
    pid_map: Dict[str, int],
    split: Dict[str, List[int]],
    ann_rows: Sequence[Dict[str, int]],
    metas: Sequence[CaseMeta],
    width: int = 6,
) -> None:
    out_dir = Path(out_dir)
    split_dir = ensure_dir(out_dir / "split")
    ensure_dir(out_dir / "full")

    write_pid_map(out_dir / "pid_map.csv", pid_map, width=width)
    write_meta_csv(out_dir / "meta.csv", metas)

    for s in ["train", "val", "test"]:
        write_list_file(split_dir / f"{s}.txt", split.get(s, []), width=width)

    write_annotation_csv(split_dir / "all_anno.csv", ann_rows)
    ann_by_split = split_annotations(ann_rows, split)
    # SANet original config reads train_anno for train and val; keep train+val together.
    write_annotation_csv(split_dir / "train_anno.csv", ann_by_split["train"] + ann_by_split["val"])
    write_annotation_csv(split_dir / "val_anno.csv", ann_by_split["val"])
    write_annotation_csv(split_dir / "test_anno.csv", ann_by_split["test"])
    write_dataset_summary(out_dir / "dataset_summary.json", dataset_name, metas, split)

    # Small config helper. Users still need to patch paths in SANet config.py.
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


def get_col(df: pd.DataFrame, aliases: Sequence[str], required: bool = True) -> Optional[str]:
    lower = {c.lower().strip(): c for c in df.columns}
    compact = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    for a in aliases:
        al = a.lower().strip()
        if al in lower:
            return lower[al]
        ac = re.sub(r"[^a-z0-9]", "", al)
        if ac in compact:
            return compact[ac]
    if required:
        raise KeyError(f"None of aliases {aliases} found. Available columns: {list(df.columns)}")
    return None


def convert_bbox_coords_by_spacing(
    bbox_zyx: Tuple[float, float, float, float, float, float],
    old_spacing_xyz: Tuple[float, float, float],
    new_spacing_xyz: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[int, int, int, int, int, int]:
    """Scale a bbox from original voxel coordinates to resampled voxel coordinates.

    Input bbox is zmin,zmax,ymin,ymax,xmin,xmax.
    Spacing is x,y,z as in SimpleITK.
    """
    zmin, zmax, ymin, ymax, xmin, xmax = bbox_zyx
    sx, sy, sz = old_spacing_xyz
    nsx, nsy, nsz = new_spacing_xyz
    return (
        int(math.floor(zmin * sz / nsz)),
        int(math.ceil(zmax * sz / nsz)),
        int(math.floor(ymin * sy / nsy)),
        int(math.ceil(ymax * sy / nsy)),
        int(math.floor(xmin * sx / nsx)),
        int(math.ceil(xmax * sx / nsx)),
    )


def link_or_copy(src: Path | str, dst: Path | str, mode: str = "symlink") -> None:
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        raise ValueError("mode must be 'copy' or 'symlink'")

def resample_to_reference(
    img: "sitk.Image",
    ref: "sitk.Image",
    is_label: bool = False,
    default_value: float = 0.0,
) -> "sitk.Image":
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(img)


def maybe_split_train_to_val(split: Dict[str, List[int]], val_frac_from_train: float, seed: int = 2026) -> Dict[str, List[int]]:
    if val_frac_from_train <= 0:
        return split
    train = list(split.get("train", []))
    if not train:
        return split
    rng = np.random.default_rng(seed)
    arr = np.array(train, dtype=int)
    rng.shuffle(arr)
    n_val = max(1, int(round(len(arr) * val_frac_from_train)))
    val_new = list(map(int, arr[:n_val]))
    train_new = list(map(int, arr[n_val:]))
    out = {k: list(v) for k, v in split.items()}
    out["train"] = sorted(train_new)
    out["val"] = sorted(list(out.get("val", [])) + val_new)
    return out
