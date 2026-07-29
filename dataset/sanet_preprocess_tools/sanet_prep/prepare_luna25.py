from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from .common import (
    CaseMeta,
    build_numeric_pid_map,
    clip_bbox_to_shape,
    derive_split_from_ids,
    ensure_dir,
    extract_first_int,
    find_files,
    get_col,
    image_stats,
    image_to_sanet_array,
    natural_key,
    normalize_original_id,
    pid_to_filename,
    read_image,
    resample_image,
    resample_to_reference,
    save_common_outputs,
    save_sanet_npy,
)


IMAGE_DIR_NAMES = ("luna25_images_nii", "luna25_images")
IMAGE_EXTS = (".nii.gz", ".nii", ".mha", ".mhd")
MASK_EXTS = (".nii.gz", ".nii", ".mha", ".mhd")


def strip_luna25_suffix(path_or_id: Path | str) -> str:
    oid = str(Path(str(path_or_id)).name)
    for suffix in [".nii.gz", ".mha.gz", ".mhd.gz", ".nii", ".mha", ".mhd", ".npy", ".npz", ".csv", ".gz"]:
        if oid.endswith(suffix):
            oid = oid[: -len(suffix)]
            break
    oid = normalize_original_id(oid)
    changed = True
    while changed:
        changed = False
        for tail in [
            "_nodule_mask",
            "_lesion_mask",
            "_tumor_mask",
            "_nodule",
            "_lesion",
            "_tumor",
            "_lungseg",
            "_lung",
            "_mask",
            "_seg",
            "_label",
            "_CT",
            "_ct",
        ]:
            if oid.endswith(tail):
                oid = oid[: -len(tail)]
                changed = True
    return oid


def collect_images(root: Path) -> Dict[str, Path]:
    files: List[Path] = []
    for dirname in IMAGE_DIR_NAMES:
        d = root / dirname
        if d.exists():
            files.extend(find_files(d, IMAGE_EXTS, recursive=True))
    if not files:
        files = [
            p
            for p in find_files(root, IMAGE_EXTS, recursive=True)
            if not any(part.lower() in {"lungseg", "output", "mask", "masks", "label", "labels", "seg"} for part in p.parts)
        ]
    images: Dict[str, Path] = {}
    for p in files:
        oid = strip_luna25_suffix(p)
        if oid in images and images[oid] != p:
            # Prefer NIfTI converted images when both .mha and .nii.gz exist.
            if "luna25_images_nii" in {part.lower() for part in p.parts}:
                images[oid] = p
            continue
        images[oid] = p
    if not images:
        raise FileNotFoundError(f"No LUNA25 CT images found under {root}")
    return images


def collect_masks(mask_root: Path) -> Dict[str, Path]:
    masks: Dict[str, Path] = {}
    for p in find_files(mask_root, MASK_EXTS, recursive=True):
        oid = strip_luna25_suffix(p)
        if oid in masks and masks[oid] != p:
            raise RuntimeError(f"Duplicate mask for {oid}: {masks[oid]} and {p}")
        masks[oid] = p
    if not masks:
        raise FileNotFoundError(f"No mask files found under {mask_root}")
    return masks


def match_by_id(case_id: str, masks: Dict[str, Path], mask_kind: str) -> Path:
    if case_id in masks:
        return masks[case_id]
    n = extract_first_int(case_id)
    matches = [oid for oid in masks if extract_first_int(oid) == n] if n is not None else []
    if len(matches) == 1:
        return masks[matches[0]]
    raise FileNotFoundError(f"No {mask_kind} mask matched case {case_id}. Example mask ids: {list(sorted(masks, key=natural_key))[:5]}")


def crop_to_lung_mask_bbox(img: "sitk.Image", lung_mask_img: "sitk.Image") -> "sitk.Image":
    lung_mask_ref = resample_to_reference(lung_mask_img, img, is_label=True, default_value=0)
    coords = np.where(sitk.GetArrayFromImage(lung_mask_ref) > 0)
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


def bbox_with_margin(
    bbox: Tuple[int, int, int, int, int, int],
    shape_zyx: Tuple[int, int, int],
    margin: int,
) -> Optional[Tuple[int, int, int, int, int, int]]:
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    m = int(margin)
    return clip_bbox_to_shape((zmin - m, zmax + m, ymin - m, ymax + m, xmin - m, xmax + m), shape_zyx)


def extract_lesion_bboxes_from_mask(
    mask_zyx: np.ndarray,
    min_voxels: int = 1,
    connectivity: int = 26,
) -> List[Tuple[int, int, int, int, int, int]]:
    """Extract one bbox per lesion from a binary or multi-label lesion mask."""
    try:
        from scipy import ndimage as ndi
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scipy is required for connected component extraction. Install with `pip install scipy`.") from exc

    arr = np.asarray(mask_zyx)
    labels = [v for v in np.unique(arr) if v != 0]
    if not labels:
        return []

    conn = 3 if int(connectivity) >= 26 else (2 if int(connectivity) >= 18 else 1)
    structure = ndi.generate_binary_structure(3, conn)
    bboxes: List[Tuple[int, int, int, int, int, int]] = []
    for label in labels:
        lab_arr, n_comp = ndi.label(arr == label, structure=structure)
        for comp_id in range(1, n_comp + 1):
            coords = np.where(lab_arr == comp_id)
            if len(coords[0]) < min_voxels:
                continue
            z, y, x = coords
            bboxes.append((int(z.min()), int(z.max()), int(y.min()), int(y.max()), int(x.min()), int(x.max())))
    return sorted(bboxes, key=lambda b: (b[0], b[2], b[4]))


def split_from_table(table_path: Path, case_ids: List[str], pid_map: Dict[str, int]) -> Dict[str, List[int]]:
    if str(table_path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(table_path)
    else:
        df = pd.read_csv(table_path)
    id_col = get_col(df, ["seriesinstanceuid", "series_uid", "seriesuid", "case", "id", "scan", "scan_id", "filename", "file"])
    split_col = get_col(df, ["split", "set", "subset", "train_test", "dataset"], required=False)
    if split_col is None:
        raise KeyError("Split table must contain a split/set column with train/val/test values.")

    known = set(case_ids)
    out = {"train": [], "val": [], "test": []}
    for _, row in df.iterrows():
        oid = strip_luna25_suffix(str(row[id_col]))
        if oid not in known:
            continue
        split_name = str(row[split_col]).lower()
        if "train" in split_name:
            out["train"].append(pid_map[oid])
        elif "val" in split_name or "valid" in split_name:
            out["val"].append(pid_map[oid])
        elif "test" in split_name:
            out["test"].append(pid_map[oid])
    if not out["train"] or not out["test"]:
        raise RuntimeError(f"Split table {table_path} did not yield non-empty train/test splits.")
    return out


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    full_dir = ensure_dir(out_dir / "full")

    images = collect_images(root)
    lung_masks = collect_masks(Path(args.lung_mask_root))
    lesion_masks = collect_masks(Path(args.lesion_mask_root))
    case_ids = list(images)
    pid_map = build_numeric_pid_map(case_ids)

    all_rows: List[Dict[str, int]] = []
    metas: List[CaseMeta] = []
    for oid, img_path in tqdm(sorted(images.items(), key=lambda kv: natural_key(kv[0])), desc="LUNA25 -> SANet npy"):
        lung_mask_path = match_by_id(oid, lung_masks, "lung")
        lesion_mask_path = match_by_id(oid, lesion_masks, "lesion")

        img = read_image(img_path)
        orig_shape, orig_spacing, orig_min, orig_max = image_stats(img)
        img_cropped = crop_to_lung_mask_bbox(img, read_image(lung_mask_path))
        img_1mm = resample_image(img_cropped, out_spacing_xyz=(1.0, 1.0, 1.0), is_label=False, default_value=args.pad_hu)

        lesion_mask_ref = resample_to_reference(read_image(lesion_mask_path), img_1mm, is_label=True, default_value=0)
        lesion_mask_arr = sitk.GetArrayFromImage(lesion_mask_ref)

        arr = image_to_sanet_array(
            img_1mm,
            assume_already_0_255=args.assume_already_0_255,
            auto_skip_window_if_0_255=args.auto_skip_window_if_0_255,
        )
        proc_shape = tuple(arr.shape[1:])
        sanet_pid = pid_map[oid]
        save_sanet_npy(arr, full_dir / f"{pid_to_filename(sanet_pid, args.pid_width)}_zoom.npy")

        bboxes = extract_lesion_bboxes_from_mask(
            lesion_mask_arr,
            min_voxels=args.min_mask_voxels,
            connectivity=args.component_connectivity,
        )
        n_boxes = 0
        for nodule_id, bbox in enumerate(bboxes):
            bbox = bbox_with_margin(bbox, proc_shape, args.bbox_margin)
            if bbox is None:
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
            n_boxes += 1

        metas.append(
            CaseMeta(
                original_id=oid,
                sanet_pid=sanet_pid,
                image_path=str(img_path),
                original_shape_zyx=orig_shape,
                processed_shape_zyx=proc_shape,
                original_spacing_xyz=orig_spacing,
                processed_spacing_xyz=(1.0, 1.0, 1.0),
                original_min=orig_min,
                original_max=orig_max,
                processed_min=float(arr.min()),
                processed_max=float(arr.max()),
                n_bboxes=n_boxes,
            )
        )

    if args.split_table:
        split = split_from_table(Path(args.split_table), case_ids, pid_map)
    elif args.split_dir:
        split = derive_split_from_ids(case_ids, args.split_dir, pid_map, allow_fallback=False)
    else:
        split = derive_split_from_ids(
            case_ids,
            None,
            pid_map,
            ratio=tuple(args.fallback_ratio),
            seed=args.seed,
            allow_fallback=args.allow_fallback_split,
        )
    save_common_outputs(out_dir, "LUNA25", pid_map, split, all_rows, metas, width=args.pid_width)
    print(f"Done. Output: {out_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare LUNA25 CT/lung masks/lesion masks to SANet/PN9-style npy and bbox CSV.")
    p.add_argument("--root", default="/data/医保大赛/code/SANet/data/luna25", help="LUNA25 root directory.")
    p.add_argument("--out-dir", default="../../data/LUNA25")
    p.add_argument("--lung-mask-root", default="/data/医保大赛/code/SANet/data/luna25/lungseg",
                   help="Directory containing lung masks used to crop CT before saving npy.")
    p.add_argument("--lesion-mask-root", default="/data/医保大赛/code/SANet/data/luna25/output",
                   help="Directory containing per-case nodule/lesion masks. A multi-label or binary mask may contain multiple lesions.")
    p.add_argument("--split-dir", default=None, help="Directory containing train.txt/val.txt/test.txt.")
    p.add_argument("--split-table", default=None, help="CSV/XLSX table containing SeriesInstanceUID and split columns.")
    p.add_argument("--allow-fallback-split", action="store_true", help="If no split is provided, use deterministic fallback ratio.")
    p.add_argument("--fallback-ratio", nargs=3, type=float, default=[7, 1, 2], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--min-mask-voxels", type=int, default=1,
                   help="Minimum voxels for a connected lesion component to be kept after 1mm resampling.")
    p.add_argument("--component-connectivity", type=int, default=26, choices=[6, 18, 26])
    p.add_argument("--bbox-margin", type=int, default=1, help="Voxel margin added to each lesion bbox after extraction.")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--pid-width", type=int, default=5)
    p.add_argument("--pad-hu", type=float, default=-1200.0)
    p.add_argument("--assume-already-0-255", action="store_true")
    p.add_argument("--auto-skip-window-if-0-255", action="store_true")
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())


"""
Example:
python3 -m sanet_prep.prepare_luna25 --allow-fallback-split
"""
