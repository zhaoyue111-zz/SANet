from __future__ import annotations

import argparse
import json
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
    natural_key,
    normalize_original_id,
    pid_to_filename,
    read_image,
    resample_image,
    resample_to_reference,
    save_common_outputs,
    save_sanet_npy,
)


IMAGE_EXTS = (".nii.gz", ".nii", ".mha", ".mhd")


def strip_lndb_suffix(path_or_id: Path | str) -> str:
    oid = str(Path(str(path_or_id)).name)
    for suffix in [".nii.gz", ".mha.gz", ".mhd.gz", ".nii", ".mha", ".mhd", ".gz"]:
        if oid.endswith(suffix):
            oid = oid[: -len(suffix)]
            break
    oid = normalize_original_id(oid)
    changed = True
    while changed:
        changed = False
        for tail in ["_ct", "_CT", "_image", "_img", "_scan", "_mask", "_label", "_seg"]:
            if oid.endswith(tail):
                oid = oid[: -len(tail)]
                changed = True
    return oid


def strip_lndb_nodule_mask_suffix(path_or_id: Path | str) -> str:
    oid = str(Path(str(path_or_id)).name)
    for suffix in [".nii.gz", ".mha.gz", ".mhd.gz", ".nii", ".mha", ".mhd", ".gz"]:
        if oid.endswith(suffix):
            oid = oid[: -len(suffix)]
            break
    marker = "_nid"
    if marker in oid:
        oid = oid.split(marker)[0]
    return strip_lndb_suffix(oid)


def collect_images(root: Path) -> Dict[str, Path]:
    files = [
        p for p in find_files(root, IMAGE_EXTS, recursive=True)
        if p.name.lower().endswith("_ct.nii.gz")
        or p.name.lower().endswith("_ct.nii")
        or p.name.lower().endswith("_ct.mha")
        or p.name.lower().endswith("_ct.mhd")
    ]
    if not files:
        files = [
            p for p in find_files(root, IMAGE_EXTS, recursive=True)
            if not any(token in p.name.lower() for token in ["mask", "label", "seg", "_nid"])
        ]

    images: Dict[str, Path] = {}
    for path in files:
        oid = strip_lndb_suffix(path)
        if oid in images and images[oid] != path:
            raise RuntimeError(f"Duplicate LNDb CT for {oid}: {images[oid]} and {path}")
        images[oid] = path
    if not images:
        raise FileNotFoundError(f"No LNDb CT files found under {root}")
    return images


def collect_nodule_masks(root: Path) -> Dict[str, List[Path]]:
    masks: Dict[str, List[Path]] = {}
    for path in find_files(root, IMAGE_EXTS, recursive=True):
        name = path.name.lower()
        if "_nid" not in name or "mask" not in name:
            continue
        oid = strip_lndb_nodule_mask_suffix(path)
        masks.setdefault(oid, []).append(path)
    if not masks:
        raise FileNotFoundError(f"No LNDb nodule mask files like *_nid*_mask.nii.gz found under {root}")
    return {k: sorted(v, key=lambda p: natural_key(p.name)) for k, v in masks.items()}


def collect_lung_masks(root: Path) -> Dict[str, Path]:
    masks: Dict[str, Path] = {}
    for path in find_files(root, IMAGE_EXTS, recursive=True):
        oid = strip_lndb_suffix(path)
        if oid in masks and masks[oid] != path:
            raise RuntimeError(f"Duplicate LNDb lung mask for {oid}: {masks[oid]} and {path}")
        masks[oid] = path
    if not masks:
        raise FileNotFoundError(f"No LNDb lung masks found under {root}")
    return masks


def match_by_number(case_id: str, paths: Dict[str, Path], kind: str) -> Path:
    if case_id in paths:
        return paths[case_id]
    case_num = extract_first_int(case_id)
    matches = [oid for oid in paths if extract_first_int(oid) == case_num] if case_num is not None else []
    if len(matches) == 1:
        return paths[matches[0]]
    raise FileNotFoundError(f"No {kind} matched case {case_id}. Examples: {list(sorted(paths, key=natural_key))[:5]}")


def match_mask_list_by_number(case_id: str, masks: Dict[str, List[Path]]) -> List[Path]:
    if case_id in masks:
        return masks[case_id]
    case_num = extract_first_int(case_id)
    matches = [oid for oid in masks if extract_first_int(oid) == case_num] if case_num is not None else []
    if len(matches) == 1:
        return masks[matches[0]]
    raise FileNotFoundError(f"No nodule masks matched case {case_id}. Examples: {list(sorted(masks, key=natural_key))[:5]}")


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


def lndb_image_to_sanet_array(img: "sitk.Image", input_scale: str = "auto") -> np.ndarray:
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    amin, amax = float(np.min(arr)), float(np.max(arr))

    if input_scale == "auto":
        if amin >= -1e-6 and amax <= 1.0 + 1e-6:
            out = np.clip(arr, 0.0, 1.0) * 255.0
        elif amin >= -1e-6 and amax <= 255.0 + 1e-6:
            out = np.clip(arr, 0.0, 255.0)
        else:
            out = np.clip(arr, -1200.0, 600.0)
            out = (out + 1200.0) / 1800.0 * 255.0
    elif input_scale == "zero_one":
        out = np.clip(arr, 0.0, 1.0) * 255.0
    elif input_scale == "zero_255":
        out = np.clip(arr, 0.0, 255.0)
    elif input_scale == "hu":
        out = np.clip(arr, -1200.0, 600.0)
        out = (out + 1200.0) / 1800.0 * 255.0
    else:
        raise ValueError(f"Unknown input_scale: {input_scale}")

    return out.astype(np.float32, copy=False)[None, ...]


def bbox_from_mask_on_reference(
    mask_path: Path,
    ref_img: "sitk.Image",
    min_voxels: int,
    margin: int,
) -> Optional[Tuple[int, int, int, int, int, int]]:
    mask_ref = resample_to_reference(read_image(mask_path), ref_img, is_label=True, default_value=0)
    mask_arr = sitk.GetArrayFromImage(mask_ref) > 0
    coords = np.where(mask_arr)
    if len(coords[0]) < min_voxels:
        return None

    z, y, x = coords
    zmin, zmax = int(z.min()) - margin, int(z.max()) + margin
    ymin, ymax = int(y.min()) - margin, int(y.max()) + margin
    xmin, xmax = int(x.min()) - margin, int(x.max()) + margin
    return clip_bbox_to_shape((zmin, zmax, ymin, ymax, xmin, xmax), tuple(int(v) for v in mask_arr.shape))


def split_from_table(table_path: Path, case_ids: List[str], pid_map: Dict[str, int]) -> Dict[str, List[int]]:
    df = pd.read_excel(table_path) if str(table_path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(table_path)
    id_col = get_col(df, ["patient", "pid", "case", "id", "scan", "scan_id", "filename", "file"])
    split_col = get_col(df, ["split", "set", "subset", "train_test", "dataset"])

    known = set(case_ids)
    out = {"train": [], "val": [], "test": []}
    for _, row in df.iterrows():
        oid = strip_lndb_suffix(str(row[id_col]))
        if oid not in known:
            n = extract_first_int(oid)
            matches = [x for x in known if extract_first_int(x) == n] if n is not None else []
            if len(matches) != 1:
                continue
            oid = matches[0]

        split_name = str(row[split_col]).strip().lower()
        if "unassigned" in split_name or split_name in {"", "nan", "none"}:
            continue
        if "train" in split_name:
            out["train"].append(pid_map[oid])
        elif "val" in split_name or "valid" in split_name:
            out["val"].append(pid_map[oid])
        elif "test" in split_name:
            out["test"].append(pid_map[oid])

    for key in out:
        seen = set()
        out[key] = [pid for pid in out[key] if not (pid in seen or seen.add(pid))]
    if not out["train"] or not out["test"]:
        raise RuntimeError(f"Split table {table_path} did not yield non-empty train/test splits.")
    return out


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    full_dir = ensure_dir(out_dir / "full")

    images = collect_images(root)
    nodule_masks = collect_nodule_masks(root if args.nodule_mask_root is None else Path(args.nodule_mask_root))
    lung_masks = collect_lung_masks(Path(args.lung_mask_root)) if args.lung_mask_root else {}

    case_ids = sorted(images, key=natural_key)
    if args.max_cases and args.max_cases > 0:
        case_ids = case_ids[:args.max_cases]
    pid_map = build_numeric_pid_map(case_ids)

    all_rows: List[Dict[str, int]] = []
    metas: List[CaseMeta] = []
    warnings: List[Dict[str, object]] = []

    for oid in tqdm(case_ids, desc="LNDb -> SANet npy"):
        img_path = images[oid]
        img = read_image(img_path)
        orig_shape, orig_spacing, orig_min, orig_max = image_stats(img)

        if args.lung_mask_root:
            lung_mask_path = match_by_number(oid, lung_masks, "lung mask")
            img_proc = crop_to_lung_mask_bbox(img, read_image(lung_mask_path))
        else:
            img_proc = img

        img_1mm = resample_image(img_proc, out_spacing_xyz=(1.0, 1.0, 1.0), is_label=False, default_value=args.pad_value)
        arr = lndb_image_to_sanet_array(img_1mm, input_scale=args.input_scale)
        proc_shape = tuple(int(v) for v in arr.shape[1:])

        if float(arr.max() - arr.min()) < args.min_output_dynamic_range:
            warnings.append({
                "case": oid,
                "warning": "low_output_dynamic_range",
                "processed_min": float(arr.min()),
                "processed_max": float(arr.max()),
            })

        sanet_pid = pid_map[oid]
        save_sanet_npy(arr, full_dir / f"{pid_to_filename(sanet_pid, args.pid_width)}_zoom.npy")

        mask_paths = match_mask_list_by_number(oid, nodule_masks)
        n_boxes = 0
        for nodule_id, mask_path in enumerate(mask_paths):
            bbox = bbox_from_mask_on_reference(
                mask_path,
                img_1mm,
                min_voxels=args.min_mask_voxels,
                margin=args.bbox_margin,
            )
            if bbox is None:
                warnings.append({"case": oid, "mask": str(mask_path), "warning": "empty_or_too_small_mask"})
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

    save_common_outputs(out_dir, "LNDb", pid_map, split, all_rows, metas, width=args.pid_width)
    with (out_dir / "preprocess_warnings.json").open("w", encoding="utf-8") as f:
        json.dump(warnings, f, indent=2, ensure_ascii=False)
    print(f"Done. Output: {out_dir}")
    print(f"Cases: {len(metas)}, boxes: {len(all_rows)}, warnings: {len(warnings)}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare LNDb normalized CT + nodule masks to SANet/PN9-style npy.")
    p.add_argument("--root", default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/LNDb/LNDb",
                   help="Directory containing LNDb *_ct.nii.gz and *_nid*_mask.nii.gz files.")
    p.add_argument("--out-dir", default="data/LNDB")
    p.add_argument("--nodule-mask-root", default=None,
                   help="Directory containing LNDb nodule masks. Defaults to --root.")
    p.add_argument("--lung-mask-root", default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/lungseg/LNDb",
                   help="Directory containing lung masks used to crop CT before saving npy. Empty string disables lung cropping.")
    p.add_argument("--split-dir", default=None, help="Directory containing train.txt/val.txt/test.txt.")
    p.add_argument("--split-table", default=None,
                   help="CSV/XLSX with id and split columns. The provided LNDb metadata.csv has UNASSIGNED splits, so fallback split is usually needed.")
    p.add_argument("--allow-fallback-split", action="store_true",
                   help="If no split-dir/table is provided, use deterministic fallback split.")
    p.add_argument("--fallback-ratio", nargs=3, type=float, default=[7, 1, 2], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--input-scale", choices=["auto", "zero_one", "zero_255", "hu"], default="auto",
                   help="LNDb CT files here are normalized 0..1; auto maps them to 0..255.")
    p.add_argument("--min-mask-voxels", type=int, default=1)
    p.add_argument("--bbox-margin", type=int, default=1)
    p.add_argument("--min-output-dynamic-range", type=float, default=10.0,
                   help="Warn if saved [0,255] image range is smaller than this.")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--pid-width", type=int, default=5)
    p.add_argument("--pad-value", type=float, default=0.0,
                   help="Padding value in source scale before intensity conversion. For normalized LNDb, 0 maps to black.")
    p.add_argument("--max-cases", type=int, default=0, help="Debug only: process the first N cases.")
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())


'''
python -m dataset.sanet_preprocess_tools.sanet_prep.prepare_lndb --out-dir data/LNDB --allow-fallback-split
'''
