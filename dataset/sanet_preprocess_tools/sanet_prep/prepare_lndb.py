from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import (
    CaseMeta,
    build_numeric_pid_map,
    clip_bbox_to_shape,
    convert_bbox_coords_by_spacing,
    derive_split_from_ids,
    ensure_dir,
    extract_first_int,
    find_files,
    get_col,
    image_stats,
    image_to_sanet_array,
    normalize_original_id,
    physical_center_radius_to_bbox_zyx,
    pid_to_filename,
    read_image,
    resample_image,
    resample_to_reference,
    save_common_outputs,
    save_sanet_npy,
)


def collect_images(image_root: Path) -> Dict[str, Path]:
    files = find_files(image_root, [".mhd", ".nii.gz", ".nii"], recursive=True)
    # Filter out obvious masks/labels if image_root is a mixed folder.
    files = [p for p in files if not any(t in p.name.lower() for t in ["mask", "label", "seg", "tumor"])]
    if not files:
        raise FileNotFoundError(f"No CT image files (.mhd/.nii.gz/.nii) found under {image_root}")
    return {normalize_original_id(p): p for p in files}


def collect_masks_by_case(mask_root: Path) -> Dict[int, List[Path]]:
    all_files = find_files(mask_root, [".mhd", ".nii.gz", ".nii"], recursive=True)
    all_masks = [
        path for path in all_files
        if any(token in path.name.lower() for token in ["mask", "label", "seg"])
    ]
    masks_by_case: Dict[int, List[Path]] = {}
    for path in all_masks:
        case_number = extract_first_int(normalize_original_id(path))
        if case_number is not None:
            masks_by_case.setdefault(case_number, []).append(path)
    return masks_by_case


def collect_lung_masks_by_case(mask_root: Path) -> Dict[int, Path]:
    all_files = find_files(mask_root, [".mhd", ".nii.gz", ".nii"], recursive=True)
    masks_by_case: Dict[int, Path] = {}
    for path in all_files:
        case_number = extract_first_int(normalize_original_id(path))
        if case_number is not None:
            masks_by_case.setdefault(case_number, path)
    return masks_by_case


def crop_to_mask_bbox(img, mask_img, default_value: float = 0.0):
    """Crop an image to the foreground bbox of a mask resampled on the image grid."""
    import SimpleITK as sitk

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
    return sitk.RegionOfInterest(img, size_xyz, index_xyz), index_xyz, size_xyz


def expand_bbox_by_one_voxel(
    bbox: Tuple[int, int, int, int, int, int],
    shape_zyx: Tuple[int, int, int],
) -> Optional[Tuple[int, int, int, int, int, int]]:
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    return clip_bbox_to_shape((zmin - 1, zmax + 1, ymin - 1, ymax + 1, xmin - 1, xmax + 1), shape_zyx)


def bboxes_from_masks_on_reference(
    mask_paths: List[Path],
    ref_img,
    min_voxels: int,
) -> List[Tuple[int, int, int, int, int, int]]:
    """Return one expanded bbox per LNDb nodule mask on the processed CT grid."""
    import SimpleITK as sitk

    boxes = []
    ref_shape = tuple(int(v) for v in sitk.GetArrayFromImage(ref_img).shape)
    for mask_path in mask_paths:
        mask = read_image(mask_path)
        mask_ref = resample_to_reference(mask, ref_img, is_label=True, default_value=0)
        coords = np.where(sitk.GetArrayFromImage(mask_ref) > 0)
        if len(coords[0]) < min_voxels:
            continue
        z, y, x = coords
        box = (
            int(z.min()), int(z.max()),
            int(y.min()), int(y.max()),
            int(x.min()), int(x.max()),
        )
        expanded = expand_bbox_by_one_voxel(box, ref_shape)
        if expanded is not None:
            boxes.append(expanded)
    return boxes


def load_generic_annotation_csv(csv_path: Path) -> Tuple[pd.DataFrame, str]:
    """Load bbox or center+diameter annotations.

    Returns (df, mode), where mode is 'bbox' or 'center'.
    BBox columns are assumed x/y/z image voxel coordinates unless --bbox-space says physical.
    Center columns are interpreted by --center-space.
    """
    df = pd.read_csv(csv_path)
    id_col = get_col(df, ["pid", "patient", "patient_id", "scan", "scan_id", "LNDbID", "file", "filename", "case", "id"])
    # bbox aliases
    x_min = get_col(df, ["xmin", "x_min", "min_x", "x1"], required=False)
    x_max = get_col(df, ["xmax", "x_max", "max_x", "x2"], required=False)
    y_min = get_col(df, ["ymin", "y_min", "min_y", "y1"], required=False)
    y_max = get_col(df, ["ymax", "y_max", "max_y", "y2"], required=False)
    z_min = get_col(df, ["zmin", "z_min", "min_z", "z1"], required=False)
    z_max = get_col(df, ["zmax", "z_max", "max_z", "z2"], required=False)
    if all(v is not None for v in [x_min, x_max, y_min, y_max, z_min, z_max]):
        out = pd.DataFrame(
            {
                "original_id": df[id_col].astype(str).map(normalize_original_id),
                "xmin": df[x_min].astype(float),
                "xmax": df[x_max].astype(float),
                "ymin": df[y_min].astype(float),
                "ymax": df[y_max].astype(float),
                "zmin": df[z_min].astype(float),
                "zmax": df[z_max].astype(float),
            }
        )
        return out, "bbox"

    # center aliases: LNDb files may use x,y,z or ctrX/ctrY/ctrZ style names.
    cx = get_col(df, ["x", "coordx", "coord_x", "center_x", "x_center", "ctrx", "centroid_x"])
    cy = get_col(df, ["y", "coordy", "coord_y", "center_y", "y_center", "ctry", "centroid_y"])
    cz = get_col(df, ["z", "coordz", "coord_z", "center_z", "z_center", "ctrz", "centroid_z"])
    diam = get_col(df, ["diameter", "diameter_mm", "diam", "d", "size", "VolumeEquivalentDiameter"], required=False)
    rad = get_col(df, ["radius", "radius_mm", "r"], required=False)
    if diam is None and rad is None:
        raise KeyError("Annotation CSV has center columns but no diameter/radius column.")
    out = pd.DataFrame({"original_id": df[id_col].astype(str).map(normalize_original_id), "x": df[cx].astype(float), "y": df[cy].astype(float), "z": df[cz].astype(float)})
    if diam is not None:
        out["radius"] = df[diam].astype(float) / 2.0
    else:
        out["radius"] = df[rad].astype(float)
    return out, "center"


def match_id(anno_id: str, case_ids: List[str]) -> str:
    if anno_id in case_ids:
        return anno_id
    ai = extract_first_int(anno_id)
    matches = [c for c in case_ids if extract_first_int(c) == ai] if ai is not None else []
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"Cannot match annotation id {anno_id!r} to image cases. Example: {case_ids[:5]}")


def run(args: argparse.Namespace) -> None:
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    full_dir = ensure_dir(out_dir / "full")
    cases = collect_images(image_root)
    case_ids = list(cases)
    pid_map = build_numeric_pid_map(case_ids)

    ann_df = None
    ann_mode = None
    if args.anno_csv:
        ann_df, ann_mode = load_generic_annotation_csv(Path(args.anno_csv))
        ann_df["case_id"] = ann_df["original_id"].map(lambda x: match_id(x, case_ids))

    use_masks = bool(args.mask_root) or ann_df is None
    mask_root = Path(args.mask_root) if args.mask_root else image_root
    masks_by_case = collect_masks_by_case(mask_root) if use_masks else {}
    lung_mask_root = Path(args.lung_mask_root)
    lung_masks_by_case = collect_lung_masks_by_case(lung_mask_root)

    all_rows: List[Dict[str, int]] = []
    metas: List[CaseMeta] = []
    for oid, img_path in tqdm(sorted(cases.items()), desc="LNDb -> SANet npy"):
        img = read_image(img_path)
        orig_shape, orig_spacing, orig_min, orig_max = image_stats(img)
        case_number = extract_first_int(oid)
        lung_mask_path = lung_masks_by_case.get(case_number)
        if lung_mask_path is None:
            raise FileNotFoundError(f"No lung mask matched for {oid} under {lung_mask_root}")
        img_cropped, _, _ = crop_to_mask_bbox(img, read_image(lung_mask_path), default_value=args.pad_hu)
        img_1mm = resample_image(img_cropped, out_spacing_xyz=(1.0, 1.0, 1.0), is_label=False, default_value=args.pad_hu)
        arr = image_to_sanet_array(
            img_1mm,
            assume_already_0_255=args.assume_already_0_255,
            auto_skip_window_if_0_255=args.auto_skip_window_if_0_255,
        )
        proc_shape = tuple(arr.shape[1:])
        sanet_pid = pid_map[oid]
        save_sanet_npy(arr, full_dir / f"{pid_to_filename(sanet_pid, args.pid_width)}_zoom.npy")

        bboxes: List[Tuple[int, int, int, int, int, int]] = []
        if use_masks:
            mask_paths = masks_by_case.get(case_number, [])
            if not mask_paths and not args.allow_missing_masks:
                raise FileNotFoundError(f"No mask matched for {oid} under {mask_root}")
            bboxes.extend(
                bboxes_from_masks_on_reference(
                    mask_paths, img_1mm, min_voxels=args.min_mask_voxels
                )
            )
        elif ann_df is not None:
            sub = ann_df[ann_df["case_id"] == oid]
            for _, r in sub.iterrows():
                if ann_mode == "bbox":
                    b = (r.zmin, r.zmax, r.ymin, r.ymax, r.xmin, r.xmax)
                    if args.bbox_space == "voxel":
                        b = convert_bbox_coords_by_spacing(b, old_spacing_xyz=orig_spacing, new_spacing_xyz=(1.0, 1.0, 1.0))
                    else:
                        # physical min/max bbox: use center/radius inferred from physical extents
                        cx = (float(r.xmin) + float(r.xmax)) / 2
                        cy = (float(r.ymin) + float(r.ymax)) / 2
                        cz = (float(r.zmin) + float(r.zmax)) / 2
                        rx = abs(float(r.xmax) - float(r.xmin)) / 2
                        ry = abs(float(r.ymax) - float(r.ymin)) / 2
                        rz = abs(float(r.zmax) - float(r.zmin)) / 2
                        b = physical_center_radius_to_bbox_zyx(img_1mm, (cx, cy, cz), (rx, ry, rz))
                    bboxes.append(b)
                else:
                    if args.center_space == "physical":
                        bboxes.append(physical_center_radius_to_bbox_zyx(img_1mm, (r.x, r.y, r.z), float(r.radius)))
                    else:
                        # center is original voxel coords; radius is assumed mm unless --radius-space voxel.
                        x, y, z = float(r.x), float(r.y), float(r.z)
                        if args.radius_space == "voxel":
                            rx = ry = rz = float(r.radius)
                        else:
                            sx, sy, sz = orig_spacing
                            rx, ry, rz = float(r.radius) / sx, float(r.radius) / sy, float(r.radius) / sz
                        b0 = (z - rz, z + rz, y - ry, y + ry, x - rx, x + rx)
                        bboxes.append(convert_bbox_coords_by_spacing(b0, old_spacing_xyz=orig_spacing, new_spacing_xyz=(1.0, 1.0, 1.0)))
        else:
            raise RuntimeError("Either --mask-root or --anno-csv must be provided to generate labels.")

        n_boxes = 0
        nodule_id = 0
        for b in bboxes:
            b = clip_bbox_to_shape(b, proc_shape)
            if b is None:
                continue
            zmin, zmax, ymin, ymax, xmin, xmax = b
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

    if args.split_dir:
        split = derive_split_from_ids(case_ids, args.split_dir, pid_map, allow_fallback=False)
    else:
        split = derive_split_from_ids(
            case_ids,
            None,
            pid_map,
            ratio=(7, 1, 2),
            seed=args.seed,
            allow_fallback=args.allow_fallback_split,
        )
    save_common_outputs(out_dir, "LNDb", pid_map, split, all_rows, metas, width=args.pid_width)
    print(f"Done. Output: {out_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare LNDb MHD/NIfTI + masks/annotations to SANet/PN9-style npy.")
    p.add_argument("--image-root", default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/LNDb/LNDb", help="Directory containing LNDb CT images (.mhd/.nii.gz/.nii).")
    p.add_argument("--out-dir", default="../../data/LNDB")
    p.add_argument("--mask-root", default=None,
                   help="Directory containing nodule masks. Defaults to --image-root.")
    p.add_argument("--lung-mask-root", default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/lungseg/LNDb",
                   help="Directory containing lung masks used to crop CT before saving npy.")
    p.add_argument("--anno-csv", default=None, help="CSV with bbox or center+diameter annotations if masks are unavailable.")
    p.add_argument("--split-dir", default=None, help="Directory containing official/author train.txt/val.txt/test.txt.")
    p.add_argument("--allow-fallback-split", action="store_true", help="If no split-dir is provided, use deterministic 7:1:2 split.")
    p.add_argument("--bbox-space", choices=["voxel", "physical"], default="voxel",help="指定anno.csv中bbox坐标的空间类型，可忽略，LNDB数据集没有CSV文件")
    p.add_argument("--center-space", choices=["voxel", "physical"], default="physical")
    p.add_argument("--radius-space", choices=["voxel", "physical_mm"], default="physical_mm")
    p.add_argument("--min-mask-voxels", type=int, default=1,help="最小mask体素数,设为1：保留所有mask（包括单像素）")
    p.add_argument("--allow-missing-masks", action="store_true",help="是否允许mask缺失，如果不允许则会在找不到mask时抛出异常")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--pid-width", type=int, default=5,
                   help="Width of zero-padding for SANet pid filenames, e.g. 00001_zoom.npy")
    p.add_argument("--pad-hu", type=float, default=-1200.0,
                   help="HU value to pad if resampling creates out-of-bounds areas; default 0 for histopathology.")
    p.add_argument("--assume-already-0-255", action="store_true",
                   help="Use if MHD values are already 0-255; do not HU-window.")
    p.add_argument("--auto-skip-window-if-0-255", action="store_true",
                   help="Skip HU window automatically if each processed image min/max are within [0,255].")
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())

'''
python3 -m sanet_prep.prepare_lndb --allow-fallback-split --auto-skip-window-if-0-255
'''
