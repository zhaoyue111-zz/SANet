'''
Histopathology_sanet_output/
├── full/
│   └── 000001_zoom.npy          # shape=[1, D, H, W], 值域[0,255]
├── split/
│   ├── train.txt                # 病例ID列表
│   ├── val.txt
│   ├── test.txt
│   ├── train_anno.csv           # pid,zmin,zmax,ymin,ymax,xmin,xmax
│   ├── val_anno.csv
│   └── test_anno.csv
├── pid_map.csv                  # original_id → sanet_pid 映射
├── meta.csv                     # 每个病例的元数据
├── dataset_summary.json         # 数据集统计信息
└── sanet_config_example.json    # 配置示例
'''

from __future__ import annotations

import argparse
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from tqdm import tqdm

from .common import (
    CaseMeta,
    build_numeric_pid_map,
    clip_bbox_to_shape,
    convert_bbox_coords_by_spacing,
    ensure_dir,
    extract_first_int,
    find_files,
    get_col,
    image_stats,
    image_to_sanet_array,
    maybe_split_train_to_val,
    normalize_original_id,
    pid_to_filename,
    read_image,
    resample_image,
    resample_to_reference,
    save_common_outputs,
    save_sanet_npy,
    bbox_from_mask_array,
)

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError(
        "SimpleITK is required. Install it with `pip install SimpleITK`."
    ) from exc


def collect_mhd_cases(mhd_root: Path) -> Dict[str, Path]:
    files = find_files(mhd_root, [".mhd"], recursive=True)
    if not files:
        raise FileNotFoundError(f"No .mhd files found under {mhd_root}")
    cases: Dict[str, Path] = {}
    for p in files:
        oid = normalize_original_id(p)
        cases[oid] = p
    return cases


def load_all_anno_3d(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Flexible aliases for the released CSV and common variants.
    id_col = get_col(df, ["pid", "patient", "patient_id", "series", "series_id", "filename", "file", "image", "case", "id"])
    x_min = get_col(df, ["xmin", "x_min", "min_x", "x1", "x lower", "x min"])
    x_max = get_col(df, ["xmax", "x_max", "max_x", "x2", "x upper", "x max"])
    y_min = get_col(df, ["ymin", "y_min", "min_y", "y1", "y lower", "y min"])
    y_max = get_col(df, ["ymax", "y_max", "max_y", "y2", "y upper", "y max"])
    z_min = get_col(df, ["zmin", "z_min", "min_z", "z1", "z lower", "z min"])
    z_max = get_col(df, ["zmax", "z_max", "max_z", "z2", "z upper", "z max"])
    index_col = get_col(df, ["index", "idx", "nodule_id", "id"])
    x_center_col = get_col(df, ["x_center", "xcenter", "center_x"], required=False)
    y_center_col = get_col(df, ["y_center", "ycenter", "center_y"], required=False)
    z_center_col = get_col(df, ["z_center", "zcenter", "center_z"], required=False)

    # Extract patient ID from image column (e.g., "0001_41.bmp" -> "1")
    def extract_patient_id(image_name: str) -> str:
        # Format: "XXXX_YY.bmp" -> extract XXXX and convert to int to remove leading zeros
        parts = str(image_name).split("_")
        if parts:
            patient_part = parts[0]
            # Convert to int to remove leading zeros: "0001" -> 1
            try:
                return str(int(patient_part))
            except ValueError:
                return patient_part
        return str(image_name)

    out = pd.DataFrame(
        {
            "index": df[index_col].astype(int) if index_col else range(len(df)),
            "original_id_raw": df[id_col].astype(str),
            "xmin": df[x_min].astype(float),
            "xmax": df[x_max].astype(float),
            "ymin": df[y_min].astype(float),
            "ymax": df[y_max].astype(float),
            "zmin": df[z_min].astype(float),
            "zmax": df[z_max].astype(float),
            "x_center": df[x_center_col].astype(float) if x_center_col else (df[x_min].astype(float) + df[x_max].astype(float)) / 2.0,
            "y_center": df[y_center_col].astype(float) if y_center_col else (df[y_min].astype(float) + df[y_max].astype(float)) / 2.0,
            "z_center": df[z_center_col].astype(float) if z_center_col else (df[z_min].astype(float) + df[z_max].astype(float)) / 2.0,
        }
    )
    # Extract patient ID from image name for matching with MHD cases
    out["original_id"] = out["original_id_raw"].map(extract_patient_id)
    return out


def match_annotation_id_to_case(anno_id: str, case_ids: List[str]) -> Optional[str]:
    if anno_id in case_ids:
        return anno_id
    ai = extract_first_int(anno_id)
    matches = [c for c in case_ids if extract_first_int(c) == ai] if ai is not None else []
    if len(matches) == 1:
        return matches[0]
    # Return None if cannot match (annotation exists but no corresponding MHD file)
    return None


def _interval_gap(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return max(0.0, max(a_min - b_max - 1.0, b_min - a_max - 1.0))


def compute_bbox_center_distance(
    bbox1: Tuple[int, int, int, int, int, int],
    bbox2: Tuple[int, int, int, int, int, int],
) -> float:
    """计算两个 bbox 中心点之间的距离。"""
    zmin1, zmax1, ymin1, ymax1, xmin1, xmax1 = bbox1
    zmin2, zmax2, ymin2, ymax2, xmin2, xmax2 = bbox2
    
    center1 = ((zmin1 + zmax1) / 2, (ymin1 + ymax1) / 2, (xmin1 + xmax1) / 2)
    center2 = ((zmin2 + zmax2) / 2, (ymin2 + ymax2) / 2, (xmin2 + xmax2) / 2)
    
    return math.sqrt(
        (center1[0] - center2[0]) ** 2 +
        (center1[1] - center2[1]) ** 2 +
        (center1[2] - center2[2]) ** 2
    )


def compute_bbox_iou(
    bbox1: Tuple[int, int, int, int, int, int],
    bbox2: Tuple[int, int, int, int, int, int],
) -> float:
    zmin1, zmax1, ymin1, ymax1, xmin1, xmax1 = bbox1
    zmin2, zmax2, ymin2, ymax2, xmin2, xmax2 = bbox2
    
    inter_zmin = max(zmin1, zmin2)
    inter_zmax = min(zmax1, zmax2)
    inter_ymin = max(ymin1, ymin2)
    inter_ymax = min(ymax1, ymax2)
    inter_xmin = max(xmin1, xmin2)
    inter_xmax = min(xmax1, xmax2)
    
    if inter_zmax < inter_zmin or inter_ymax < inter_ymin or inter_xmax < inter_xmin:
        return 0.0
    
    inter_vol = (inter_zmax - inter_zmin + 1) * (inter_ymax - inter_ymin + 1) * (inter_xmax - inter_xmin + 1)
    
    vol1 = (zmax1 - zmin1 + 1) * (ymax1 - ymin1 + 1) * (xmax1 - xmin1 + 1)
    vol2 = (zmax2 - zmin2 + 1) * (ymax2 - ymin2 + 1) * (xmax2 - xmin2 + 1)
    
    union_vol = vol1 + vol2 - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0


def compute_union_bbox(
    bbox1: Tuple[int, int, int, int, int, int],
    bbox2: Tuple[int, int, int, int, int, int],
) -> Tuple[int, int, int, int, int, int]:
    zmin1, zmax1, ymin1, ymax1, xmin1, xmax1 = bbox1
    zmin2, zmax2, ymin2, ymax2, xmin2, xmax2 = bbox2
    return (
        min(zmin1, zmin2), max(zmax1, zmax2),
        min(ymin1, ymin2), max(ymax1, ymax2),
        min(xmin1, xmin2), max(xmax1, xmax2),
    )


def load_pred_nodule_masks(pred_root: Path) -> Dict[str, Path]:
    files = find_files(pred_root, [".nii.gz"], recursive=False)
    masks: Dict[str, Path] = {}
    for p in files:
        oid = normalize_original_id(p)
        masks[oid] = p
    return masks


def load_lung_masks(lung_mask_root: Path) -> Dict[str, Path]:
    files = find_files(lung_mask_root, [".nii.gz"], recursive=False)
    masks: Dict[str, Path] = {}
    for p in files:
        oid = normalize_original_id(p)
        masks[oid] = p
    return masks


def extract_bboxes_from_pred_mask(pred_mask_path: Path, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> List[Tuple[int, int, int, int, int, int]]:
    mask_img = sitk.ReadImage(str(pred_mask_path))
    mask_arr = sitk.GetArrayFromImage(mask_img)
    mask_arr = (mask_arr > 0).astype(np.uint8)
    
    if not mask_arr.any():
        return []
    
    bboxes = bbox_from_mask_array(mask_arr, min_voxels=1, morph_close=True)
    return bboxes


def resample_and_crop_ct_with_lung_mask(
    ct_img: "sitk.Image",
    lung_mask_img: "sitk.Image",
    out_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    pad_value: float = 0.0,
) -> Tuple["sitk.Image", Tuple[int, int, int]]:
    """Crop CT on its native grid using the lung mask, then resample to output spacing.

    The Histopathology annotations are in original voxel coordinates. Computing the
    crop bbox on the CT native grid keeps the crop offset in the same coordinate
    system before converting both annotation boxes and offset to 1mm space.
    """
    lung_mask_ref = resample_to_reference(lung_mask_img, ct_img, is_label=True, default_value=0)
    mask_binary = sitk.GetArrayFromImage(lung_mask_ref) > 0

    if not mask_binary.any():
        ct_resampled = resample_image(ct_img, out_spacing_xyz=out_spacing, is_label=False, default_value=pad_value)
        return ct_resampled, (0, 0, 0)

    z_indices, y_indices, x_indices = np.where(mask_binary)
    zmin, zmax = int(z_indices.min()), int(z_indices.max())
    ymin, ymax = int(y_indices.min()), int(y_indices.max())
    xmin, xmax = int(x_indices.min()), int(x_indices.max())

    index_xyz = [xmin, ymin, zmin]
    size_xyz = [xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1]
    ct_cropped = sitk.RegionOfInterest(ct_img, size_xyz, index_xyz)
    ct_cropped_1mm = resample_image(ct_cropped, out_spacing_xyz=out_spacing, is_label=False, default_value=pad_value)

    sx, sy, sz = ct_img.GetSpacing()
    nsx, nsy, nsz = out_spacing
    crop_offset = (
        int(math.floor(zmin * sz / nsz)),
        int(math.floor(ymin * sy / nsy)),
        int(math.floor(xmin * sx / nsx)),
    )
    return ct_cropped_1mm, crop_offset


def merge_slice_boxes(
    rows: pd.DataFrame,
    xy_gap: float = 3.0,
    max_missing_slices: float = 1.0,
) -> List[Tuple[int, int, int, int, int, int]]:
    """Merge per-slice boxes using bbox continuity.

    The released center columns contain known typos, so they must not drive
    lesion grouping.
    """
    records = rows.to_dict("records")
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i, left in enumerate(records):
        for j in range(i + 1, len(records)):
            right = records[j]
            z_gap = _interval_gap(left["zmin"], left["zmax"], right["zmin"], right["zmax"])
            y_gap = _interval_gap(left["ymin"], left["ymax"], right["ymin"], right["ymax"])
            x_gap = _interval_gap(left["xmin"], left["xmax"], right["xmin"], right["xmax"])
            if z_gap <= max_missing_slices and y_gap <= xy_gap and x_gap <= xy_gap:
                union(i, j)

    groups: Dict[int, List[dict]] = {}
    for index, record in enumerate(records):
        groups.setdefault(find(index), []).append(record)

    boxes = []
    for group in groups.values():
        boxes.append((
            int(min(row["zmin"] for row in group)),
            int(max(row["zmax"] for row in group)),
            int(min(row["ymin"] for row in group)),
            int(max(row["ymax"] for row in group)),
            int(min(row["xmin"] for row in group)),
            int(max(row["xmax"] for row in group)),
        ))
    return sorted(boxes, key=lambda box: (box[0], box[2], box[4]))


def run(args: argparse.Namespace) -> None:
    mhd_root = Path(args.mhd_root)
    anno_csv = Path(args.all_anno_3d)
    out_dir = Path(args.out_dir)
    full_dir = ensure_dir(out_dir / "full")
    
    pred_root = Path(args.pred_root) if args.pred_root else None
    lung_mask_root = Path(args.lung_mask_root) if args.lung_mask_root else None
    
    use_lung_crop = lung_mask_root is not None
    use_pred = pred_root is not None
    
    cases = collect_mhd_cases(mhd_root)
    case_ids = list(cases)
    pid_map = build_numeric_pid_map(case_ids)
    
    ann = load_all_anno_3d(anno_csv)
    ann["case_id"] = ann["original_id"].map(lambda x: match_annotation_id_to_case(x, case_ids))
    # Drop annotations that cannot be matched to MHD files
    ann = ann.dropna(subset=["case_id"])
    ann["case_id"] = ann["case_id"].astype(str)
    
    gt_boxes_by_case: Dict[str, List[Tuple[int, int, int, int, int, int]]] = {
        cid: [] for cid in case_ids
    }
    for case_id, group in ann.groupby("case_id"):
        for box in merge_slice_boxes(group):
            gt_boxes_by_case[case_id].append(box)
    
    pred_masks = load_pred_nodule_masks(pred_root) if use_pred else {}
    lung_masks = load_lung_masks(lung_mask_root) if use_lung_crop else {}
    
    union_bbox_cases = {str(x) for x in [21, 42, 46, 65,61,78]}
    pred_only_cases = {str(x) for x in [2, 30, 38, 41, 47, 54, 56, 72, 88]}
    
    all_rows: List[Dict[str, int]] = []
    metas: List[CaseMeta] = []
    
    for oid, img_path in tqdm(sorted(cases.items()), desc="Histopathology MHD -> SANet npy"):
        img = read_image(img_path)
        orig_shape, orig_spacing, orig_min, orig_max = image_stats(img)
        sanet_pid = pid_map[oid]
        output_path = full_dir / f"{pid_to_filename(sanet_pid, args.pid_width)}_zoom.npy"
        
        if args.reuse_existing_volumes and output_path.exists():
            arr = np.load(output_path, mmap_mode="r")
            proc_shape = tuple(arr.shape[1:])
            crop_offset = (0, 0, 0)
        else:
            if use_lung_crop and oid in lung_masks:
                lung_mask_img = sitk.ReadImage(str(lung_masks[oid]))
                img_1mm, crop_offset = resample_and_crop_ct_with_lung_mask(
                    img, lung_mask_img,
                    out_spacing=(1.0, 1.0, 1.0),
                    pad_value=args.pad_hu,
                )
                arr = image_to_sanet_array(
                    img_1mm,
                    assume_already_0_255=args.assume_already_0_255,
                    auto_skip_window_if_0_255=args.auto_skip_window_if_0_255,
                )
            else:
                img_1mm = resample_image(
                    img,
                    out_spacing_xyz=(1.0, 1.0, 1.0),
                    is_label=False,
                    default_value=args.pad_hu,
                )
                arr = image_to_sanet_array(
                    img_1mm,
                    assume_already_0_255=args.assume_already_0_255,
                    auto_skip_window_if_0_255=args.auto_skip_window_if_0_255,
                )
                crop_offset = (0, 0, 0)
            
            save_sanet_npy(arr, output_path)
        
        proc_shape = tuple(arr.shape[1:])
        
        if use_pred and oid in union_bbox_cases:
            gt_boxes_orig = gt_boxes_by_case.get(oid, [])
            gt_boxes_1mm = []
            for b in gt_boxes_orig:
                b1 = convert_bbox_coords_by_spacing(b, old_spacing_xyz=orig_spacing, new_spacing_xyz=(1.0, 1.0, 1.0))
                if b1 is not None:
                    gt_boxes_1mm.append(b1)
            
            pred_mask_path = pred_masks.get(oid)
            pred_boxes_orig = []
            if pred_mask_path:
                pred_boxes_orig = extract_bboxes_from_pred_mask(pred_mask_path)
            
            # Convert pred bboxes from original pixel space to 1mm space
            pred_boxes_1mm = []
            for b in pred_boxes_orig:
                b1 = convert_bbox_coords_by_spacing(b, old_spacing_xyz=orig_spacing, new_spacing_xyz=(1.0, 1.0, 1.0))
                if b1 is not None:
                    pred_boxes_1mm.append(b1)
            
            final_boxes = []
            for gt_box in gt_boxes_1mm:
                best_iou = -1.0
                best_pred_box = None
                for pred_box in pred_boxes_1mm:
                    iou = compute_bbox_iou(gt_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_box = pred_box
                
                # 如果所有 IoU 都是 0，使用中心点距离选择最接近的 pred box
                if best_iou == 0.0:
                    min_dist = float('inf')
                    closest_pred_box = None
                    for pred_box in pred_boxes_1mm:
                        dist = compute_bbox_center_distance(gt_box, pred_box)
                        if dist < min_dist:
                            min_dist = dist
                            closest_pred_box = pred_box
                    if closest_pred_box is not None:
                        best_pred_box = closest_pred_box
                
                if best_pred_box is not None:
                    union_box = compute_union_bbox(gt_box, best_pred_box)
                    final_boxes.append(union_box)
                else:
                    final_boxes.append(gt_box)
        elif use_pred and oid in pred_only_cases:
            gt_boxes_orig = gt_boxes_by_case.get(oid, [])
            gt_boxes_1mm = []
            for b in gt_boxes_orig:
                b1 = convert_bbox_coords_by_spacing(b, old_spacing_xyz=orig_spacing, new_spacing_xyz=(1.0, 1.0, 1.0))
                if b1 is not None:
                    gt_boxes_1mm.append(b1)
            
            pred_mask_path = pred_masks.get(oid)
            pred_boxes_orig = []
            if pred_mask_path:
                pred_boxes_orig = extract_bboxes_from_pred_mask(pred_mask_path)
            
            # Convert pred bboxes from original pixel space to 1mm space
            pred_boxes_1mm = []
            for b in pred_boxes_orig:
                b1 = convert_bbox_coords_by_spacing(b, old_spacing_xyz=orig_spacing, new_spacing_xyz=(1.0, 1.0, 1.0))
                if b1 is not None:
                    pred_boxes_1mm.append(b1)
            
            final_boxes = []
            for gt_box in gt_boxes_1mm:
                best_iou = 0.0
                best_pred_box = None
                for pred_box in pred_boxes_1mm:
                    iou = compute_bbox_iou(gt_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_box = pred_box
                
                if best_iou > 0.0 and best_pred_box is not None:
                    final_boxes.append(best_pred_box)
                else:
                    final_boxes.append(gt_box)
        else:
            gt_boxes_orig = gt_boxes_by_case.get(oid, [])
            final_boxes = []
            for b in gt_boxes_orig:
                b1 = convert_bbox_coords_by_spacing(b, old_spacing_xyz=orig_spacing, new_spacing_xyz=(1.0, 1.0, 1.0))
                if b1 is not None:
                    final_boxes.append(b1)
        
        n_boxes = 0
        nodule_id = 0
        for b in final_boxes:
            b_adjusted = (
                b[0] - crop_offset[0],
                b[1] - crop_offset[0],
                b[2] - crop_offset[1],
                b[3] - crop_offset[1],
                b[4] - crop_offset[2],
                b[5] - crop_offset[2],
            )
            b_clipped = clip_bbox_to_shape(b_adjusted, proc_shape)
            if b_clipped is None:
                continue
            zmin, zmax, ymin, ymax, xmin, xmax = b_clipped
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
    
    split = None
    if args.split_dir:
        from .common import derive_split_from_ids
        split = derive_split_from_ids(case_ids, args.split_dir, pid_map, allow_fallback=False)
    else:
        from .common import derive_split_from_ids
        split = derive_split_from_ids(
            case_ids,
            args.split_dir,
            pid_map,
            ratio=(7, 1, 2),
            seed=args.seed,
            allow_fallback=args.allow_fallback_split,
        )
    save_common_outputs(out_dir, "Histopathology-based Dataset", pid_map, split, all_rows, metas, width=args.pid_width)
    print(f"Done. Output: {out_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare Histopathology-based Dataset MHD_3D to SANet/PN9-style npy.")
    p.add_argument("--mhd-root", default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/Histopathology/MHD_3D",  help="Path to MHD_3D directory containing .mhd/.raw files.")
    p.add_argument("--all-anno-3d", default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/Histopathology/all_anno_3D.csv", help="Path to all_anno_3D.csv.")
    p.add_argument("--out-dir", default="../../data/histopathology", help="Output SANet-ready directory.")
    p.add_argument("--split-dir", default=None, help="Original split directory containing train.txt/test.txt[/val.txt], if available.")
    p.add_argument("--allow-fallback-split", action="store_true", help="If no split-dir is provided, use deterministic 7:1:2 train/val/test split.")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--pid-width", type=int, default=5, help="Width of zero-padding for SANet pid filenames, e.g. 00001_zoom.npy")
    p.add_argument("--pad-hu", type=float, default=0.0, help="HU value to pad if resampling creates out-of-bounds areas; default 0 for histopathology.")
    p.add_argument("--assume-already-0-255", action="store_true", help="Use if MHD values are already 0-255; do not HU-window.")
    p.add_argument("--auto-skip-window-if-0-255", action="store_true", help="Skip HU window automatically if each processed image min/max are within [0,255].")
    p.add_argument(
        "--reuse-existing-volumes",
        action="store_true",
        help="Keep existing *_zoom.npy files and rebuild annotations/metadata only.",
    )
    p.add_argument("--pred-root", default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/Histopathology/model_nodule_detect_output", help="Path to predicted nodule masks directory (nii.gz files).")
    p.add_argument("--lung-mask-root", default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/lungseg/Histopathology", help="Path to lung segmentation masks directory (nii.gz files).")
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())

'''
随机按照7:1:2比例划分train/val/test，如果原始数据没有提供划分文件。可以使用以下命令运行：
python3 -m sanet_prep.prepare_histopathology --allow-fallback-split --auto-skip-window-if-0-255
'''

'''
 python3 -m sanet_prep.prepare_histopathology 
 --allow-fallback-split
 --pred-root "/media/SENSETIME\yangtingting/T7/医保大赛数据/Histopathology/model_nodule_detect_output" 
 --lung-mask-root "/media/SENSETIME\yangtingting/T7/医保大赛数据/lungseg/Histopathology" 
 --out-dir "../../data/histopathology"
'''