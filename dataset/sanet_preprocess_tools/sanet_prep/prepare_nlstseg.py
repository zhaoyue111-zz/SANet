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
    maybe_split_train_to_val,
    normalize_original_id,
    pid_to_filename,
    read_image,
    resample_image,
    resample_to_reference,
    save_common_outputs,
    save_sanet_npy,
)

'''
问题：有大肿瘤和小结节。病灶体积范围是 0.03 cm³ 到 372.21 cm³，并且 500 个病灶小于 5 cm³
当前已兼顾保留小结节和避免大肿瘤碎片化。
'''


MASK_KEYWORDS = ("tumor", "nodule", "lesion", "mask", "label", "seg")
EXCLUDED_DIR_NAMES = {"ge3mm"}


def _strip_nlstseg_suffix(oid: str) -> str:
    """Normalize NLSTSeg image/mask ids so CT, tumor and nodule masks can be paired."""
    s = normalize_original_id(oid)
    changed = True
    while changed:
        changed = False
        for tail in ["_CT", "_ct", "_tumor", "_Tumor", "_nodule", "_Nodule", "_lesion", "_Lesion", "_mask", "_label", "_seg"]:
            if s.endswith(tail):
                s = s[: -len(tail)]
                changed = True
    return s


def _is_mask_file(path: Path) -> bool:
    name = path.name.lower()
    return any(k in name for k in MASK_KEYWORDS)


def _is_excluded_path(path: Path) -> bool:
    return any(part.lower() in EXCLUDED_DIR_NAMES for part in path.parts)


def collect_lung_masks(lung_mask_root: Path) -> Dict[str, Path]:
    files = find_files(lung_mask_root, [".nii.gz", ".nii"], recursive=True)
    masks: Dict[str, Path] = {}
    for p in files:
        oid = _strip_nlstseg_suffix(str(p))
        masks[oid] = p
    return masks


def crop_to_lung_mask_bbox(
    img: "sitk.Image",
    lung_mask_img: "sitk.Image",
) -> "sitk.Image":
    """Crop CT on its native grid using a lung mask resampled to the CT reference."""
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


def expand_bbox_by_one_voxel(
    bbox: Tuple[int, int, int, int, int, int],
    shape_zyx: Tuple[int, int, int],
) -> Optional[Tuple[int, int, int, int, int, int]]:
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    return clip_bbox_to_shape((zmin - 1, zmax + 1, ymin - 1, ymax + 1, xmin - 1, xmax + 1), shape_zyx)


def combine_and_resample_masks(mask_paths: List[Path], ref_img: "sitk.Image") -> np.ndarray:
    """Combine all tumor/nodule/lesion masks as one positive binary mask on ref_img grid."""
    combined = None
    for mask_path in mask_paths:
        mask = read_image(mask_path)
        mask_ref = resample_to_reference(mask, ref_img, is_label=True, default_value=0)
        mask_arr = sitk.GetArrayFromImage(mask_ref) > 0
        combined = mask_arr if combined is None else (combined | mask_arr)
    if combined is None:
        raise RuntimeError("No mask was combined. Check mask_paths.")
    return combined


def _bbox_gap(b1: Tuple[int, int, int, int, int, int], b2: Tuple[int, int, int, int, int, int]) -> int:
    """Chebyshev gap between two z/y/x bboxes. 0 means touching or overlapping."""
    z1min, z1max, y1min, y1max, x1min, x1max = b1
    z2min, z2max, y2min, y2max, x2min, x2max = b2
    gz = max(0, max(z1min - z2max - 1, z2min - z1max - 1))
    gy = max(0, max(y1min - y2max - 1, y2min - y1max - 1))
    gx = max(0, max(x1min - x2max - 1, x2min - x1max - 1))
    return int(max(gz, gy, gx))


def _union_bbox(boxes: List[Tuple[int, int, int, int, int, int]]) -> Tuple[int, int, int, int, int, int]:
    arr = np.array(boxes, dtype=int)
    return (
        int(arr[:, 0].min()), int(arr[:, 1].max()),
        int(arr[:, 2].min()), int(arr[:, 3].max()),
        int(arr[:, 4].min()), int(arr[:, 5].max()),
    )


def extract_lesion_bboxes_from_binary_mask(
    mask_zyx: np.ndarray,
    min_voxels: int = 10,
    connectivity: int = 26,
    close_iterations: int = 1,
    fill_holes: bool = True,
    merge_satellites: bool = True,
    satellite_max_voxels: int = 50,
    large_component_min_voxels: int = 200,
    merge_gap_voxels: int = 3,
) -> List[Tuple[int, int, int, int, int, int]]:
    """Extract one bbox per positive lesion while avoiding tiny satellite fragments.

    All mask values >0 are treated as the same positive class.  The algorithm keeps
    small true nodules if they are standalone components, but merges/removes tiny
    speckles around a large tumor so one big tumor is not written as many boxes.
    """
    try:
        from scipy import ndimage as ndi
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scipy is required. Install with `pip install scipy`.") from exc

    mask = np.asarray(mask_zyx) > 0
    if not mask.any():
        return []

    # 26-connectivity is safer after resampling: diagonal/contacting tumor voxels stay together.
    conn = 3 if int(connectivity) >= 26 else (2 if int(connectivity) >= 18 else 1)
    structure = ndi.generate_binary_structure(3, conn)

    if close_iterations > 0:
        mask = ndi.binary_closing(mask, structure=structure, iterations=int(close_iterations))
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    lab, n = ndi.label(mask, structure=structure)
    comps = []
    for i in range(1, n + 1):
        coords = np.where(lab == i)
        size = int(len(coords[0]))
        if size <= 0:
            continue
        z, y, x = coords
        bbox = (int(z.min()), int(z.max()), int(y.min()), int(y.max()), int(x.min()), int(x.max()))
        comps.append({"bbox": bbox, "size": size})

    if not comps:
        return []

    # Merge only tiny satellite fragments into nearby large components.  This avoids
    # merging two real, separate nodules, while fixing artifacts like 1-4 voxel crumbs.
    if merge_satellites and len(comps) > 1:
        parent = list(range(len(comps)))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                si, sj = comps[i]["size"], comps[j]["size"]
                small_near_large = (
                    min(si, sj) <= satellite_max_voxels
                    and max(si, sj) >= large_component_min_voxels
                    and _bbox_gap(comps[i]["bbox"], comps[j]["bbox"]) <= merge_gap_voxels
                )
                if small_near_large:
                    union(i, j)

        groups: Dict[int, List[dict]] = {}
        for i, c in enumerate(comps):
            groups.setdefault(find(i), []).append(c)
        comps = []
        for group in groups.values():
            comps.append({
                "bbox": _union_bbox([g["bbox"] for g in group]),
                "size": int(sum(g["size"] for g in group)),
            })

    bboxes: List[Tuple[int, int, int, int, int, int]] = []
    for c in comps:
        # Remove isolated speckles, but keep standalone small nodules above min_voxels.
        if c["size"] < min_voxels:
            continue
        bboxes.append(c["bbox"])

    return sorted(bboxes, key=lambda b: (b[0], b[2], b[4]))

def collect_nlstseg_pairs(root: Path) -> Dict[str, Tuple[Path, List[Path]]]:
    nii_files = [p for p in find_files(root, [".nii.gz", ".nii"], recursive=True) if not _is_excluded_path(p)]
    images = []
    for p in nii_files:
        name = p.name.lower()
        if _is_mask_file(p):
            continue
        if name.endswith("_ct.nii.gz") or name.endswith("_ct.nii") or "ct" in name:
            images.append(p)
    if not images:
        # fallback: every nii that is not a mask could be an image
        images = [p for p in nii_files if not _is_mask_file(p)]

    masks = [p for p in nii_files if _is_mask_file(p)]
    pairs: Dict[str, Tuple[Path, List[Path]]] = {}
    for img in images:
        oid = _strip_nlstseg_suffix(str(img))
        candidates = [m for m in masks if _strip_nlstseg_suffix(str(m)) == oid]
        if not candidates:
            n = extract_first_int(oid)
            if n is not None:
                candidates = [m for m in masks if extract_first_int(_strip_nlstseg_suffix(str(m))) == n]
        if not candidates:
            raise FileNotFoundError(f"Expected at least one tumor/nodule/lesion mask for image {img}, got 0 candidates.")
        pairs[oid] = (img, sorted(candidates, key=lambda p: str(p)))
    if not pairs:
        raise FileNotFoundError(f"No NLSTseg image/mask NIfTI pairs found under {root}")
    return pairs


def split_from_table(table_path: Path, case_ids: List[str], pid_map: Dict[str, int]) -> Dict[str, List[int]]:
    if str(table_path).lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(table_path)
    else:
        df = pd.read_csv(table_path)
    id_col = get_col(df, ["pid", "patient", "patient_id", "case", "id", "scan", "scan_id", "filename", "file"])
    split_col = get_col(df, ["split", "set", "subset", "train_test", "dataset"], required=False)
    if split_col is None:
        raise KeyError("Split table must contain a split/set column with train/val/test values.")
    known = set(case_ids)
    out = {"train": [], "val": [], "test": []}
    for _, r in df.iterrows():
        oid = normalize_original_id(str(r[id_col]))
        if oid not in known:
            n = extract_first_int(oid)
            matches = [x for x in known if extract_first_int(x) == n] if n is not None else []
            if len(matches) == 1:
                oid = matches[0]
            else:
                continue
        s = str(r[split_col]).lower()
        if "train" in s:
            out["train"].append(pid_map[oid])
        elif "val" in s or "valid" in s:
            out["val"].append(pid_map[oid])
        elif "test" in s:
            out["test"].append(pid_map[oid])
    if not out["train"] or not out["test"]:
        raise RuntimeError(f"Split table {table_path} did not yield non-empty train/test splits.")
    return out


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    full_dir = ensure_dir(out_dir / "full")

    pairs = collect_nlstseg_pairs(root)
    case_ids = list(pairs)
    pid_map = build_numeric_pid_map(case_ids)
    lung_masks = collect_lung_masks(Path(args.lung_mask_root)) if args.lung_mask_root else {}
    missing_lung_masks = [oid for oid in case_ids if oid not in lung_masks]
    if missing_lung_masks:
        raise RuntimeError(
            f"{len(missing_lung_masks)} NLSTseg case(s) do not have lung masks. "
            f"Examples: {missing_lung_masks[:5]}"
        )

    all_rows: List[Dict[str, int]] = []
    metas: List[CaseMeta] = []
    for oid, (img_path, mask_paths) in tqdm(sorted(pairs.items()), desc="NLSTseg -> SANet npy"):
        img = read_image(img_path)
        orig_shape, orig_spacing, orig_min, orig_max = image_stats(img)
        if args.lung_mask_root:
            lung_mask_img = read_image(lung_masks[oid])
            img_cropped = crop_to_lung_mask_bbox(img, lung_mask_img)
        else:
            img_cropped = img
        img_1mm = resample_image(img_cropped, out_spacing_xyz=(1.0, 1.0, 1.0), is_label=False, default_value=args.pad_hu)
        mask_1mm_arr = combine_and_resample_masks(mask_paths, img_1mm)
        arr = image_to_sanet_array(
            img_1mm,
            assume_already_0_255=args.assume_already_0_255,
            auto_skip_window_if_0_255=args.auto_skip_window_if_0_255,
        )
        proc_shape = tuple(arr.shape[1:])
        sanet_pid = pid_map[oid]
        save_sanet_npy(arr, full_dir / f"{pid_to_filename(sanet_pid, args.pid_width)}_zoom.npy")

        bboxes = extract_lesion_bboxes_from_binary_mask(
            mask_1mm_arr,
            min_voxels=args.min_mask_voxels,
            connectivity=args.component_connectivity,
            close_iterations=args.close_iterations,
            fill_holes=args.fill_holes,
            merge_satellites=args.merge_satellites,
            satellite_max_voxels=args.satellite_max_voxels,
            large_component_min_voxels=args.large_component_min_voxels,
            merge_gap_voxels=args.merge_gap_voxels,
        )
        bboxes = [b for b in (expand_bbox_by_one_voxel(b, proc_shape) for b in bboxes) if b is not None]
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
    save_common_outputs(out_dir, "NLSTseg", pid_map, split, all_rows, metas, width=args.pid_width)
    print(f"Done. Output: {out_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare NLSTseg NIfTI CT/masks to SANet/PN9-style npy.")
    p.add_argument("--root", default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/NLSTseg", help="Extracted NLSTseg directory containing *_CT.nii.gz and *_tumor.nii.gz files; ge3mm is excluded.")
    p.add_argument("--out-dir", default="../../data/NLSTSeg")
    p.add_argument("--lung-mask-root", default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/lungseg/NLSTseg", help="Directory containing lung masks used to crop CT before saving npy.")
    p.add_argument("--split-dir", default=None, help="Directory containing author/original train.txt/val.txt/test.txt.")
    p.add_argument("--split-table", default=None, help="CSV/XLSX table containing id and split columns, e.g. from 1_Table.zip if available.")
    p.add_argument("--allow-fallback-split", action="store_true", help="If no author split is found, use deterministic fallback ratio.")
    p.add_argument("--fallback-ratio", nargs=3, type=float, default=[7, 1, 2], metavar=("TRAIN", "VAL", "TEST"),
                   help="Fallback split ratio when no author split is found. Default 7:1:2 (SANet requires validation set).")
    p.add_argument("--min-mask-voxels", type=int, default=10,
                   help="Minimum voxels for a connected component to be kept. After 1mm resampling, 10 filters speckles while preserving small nodules.")
    p.add_argument("--component-connectivity", type=int, default=26, choices=[6, 18, 26],
                   help="Connectivity for lesion components. Use 26 to avoid splitting diagonally connected tumor voxels.")
    p.add_argument("--close-iterations", type=int, default=0,
                   help="3D binary closing iterations before connected components. Default 0 keeps the true mask geometry.")
    p.add_argument("--fill-holes", action="store_true", help="Fill binary holes before component extraction. Default keeps the true mask geometry.")
    p.add_argument("--merge-satellites", action="store_true", help="Merge tiny nearby fragments into large tumors. Default keeps mask components separate.")
    p.add_argument("--satellite-max-voxels", type=int, default=50,
                   help="Components <= this size may be treated as tiny satellites if close to a large tumor.")
    p.add_argument("--large-component-min-voxels", type=int, default=200,
                   help="Minimum size of a component to be considered a large tumor for satellite merging.")
    p.add_argument("--merge-gap-voxels", type=int, default=3,
                   help="Max bbox gap in 1mm voxels for merging tiny satellite fragments into a large tumor.")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--pid-width", type=int, default=5)
    p.add_argument("--pad-hu", type=float, default=-1200.0)
    p.add_argument("--assume-already-0-255", action="store_true")
    p.add_argument("--auto-skip-window-if-0-255", action="store_true")
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())

'''
python3 -m sanet_prep.prepare_nlstseg --allow-fallback-split --auto-skip-window-if-0-255
'''