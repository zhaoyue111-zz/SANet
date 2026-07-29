from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from .common import (
    clip_bbox_to_shape,
    ensure_dir,
    extract_first_int,
    find_files,
    link_or_copy,
    natural_key,
    save_sanet_npy,
    write_pid_map,
)


SANET_ANNO_COLUMNS = ["pid", "zmin", "zmax", "ymin", "ymax", "xmin", "xmax", "nodule_id"]


def find_split_file(root: Path, name: str) -> Path:
    candidates = list(root.rglob(name))
    if not candidates:
        raise FileNotFoundError(f"Cannot find {name} under {root}")
    # Prefer shallower path to avoid nested duplicates.
    return sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]


def npy_stem_to_pid(stem: str) -> int:
    s = stem.replace("_zoom", "")
    n = extract_first_int(s)
    if n is None:
        raise ValueError(f"PN9 filename stem {stem!r} is not numeric; original SANet expects numeric pids.")
    return int(n)


def strip_image_suffix(path_or_id: Path | str) -> str:
    name = Path(str(path_or_id)).name
    for suffix in [".nii.gz", ".mha.gz", ".mhd.gz", ".nii", ".mha", ".mhd", ".gz", ".npy"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("_zoom", "")


def collect_lung_masks(root: Path) -> Dict[str, Path]:
    masks: Dict[str, Path] = {}
    for path in find_files(root, [".nii.gz", ".nii", ".mha", ".mhd"], recursive=True):
        oid = strip_image_suffix(path)
        if oid in masks and masks[oid] != path:
            raise RuntimeError(f"Duplicate PN9 lung mask for {oid}: {masks[oid]} and {path}")
        masks[oid] = path
    if not masks:
        raise FileNotFoundError(f"No PN9 lung masks found under {root}")
    return masks


def match_lung_mask(case_id: str, masks: Dict[str, Path]) -> Path:
    if case_id in masks:
        return masks[case_id]
    case_num = extract_first_int(case_id)
    matches = [oid for oid in masks if extract_first_int(oid) == case_num] if case_num is not None else []
    if len(matches) == 1:
        return masks[matches[0]]
    raise FileNotFoundError(
        f"No lung mask matched PN9 case {case_id}. Examples: {list(sorted(masks, key=natural_key))[:5]}"
    )


def lung_crop_bbox(mask_path: Path, shape_zyx: tuple[int, int, int], margin: int) -> tuple[int, int, int, int, int, int]:
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) > 0
    if mask.shape != shape_zyx:
        raise ValueError(f"Lung mask {mask_path} shape {mask.shape} does not match image shape {shape_zyx}")
    coords = np.where(mask)
    if len(coords[0]) == 0:
        raise ValueError(f"Lung mask {mask_path} is empty.")

    z, y, x = coords
    bbox = (
        int(z.min()) - margin,
        int(z.max()) + margin,
        int(y.min()) - margin,
        int(y.max()) + margin,
        int(x.min()) - margin,
        int(x.max()) + margin,
    )
    clipped = clip_bbox_to_shape(bbox, shape_zyx)
    if clipped is None:
        raise ValueError(f"Lung crop bbox for {mask_path} is outside image shape {shape_zyx}")
    return clipped


def crop_sanet_array(arr: np.ndarray, bbox: tuple[int, int, int, int, int, int]) -> np.ndarray:
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    return arr[:, zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]


def read_id_list(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [line.strip().split()[0] for line in f if line.strip() and not line.lstrip().startswith("#")]


def write_id_list(path: Path, ids: Iterable[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for pid in ids:
            f.write(f"{pid}\n")


def normalize_annotation_csv(src: Path) -> pd.DataFrame:
    df = pd.read_csv(src)
    required = ["pid", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{src} missing required columns: {missing}. Available columns: {list(df.columns)}")

    out = pd.DataFrame({
        "pid": df["pid"].astype(int),
        "zmin": df["zmin"].astype(int),
        "zmax": df["zmax"].astype(int),
        "ymin": df["ymin"].astype(int),
        "ymax": df["ymax"].astype(int),
        "xmin": df["xmin"].astype(int),
        "xmax": df["xmax"].astype(int),
        "nodule_id": df["nodule_id"].astype(int) if "nodule_id" in df.columns else range(len(df)),
    })
    bad = out[(out.zmax < out.zmin) | (out.ymax < out.ymin) | (out.xmax < out.xmin)]
    if len(bad) > 0:
        raise ValueError(f"{src} has {len(bad)} invalid bbox rows with max < min.")
    return out[SANET_ANNO_COLUMNS]


def write_annotation_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, columns=SANET_ANNO_COLUMNS, quoting=csv.QUOTE_MINIMAL)


def adjust_annotations_for_lung_crop(
    df: pd.DataFrame,
    crop_meta: Dict[int, Dict[str, object]],
) -> pd.DataFrame:
    rows = []
    dropped = 0
    for _, row in df.iterrows():
        pid = int(row["pid"])
        meta = crop_meta.get(pid)
        if meta is None:
            rows.append(row.to_dict())
            continue

        z0, _, y0, _, x0, _ = meta["bbox"]
        adjusted = row.to_dict()
        adjusted["zmin"] = int(adjusted["zmin"]) - int(z0)
        adjusted["zmax"] = int(adjusted["zmax"]) - int(z0)
        adjusted["ymin"] = int(adjusted["ymin"]) - int(y0)
        adjusted["ymax"] = int(adjusted["ymax"]) - int(y0)
        adjusted["xmin"] = int(adjusted["xmin"]) - int(x0)
        adjusted["xmax"] = int(adjusted["xmax"]) - int(x0)

        clipped = clip_bbox_to_shape(
            (
                adjusted["zmin"],
                adjusted["zmax"],
                adjusted["ymin"],
                adjusted["ymax"],
                adjusted["xmin"],
                adjusted["xmax"],
            ),
            meta["shape_zyx"],
        )
        if clipped is None:
            dropped += 1
            continue
        (
            adjusted["zmin"],
            adjusted["zmax"],
            adjusted["ymin"],
            adjusted["ymax"],
            adjusted["xmin"],
            adjusted["xmax"],
        ) = clipped
        rows.append(adjusted)

    if dropped:
        print(f"warning: dropped {dropped} annotation boxes outside lung-cropped PN9 images")
    return pd.DataFrame(rows, columns=SANET_ANNO_COLUMNS)


def filter_annos_by_ids(df: pd.DataFrame, ids: Iterable[str]) -> pd.DataFrame:
    pid_set = {int(pid) for pid in ids}
    return df[df["pid"].astype(int).isin(pid_set)].copy()


def validate_split_images(split_ids: Dict[str, List[str]], npy_ids: set[str]) -> None:
    missing = {}
    for name, ids in split_ids.items():
        bad = [pid for pid in ids if pid not in npy_ids]
        if bad:
            missing[name] = bad[:5]
    if missing:
        raise FileNotFoundError(f"Some split ids have no *_zoom.npy under PN9 root: {missing}")


def select_unique_npy_paths(npys: List[Path], split_ids: Dict[str, List[str]]) -> Dict[str, Path]:
    """Return one npy per pid, preferring the matching PN9 train/test subtree."""
    by_pid: Dict[str, List[Path]] = {}
    for p in npys:
        by_pid.setdefault(p.stem.replace("_zoom", ""), []).append(p)

    test_ids = set(split_ids.get("test", []))
    selected: Dict[str, Path] = {}
    for pid, paths in by_pid.items():
        paths = sorted(paths, key=lambda p: str(p))
        if pid in test_ids:
            preferred = [p for p in paths if "/test/" in str(p).replace("\\", "/")]
        else:
            preferred = [p for p in paths if "/train/" in str(p).replace("\\", "/")]
        selected[pid] = preferred[0] if preferred else paths[0]
    return selected


def run(args: argparse.Namespace) -> None:
    root = Path(args.pn9_root)
    out_dir = Path(args.out_dir)
    full_dir = ensure_dir(out_dir / "full")
    split_dir = ensure_dir(out_dir / "split")
    lung_masks = collect_lung_masks(Path(args.lung_mask_root)) if args.lung_mask_root else {}

    npys = find_files(root, [".npy"], recursive=True)
    npys = [p for p in npys if p.name.endswith("_zoom.npy")]
    if not npys:
        raise FileNotFoundError(f"No *_zoom.npy found under {root}; uncompress PN9 first.")

    split_files = {
        "train": find_split_file(root, "train.txt"),
        "val": find_split_file(root, "val.txt"),
        "test": find_split_file(root, "test.txt"),
    }
    train_full_file = find_split_file(root, "train_full.txt")
    split_ids = {name: read_id_list(path) for name, path in split_files.items()}
    split_ids["train_full"] = read_id_list(train_full_file)
    selected_npys = select_unique_npy_paths(npys, split_ids)

    pid_map: Dict[str, int] = {}
    crop_meta: Dict[int, Dict[str, object]] = {}
    meta_rows = []
    npy_ids = set()
    for original, p in tqdm(sorted(selected_npys.items()), desc="Prepare PN9 npy"):
        pid = npy_stem_to_pid(p.stem)
        pid_map[original] = pid
        npy_ids.add(original)
        dst = full_dir / f"{original}_zoom.npy"
        # Keep original PN9 filename, because original split txt already refers to it.
        if args.lung_mask_root:
            arr = np.load(p).astype(np.float32, copy=False)
            if arr.ndim != 4 or arr.shape[0] != 1:
                raise ValueError(f"PN9 array must have shape [1,D,H,W], got {arr.shape}: {p}")
            mask_path = match_lung_mask(original, lung_masks)
            bbox = lung_crop_bbox(mask_path, tuple(int(v) for v in arr.shape[1:]), args.lung_crop_margin)
            arr = crop_sanet_array(arr, bbox)
            save_sanet_npy(arr, dst)
            crop_meta[pid] = {
                "bbox": bbox,
                "shape_zyx": tuple(int(v) for v in arr.shape[1:]),
                "mask_path": str(mask_path),
            }
        else:
            link_or_copy(p, dst, mode=args.mode)
        if args.scan_meta:
            arr = np.load(dst, mmap_mode="r")
            row = {
                "original_id": original,
                "sanet_pid": pid,
                "path": str(p),
                "out_path": str(dst),
                "shape": "x".join(map(str, arr.shape)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
            if pid in crop_meta:
                row["lung_mask_path"] = crop_meta[pid]["mask_path"]
                row["lung_crop_bbox_zyx"] = ",".join(map(str, crop_meta[pid]["bbox"]))
            meta_rows.append(row)

    validate_split_images(split_ids, npy_ids)

    for name in ["train", "val", "test", "train_full"]:
        write_id_list(split_dir / f"{name}.txt", split_ids[name])

    train_anno = normalize_annotation_csv(find_split_file(root, "train_anno.csv"))
    test_anno = normalize_annotation_csv(find_split_file(root, "test_anno.csv"))
    if crop_meta:
        train_anno = adjust_annotations_for_lung_crop(train_anno, crop_meta)
        test_anno = adjust_annotations_for_lung_crop(test_anno, crop_meta)
    val_anno = filter_annos_by_ids(train_anno, split_ids["val"])
    all_anno = pd.concat([train_anno, test_anno], ignore_index=True)

    # Current SANet config uses train_anno for both train and val readers, so keep
    # the official PN9 train_full annotations there and also write val_anno for QA.
    write_annotation_csv(split_dir / "train_anno.csv", train_anno)
    write_annotation_csv(split_dir / "val_anno.csv", val_anno)
    write_annotation_csv(split_dir / "test_anno.csv", test_anno)
    write_annotation_csv(split_dir / "all_anno.csv", all_anno)

    write_pid_map(out_dir / "pid_map.csv", pid_map, width=args.pid_width)
    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(out_dir / "meta.csv", index=False)

    summary = {
        "dataset_name": "PN9",
        "note": "PN9 npy is already 1mm isotropic and intensity [0,255]. This script optionally crops npy volumes to the lung-mask bbox and shifts annotations into the cropped coordinate frame.",
        "num_source_npy_files": len(npys),
        "num_unique_cases": len(selected_npys),
        "lung_mask_root": str(args.lung_mask_root) if args.lung_mask_root else "",
        "lung_crop_margin": int(args.lung_crop_margin),
        "num_lung_cropped_cases": len(crop_meta),
        "split_counts": {k: len(v) for k, v in split_ids.items() if k != "train_full"},
        "num_boxes": {
            "train_anno": int(len(train_anno)),
            "val_anno": int(len(val_anno)),
            "test_anno": int(len(test_anno)),
            "all_anno": int(len(all_anno)),
        },
        "annotation_csv": SANET_ANNO_COLUMNS,
        "out_full": str(full_dir),
        "out_split": str(split_dir),
    }
    with (out_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Done. Output: {out_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Normalize downloaded PN9 npy layout for SANet.")
    p.add_argument("--pn9-root", required=True, help="Extracted PN9 npy directory containing *_zoom.npy and split/annotation files.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=["symlink", "copy"], default="copy")
    p.add_argument("--lung-mask-root", default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/lungseg/PN9",
                   help="Directory containing PN9 lung masks used to crop npy before saving. Empty string disables lung cropping.")
    p.add_argument("--lung-crop-margin", type=int, default=5,
                   help="Voxel margin added around the lung mask bbox before cropping.")
    p.add_argument("--scan-meta", action="store_true", help="Read each npy to record shape/min/max; slower.")
    p.add_argument("--pid-width", type=int, default=5)
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())

r'''
PYTHONPATH=/data/医保大赛/code/SANet/dataset/sanet_preprocess_tools \
  python3 -m sanet_prep.prepare_pn9 \
    --pn9-root '/media/SENSETIME\yangtingting/T7/医保大赛数据/PN9/npy' \
    --out-dir /data/医保大赛/code/SANet/data/PN9
'''
