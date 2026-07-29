from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from .common import clip_bbox_to_shape, ensure_dir, extract_first_int, find_files, link_or_copy, natural_key, save_sanet_npy


def strip_case_suffix(path_or_id: Path | str) -> str:
    name = Path(str(path_or_id)).name
    for suffix in [".nii.gz", ".mha.gz", ".mhd.gz", ".nii", ".mha", ".mhd", ".gz", ".npy"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("_zoom", "")


def pid_keys(value: object) -> set[str]:
    raw = str(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    keys = {raw, raw.replace("_zoom", "")}
    n = extract_first_int(raw)
    if n is not None:
        keys.update({str(n), f"{n:05d}", f"{n:06d}"})
    return keys


def read_expected_stems(pid_map_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pid_map_csv, dtype=str)
    if "sanet_filename_stem" not in df.columns:
        raise KeyError(f"{pid_map_csv} must contain column 'sanet_filename_stem'. Columns: {list(df.columns)}")
    if "original_id" not in df.columns:
        df["original_id"] = df["sanet_filename_stem"]
    df["sanet_filename_stem"] = df["sanet_filename_stem"].astype(str).str.strip().str.replace("_zoom", "", regex=False)
    df["original_id"] = df["original_id"].astype(str).str.strip().str.replace("_zoom", "", regex=False)
    return df


def existing_full_stems(full_dir: Path) -> set[str]:
    if not full_dir.exists():
        return set()
    return {p.name[:-len("_zoom.npy")] for p in full_dir.glob("*_zoom.npy")}


def collect_source_npys(pn9_root: Path) -> Dict[str, Path]:
    paths = [p for p in find_files(pn9_root, [".npy"], recursive=True) if p.name.endswith("_zoom.npy")]
    out: Dict[str, Path] = {}
    for path in sorted(paths, key=lambda p: str(p)):
        stem = strip_case_suffix(path)
        keys = pid_keys(stem)
        for key in keys:
            out.setdefault(key, path)
    if not out:
        raise FileNotFoundError(f"No *_zoom.npy found under {pn9_root}")
    return out


def collect_lung_masks(root: Path) -> Dict[str, Path]:
    masks: Dict[str, Path] = {}
    for path in find_files(root, [".nii.gz", ".nii", ".mha", ".mhd"], recursive=True):
        case_id = strip_case_suffix(path)
        for key in pid_keys(case_id):
            masks.setdefault(key, path)
    if not masks:
        raise FileNotFoundError(f"No lung masks found under {root}")
    return masks


def resolve_source(row: pd.Series, source_npys: Dict[str, Path]) -> Optional[Path]:
    keys = set()
    keys.update(pid_keys(row["sanet_filename_stem"]))
    keys.update(pid_keys(row["original_id"]))
    for key in keys:
        if key in source_npys:
            return source_npys[key]
    return None


def resolve_mask(row: pd.Series, lung_masks: Dict[str, Path]) -> Path:
    keys = set()
    keys.update(pid_keys(row["sanet_filename_stem"]))
    keys.update(pid_keys(row["original_id"]))
    for key in keys:
        if key in lung_masks:
            return lung_masks[key]
    examples = sorted({str(p) for p in lung_masks.values()}, key=natural_key)[:5]
    raise FileNotFoundError(f"No lung mask matched {row.to_dict()}. Examples: {examples}")


def lung_crop_bbox(mask_path: Path, shape_zyx: tuple[int, int, int], margin: int) -> tuple[int, int, int, int, int, int]:
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) > 0
    if mask.shape != shape_zyx:
        raise ValueError(f"Lung mask {mask_path} shape {mask.shape} does not match image shape {shape_zyx}")
    coords = np.where(mask)
    if len(coords[0]) == 0:
        raise ValueError(f"Lung mask {mask_path} is empty")
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


def write_list(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def repair_missing(args: argparse.Namespace) -> None:
    sanet_dir = Path(args.sanet_dir)
    full_dir = Path(args.full_dir) if args.full_dir else sanet_dir / "full"
    pid_map_csv = Path(args.pid_map_csv) if args.pid_map_csv else sanet_dir / "pid_map.csv"
    report_dir = ensure_dir(Path(args.report_dir) if args.report_dir else sanet_dir / "repair_reports")

    pid_map = read_expected_stems(pid_map_csv)
    existing = existing_full_stems(full_dir)
    missing_df = pid_map[~pid_map["sanet_filename_stem"].isin(existing)].copy()

    missing_rows = missing_df[["original_id", "sanet_filename_stem"]].to_dict("records")
    write_list(report_dir / "missing_pn9_files.csv", missing_rows, ["original_id", "sanet_filename_stem"])
    print(f"Expected: {len(pid_map)}, existing: {len(existing)}, missing: {len(missing_df)}")
    print(f"Missing list: {report_dir / 'missing_pn9_files.csv'}")
    if len(missing_df) == 0 or args.check_only:
        return

    source_npys = collect_source_npys(Path(args.pn9_root))
    lung_masks = collect_lung_masks(Path(args.lung_mask_root)) if args.lung_mask_root else {}
    ensure_dir(full_dir)

    converted = []
    unresolved = []
    for _, row in tqdm(list(missing_df.iterrows()), desc="Repair PN9 missing npy"):
        source = resolve_source(row, source_npys)
        dst = full_dir / f"{row['sanet_filename_stem']}_zoom.npy"
        if source is None:
            unresolved.append({
                "original_id": row["original_id"],
                "sanet_filename_stem": row["sanet_filename_stem"],
                "reason": "source *_zoom.npy not found",
            })
            continue
        if dst.exists() and not args.overwrite:
            continue

        meta = {
            "original_id": row["original_id"],
            "sanet_filename_stem": row["sanet_filename_stem"],
            "source_path": str(source),
            "out_path": str(dst),
        }
        if args.lung_mask_root:
            arr = np.load(source).astype(np.float32, copy=False)
            if arr.ndim != 4 or arr.shape[0] != 1:
                raise ValueError(f"PN9 array must have shape [1,D,H,W], got {arr.shape}: {source}")
            mask_path = resolve_mask(row, lung_masks)
            bbox = lung_crop_bbox(mask_path, tuple(int(v) for v in arr.shape[1:]), args.lung_crop_margin)
            cropped = crop_sanet_array(arr, bbox)
            save_sanet_npy(cropped, dst)
            meta.update({
                "mode": "lung_crop",
                "lung_mask_path": str(mask_path),
                "lung_crop_bbox_zyx": ",".join(map(str, bbox)),
                "source_shape": "x".join(map(str, arr.shape)),
                "out_shape": "x".join(map(str, cropped.shape)),
            })
        else:
            link_or_copy(source, dst, mode=args.mode)
            meta.update({"mode": args.mode})
        converted.append(meta)

    write_list(
        report_dir / "converted_pn9_files.csv",
        converted,
        [
            "original_id", "sanet_filename_stem", "source_path", "out_path", "mode",
            "lung_mask_path", "lung_crop_bbox_zyx", "source_shape", "out_shape",
        ],
    )
    write_list(report_dir / "unresolved_pn9_files.csv", unresolved, ["original_id", "sanet_filename_stem", "reason"])
    with (report_dir / "repair_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "sanet_dir": str(sanet_dir),
            "full_dir": str(full_dir),
            "pid_map_csv": str(pid_map_csv),
            "pn9_root": str(args.pn9_root),
            "lung_mask_root": str(args.lung_mask_root),
            "expected": int(len(pid_map)),
            "existing_before": int(len(existing)),
            "missing_before": int(len(missing_df)),
            "converted": int(len(converted)),
            "unresolved": int(len(unresolved)),
        }, f, indent=2, ensure_ascii=False)

    print(f"Converted: {len(converted)}")
    print(f"Unresolved: {len(unresolved)}")
    print(f"Report dir: {report_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detect and repair missing PN9 SANet full npy files from pid_map.csv.")
    p.add_argument("--sanet-dir", required=True, help="Prepared SANet PN9 directory containing pid_map.csv and full/.")
    p.add_argument("--pn9-root", required=True, help="Original/local PN9 npy root containing *_zoom.npy files.")
    p.add_argument("--pid-map-csv", default="", help="Defaults to <sanet-dir>/pid_map.csv.")
    p.add_argument("--full-dir", default="", help="Defaults to <sanet-dir>/full.")
    p.add_argument("--report-dir", default="", help="Defaults to <sanet-dir>/repair_reports.")
    p.add_argument("--mode", choices=["copy", "symlink"], default="copy", help="Used only when --lung-mask-root is empty.")
    p.add_argument("--lung-mask-root", default=r"/media/SENSETIME\yangtingting/T7/医保大赛数据/lungseg/PN9",
                   help="PN9 lung mask root. Pass empty string '' to disable lung cropping.")
    p.add_argument("--lung-crop-margin", type=int, default=5)
    p.add_argument("--check-only", action="store_true", help="Only write missing_pn9_files.csv; do not convert.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output npy if it already exists.")
    return p


if __name__ == "__main__":
    repair_missing(build_argparser().parse_args())

'''
python -m sanet_prep.repair_pn9_missing \
    --sanet-dir /data/医保大赛/code/SANet/data/PN9 \
    --pid-map-csv /data/医保大赛/code/SANet/data/PN9/missing_pn9_files.csv \
    --pn9-root '/media/SENSETIME\yangtingting/T7/医保大赛数据/PN9/npy' \
    --lung-mask-root '/media/SENSETIME\yangtingting/T7/医保大赛数据/lungseg/PN9'
'''