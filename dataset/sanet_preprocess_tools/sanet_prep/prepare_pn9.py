from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import ensure_dir, extract_first_int, find_files, link_or_copy, write_pid_map


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
    meta_rows = []
    npy_ids = set()
    for original, p in tqdm(sorted(selected_npys.items()), desc="Link/copy PN9 npy"):
        pid = npy_stem_to_pid(p.stem)
        pid_map[original] = pid
        npy_ids.add(original)
        dst = full_dir / f"{original}_zoom.npy"
        # Keep original PN9 filename, because original split txt already refers to it.
        link_or_copy(p, dst, mode=args.mode)
        if args.scan_meta:
            arr = np.load(p, mmap_mode="r")
            meta_rows.append({"original_id": original, "sanet_pid": pid, "path": str(p), "shape": "x".join(map(str, arr.shape)), "min": float(np.min(arr)), "max": float(np.max(arr))})

    validate_split_images(split_ids, npy_ids)

    for name in ["train", "val", "test", "train_full"]:
        write_id_list(split_dir / f"{name}.txt", split_ids[name])

    train_anno = normalize_annotation_csv(find_split_file(root, "train_anno.csv"))
    test_anno = normalize_annotation_csv(find_split_file(root, "test_anno.csv"))
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
        "note": "PN9 npy is already preprocessed by SANet authors: 1mm isotropic and intensity [0,255]. This script only normalizes folder layout.",
        "num_source_npy_files": len(npys),
        "num_unique_cases": len(selected_npys),
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
    p.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    p.add_argument("--scan-meta", action="store_true", help="Read each npy to record shape/min/max; slower.")
    p.add_argument("--pid-width", type=int, default=6)
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())

'''
PYTHONPATH=/data/医保大赛/code/SANet/dataset/sanet_preprocess_tools \
  python3 -m sanet_prep.prepare_pn9 \
    --pn9-root '/media/SENSETIME\yangtingting/T7/医保大赛数据/PN9/npy' \
    --out-dir /data/医保大赛/code/SANet/data/PN9

'''