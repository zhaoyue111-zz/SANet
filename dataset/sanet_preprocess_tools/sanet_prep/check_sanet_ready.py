'''
检查SANet-ready数据集是否符合要求
'''

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

REQUIRED_COLS = ["pid", "zmin", "zmax", "ymin", "ymax", "xmin", "xmax"]


def read_split(path: Path):
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


def run(args):
    root = Path(args.sanet_dir)
    full = root / "full"
    split = root / "split"
    assert full.exists(), f"Missing {full}"
    assert split.exists(), f"Missing {split}"

    ids = {p.stem.replace("_zoom", ""): p for p in full.glob("*_zoom.npy")}
    print(f"Found {len(ids)} *_zoom.npy files")
    for name in ["train", "val", "test"]:
        pids = read_split(split / f"{name}.txt")
        missing = [pid for pid in pids if pid not in ids]
        print(f"{name}: {len(pids)} cases, missing npy={len(missing)}")
        if missing[:5]:
            print("  examples missing:", missing[:5])

    shape_cache: Dict[int, tuple] = {}
    intensity_ranges = []
    checked = 0
    for stem, p in list(ids.items())[: args.max_npy_check]:
        arr = np.load(p, mmap_mode="r")
        assert arr.ndim == 4 and arr.shape[0] == 1, f"{p} should be [1,D,H,W], got {arr.shape}"
        assert arr.min() >= -1e-3 and arr.max() <= 255 + 1e-3, f"{p} intensity not in [0,255]: {arr.min()}..{arr.max()}"
        intensity_ranges.append((float(arr.min()), float(arr.max())))
        try:
            shape_cache[int(stem)] = tuple(arr.shape[1:])
        except ValueError:
            pass
        checked += 1
    print(f"Checked {checked} npy files for shape/intensity")
    if intensity_ranges and max(vmax for _, vmax in intensity_ranges) <= 1.0 + 1e-3:
        raise AssertionError(
            "All checked volumes are in [0,1], but SANet expects [0,255]. "
            "Scale normalized CT values by 255 during preprocessing."
        )

    total_boxes = 0
    for csv_name in ["train_anno.csv", "val_anno.csv", "test_anno.csv", "all_anno.csv"]:
        csv_path = split / csv_name
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
        assert not missing_cols, f"{csv_path} missing columns {missing_cols}"
        bad_order = df[(df.zmax < df.zmin) | (df.ymax < df.ymin) | (df.xmax < df.xmin)]
        assert len(bad_order) == 0, f"{csv_path} has invalid min/max rows: {len(bad_order)}"
        # Bounds check for pids already sampled in max_npy_check.
        bad_bounds = []
        for _, r in df.iterrows():
            pid = int(r.pid)
            if pid not in shape_cache:
                continue
            D, H, W = shape_cache[pid]
            if not (0 <= r.zmin <= r.zmax < D and 0 <= r.ymin <= r.ymax < H and 0 <= r.xmin <= r.xmax < W):
                bad_bounds.append(pid)
        assert not bad_bounds, f"{csv_path} has boxes outside image bounds; examples: {bad_bounds[:5]}"
        print(f"{csv_name}: {len(df)} boxes OK")
        if csv_name == "all_anno.csv":
            total_boxes = len(df)

    if total_boxes == 0 and not args.allow_empty_annotations:
        raise AssertionError(
            "all_anno.csv contains 0 boxes. Check mask/annotation matching, or "
            "pass --allow-empty-annotations for an intentionally negative dataset."
        )

    print("SANet-ready validation finished.")


def build_argparser():
    p = argparse.ArgumentParser(description="Validate SANet-ready dataset folder.")
    p.add_argument("--sanet-dir", default="/data/医保大赛/code/SANet/data/NLSTSeg")
    p.add_argument("--max-npy-check", type=int, default=100, help="Limit slow npy min/max reads; use a larger value for final QA.")
    p.add_argument("--allow-empty-annotations", action="store_true")
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())
