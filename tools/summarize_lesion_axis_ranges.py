#!/usr/bin/env python3
"""Summarize lesion extents from data/*/split/all_anno.csv."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["zmin", "zmax", "ymin", "ymax", "xmin", "xmax"]


def lesion_ranges(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("missing required columns: %s" % ", ".join(missing))

    values = df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid = values.notna().all(axis=1)
    values = values[valid]
    if values.empty:
        return {
            "count": 0,
            "max_z": np.nan,
            "max_y": np.nan,
            "max_x": np.nan,
        }

    z_range = values["zmax"] - values["zmin"] + 1
    y_range = values["ymax"] - values["ymin"] + 1
    x_range = values["xmax"] - values["xmin"] + 1
    return {
        "count": int(len(values)),
        "max_z": float(z_range.max()),
        "max_y": float(y_range.max()),
        "max_x": float(x_range.max()),
    }


def find_anno_files(data_root):
    return sorted(Path(data_root).glob("*/split/all_anno.csv"))


def main():
    parser = argparse.ArgumentParser(
        description="Read all data/*/split/all_anno.csv files and report max lesion extent per axis."
    )
    parser.add_argument(
        "--data-root",
        default="/data/医保大赛/code/SANet/data",
        help="Directory containing dataset subdirectories. Default: %(default)s",
    )
    args = parser.parse_args()

    anno_files = find_anno_files(args.data_root)
    if not anno_files:
        raise FileNotFoundError("No */split/all_anno.csv files found under %s" % args.data_root)

    rows = []
    all_frames = []
    for path in anno_files:
        dataset = path.parents[1].name
        df = pd.read_csv(path)
        stats = lesion_ranges(df)
        rows.append((dataset, path, stats))
        all_frames.append(df)

    print("dataset,count,max_z_range,max_y_range,max_x_range,path")
    for dataset, path, stats in rows:
        print("%s,%d,%g,%g,%g,%s" % (
            dataset,
            stats["count"],
            stats["max_z"],
            stats["max_y"],
            stats["max_x"],
            path,
        ))

    total_stats = lesion_ranges(pd.concat(all_frames, ignore_index=True))
    print("ALL,%d,%g,%g,%g,%s" % (
        total_stats["count"],
        total_stats["max_z"],
        total_stats["max_y"],
        total_stats["max_x"],
        args.data_root,
    ))


if __name__ == "__main__":
    main()
