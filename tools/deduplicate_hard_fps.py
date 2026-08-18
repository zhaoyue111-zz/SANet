#!/usr/bin/env python3
"""Deduplicate mined SANet hard false-positive boxes."""

import argparse
import math
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "/mnt/afs2/code/SANet/hard_examples_v3/train/hard_fps.csv"
DEFAULT_IOM_THRESHOLD = 0.50
DEFAULT_DISTANCE_THRESHOLD = 0.5
EPS = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate hard FP boxes by descending probability. A lower-score "
            "box is removed when IoM >= threshold and normalized center distance "
            "<= threshold relative to an already kept box."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT,
        help="Input hard_fps.csv path. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output CSV path. The input file is never overwritten.",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Only deduplicate rows whose dataset column matches this value. Empty means all datasets.",
    )
    parser.add_argument(
        "--pid",
        default="",
        help="Only deduplicate rows whose pid column matches this value. Empty means all pids.",
    )
    parser.add_argument(
        "--iom-threshold",
        type=float,
        default=DEFAULT_IOM_THRESHOLD,
        help="IoM threshold for duplicate boxes. Default: %(default)s",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=DEFAULT_DISTANCE_THRESHOLD,
        help="Normalized center distance threshold for duplicate boxes. Default: %(default)s",
    )
    return parser.parse_args()


def normalize_pid(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def validate_columns(df, path):
    required = {
        "dataset",
        "pid",
        "center_x",
        "center_y",
        "center_z",
        "diameter",
        "probability",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("%s is missing columns: %s" % (path, ", ".join(sorted(missing))))


def box_bounds(row):
    radius = float(row["diameter"]) / 2.0
    return (
        float(row["center_x"]) - radius,
        float(row["center_x"]) + radius,
        float(row["center_y"]) - radius,
        float(row["center_y"]) + radius,
        float(row["center_z"]) - radius,
        float(row["center_z"]) + radius,
    )


def box_volume(row):
    diameter = max(float(row["diameter"]), 0.0)
    return diameter ** 3


def intersection_volume(a, b):
    ax0, ax1, ay0, ay1, az0, az1 = box_bounds(a)
    bx0, bx1, by0, by1, bz0, bz1 = box_bounds(b)
    dx = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    dy = max(0.0, min(ay1, by1) - max(ay0, by0))
    dz = max(0.0, min(az1, bz1) - max(az0, bz0))
    return dx * dy * dz


def iom(a, b):
    denominator = min(box_volume(a), box_volume(b))
    if denominator <= EPS:
        return 0.0
    return intersection_volume(a, b) / denominator


def normalized_center_distance(a, b):
    dx = float(a["center_x"]) - float(b["center_x"])
    dy = float(a["center_y"]) - float(b["center_y"])
    dz = float(a["center_z"]) - float(b["center_z"])
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    scale = max((float(a["diameter"]) + float(b["diameter"])) / 2.0, EPS)
    return distance / scale


def is_duplicate(candidate, kept, iom_threshold, distance_threshold):
    return (
        iom(candidate, kept) >= iom_threshold
        and normalized_center_distance(candidate, kept) <= distance_threshold
    )


def deduplicate_group(group, iom_threshold, distance_threshold):
    sorted_group = group.sort_values(
        by="probability",
        ascending=False,
        kind="mergesort",
    )

    kept_rows = []
    kept_indices = []
    removed_indices = []
    for index, row in sorted_group.iterrows():
        if any(is_duplicate(row, kept, iom_threshold, distance_threshold) for kept in kept_rows):
            removed_indices.append(index)
            continue
        kept_rows.append(row)
        kept_indices.append(index)

    return kept_indices, removed_indices


def build_filter(df, dataset, pid):
    mask = pd.Series(True, index=df.index)
    if dataset:
        mask &= df["dataset"].astype(str) == dataset
    if pid:
        target = normalize_pid(pid)
        mask &= df["pid"].map(normalize_pid) == target
    return mask


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.resolve() == output_path.resolve():
        raise ValueError("output path must not be the same as input path")

    df = pd.read_csv(input_path)
    validate_columns(df, input_path)

    selected_mask = build_filter(df, args.dataset.strip(), args.pid.strip())
    selected = df[selected_mask].copy()

    keep_selected = []
    remove_selected = []
    if not selected.empty:
        for _, group in selected.groupby(["dataset", selected["pid"].map(normalize_pid)], sort=False):
            kept, removed = deduplicate_group(
                group,
                iom_threshold=args.iom_threshold,
                distance_threshold=args.distance_threshold,
            )
            keep_selected.extend(kept)
            remove_selected.extend(removed)

    output = df.drop(index=remove_selected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print("input rows: %d" % len(df))
    print("selected rows: %d" % len(selected))
    print("kept selected rows: %d" % len(keep_selected))
    print("removed duplicate rows: %d" % len(remove_selected))
    print("output rows: %d" % len(output))
    print("wrote: %s" % output_path)


if __name__ == "__main__":
    main()

'''
# 全部数据去重，写到新文件
  python tools/deduplicate_hard_fps.py -o train_hard_examples/hard_fps_dedup.csv

# 只处理某个数据集，其它行原样保留
  python tools/deduplicate_hard_fps.py --dataset LNDB -o train_hard_examples/hard_fps_dedup_lndb.csv

# 只处理某个 dataset + pid，其它行原样保留
  python tools/deduplicate_hard_fps.py --dataset LNDB --pid 75 -o train_hard_examples/hard_fps_dedup_lndb_75.csv
'''