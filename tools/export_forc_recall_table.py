#!/usr/bin/env python3
"""Export per-dataset recall tables from SANet FROC result files."""

import argparse
import csv
import math
import re
from bisect import bisect_right
from pathlib import Path

'''
输出一个表格，每个数据集per scan（0.5 1 2 4 8 16 不限制阈值）下的recall各是多少
  '''

DEFAULT_POINTS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan result directories, read FROC curves, and write one recall "
            "table per dataset."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="/mnt/afs2/code/SANet/test_output_ensenmble/res/48",
        help="Experiment root, e.g. test_output/res/14_pretrained_rcnn20",
    )
    parser.add_argument(
        "--result-dir",
        default="res_froc_eval_diamter05",
        help=(
            "Result subdirectory under each dataset FROC folder. Default: "
            "auto-detect res and res_froc_eval"
        ),
    )
    parser.add_argument(
        "--points",
        default=",".join(str(x).rstrip("0").rstrip(".") for x in DEFAULT_POINTS),
        help="Comma/space separated FP/scan points. Default: 0.5,1,2,4,8,16",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=6,
        help="Number of decimal places to print. Default: 6",
    )
    return parser.parse_args()


def parse_points(text):
    points = []
    for item in re.split(r"[,\s]+", text.strip()):
        if not item:
            continue
        points.append(float(item))
    if not points:
        raise ValueError("at least one FP/scan point is required")
    return points


def find_result_files(root, result_dir=None):
    root = Path(root)
    if result_dir is not None:
        return sorted(root.rglob(f"FROC/{result_dir}/froc_results*.txt"))

    files = []
    for pattern in ("FROC/res/froc_results*.txt", "FROC/res_froc_eval/froc_results*.txt"):
        files.extend(root.rglob(pattern))
    return sorted({path.resolve(): path for path in files}.values())


def read_froc_curve(path):
    fps = []
    recalls = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                fp_per_scan = float(row[0])
                recall = float(row[1])
            except ValueError:
                continue
            if math.isfinite(fp_per_scan):
                fps.append(fp_per_scan)
                recalls.append(recall)

    if not fps:
        raise ValueError(f"no numeric FROC rows found in {path}")

    order = sorted(range(len(fps)), key=lambda i: fps[i])
    return [fps[i] for i in order], [recalls[i] for i in order]


def interp_recall(fps, recalls, target):
    if target <= fps[0]:
        idx = bisect_right(fps, target) - 1
        return recalls[idx] if idx >= 0 and fps[idx] == target else recalls[0]
    if target >= fps[-1]:
        idx = bisect_right(fps, target) - 1
        return recalls[idx]

    right = bisect_right(fps, target)
    left = right - 1
    if fps[left] == target:
        return recalls[left]

    span = fps[right] - fps[left]
    if span == 0:
        return recalls[left]
    ratio = (target - fps[left]) / span
    return recalls[left] + ratio * (recalls[right] - recalls[left])


def read_unlimited_recall(froc_path, fallback):
    analysis_path = froc_path.with_name("CADAnalysis.txt")
    if analysis_path.exists():
        for line in analysis_path.read_text(errors="replace").splitlines():
            match = re.search(r"\bSensitivity:\s*([0-9.eE+-]+)", line)
            if match:
                return float(match.group(1))
    return fallback


def infer_dataset_name(path):
    parts = path.parts
    try:
        froc_idx = parts.index("FROC")
    except ValueError:
        return path.parent.name
    return parts[froc_idx - 1] if froc_idx >= 1 else path.parent.name


def markdown_table(headers, rows):
    widths = [len(header) for header in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def render(row):
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render(headers), sep] + [render(row) for row in rows])


def format_value(value, digits):
    return f"{value:.{digits}f}"


def main():
    args = parse_args()
    points = parse_points(args.points)
    result_files = find_result_files(args.root, args.result_dir)
    if not result_files:
        search = args.result_dir if args.result_dir else "res or res_froc_eval"
        raise SystemExit(f"No FROC result files found under {args.root} for {search}")

    grouped = {}
    for result_file in result_files:
        dataset = infer_dataset_name(result_file)
        current = grouped.get(dataset)
        if current is None:
            grouped[dataset] = result_file
            continue

        current_dir = current.parent.name
        new_dir = result_file.parent.name
        priority = {"res_froc_eval": 2, "res": 1}
        current_score = priority.get(current_dir, 0)
        new_score = priority.get(new_dir, 0)
        if new_score > current_score:
            grouped[dataset] = result_file
        elif new_score == current_score and result_file.stat().st_mtime > current.stat().st_mtime:
            grouped[dataset] = result_file

    for dataset, result_file in sorted(grouped.items()):
        fps, recalls = read_froc_curve(result_file)
        unlimited = read_unlimited_recall(result_file, max(recalls))
        row_values = [dataset]
        row_values.extend(format_value(interp_recall(fps, recalls, point), args.digits) for point in points)
        row_values.append(format_value(unlimited, args.digits))

        headers = ["dataset"] + [f"recall@{point:g}/scan" for point in points] + ["recall@unlimited"]
        table = markdown_table(headers, [row_values])

        out_dir = result_file.parent
        md_path = out_dir / "recall_table.md"
        csv_path = out_dir / "recall_table.csv"

        # md_path.write_text(table + "\n", encoding="utf-8")
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(row_values)

        # print(f"Wrote {md_path}")
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()