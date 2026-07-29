#!/usr/bin/env python3
"""Summarize recall at selected FP/scan points from SANet FROC outputs."""

import argparse
import csv
import math
import re
from bisect import bisect_right
from pathlib import Path


DEFAULT_POINTS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find FROC/res/froc_results.txt files and report recall at selected "
            "FP/scan points plus recall with no score threshold."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["test_output/res"],
        help=(
            "Experiment/result directories or froc_results.txt files. "
            "Default: test_output/res"
        ),
    )
    parser.add_argument(
        "--points",
        default=",".join(str(x).rstrip("0").rstrip(".") for x in DEFAULT_POINTS),
        help="Comma/space separated FP/scan points. Default: 0.5,1,2,4,8,16",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional CSV output path.",
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


def find_froc_files(paths):
    files = []
    seen = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.rglob("FROC/res/froc_results.txt"))
        else:
            candidates = []

        for candidate in candidates:
            if candidate.name != "froc_results.txt":
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                files.append(candidate)
                seen.add(resolved)
    return files


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
            if math.isfinite(fp_per_scan) and math.isfinite(recall):
                fps.append(fp_per_scan)
                recalls.append(recall)

    if not fps:
        raise ValueError("no numeric FROC rows found")

    order = sorted(range(len(fps)), key=lambda i: fps[i])
    return [fps[i] for i in order], [recalls[i] for i in order]


def interp_recall(fps, recalls, target):
    """Linear interpolation matching numpy.interp's rightmost duplicate behavior."""
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


def infer_names(path):
    parts = path.parts
    try:
        froc_idx = parts.index("FROC")
    except ValueError:
        return "", path.parent.name

    dataset = parts[froc_idx - 1] if froc_idx >= 1 else ""
    experiment = parts[froc_idx - 2] if froc_idx >= 2 else ""
    return experiment, dataset


def format_number(value, digits):
    return f"{value:.{digits}f}"


def markdown_table(rows, headers):
    widths = [len(header) for header in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def render(row):
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render(headers), sep] + [render(row) for row in rows])


def main():
    args = parse_args()
    points = parse_points(args.points)
    froc_files = find_froc_files(args.paths)
    if not froc_files:
        searched = ", ".join(args.paths)
        raise SystemExit(f"No FROC/res/froc_results.txt files found under: {searched}")

    csv_rows = []
    display_rows = []
    for froc_file in froc_files:
        fps, recalls = read_froc_curve(froc_file)
        experiment, dataset = infer_names(froc_file)
        point_recalls = [interp_recall(fps, recalls, point) for point in points]
        unlimited = read_unlimited_recall(froc_file, max(recalls))

        csv_row = {
            "experiment": experiment,
            "dataset": dataset,
            "froc_file": str(froc_file),
        }
        for point, recall in zip(points, point_recalls):
            csv_row[f"recall@{point:g}fp/scan"] = recall
        csv_row["recall@unlimited"] = unlimited
        csv_rows.append(csv_row)

        display_rows.append(
            [experiment, dataset]
            + [format_number(value, args.digits) for value in point_recalls]
            + [format_number(unlimited, args.digits)]
        )

    headers = (
        ["experiment", "dataset"]
        + [f"recall@{point:g}/scan" for point in points]
        + ["recall@unlimited"]
    )
    print(markdown_table(display_rows, headers))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(csv_rows[0].keys())
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nWrote CSV: {output_path}")


if __name__ == "__main__":
    main()
