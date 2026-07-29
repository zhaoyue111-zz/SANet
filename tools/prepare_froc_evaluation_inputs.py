#!/usr/bin/env python3
"""Prepare extra CSV inputs required by froc_evaluation/noduleCADEvaluationLUNA16.py."""

import argparse
import csv
from pathlib import Path

'''
需要空的annotations_excluded.csv和有效的seriesuids.csv
'''
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Given a dataset result directory, create annotations_excluded.csv "
            "and seriesuids.csv for froc_evaluation/noduleCADEvaluationLUNA16.py."
        )
    )
    parser.add_argument(
        "dataset_result_dir",
        help=(
            "Dataset result dir, e.g. "
            "test_output/res/14_pretrained_rcnn20/histopathology"
        ),
    )
    parser.add_argument(
        "--out-dir",
        help="Output dir for generated CSV files. Default: <dataset_result_dir>/FROC",
    )
    parser.add_argument(
        "--prefer",
        choices=("annotations", "results", "detections"),
        default="annotations",
        help="Where to get seriesuids first. Default: annotations",
    )
    return parser.parse_args()


def read_pid_column(csv_path):
    if not csv_path.exists():
        return []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "pid" not in reader.fieldnames:
            return []
        return [row["pid"] for row in reader if row.get("pid")]


def rewrite_pid_column(src_path, dst_path):
    with src_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"missing CSV header: {src_path}")
        rows = list(reader)

    if "pid" not in reader.fieldnames:
        raise ValueError(f"missing pid column: {src_path}")

    for row in rows:
        row["pid"] = normalize_pid(row["pid"])

    with dst_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_pid(pid):
    text = str(pid).strip()
    if not text:
        return ""
    try:
        return f"{int(float(text)):05d}"
    except ValueError:
        return text


def unique_in_order(values):
    seen = set()
    output = []
    for value in values:
        normalized = normalize_pid(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def seriesuids_from_detections(dataset_dir):
    pids = []
    for path in sorted(dataset_dir.glob("*_detections.npy")):
        pids.append(path.name[: -len("_detections.npy")])
    return unique_in_order(pids)


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_result_dir)
    froc_dir = dataset_dir / "FROC"
    out_dir = Path(args.out_dir) if args.out_dir else froc_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "annotations": lambda: unique_in_order(read_pid_column(froc_dir / "annotations.csv")),
        "results": lambda: unique_in_order(read_pid_column(froc_dir / "results.csv")),
        "detections": lambda: seriesuids_from_detections(dataset_dir),
    }

    order = [args.prefer] + [name for name in ("annotations", "results", "detections") if name != args.prefer]
    seriesuids = []
    used_source = ""
    for source_name in order:
        seriesuids = sources[source_name]()
        if seriesuids:
            used_source = source_name
            break

    if not seriesuids:
        raise SystemExit(f"No seriesuids found under: {dataset_dir}")

    seriesuids_path = out_dir / "seriesuids.csv"
    with seriesuids_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for pid in seriesuids:
            writer.writerow([pid])

    excluded_path = out_dir / "annotations_excluded.csv"
    with excluded_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pid", "center_x", "center_y", "center_z", "diameter"])

    annotations_path = out_dir / "annotations_froc_eval.csv"
    results_path = out_dir / "results_froc_eval.csv"
    rewrite_pid_column(froc_dir / "annotations.csv", annotations_path)
    rewrite_pid_column(froc_dir / "results.csv", results_path)

    print(f"seriesuids: {seriesuids_path} ({len(seriesuids)} cases, from {used_source})")
    print(f"excluded annotations: {excluded_path}")
    print(f"normalized annotations: {annotations_path}")
    print(f"normalized results: {results_path}")


if __name__ == "__main__":
    main()

'''
python tools/prepare_froc_evaluation_inputs.py     test_output/res/14_pretrained_rcnn20/histopathology
'''