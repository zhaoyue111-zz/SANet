#!/usr/bin/env python3
"""Prepare CSV inputs required by froc_evaluation/noduleCADEvaluationLUNA16.py.

For the new organized DDP test flow, test.py writes FROC/seriesuids.csv directly
from test.txt. This helper prefers that complete list, so scans with neither GT
nor predictions are still counted by FROC.

No *_detections.npy file is required by the new flow. Legacy detection-file
fallback is kept only for backward compatibility.
"""

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Given a result directory, create normalized FROC inputs and "
            "preserve/use the complete seriesuids.csv when available."
        )
    )
    parser.add_argument(
        "dataset_result_dir",
        help="Result dir containing FROC/, e.g. test_output/luna25_organized",
    )
    parser.add_argument(
        "--out-dir",
        help="Output dir. Default: <dataset_result_dir>/FROC",
    )
    parser.add_argument(
        "--prefer",
        choices=("seriesuids", "annotations", "results", "detections"),
        default="seriesuids",
        help=(
            "Where to get seriesuids first. Default: seriesuids. "
            "Use the test.py-generated complete list for organized DDP test."
        ),
    )
    return parser.parse_args()


def normalize_pid(pid):
    text = str(pid).strip()
    if not text:
        return ""
    # Preserve DICOM UIDs and other non-numeric IDs. Keep old numeric PID
    # normalization for legacy SANet datasets.
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


def read_pid_column(csv_path):
    if not csv_path.exists():
        return []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "pid" not in reader.fieldnames:
            return []
        return [row["pid"] for row in reader if row.get("pid")]


def read_seriesuids_csv(csv_path):
    """Read the one-column, no-header seriesuids.csv used by LUNA FROC."""
    if not csv_path.exists():
        return []

    values = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            value = str(row[0]).strip()
            if value.lower() in {"pid", "seriesuid", "seriesuids"}:
                continue
            if value:
                values.append(value)
    return unique_in_order(values)


def rewrite_pid_column(src_path, dst_path):
    if not src_path.exists():
        raise FileNotFoundError(f"Missing required CSV: {src_path}")

    with src_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"missing CSV header: {src_path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if "pid" not in fieldnames:
        raise ValueError(f"missing pid column: {src_path}")

    for row in rows:
        row["pid"] = normalize_pid(row["pid"])

    with dst_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def seriesuids_from_detections(dataset_dir):
    # Legacy fallback only. New test.py deliberately creates no such files.
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

    seriesuids_existing = froc_dir / "seriesuids.csv"

    sources = {
        "seriesuids": lambda: read_seriesuids_csv(seriesuids_existing),
        "annotations": lambda: unique_in_order(
            read_pid_column(froc_dir / "annotations.csv")
        ),
        "results": lambda: unique_in_order(
            read_pid_column(froc_dir / "results.csv")
        ),
        "detections": lambda: seriesuids_from_detections(dataset_dir),
    }

    source_order = ("seriesuids", "annotations", "results", "detections")
    order = [args.prefer] + [
        name for name in source_order if name != args.prefer
    ]

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
    with seriesuids_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for pid in seriesuids:
            writer.writerow([pid])

    excluded_path = out_dir / "annotations_excluded.csv"
    with excluded_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pid", "center_x", "center_y", "center_z", "diameter"])

    annotations_path = out_dir / "annotations_froc_eval.csv"
    results_path = out_dir / "results_froc_eval.csv"
    rewrite_pid_column(froc_dir / "annotations.csv", annotations_path)
    rewrite_pid_column(froc_dir / "results.csv", results_path)

    print(
        f"seriesuids: {seriesuids_path} "
        f"({len(seriesuids)} cases, from {used_source})"
    )
    print(f"excluded annotations: {excluded_path}")
    print(f"normalized annotations: {annotations_path}")
    print(f"normalized results: {results_path}")


if __name__ == "__main__":
    main()
