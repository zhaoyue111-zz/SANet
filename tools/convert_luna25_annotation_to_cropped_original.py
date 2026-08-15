#!/usr/bin/env python3
"""Convert full-CT bbox coordinates to cropped original-resolution coordinates.

The source annotation is defined on the original full CT grid (ZYX).  The
organized NPZ stores ``image_original`` after cropping with
``mask_size_original``.  This script subtracts the crop start index from each
source bbox and writes the result to the organized annotation CSV.

The bbox corners are inclusive, while ``mask_size_original`` uses the
half-open convention ``[start, end)`` described in ``npz_buffer.md``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CSV = REPO_ROOT / "luna25_npz" / "annotation.csv"
DEFAULT_ORGANIZED_CSV = REPO_ROOT / "luna25_organized" / "annotation.csv"
DEFAULT_NPZ_DIR = REPO_ROOT / "luna25_npz"
BBOX_COLUMNS = (
    "bbox_min_z", "bbox_min_y", "bbox_min_x",
    "bbox_max_z", "bbox_max_y", "bbox_max_x",
)
SOURCE_REQUIRED_COLUMNS = (
    "patient_id", "studyInstanceUID", "seriesInstanceUID", "lesion_id",
) + BBOX_COLUMNS
TARGET_COLUMNS = [
    "pid", "patient_id", "studyInstanceUID", "seriesInstanceUID",
    "lesion_id", "nodule_id", *BBOX_COLUMNS,
]


def scalar_string(value) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(value)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def project_source_rows(fields: list[str], rows: list[dict[str, str]]):
    """Extract only source identity, lesion, and bbox fields."""
    missing = [column for column in SOURCE_REQUIRED_COLUMNS if column not in fields]
    if missing:
        raise ValueError("Source annotation missing required columns: %s" % missing)

    projected = []
    for row in rows:
        projected_row = {}
        for column in SOURCE_REQUIRED_COLUMNS:
            projected_row[column] = row[column]
        projected.append(projected_row)
    return projected


def bbox_from_row(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.array([int(row[f"bbox_min_{axis}"]) for axis in "zyx"], dtype=int)
    maximum = np.array([int(row[f"bbox_max_{axis}"]) for axis in "zyx"], dtype=int)
    if np.any(minimum > maximum):
        raise ValueError(f"Invalid bbox corners for series={row.get('seriesInstanceUID')}: {row}")
    return minimum, maximum


def assert_bbox_in_shape(minimum: np.ndarray, maximum: np.ndarray,
                         shape: tuple[int, int, int], label: str) -> None:
    shape_array = np.asarray(shape, dtype=int)
    if np.any(minimum < 0) or np.any(maximum >= shape_array):
        raise ValueError(
            f"{label} bbox {minimum.tolist()}..{maximum.tolist()} is outside "
            f"shape {tuple(shape)}"
        )


def convert_annotations(source_rows: list[dict[str, str]], npz_dir: Path):
    converted_rows = []
    nodule_counts = {}
    stats = {
        "source_rows": len(source_rows),
        "full_shape_ok": 0,
        "inside_crop_ok": 0,
        "cropped_shape_ok": 0,
    }

    for pid, source_row in enumerate(source_rows):
        pid = str(pid)
        patient_id = str(source_row["patient_id"])
        nodule_id = nodule_counts.get(patient_id, 0)
        nodule_counts[patient_id] = nodule_id + 1
        series_uid = source_row["seriesInstanceUID"]
        npz_path = npz_dir / f"{series_uid}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing NPZ for pid={pid}: {npz_path}")

        with np.load(npz_path, allow_pickle=True) as data:
            stored_uid = scalar_string(data["seriesUID"])
            full_shape = tuple(np.asarray(data["image_shape_original"][0], dtype=int).tolist())
            crop_box = np.asarray(data["mask_size_original"], dtype=int)
            cropped_shape = tuple(np.asarray(data["image_original"]).shape)

        if stored_uid != series_uid:
            raise ValueError(f"NPZ seriesUID mismatch for pid={pid}: {stored_uid} != {series_uid}")
        if crop_box.shape != (3, 2):
            raise ValueError(f"Invalid mask_size_original shape for pid={pid}: {crop_box.shape}")
        if tuple((crop_box[:, 1] - crop_box[:, 0]).tolist()) != cropped_shape:
            raise ValueError(
                f"image_original shape {cropped_shape} does not match "
                f"mask_size_original for pid={pid}"
            )

        minimum, maximum = bbox_from_row(source_row)
        assert_bbox_in_shape(minimum, maximum, full_shape, f"Source pid={pid}")
        stats["full_shape_ok"] += 1

        crop_start = crop_box[:, 0]
        crop_end = crop_box[:, 1]
        if np.any(minimum < crop_start) or np.any(maximum >= crop_end):
            raise ValueError(
                f"Source pid={pid} bbox {minimum.tolist()}..{maximum.tolist()} "
                f"is outside mask_size_original {crop_box.tolist()}"
            )
        stats["inside_crop_ok"] += 1

        cropped_minimum = minimum - crop_start
        cropped_maximum = maximum - crop_start
        assert_bbox_in_shape(
            cropped_minimum, cropped_maximum, cropped_shape,
            f"Cropped-original pid={pid}",
        )
        stats["cropped_shape_ok"] += 1

        output_row = {
            "pid": pid,
            "patient_id": patient_id,
            "studyInstanceUID": str(source_row["studyInstanceUID"]),
            "seriesInstanceUID": series_uid,
            "lesion_id": str(source_row["lesion_id"]),
            "nodule_id": str(nodule_id),
        }
        for axis_index, axis in enumerate("zyx"):
            output_row[f"bbox_min_{axis}"] = str(int(cropped_minimum[axis_index]))
            output_row[f"bbox_max_{axis}"] = str(int(cropped_maximum[axis_index]))
        converted_rows.append(output_row)

    return converted_rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--organized-csv", type=Path, default=DEFAULT_ORGANIZED_CSV)
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate coordinates without rewriting organized CSV")
    args = parser.parse_args()

    source_fields, source_rows = read_csv(args.source_csv)
    if not source_rows:
        raise ValueError("Source annotation has no data rows: %s" % args.source_csv)
    source_rows = project_source_rows(source_fields, source_rows)
    converted_rows, stats = convert_annotations(source_rows, args.npz_dir)

    print("Source annotation coordinate check:")
    print(f"  rows: {stats['source_rows']}")
    print(f"  in original full-CT image_shape_original: {stats['full_shape_ok']}/{stats['source_rows']}")
    print(f"  inside mask_size_original crop: {stats['inside_crop_ok']}/{stats['source_rows']}")
    print(f"  valid after crop in image_original: {stats['cropped_shape_ok']}/{stats['source_rows']}")
    print("  conclusion: source bbox coordinates are original full-CT ZYX coordinates")
    print("  output: converted to crop-local original-resolution image_original coordinates")

    if args.check_only:
        return

    with args.organized_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TARGET_COLUMNS)
        writer.writeheader()
        writer.writerows(converted_rows)
    print(f"Wrote converted annotation: {args.organized_csv}")


if __name__ == "__main__":
    main()
