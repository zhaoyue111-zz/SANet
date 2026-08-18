#!/usr/bin/env python3
"""Convert full-CT bbox coordinates to cropped original-resolution coordinates.

The source annotation is defined on the original full CT grid (ZYX).  The
organized NPZ stores ``image_original`` after cropping with
``mask_size_original``.  This script subtracts the crop start index from each
source bbox and writes the result to the organized annotation CSV.

NPZ layout under ``luna25_organized``:
``{npz_dir}/{patient_id}/{studyInstanceUID}/{seriesInstanceUID}/{seriesInstanceUID}.npz``

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
DEFAULT_NPZ_DIR = REPO_ROOT / "luna25_organized"
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


def resolve_npz_path(npz_dir: Path, patient_id: str, study_uid: str, series_uid: str) -> Path:
    """Resolve organized NPZ path:

    ``{npz_dir}/{patient_id}/{studyInstanceUID}/{seriesInstanceUID}/{seriesInstanceUID}.npz``
    """
    candidates = [
        npz_dir / patient_id / study_uid / series_uid / f"{series_uid}.npz",
        # Fallback: recursive search by series UID filename.
    ]
    for path in candidates:
        if path.is_file():
            return path

    matches = sorted(npz_dir.glob(f"**/patient_*/study_*/{series_uid}/{series_uid}.npz"))
    if not matches:
        matches = sorted(npz_dir.glob(f"**/{series_uid}.npz"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple NPZ files for series={series_uid}: {matches}"
        )
    raise FileNotFoundError(
        f"Missing NPZ for patient={patient_id}, study={study_uid}, series={series_uid} "
        f"under {npz_dir}"
    )


def read_csv(
    path: Path,
    limit: int | None = None,
    offset: int = 0,
) -> tuple[list[str], list[dict[str, str]]]:
    """Stream-read CSV. With ``limit``, stop after N data rows (skips full-file load)."""
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be > 0, got {limit}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fields = list(reader.fieldnames)
        rows: list[dict[str, str]] = []
        for index, row in enumerate(reader):
            if index < offset:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
        return fields, rows


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


def has_complete_bbox(row: dict[str, str]) -> bool:
    """Return False when any bbox field is missing/blank."""
    for column in BBOX_COLUMNS:
        value = row.get(column)
        if value is None:
            return False
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none", "null"}:
            return False
    return True


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
        "skipped_empty_bbox": 0,
        "full_shape_ok": 0,
        "inside_crop_ok": 0,
        "cropped_shape_ok": 0,
    }

    for source_index, source_row in enumerate(source_rows):
        if not has_complete_bbox(source_row):
            stats["skipped_empty_bbox"] += 1
            print(
                "Skip empty bbox: row=%d patient=%s series=%s lesion=%s"
                % (
                    source_index,
                    source_row.get("patient_id"),
                    source_row.get("seriesInstanceUID"),
                    source_row.get("lesion_id"),
                )
            )
            continue

        patient_id = str(source_row["patient_id"])
        nodule_id = nodule_counts.get(patient_id, 0)
        nodule_counts[patient_id] = nodule_id + 1
        series_uid = source_row["seriesInstanceUID"]
        study_uid = str(source_row["studyInstanceUID"])
        npz_path = resolve_npz_path(npz_dir, patient_id, study_uid, series_uid)
        pid = str(len(converted_rows))

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
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only read/convert the first N data rows (after --offset). "
             "Use this for quick smoke tests on large CSVs.",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Skip the first N data rows before applying --limit.",
    )
    parser.add_argument("--check-only", action="store_true",
                        help="Validate coordinates without rewriting organized CSV")
    args = parser.parse_args()

    source_fields, source_rows = read_csv(
        args.source_csv, limit=args.limit, offset=args.offset,
    )
    if not source_rows:
        raise ValueError("Source annotation has no data rows: %s" % args.source_csv)
    source_rows = project_source_rows(source_fields, source_rows)
    if args.limit is not None or args.offset:
        print(
            f"Smoke subset: offset={args.offset}, "
            f"limit={args.limit}, loaded={len(source_rows)}"
        )
    converted_rows, stats = convert_annotations(source_rows, args.npz_dir)

    print("Source annotation coordinate check:")
    print(f"  rows: {stats['source_rows']}")
    print(f"  skipped empty bbox: {stats['skipped_empty_bbox']}")
    kept = stats['source_rows'] - stats['skipped_empty_bbox']
    print(f"  in original full-CT image_shape_original: {stats['full_shape_ok']}/{kept}")
    print(f"  inside mask_size_original crop: {stats['inside_crop_ok']}/{kept}")
    print(f"  valid after crop in image_original: {stats['cropped_shape_ok']}/{kept}")
    print("  conclusion: source bbox coordinates are original full-CT ZYX coordinates")
    print("  output: converted to crop-local original-resolution image_original coordinates")

    if args.check_only:
        return

    if not converted_rows:
        raise ValueError("No rows left after skipping empty bbox annotations")

    with args.organized_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TARGET_COLUMNS)
        writer.writeheader()
        writer.writerows(converted_rows)
    print(f"Wrote converted annotation: {args.organized_csv}")


if __name__ == "__main__":
    main()
