#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计 NLSTseg CT 的层厚信息。

默认扫描：
  /media/SENSETIME\\yangtingting/T7/医保大赛数据/NLSTseg/**/*_CT.nii.gz

输出：
  1. nlstseg_spacing_summary.csv：每个病例的 spacing 和图像尺寸。
  2. nlstseg_thickness_distribution.csv：层厚四舍五入后的计数与占比。

仅读取 NIfTI 头信息，不加载完整 CT 体数据。
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import SimpleITK as sitk


DEFAULT_ROOT = Path(
    r"/media/SENSETIME\yangtingting/T7/医保大赛数据/NLSTseg"
)


def find_ct_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*_CT.nii.gz") if path.is_file())


def case_id_from_path(path: Path) -> str:
    suffix = "_CT.nii.gz"
    return path.name[: -len(suffix)]


def read_image_information(path: Path) -> dict[str, object]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()

    spacing_x, spacing_y, spacing_z = map(float, reader.GetSpacing())
    size_x, size_y, size_z = map(int, reader.GetSize())
    return {
        "file": str(path),
        "case_id": case_id_from_path(path),
        "spacing_x": spacing_x,
        "spacing_y": spacing_y,
        "spacing_z_slice_thickness": spacing_z,
        "size_x": size_x,
        "size_y": size_y,
        "size_z_slices": size_z,
        "error": "",
    }


def round_half_up(value: float, digits: int) -> Decimal:
    quantum = Decimal(1).scaleb(-digits)
    return Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="NLSTseg 数据集根目录。",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("nlstseg_spacing_summary.csv"),
        help="每例 spacing 明细 CSV。",
    )
    parser.add_argument(
        "--distribution-csv",
        type=Path,
        default=Path("nlstseg_thickness_distribution.csv"),
        help="四舍五入后的层厚分布 CSV。",
    )
    parser.add_argument(
        "--round-digits",
        type=int,
        default=1,
        help="层厚四舍五入保留位数，默认 1，即精确到 0.1 mm。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.round_digits < 0:
        raise ValueError("--round-digits 必须大于或等于 0")

    root = args.root.resolve()
    files = find_ct_files(root)
    if not files:
        raise RuntimeError(f"没有在 {root} 下找到 *_CT.nii.gz 文件")

    rows: list[dict[str, object]] = []
    for index, path in enumerate(files, start=1):
        try:
            rows.append(read_image_information(path))
        except Exception as error:
            rows.append(
                {
                    "file": str(path),
                    "case_id": case_id_from_path(path),
                    "spacing_x": "",
                    "spacing_y": "",
                    "spacing_z_slice_thickness": "",
                    "size_x": "",
                    "size_y": "",
                    "size_z_slices": "",
                    "error": str(error),
                }
            )
        if index % 100 == 0 or index == len(files):
            print(f"已读取 {index}/{len(files)} 个 CT 头信息")

    detail_fields = (
        "file",
        "case_id",
        "spacing_x",
        "spacing_y",
        "spacing_z_slice_thickness",
        "size_x",
        "size_y",
        "size_z_slices",
        "error",
    )
    write_csv(args.out_csv, detail_fields, rows)

    valid_rows = [
        row for row in rows if row["spacing_z_slice_thickness"] != ""
    ]
    thicknesses = [
        float(row["spacing_z_slice_thickness"]) for row in valid_rows
    ]
    rounded_counts = Counter(
        round_half_up(value, args.round_digits) for value in thicknesses
    )
    distribution_rows = [
        {
            "rounded_thickness_mm": f"{thickness:.{args.round_digits}f}",
            "count": count,
            "percentage": f"{count / len(thicknesses) * 100:.4f}",
        }
        for thickness, count in sorted(rounded_counts.items())
    ]
    write_csv(
        args.distribution_csv,
        ("rounded_thickness_mm", "count", "percentage"),
        distribution_rows,
    )

    print("\n========== NLSTseg 层厚统计 ==========")
    print(f"CT 文件总数: {len(rows)}")
    print(f"成功读取: {len(valid_rows)}")
    print(f"读取失败: {len(rows) - len(valid_rows)}")
    print(f"层厚范围: {min(thicknesses):.6f} - {max(thicknesses):.6f} mm")
    print(f"平均层厚: {statistics.fmean(thicknesses):.6f} mm")
    print(f"中位层厚: {statistics.median(thicknesses):.6f} mm")
    print(
        f"\n层厚分布（ROUND_HALF_UP，保留 {args.round_digits} 位小数）:"
    )
    for row in distribution_rows:
        print(
            f"  {row['rounded_thickness_mm']} mm: "
            f"{row['count']}/{len(thicknesses)} ({row['percentage']}%)"
        )
    print(f"\n每例明细: {args.out_csv.resolve()}")
    print(f"层厚分布: {args.distribution_csv.resolve()}")


if __name__ == "__main__":
    main()
