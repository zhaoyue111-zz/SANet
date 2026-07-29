from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

'''
统计同目录下lndb
  histopathology NLSTseg PN9 /media/SENSETIME\yangtingting/T7/医保大赛数据/
  SANet_data/LUNA16 这些数据集的训练 验证 测试集数量
'''

DEFAULT_DATASETS: List[Tuple[str, Path]] = [
    ("lndb", Path("/data/医保大赛/code/SANet/data/LNDB")),
    ("PN9",Path("/data/医保大赛/code/SANet/data/PN9")),
    ("histopathology", Path("/data/医保大赛/code/SANet/data/histopathology")),
    ("NLSTseg", Path("/data/医保大赛/code/SANet/data/NLSTSeg")),
    ("LUNA16", Path(r"/media/SENSETIME\yangtingting/T7/医保大赛数据/SANet_data/LUNA16")),
]


def count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                count += 1
    return count


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def dataset_counts(dataset_root: Path) -> Dict[str, int]:
    split_dir = dataset_root / "split"
    train = count_nonempty_lines(split_dir / "train.txt")
    val = count_nonempty_lines(split_dir / "val.txt")
    test = count_nonempty_lines(split_dir / "test.txt")
    return {
        "train": train,
        "val": val,
        "test": test,
        "total": train + val + test,
        "train_boxes": count_csv_rows(split_dir / "train_anno.csv"),
        "val_boxes": count_csv_rows(split_dir / "val_anno.csv"),
        "test_boxes": count_csv_rows(split_dir / "test_anno.csv"),
        "all_boxes": count_csv_rows(split_dir / "all_anno.csv"),
    }


def print_table(rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    headers = ["dataset", "train", "val", "test", "total", "train_boxes", "val_boxes", "test_boxes", "all_boxes", "path"]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row[h])))

    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(str(row[h]).ljust(widths[h]) for h in headers))


def parse_dataset_arg(value: str) -> Tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), Path(path.strip())
    p = Path(value)
    return p.name, p


def run(args: argparse.Namespace) -> None:
    datasets = list(DEFAULT_DATASETS)
    datasets.extend(parse_dataset_arg(v) for v in args.dataset)

    rows: List[Dict[str, object]] = []
    for name, root in datasets:
        counts = dataset_counts(root)
        rows.append({
            "dataset": name,
            **counts,
            "path": str(root),
        })
    print_table(rows)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Count train/val/test case numbers for SANet prepared datasets.")
    p.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional dataset root to count. Can be passed multiple times; NAME= is optional.",
    )
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())
