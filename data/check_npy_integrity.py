import argparse
import os
from pathlib import Path

import numpy as np


def iter_npy_files(data_root, dataset):
    root = Path(data_root)
    datasets = [root / dataset] if dataset else sorted(p for p in root.iterdir() if p.is_dir())
    for dataset_dir in datasets:
        full_dir = dataset_dir / "full"
        if not full_dir.is_dir():
            continue
        for path in sorted(full_dir.glob("*_zoom.npy")):
            yield dataset_dir.name, path


def main():
    parser = argparse.ArgumentParser(description="Check SANet *_zoom.npy files can be loaded.")
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--dataset", default=None, help="Dataset name under data root. Default: all.")
    parser.add_argument("--header-only", action="store_true", help="Only check npy header via mmap.")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    total = 0
    bad = []
    for dataset, path in iter_npy_files(args.data_root, args.dataset):
        total += 1
        try:
            arr = np.load(path, mmap_mode="r" if args.header_only else None)
            _ = arr.shape
            _ = arr.dtype
            if not args.header_only:
                _ = arr.sum(dtype=np.float64)
        except Exception as exc:
            size = path.stat().st_size if path.exists() else -1
            msg = "%s\t%s\t%d bytes\t%s" % (dataset, path, size, repr(exc))
            print("BAD\t" + msg)
            bad.append(msg)
            if args.stop_on_error:
                break

    print("checked=%d bad=%d" % (total, len(bad)))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
