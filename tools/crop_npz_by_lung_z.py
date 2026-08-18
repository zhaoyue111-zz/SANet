#!/usr/bin/env python3
"""Crop an NPZ volume in Z only using the full-grid lung mask.

Reads HU from a full-CT ``.nii.gz`` (not from NPZ ``image_original``, which is
already lung-bbox cropped). Z extent comes from the source NPZ ``raw_lrmask``.
Writes a **new** NPZ; the source NPZ is never modified.

Crop rule:
  - Full CT from ``--ct`` / ``--ct-dir/{seriesUID}.nii.gz``
  - Z extent from NPZ ``raw_lrmask > 0``, expanded by ``--z-pad`` (default 10)
  - Keep full Y/X (no crop on those axes)
  - Update ``image_original`` / ``mask_original`` / ``raw_lrmask`` / crop meta
  - Copy ``image`` / ``mask`` / ``mask_lr`` unchanged (not recomputed)
  - Copy ``origin`` / ``spacing`` / ``direction`` / ``seriesUID`` unchanged

Output path:
  - Parent dirs are created if missing
  - If the target file already exists, append a suffix before ``.npz``
    (default ``_zcrop``), then ``_zcrop2``, ``_zcrop3``, ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


DEFAULT_CT_DIR = Path(__file__).resolve().parents[1] / "data" / "luna25" / "luna25_images_nii"


def scalar_string(value: Any) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(value)


def lung_z_bounds(raw_lrmask: np.ndarray, z_pad: int) -> tuple[int, int]:
    """Return half-open [z0, z1) after expanding lung Z extent by ``z_pad``."""
    hits = np.where(np.asarray(raw_lrmask) > 0)
    if hits[0].size == 0:
        raise ValueError("raw_lrmask has no lung voxels (all <= 0)")
    z_min = int(hits[0].min())
    z_max = int(hits[0].max())  # inclusive
    z_size = int(raw_lrmask.shape[0])
    z0 = max(0, z_min - int(z_pad))
    z1 = min(z_size, z_max + 1 + int(z_pad))
    if z1 <= z0:
        raise ValueError("Invalid Z crop range after padding: [%d, %d)" % (z0, z1))
    return z0, z1


def unique_output_path(path: Path, suffix: str = "_zcrop") -> Path:
    """Create parent dirs; if ``path`` exists, insert suffix / suffixN before .npz."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return path

    stem = path.stem
    # Path(...).stem on ``foo.npz`` is ``foo``; on ``foo.nii.gz`` would be wrong,
    # but outputs are always ``.npz`` here.
    if path.name.endswith(".npz"):
        stem = path.name[: -len(".npz")]
    ext = ".npz"
    candidate = path.with_name("%s%s%s" % (stem, suffix, ext))
    if not candidate.exists():
        return candidate
    index = 2
    while True:
        candidate = path.with_name("%s%s%d%s" % (stem, suffix, index, ext))
        if not candidate.exists():
            return candidate
        index += 1


def resolve_ct_path(series_uid: str, ct: Path | None, ct_dir: Path | None) -> Path:
    if ct is not None:
        path = ct.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("CT NIfTI not found: %s" % path)
        return path
    if ct_dir is None:
        raise ValueError("Provide --ct or --ct-dir")
    path = ct_dir.expanduser().resolve() / ("%s.nii.gz" % series_uid)
    if not path.is_file():
        raise FileNotFoundError(
            "CT NIfTI not found for seriesUID=%s under %s" % (series_uid, ct_dir)
        )
    return path


def load_ct_zyx(ct_path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(ct_path))
    # SimpleITK GetArrayFromImage -> (Z, Y, X), matching NPZ raw_lrmask.
    return np.asarray(sitk.GetArrayFromImage(image))


def build_cropped_payload(
    d: dict,
    full_hu: np.ndarray,
    z_pad: int = 10,
) -> tuple[dict, int, int]:
    required = {
        "seriesUID", "origin", "spacing", "direction",
        "image_shape", "image_shape_original",
        "mask_size", "mask_size_original",
        "raw_lrmask",
        "image", "mask", "mask_lr",
    }
    missing = sorted(required.difference(d.keys()))
    if missing:
        raise KeyError("NPZ missing keys: %s" % missing)

    raw_lrmask = np.asarray(d["raw_lrmask"])
    full_shape = tuple(np.asarray(d["image_shape_original"][0], dtype=int).tolist())
    if tuple(raw_lrmask.shape) != full_shape:
        raise ValueError(
            "raw_lrmask shape %s != image_shape_original %s"
            % (raw_lrmask.shape, full_shape)
        )
    if tuple(full_hu.shape) != full_shape:
        raise ValueError(
            "CT NIfTI shape %s != image_shape_original / raw_lrmask %s"
            % (full_hu.shape, full_shape)
        )

    z0, z1 = lung_z_bounds(raw_lrmask, z_pad=z_pad)
    y_size = full_shape[1]
    x_size = full_shape[2]

    full_lung = (raw_lrmask > 0).astype(np.uint8)

    image_original = np.ascontiguousarray(full_hu[z0:z1, :, :])
    mask_original = np.ascontiguousarray(full_lung[z0:z1, :, :])
    raw_lrmask_crop = np.ascontiguousarray(raw_lrmask[z0:z1, :, :])

    # Store as int16 HU when possible (matches typical CT NIfTI / old NPZ).
    if np.issubdtype(image_original.dtype, np.floating):
        image_original = np.rint(image_original).astype(np.int16)
    elif image_original.dtype != np.int16:
        image_original = image_original.astype(np.int16, copy=False)

    mask_size_original = np.asarray(
        [[z0, z1], [0, y_size], [0, x_size]],
        dtype=np.int64,
    )
    image_shape = np.asarray(d["image_shape"], dtype=np.int64).copy()
    image_shape[0] = np.asarray(image_original.shape, dtype=np.int64)

    out = {
        "seriesUID": np.asarray(scalar_string(d["seriesUID"])),
        "origin": np.asarray(d["origin"]),
        "spacing": np.asarray(d["spacing"]),
        "direction": np.asarray(d["direction"]),
        "image_shape_original": np.asarray(d["image_shape_original"]),
        "image_shape": image_shape,
        "mask_size_original": mask_size_original,
        "mask_size": np.asarray(d["mask_size"]),
        "image_original": image_original,
        "mask_original": mask_original,
        "raw_lrmask": raw_lrmask_crop,
        # 1mm working volumes: copy unchanged (not recomputed).
        "image": np.asarray(d["image"]),
        "mask": np.asarray(d["mask"]),
        "mask_lr": np.asarray(d["mask_lr"]),
    }
    return out, z0, z1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True, help="Source NPZ path")
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Destination NPZ path (dirs created; existing names get a suffix)",
    )
    parser.add_argument(
        "--ct", type=Path, default=None,
        help="Full-CT NIfTI (.nii.gz). If omitted, use --ct-dir/{seriesUID}.nii.gz",
    )
    parser.add_argument(
        "--ct-dir", type=Path, default=DEFAULT_CT_DIR,
        help="Directory of {seriesUID}.nii.gz (default: %s)" % DEFAULT_CT_DIR,
    )
    parser.add_argument("--z-pad", type=int, default=10, help="Extra voxels on each Z side")
    parser.add_argument(
        "--suffix", default="_zcrop",
        help="Suffix inserted before .npz when output already exists",
    )
    args = parser.parse_args()

    if args.z_pad < 0:
        raise ValueError("--z-pad must be >= 0")
    if not args.input.is_file():
        raise FileNotFoundError("Input NPZ not found: %s" % args.input)

    source = dict(np.load(args.input, allow_pickle=True))
    series_uid = scalar_string(source["seriesUID"])
    ct_path = resolve_ct_path(series_uid, args.ct, None if args.ct is not None else args.ct_dir)
    full_hu = load_ct_zyx(ct_path)

    payload, z0, z1 = build_cropped_payload(source, full_hu=full_hu, z_pad=args.z_pad)

    out_path = unique_output_path(args.output, suffix=args.suffix)
    np.savez_compressed(out_path, **payload)

    print("Wrote: %s" % out_path)
    print("CT source: %s" % ct_path)
    print(
        "Z crop (half-open): [%d, %d)  pad=%d  image_original.shape=%s"
        % (z0, z1, args.z_pad, payload["image_original"].shape)
    )
    print("mask_size_original=\n%s" % np.asarray(payload["mask_size_original"]))
    print("Copied unchanged: image / mask / mask_lr / origin / spacing / direction / seriesUID")
    print("Source NPZ left untouched: %s" % args.input.resolve())


if __name__ == "__main__":
    main()
