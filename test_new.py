#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDP inference for organized SANet NPZ data.

Expected test layout:
    data_root/{patient_id}/{studyInstanceUID}/{seriesInstanceUID}/
        {seriesInstanceUID}.npz
        (fallback: {seriesInstanceUID}_buffer.npz)

Expected test.txt:
    one entry per line:
        patient_id/studyInstanceUID/seriesInstanceUID

Expected annotation.csv columns:
    patient_id, studyInstanceUID, seriesInstanceUID,
    bbox_min_z, bbox_min_y, bbox_min_x,
    bbox_max_z, bbox_max_y, bbox_max_x

Important:
- --batch-size is a per-GPU *loading/scheduling* batch. CTs in one loader
  batch are still forwarded one-by-one, because full CT volumes have different
  spatial shapes.
- Inference uses RPN proposal boxes. Only the proposal score is ensembled:
      score = (rpn_weight * rpn_score + rcnn_weight * rcnn_prob)
              / (rpn_weight + rcnn_weight)
- No per-case *_detections.npy files are written.
- results.csv additionally stores bbox_min_z/bbox_max_z/bbox_min_y/bbox_max_y/bbox_min_x/bbox_max_x in image_original crop-local voxel coordinates, plus diameter_mm (longest side in mm using NPZ spacing_zyx).
- Before results.csv is written, predictions are mapped from the online
  resampled image back to image_original crop-local coordinates so predictions
  and annotation.csv are evaluated in the same coordinate system.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import random
import sys
from pathlib import Path

# SANet has its own legacy data_parallel calls. They must be disabled when
# one process owns one GPU under DDP.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ["SANET_DISABLE_INTERNAL_DATA_PARALLEL"] = "1"

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "build" / "box"))

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.ndimage import zoom
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler

from config import config
from net.sanet import SANet


TRIPLE_COLUMNS = ("patient_id", "studyInstanceUID", "seriesInstanceUID")
BBOX_COLUMNS = (
    "bbox_min_z", "bbox_min_y", "bbox_min_x",
    "bbox_max_z", "bbox_max_y", "bbox_max_x",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DDP SANet test for organized {series}.npz data."
    )
    parser.add_argument("--net", default="SANet", type=str)
    parser.add_argument("--weight", required=True, type=str)
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--test-list", "--test-txt", dest="test_list",
                        required=True, type=str)
    parser.add_argument("--annotation", "--annotation-csv", dest="annotation",
                        required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)

    # Per-GPU loading/scheduling batch. Forward remains one CT at a time.
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=4, type=int)

    # Used to validate the torchrun launch requested by eval.sh.
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--local-rank", "--local_rank", default=-1, type=int)
    parser.add_argument("--dist-backend", default="nccl", type=str)

    parser.add_argument("--rpn-weight", default=8.0, type=float)
    parser.add_argument("--rcnn-weight", default=2.0, type=float)

    parser.add_argument("--target-spacing-zyx", default="1.0,0.6,0.6", type=str)
    parser.add_argument("--window-min", default=-1200.0, type=float)
    parser.add_argument("--window-max", default=600.0, type=float)
    parser.add_argument("--pad-factor", default=32, type=int)

    parser.add_argument("--use-aspp", action="store_true")
    parser.add_argument("--seed", default=35202, type=int)
    parser.add_argument("--limit-test-samples", default=0, type=int,
                        help="Debug only. 0 means all test series.")
    return parser.parse_args()


def parse_spacing(text):
    try:
        values = tuple(float(v.strip()) for v in text.split(","))
    except ValueError as exc:
        raise ValueError(
            "--target-spacing-zyx must be comma-separated floats"
        ) from exc
    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError(
            "--target-spacing-zyx must contain exactly three positive values"
        )
    return np.asarray(values, dtype=np.float32)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed(args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This test.py is configured for CUDA DDP. "
            "Use torchrun with one process per GPU."
        )

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    else:
        rank = 0
        world_size = 1
        local_rank = 0 if args.local_rank < 0 else args.local_rank

    if args.num_gpus <= 0:
        raise ValueError("--num-gpus must be > 0")
    if world_size != args.num_gpus:
        raise RuntimeError(
            "torchrun world size (%d) != --num-gpus (%d). "
            "Launch with: torchrun --nproc_per_node=%d ..."
            % (world_size, args.num_gpus, args.num_gpus)
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend=args.dist_backend, init_method="env://")
        dist.barrier()

    return rank, world_size, local_rank, device, distributed


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {
        k[7:] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def read_split_entries(path):
    entries = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            parts = text.split("/")
            if len(parts) != 3 or any(
                not p or p in (".", "..") for p in parts
            ):
                raise ValueError(
                    "%s:%d must be exactly "
                    "patient_id/studyInstanceUID/seriesInstanceUID, got %r"
                    % (path, line_no, text)
                )
            entries.append("/".join(parts))

    if not entries:
        raise ValueError("Test list is empty: %s" % path)

    if len(entries) != len(set(entries)):
        raise ValueError("Duplicate series path(s) found in test list: %s" % path)

    return entries


def entry_key(entry):
    parts = entry.split("/")
    if len(parts) != 3:
        raise ValueError("Invalid organized test entry: %s" % entry)
    return tuple(parts)


def scalar_string(value):
    arr = np.asarray(value)
    if arr.size == 1:
        item = arr.reshape(-1)[0]
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    return str(value)


def pad2factor_3d(image, factor, pad_value):
    if factor <= 0:
        raise ValueError("--pad-factor must be > 0")
    shape = np.asarray(image.shape, dtype=np.int64)
    padded = np.ceil(shape / float(factor)).astype(np.int64) * factor
    pad = [(0, int(padded[i] - shape[i])) for i in range(3)]
    return np.pad(image, pad, mode="constant", constant_values=pad_value)


def resample_and_window(
    image,
    output_shape,
    window_min,
    window_max,
):
    image = np.asarray(image, dtype=np.float32)
    output_shape = np.asarray(output_shape, dtype=np.int32)
    zoom_factors = output_shape.astype(np.float32) / np.asarray(
        image.shape, dtype=np.float32
    )

    image = zoom(
        image,
        zoom_factors.tolist(),
        order=1,
        mode="nearest",
        prefilter=False,
    )

    # scipy.ndimage.zoom can differ by one voxel due to rounding. Keep the
    # exact shape used when annotation boxes were scaled.
    if tuple(image.shape) != tuple(output_shape.tolist()):
        fixed = np.full(
            tuple(output_shape.tolist()),
            window_min,
            dtype=np.float32,
        )
        slices = tuple(
            slice(0, min(image.shape[i], int(output_shape[i])))
            for i in range(3)
        )
        fixed[slices] = image[slices]
        image = fixed

    image = np.clip(image, window_min, window_max)
    image = (image - window_min) * (255.0 / (window_max - window_min))
    return image.astype(np.float32, copy=False)


class OrganizedNPZTestDataset(Dataset):
    def __init__(
        self,
        data_root,
        test_list,
        target_spacing_zyx,
        window_min,
        window_max,
        pad_factor=32,
        limit=0,
    ):
        self.data_root = Path(data_root)
        self.test_list = str(test_list)
        self.target_spacing_zyx = np.asarray(
            target_spacing_zyx, dtype=np.float32
        )
        self.window_min = float(window_min)
        self.window_max = float(window_max)
        self.pad_factor = int(pad_factor)

        if not self.window_min < self.window_max:
            raise ValueError("--window-min must be smaller than --window-max")

        self.entries = read_split_entries(self.test_list)
        if limit and limit > 0:
            self.entries = self.entries[: int(limit)]

        self.case_metadata = []
        seen_series_uids = {}

        for entry in self.entries:
            patient_id, study_uid, series_uid = entry_key(entry)
            case_dir = self.data_root / patient_id / study_uid / series_uid
            # Prefer training-style {series}.npz; keep _buffer as legacy fallback.
            candidates = (
                case_dir / ("%s.npz" % series_uid),
                case_dir / ("%s_buffer.npz" % series_uid),
            )
            npz_path = next((path for path in candidates if path.is_file()), None)
            if npz_path is None:
                raise FileNotFoundError(
                    "Missing test NPZ (tried %s)"
                    % ", ".join(str(path) for path in candidates)
                )

            with np.load(npz_path, allow_pickle=True) as data:
                if "image_shape" in data.files:
                    shape_arr = np.asarray(data["image_shape"])
                    if shape_arr.ndim >= 2:
                        image_shape = tuple(
                            int(v) for v in shape_arr.reshape(-1, 3)[0]
                        )
                    else:
                        image_shape = tuple(int(v) for v in shape_arr.reshape(3))
                else:
                    image_shape = tuple(
                        int(v) for v in np.asarray(data["image_original"]).shape
                    )

                spacing_zyx = np.asarray(
                    data["spacing"], dtype=np.float32
                ).reshape(3)

                if "seriesUID" in data.files:
                    stored_uid = scalar_string(data["seriesUID"])
                    if stored_uid != series_uid:
                        raise ValueError(
                            "NPZ seriesUID mismatch: path=%s stored=%s path_series=%s"
                            % (npz_path, stored_uid, series_uid)
                        )

            if len(image_shape) != 3 or any(v <= 0 for v in image_shape):
                raise ValueError(
                    "Invalid image_original shape %s in %s"
                    % (image_shape, npz_path)
                )
            if np.any(spacing_zyx <= 0):
                raise ValueError("Invalid spacing %s in %s" % (
                    spacing_zyx.tolist(), npz_path
                ))

            output_shape = np.maximum(
                1,
                np.rint(
                    np.asarray(image_shape, dtype=np.float32)
                    * spacing_zyx
                    / self.target_spacing_zyx
                ).astype(np.int32),
            )
            actual_scale = (
                output_shape.astype(np.float32)
                / np.asarray(image_shape, dtype=np.float32)
            )

            if series_uid in seen_series_uids:
                raise ValueError(
                    "seriesInstanceUID must be unique for FROC. Duplicate %s: "
                    "%s and %s"
                    % (series_uid, seen_series_uids[series_uid], entry)
                )
            seen_series_uids[series_uid] = entry

            self.case_metadata.append({
                "entry": entry,
                "patient_id": patient_id,
                "studyInstanceUID": study_uid,
                "seriesInstanceUID": series_uid,
                "npz_path": str(npz_path),
                "image_shape": tuple(image_shape),
                "spacing_zyx": spacing_zyx.astype(np.float32),
                "output_shape": output_shape.astype(np.int32),
                "actual_scale": actual_scale.astype(np.float32),
            })

    def __len__(self):
        return len(self.case_metadata)

    def __getitem__(self, index):
        meta = self.case_metadata[int(index)]
        with np.load(meta["npz_path"], allow_pickle=True) as data:
            image = np.asarray(data["image_original"])

        image = resample_and_window(
            image,
            meta["output_shape"],
            self.window_min,
            self.window_max,
        )
        valid_resampled_shape = np.asarray(image.shape, dtype=np.int32)

        # Match the legacy SANet whole-CT inference convention: pad only at the
        # positive end so the network sees dimensions divisible by 32.
        image = pad2factor_3d(
            image,
            factor=self.pad_factor,
            pad_value=0.0,
        )
        image = image[np.newaxis, ...]  # [C=1, Z, Y, X]
        image = (image.astype(np.float32) - 128.0) / 128.0

        return {
            "input": torch.from_numpy(image).float(),
            "entry": meta["entry"],
            "patient_id": meta["patient_id"],
            "studyInstanceUID": meta["studyInstanceUID"],
            "seriesInstanceUID": meta["seriesInstanceUID"],
            "image_shape": np.asarray(meta["image_shape"], dtype=np.int32),
            "spacing_zyx": meta["spacing_zyx"].copy(),
            "valid_resampled_shape": valid_resampled_shape,
            "actual_scale": meta["actual_scale"].copy(),
        }


class RankStridedSampler(Sampler):
    """
    DDP evaluation sampler without padding/duplication.

    torch.utils.data.DistributedSampler pads when len(dataset) is not divisible
    by world_size, which would make some CTs appear twice in results.csv.
    """

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        n = len(self.dataset)
        if self.rank >= n:
            return 0
        return (n - 1 - self.rank) // self.world_size + 1


def list_collate(batch):
    # Different CTs have different full-volume shapes; do not stack them.
    return batch


def clear_inference_cache(net):
    names = (
        "rpn_logits_flat",
        "rpn_deltas_flat",
        "rpn_window",
        "rpn_proposals",
        "raw_rpn_proposals",
        "detections",
        "ensemble_proposals",
        "rcnn_logits",
        "rcnn_deltas",
        "keeps",
        "mask_probs",
    )
    for name in names:
        if hasattr(net, name):
            setattr(net, name, None)
    gc.collect()


def ensemble_from_model(net, rpn_weight, rcnn_weight):
    """
    Return:
        boxes: [N, 6] z,y,x,d,h,w in online-resampled coordinates
        scores: [N]
    """
    proposals = getattr(net, "rpn_proposals", None)
    if proposals is None or len(proposals) == 0:
        return (
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    proposals_np = proposals.detach().cpu().numpy()
    if proposals_np.ndim != 2 or proposals_np.shape[1] < 8:
        raise RuntimeError(
            "Unexpected rpn_proposals shape: %s" % (proposals_np.shape,)
        )

    boxes = proposals_np[:, 2:8].astype(np.float32, copy=False)
    rpn_scores = proposals_np[:, 1].astype(np.float32, copy=False)

    logits = getattr(net, "rcnn_logits", None)
    if logits is None:
        raise RuntimeError(
            "RCNN logits are missing while use_rcnn=True. "
            "Check the checkpoint/model configuration."
        )
    if len(logits) != len(rpn_scores):
        raise RuntimeError(
            "RCNN logits rows (%d) != RPN proposal rows (%d)"
            % (len(logits), len(rpn_scores))
        )

    if logits.shape[1] < 2:
        raise RuntimeError(
            "Expected background + foreground RCNN classes, got %s"
            % (tuple(logits.shape),)
        )

    # This is the same foreground probability used by SANet's legacy
    # get_probability() ensemble path, but without re-decoding boxes.
    rcnn_probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    rcnn_probs = rcnn_probs.astype(np.float32, copy=False)

    denom = float(rpn_weight) + float(rcnn_weight)
    scores = (
        float(rpn_weight) * rpn_scores
        + float(rcnn_weight) * rcnn_probs
    ) / denom

    return boxes, scores.astype(np.float32, copy=False)


def inverse_resampled_box_to_original(
    box_zyxdhw,
    actual_scale,
    valid_resampled_shape,
    original_shape,
):
    """
    Invert the inclusive-coordinate transform used by organized_npz_reader.py.

    Forward GT transform:
        resampled_min = original_min * scale
        resampled_max = (original_max + 1) * scale - 1

    Therefore inversion is:
        original_min = resampled_min / scale
        original_max = (resampled_max + 1) / scale - 1
    """
    box = np.asarray(box_zyxdhw, dtype=np.float32)
    center = box[:3]
    size = box[3:6]

    if (
        not np.isfinite(box).all()
        or np.any(size <= 0)
        or np.any(actual_scale <= 0)
    ):
        return None

    # RPN/RCNN box size follows inclusive voxel-size semantics.
    resampled_min = center - (size - 1.0) / 2.0
    resampled_max = center + (size - 1.0) / 2.0

    valid_max = np.asarray(valid_resampled_shape, dtype=np.float32) - 1.0

    # Remove boxes whose center is entirely in the pad-only region.
    if np.any(center < 0) or np.any(center > valid_max):
        return None

    # Clip box extent to the real resampled CT, excluding pad2factor voxels.
    resampled_min = np.maximum(resampled_min, 0.0)
    resampled_max = np.minimum(resampled_max, valid_max)
    if np.any(resampled_max < resampled_min):
        return None

    scale = np.asarray(actual_scale, dtype=np.float32)
    original_min = resampled_min / scale
    original_max = (resampled_max + 1.0) / scale - 1.0

    original_max_valid = np.asarray(original_shape, dtype=np.float32) - 1.0
    original_min = np.maximum(original_min, 0.0)
    original_max = np.minimum(original_max, original_max_valid)
    if np.any(original_max < original_min):
        return None

    original_center = (original_min + original_max) / 2.0
    original_size = original_max - original_min + 1.0

    # Return both center/size and the exact restored corner coordinates.
    # Order:
    #   center_z, center_y, center_x, depth, height, width,
    #   bbox_min_z, bbox_min_y, bbox_min_x, bbox_max_z, bbox_max_y, bbox_max_x
    return np.concatenate(
        [original_center, original_size, original_min, original_max]
    ).astype(np.float32)


def boxes_to_result_rows(
    series_uid,
    boxes,
    scores,
    actual_scale,
    valid_resampled_shape,
    original_shape,
    spacing_zyx,
):
    rows = []
    spacing = np.asarray(spacing_zyx, dtype=np.float32).reshape(3)
    if np.any(spacing <= 0) or not np.isfinite(spacing).all():
        raise ValueError("Invalid spacing_zyx: %s" % (spacing.tolist(),))

    for box, score in zip(boxes, scores):
        restored = inverse_resampled_box_to_original(
            box,
            actual_scale=actual_scale,
            valid_resampled_shape=valid_resampled_shape,
            original_shape=original_shape,
        )
        if restored is None:
            continue

        (
            z, y, x, d, h, w,
            bbox_min_z, bbox_min_y, bbox_min_x, bbox_max_z, bbox_max_y, bbox_max_x,
        ) = [float(v) for v in restored]

        # Voxel diameter: longest axis in crop-local voxels.
        # Physical diameter: longest axis after multiplying by spacing_zyx (mm).
        size_vox = np.asarray([d, h, w], dtype=np.float32)
        size_mm = size_vox * spacing

        rows.append({
            "pid": str(series_uid),
            "center_x": x,
            "center_y": y,
            "center_z": z,
            # FROC expects one diameter. Use the maximum restored side, matching
            # the annotation conversion used for cuboid GT boxes.
            "diameter": float(np.max(size_vox)),
            "diameter_mm": float(np.max(size_mm)),

            # Exact voxel/pixel-space corners in image_original crop-local
            # coordinates, i.e. the SAME coordinate level as annotation.csv.
            "bbox_min_z": bbox_min_z,
            "bbox_max_z": bbox_max_z,
            "bbox_min_y": bbox_min_y,
            "bbox_max_y": bbox_max_y,
            "bbox_min_x": bbox_min_x,
            "bbox_max_x": bbox_max_x,

            "probability": float(score),
        })
    return rows


def build_froc_annotations(annotation_csv, dataset):
    dtype = {column: str for column in TRIPLE_COLUMNS}
    annotations = pd.read_csv(
        annotation_csv,
        dtype=dtype,
        keep_default_na=False,
    )
    required = set(TRIPLE_COLUMNS).union(BBOX_COLUMNS)
    missing = sorted(required.difference(annotations.columns))
    if missing:
        raise ValueError(
            "Annotation CSV missing columns: %s" % ", ".join(missing)
        )

    test_keys = {
        entry_key(meta["entry"])
        for meta in dataset.case_metadata
    }
    shape_by_key = {
        entry_key(meta["entry"]): np.asarray(meta["image_shape"], dtype=np.float32)
        for meta in dataset.case_metadata
    }

    rows = []
    for row in annotations.to_dict("records"):
        key = tuple(str(row[column]) for column in TRIPLE_COLUMNS)
        if key not in test_keys:
            continue

        minimum = np.asarray(
            [float(row[c]) for c in BBOX_COLUMNS[:3]],
            dtype=np.float32,
        )
        maximum = np.asarray(
            [float(row[c]) for c in BBOX_COLUMNS[3:]],
            dtype=np.float32,
        )
        image_shape = shape_by_key[key]

        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            raise ValueError("Non-finite bbox for test series %s" % (key,))
        if np.any(minimum > maximum):
            raise ValueError("Invalid bbox min>max for test series %s" % (key,))
        if np.any(minimum < 0) or np.any(maximum >= image_shape):
            raise ValueError(
                "GT bbox is outside image_original crop-local coordinates "
                "for %s: min=%s max=%s shape=%s"
                % (
                    key,
                    minimum.tolist(),
                    maximum.tolist(),
                    image_shape.astype(int).tolist(),
                )
            )

        center = (minimum + maximum) / 2.0
        sizes = maximum - minimum + 1.0
        rows.append({
            "pid": key[2],  # seriesInstanceUID
            "center_x": float(center[2]),
            "center_y": float(center[1]),
            "center_z": float(center[0]),
            "diameter": float(np.max(sizes)),
        })

    return pd.DataFrame(
        rows,
        columns=["pid", "center_x", "center_y", "center_z", "diameter"],
    )


def gather_rows(local_rows, rank, world_size, distributed):
    if not distributed:
        return local_rows

    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local_rows, gathered, dst=0)

    if rank != 0:
        return None

    merged = []
    for rows in gathered:
        if rows:
            merged.extend(rows)
    return merged


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.rpn_weight < 0 or args.rcnn_weight < 0:
        raise ValueError("--rpn-weight/--rcnn-weight must be >= 0")
    if args.rpn_weight + args.rcnn_weight <= 0:
        raise ValueError("RPN and RCNN ensemble weights cannot both be zero")

    target_spacing = parse_spacing(args.target_spacing_zyx)
    seed_everything(args.seed)

    rank = world_size = local_rank = None
    distributed = False
    try:
        rank, world_size, local_rank, device, distributed = init_distributed(args)

        logging.basicConfig(
            level=logging.INFO if rank == 0 else logging.WARNING,
            format="[%(levelname)s][%(asctime)s] %(message)s",
        )

        dataset = OrganizedNPZTestDataset(
            data_root=args.data_root,
            test_list=args.test_list,
            target_spacing_zyx=target_spacing,
            window_min=args.window_min,
            window_max=args.window_max,
            pad_factor=args.pad_factor,
            limit=args.limit_test_samples,
        )
        sampler = RankStridedSampler(dataset, rank, world_size)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            collate_fn=list_collate,
        )

        if rank == 0:
            print("========================================")
            print("SANet organized NPZ DDP test")
            print("========================================")
            print("test series           :", len(dataset))
            print("data root             :", args.data_root)
            print("test list             :", args.test_list)
            print("annotation            :", args.annotation)
            print("world size / GPUs     :", world_size)
            print("per-GPU loader batch  :", args.batch_size)
            print("forward batch         : 1 CT (sequential inside loader batch)")
            print("num workers / GPU     :", args.num_workers)
            print(
                "ensemble RPN:RCNN     : %g:%g"
                % (args.rpn_weight, args.rcnn_weight)
            )
            print(
                "target spacing ZYX    : %s"
                % ",".join("%g" % v for v in target_spacing)
            )
            print("window HU             : [%g, %g]" % (
                args.window_min, args.window_max
            ))
            print("out dir               :", args.out_dir)
            print("========================================")

        model_cfg = dict(config)
        base_model = SANet(
            model_cfg,
            mode="eval",
            use_aspp=args.use_aspp,
        ).to(device)

        checkpoint = torch.load(args.weight, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint.get("epoch", None)
        else:
            state_dict = checkpoint
            epoch = None

        state_dict = strip_module_prefix(state_dict)
        base_model.load_state_dict(state_dict, strict=True)
        base_model.set_mode("eval")
        base_model.use_rcnn = True

        if distributed:
            model = DDP(
                base_model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )
        else:
            model = base_model

        local_rows = []
        processed = 0

        for loader_batch in loader:
            # User-selected behavior B:
            # batch_size controls loading/scheduling only; full CTs are forwarded
            # one at a time to avoid variable-shape padding and unnecessary VRAM.
            for sample in loader_batch:
                clear_inference_cache(base_model)

                input_tensor = sample["input"].unsqueeze(0).to(
                    device,
                    non_blocking=True,
                )
                truth_boxes = [np.zeros((0, 6), dtype=np.float32)]
                truth_labels = [np.zeros((0,), dtype=np.int64)]

                with torch.inference_mode():
                    _ = model(input_tensor, truth_boxes, truth_labels)

                boxes, scores = ensemble_from_model(
                    base_model,
                    rpn_weight=args.rpn_weight,
                    rcnn_weight=args.rcnn_weight,
                )

                local_rows.extend(
                    boxes_to_result_rows(
                        sample["seriesInstanceUID"],
                        boxes,
                        scores,
                        actual_scale=np.asarray(
                            sample["actual_scale"], dtype=np.float32
                        ),
                        valid_resampled_shape=np.asarray(
                            sample["valid_resampled_shape"], dtype=np.int32
                        ),
                        original_shape=np.asarray(
                            sample["image_shape"], dtype=np.int32
                        ),
                        spacing_zyx=np.asarray(
                            sample["spacing_zyx"], dtype=np.float32
                        ),
                    )
                )

                processed += 1
                if rank == 0:
                    print(
                        "[rank %d] processed %d local CT(s), current=%s, candidates=%d"
                        % (
                            rank,
                            processed,
                            sample["seriesInstanceUID"],
                            len(boxes),
                        )
                    )

                del input_tensor
                clear_inference_cache(base_model)
                torch.cuda.empty_cache()

        all_rows = gather_rows(
            local_rows,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
        )

        if distributed:
            dist.barrier()

        if rank == 0:
            out_dir = Path(args.out_dir)
            froc_dir = out_dir / "FROC"
            froc_dir.mkdir(parents=True, exist_ok=True)

            series_order = {
                meta["seriesInstanceUID"]: i
                for i, meta in enumerate(dataset.case_metadata)
            }

            results_df = pd.DataFrame(
                all_rows,
                columns=[
                    "pid",
                    "center_x",
                    "center_y",
                    "center_z",
                    "diameter",
                    "diameter_mm",
                    "bbox_min_z",
                    "bbox_max_z",
                    "bbox_min_y",
                    "bbox_max_y",
                    "bbox_min_x",
                    "bbox_max_x",
                    "probability",
                ],
            )
            if len(results_df):
                results_df["_series_order"] = results_df["pid"].map(series_order)
                results_df = results_df.sort_values(
                    ["_series_order", "probability"],
                    ascending=[True, False],
                ).drop(columns=["_series_order"])

            results_path = froc_dir / "results.csv"
            results_df.to_csv(results_path, index=False)

            annotations_df = build_froc_annotations(args.annotation, dataset)
            annotations_path = froc_dir / "annotations.csv"
            annotations_df.to_csv(annotations_path, index=False)

            # Crucial: include every test series, even series with no GT and no
            # prediction, so FROC's scan denominator is correct.
            seriesuids_path = froc_dir / "seriesuids.csv"
            with seriesuids_path.open("w", encoding="utf-8") as handle:
                for meta in dataset.case_metadata:
                    handle.write(str(meta["seriesInstanceUID"]) + "\n")

            print("")
            print("[Done]")
            if epoch is not None:
                print("checkpoint epoch       :", epoch)
            print("results.csv            :", results_path)
            print("annotations.csv        :", annotations_path)
            print("seriesuids.csv         :", seriesuids_path)
            print("test series            :", len(dataset))
            print("predicted candidates   :", len(results_df))
            print("GT lesions             :", len(annotations_df))
            print("per-case detections.npy: DISABLED")

    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()