"""CT-level reader for the organized LUNA25 NPZ layout.

Each dataset item loads one cropped-original-resolution NPZ once and returns
all lesion-centered patches for that CT through ``CTBatchDataset``'s existing
crop/augmentation/finalization path.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from dataset.ct_batch_reader import CTBatchDataset, split_pos_neg_counts


class OrganizedNPZCTDataset(CTBatchDataset):
    """One item per CT, with all annotations belonging to that CT grouped."""

    def __init__(self, data_root, set_name, annotation_csv, cfg,
                 mode="train", batch_size=8, target_spacing_zyx=(1.0, 0.6, 0.6),
                 window_min=-1200.0, window_max=600.0):
        if mode not in ("train", "val"):
            raise ValueError("OrganizedNPZCTDataset supports train/val, got %s" % mode)

        self.mode = mode
        self.cfg = cfg
        self.batch_size = int(batch_size)
        self.data_root = Path(data_root)
        self.data_dir = os.fspath(self.data_root)
        self.set_name = os.fspath(set_name)
        self.annotation_csv = os.fspath(annotation_csv)
        self.dataset_name = cfg.get("dataset", "luna25_organized")
        self.augtype = cfg["augtype"]
        self.pad_value = cfg["pad_value"]
        self.partial_gt_positive_threshold = float(
            cfg.get("partial_gt_positive_threshold", 0.5)
        )
        self.target_spacing_zyx = np.asarray(target_spacing_zyx, dtype=np.float32)
        if self.target_spacing_zyx.shape != (3,) or np.any(self.target_spacing_zyx <= 0):
            raise ValueError("target_spacing_zyx must contain three positive values")
        self.window_min = float(window_min)
        self.window_max = float(window_max)
        if not self.window_min < self.window_max:
            raise ValueError("window_min must be smaller than window_max")

        with open(self.set_name, "r") as handle:
            self.filenames = [line.strip() for line in handle if line.strip()]
        if not self.filenames:
            raise ValueError("Dataset list is empty: %s" % self.set_name)

        blacklist = cfg.get("blacklist", [])
        self.filenames = [filename for filename in self.filenames if filename not in blacklist]
        self.case_npz_paths = [self._resolve_npz_path(filename) for filename in self.filenames]

        annos = pd.read_csv(self.annotation_csv)
        required = {
            "patient_id", "studyInstanceUID", "seriesInstanceUID",
            "bbox_min_z", "bbox_min_y", "bbox_min_x",
            "bbox_max_z", "bbox_max_y", "bbox_max_x",
        }
        missing = sorted(required.difference(annos.columns))
        if missing:
            raise ValueError("Annotation CSV missing columns: %s" % missing)

        annotation_by_series = {
            str(series_uid): group
            for series_uid, group in annos.groupby("seriesInstanceUID", sort=False)
        }
        self.resample_scales = []
        self.sample_bboxes = []
        self.case_series_uids = []
        for filename, npz_path in zip(self.filenames, self.case_npz_paths):
            series_uid = filename.rsplit("/", 1)[-1]
            self.case_series_uids.append(series_uid)
            if series_uid not in annotation_by_series:
                raise ValueError("No annotations found for series %s" % series_uid)

            with np.load(npz_path, allow_pickle=True) as data:
                image_shape = tuple(np.asarray(data["image_original"]).shape)
                spacing_zyx = np.asarray(data["spacing"], dtype=np.float32).reshape(3)
                stored_uid = self._scalar_string(data["seriesUID"])
            if stored_uid != series_uid:
                raise ValueError(
                    "NPZ seriesUID mismatch: path=%s stored=%s csv=%s"
                    % (npz_path, stored_uid, series_uid)
                )

            output_shape = np.maximum(
                1,
                np.rint(
                    np.asarray(image_shape, dtype=np.float32)
                    * spacing_zyx / self.target_spacing_zyx
                ).astype(np.int32),
            )
            actual_scale = output_shape.astype(np.float32) / np.asarray(image_shape, dtype=np.float32)
            self.resample_scales.append(actual_scale)

            group = annotation_by_series[series_uid]
            boxes = []
            for _, row in group.iterrows():
                minimum = np.asarray([
                    row["bbox_min_z"], row["bbox_min_y"], row["bbox_min_x"]
                ], dtype=np.float32)
                maximum = np.asarray([
                    row["bbox_max_z"], row["bbox_max_y"], row["bbox_max_x"]
                ], dtype=np.float32)
                if np.any(minimum > maximum):
                    raise ValueError("Invalid bbox for series %s" % series_uid)
                if np.any(minimum < 0) or np.any(maximum >= np.asarray(image_shape)):
                    raise ValueError(
                        "BBox outside image_original for series %s: %s..%s shape=%s"
                        % (series_uid, minimum.tolist(), maximum.tolist(), image_shape)
                    )

                # Keep inclusive bbox semantics through resampling.
                resampled_min = minimum * actual_scale
                resampled_max = (maximum + 1.0) * actual_scale - 1.0
                boxes.append(np.asarray([
                    resampled_min[0], resampled_max[0],
                    resampled_min[1], resampled_max[1],
                    resampled_min[2], resampled_max[2],
                ], dtype=np.float32))
            self.sample_bboxes.append(
                np.asarray(boxes, dtype=np.float32).reshape(-1, 6)
            )

        self.bboxes = []
        for case_index, boxes in enumerate(self.sample_bboxes):
            for box in boxes:
                self.bboxes.append([np.concatenate([[case_index], box])])
        if not self.bboxes:
            raise ValueError("No annotated boxes found for %s" % self.set_name)
        self.bboxes = np.concatenate(self.bboxes, axis=0).astype(np.float32)

        self.positives_by_ct = {
            index: [box.copy() for box in boxes]
            for index, boxes in enumerate(self.sample_bboxes)
        }
        self.pos_ct_indices = [
            index for index, boxes in self.positives_by_ct.items() if len(boxes) > 0
        ]
        self.n_pos, self.n_neg = split_pos_neg_counts(self.batch_size, 1.0)
        self.hard_fps_by_ct = {index: [] for index in range(len(self.filenames))}
        self.hard_fps = np.zeros((0, 6), dtype=np.float32)
        self.num_positive_samples = len(self.bboxes)
        self.num_negative_samples = 0
        self.num_random_samples = 0
        self.crop = self._make_crop()

        print(
            "[%s] OrganizedNPZCTDataset mode=%s ct=%d lesions=%d target_spacing_zyx=%s window=[%.1f, %.1f]"
            % (
                self.dataset_name,
                self.mode,
                len(self.filenames),
                len(self.bboxes),
                tuple(float(value) for value in self.target_spacing_zyx),
                self.window_min,
                self.window_max,
            )
        )

    @staticmethod
    def _scalar_string(value):
        array = np.asarray(value)
        return str(array.item()) if array.shape == () else str(value)

    def _resolve_npz_path(self, relative_case):
        parts = Path(relative_case).parts
        if len(parts) != 3:
            raise ValueError(
                "Split entry must be patient_id/studyInstanceUID/seriesUID: %s"
                % relative_case
            )
        series_uid = parts[-1]
        path = self.data_root.joinpath(*parts, "%s.npz" % series_uid)
        if not path.is_file():
            raise FileNotFoundError("Missing organized NPZ: %s" % path)
        return path

    def _make_crop(self):
        # CTBatchDataset.__init__ creates Crop directly; keep that behavior
        # without invoking its legacy npy/CSV discovery code.
        from dataset.bbox_reader import Crop
        return Crop(self.cfg)

    def _resample_and_window(self, image, spacing_zyx, output_shape):
        image = np.asarray(image, dtype=np.float32)
        zoom_factors = np.asarray(output_shape, dtype=np.float32) / np.asarray(image.shape, dtype=np.float32)
        image = zoom(image, zoom_factors.tolist(), order=1, mode="nearest", prefilter=False)
        if tuple(image.shape) != tuple(output_shape):
            # scipy normally returns the requested rounded shape; normalize
            # the rare one-voxel rounding discrepancy deterministically.
            fixed = np.full(tuple(output_shape), self.window_min, dtype=np.float32)
            slices = tuple(slice(0, min(image.shape[i], output_shape[i])) for i in range(3))
            fixed[slices] = image[slices]
            image = fixed
        image = np.clip(image, self.window_min, self.window_max)
        image = (image - self.window_min) * (255.0 / (self.window_max - self.window_min))
        return image.astype(np.float32, copy=False)

    def load_image(self, filename):
        case_index = self.filenames.index(filename)
        npz_path = self.case_npz_paths[case_index]
        with np.load(npz_path, allow_pickle=True) as data:
            image = np.asarray(data["image_original"])
            spacing_zyx = np.asarray(data["spacing"], dtype=np.float32).reshape(3)
        output_shape = np.maximum(
            1,
            np.rint(
                np.asarray(image.shape, dtype=np.float32)
                * spacing_zyx / self.target_spacing_zyx
            ).astype(np.int32),
        )
        image = self._resample_and_window(image, spacing_zyx, output_shape)
        return image[np.newaxis, ...]

    def _apply_full_aug(self, sample, target, bboxes, is_random_crop):
        if self.mode != "train":
            return sample, target, bboxes
        return super()._apply_full_aug(sample, target, bboxes, is_random_crop)

    def __getitem__(self, index):
        if self.mode == "train":
            return super().__getitem__(index)

        # Validation returns every lesion patch from one CT in a single item.
        ct_index = int(self.pos_ct_indices[int(index)])
        filename = self.filenames[ct_index]
        image = self.load_image(filename)
        samples = []
        for lesion_box in self.positives_by_ct[ct_index]:
            sample, target, bboxes, is_random_crop = self._crop_positive(
                image, ct_index, lesion_box
            )
            samples.append(
                self._assemble_from_crop(
                    sample, target, bboxes,
                    is_random_crop=is_random_crop,
                    force_simple_aug=False,
                )
            )
        return samples

    def __len__(self):
        return len(self.pos_ct_indices)


def build_organized_npz_datasets(data_root, annotation_csv, train_list, val_list,
                                 cfg, batch_size=8, target_spacing_zyx=(1.0, 0.6, 0.6),
                                 window_min=-1200.0, window_max=600.0):
    """Build train/val CT-level datasets for the organized NPZ layout."""
    train_dataset = OrganizedNPZCTDataset(
        data_root, train_list, annotation_csv, cfg,
        mode="train", batch_size=batch_size,
        target_spacing_zyx=target_spacing_zyx,
        window_min=window_min, window_max=window_max,
    )
    val_dataset = OrganizedNPZCTDataset(
        data_root, val_list, annotation_csv, cfg,
        mode="val", batch_size=batch_size,
        target_spacing_zyx=target_spacing_zyx,
        window_min=window_min, window_max=window_max,
    )
    return train_dataset, val_dataset
