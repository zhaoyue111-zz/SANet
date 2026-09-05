"""CT-level reader for competition organized NPZ + optional GT boost sampling.

Differences from ``organized_npz_reader.py``:
  - NPZ path comes from annotation CSV column ``path`` (split still filters cases).
  - With ``gt_boost=True`` (train only):
      * lesions matching competition boost rules get higher sampling weight
      * patients with any boost lesion: use **all** positive series under studies
        that contain at least one boost lesion
      * other patients: pick **one** positive series, preferring
        ``max(image_original) > 1000`` (lung-window heuristic), else random
  - Without ``gt_boost``: keep one random positive series per patient per epoch.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from dataset.ct_batch_reader import (
    CTBatchDataset,
    sample_pool_indices,
    split_pos_neg_counts,
)
from dataset.gt_sampling_competition import (
    DEFAULT_GT_BOOST_FACTOR,
    annotation_gt_boost_weight,
)


TRIPLE_COLUMNS = ("patient_id", "studyInstanceUID", "seriesInstanceUID")
BBOX_COLUMNS = (
    "bbox_min_z", "bbox_min_y", "bbox_min_x",
    "bbox_max_z", "bbox_max_y", "bbox_max_x",
)
BOOST_ATTR_COLUMNS = (
    "lesion_id",
    "nodule_id",
    "main_lesion",
    "detectResult",
    "noduleType",
    "symptoms",
)
LUNG_WINDOW_MAX_HU = 1000.0


def _read_split_entries(split_path):
    """Read and strictly validate patient/study/series split entries."""
    entries = []
    with open(split_path, "r") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("/")
            if len(parts) != 3 or any(not part or part in (".", "..") for part in parts):
                raise ValueError(
                    "%s:%d must be exactly patient_id/studyInstanceUID/seriesInstanceUID, got %r"
                    % (split_path, line_number, line)
                )
            entries.append("/".join(parts))
    if not entries:
        raise ValueError("Dataset list is empty: %s" % split_path)
    return entries


def _entry_key(entry):
    parts = entry.split("/")
    if len(parts) != 3:
        raise ValueError("Invalid organized split entry: %s" % entry)
    return tuple(parts)


def _check_split_isolation(train_entries, val_entries, train_path, val_path):
    """Check duplicate paths and patient-level train/validation isolation."""
    for split_name, entries in (
        ("train", train_entries),
        ("val", val_entries),
    ):
        duplicates = sorted(
            entry for entry, count in Counter(entries).items() if count > 1
        )
        if duplicates:
            raise ValueError(
                "%s contains duplicate paths (first conflicts: %s)"
                % (split_name, duplicates[:5])
            )

    train_series = set(train_entries)
    val_series = set(val_entries)
    series_overlap = sorted(train_series.intersection(val_series))
    if series_overlap:
        raise ValueError(
            "train/val series overlap (first conflicts: %s)" % series_overlap[:5]
        )

    train_patients = {_entry_key(entry)[0] for entry in train_entries}
    val_patients = {_entry_key(entry)[0] for entry in val_entries}
    patient_overlap = sorted(train_patients.intersection(val_patients))
    if patient_overlap:
        print(
            "Patient leakage between %s and %s (first conflicts: %s)"
            % (train_path, val_path, patient_overlap[:5])
        )
        raise ValueError(
            "train/val patient leakage detected (first conflicts: %s)"
            % patient_overlap[:5]
        )


def _series_path_from_rows(rows, case_key):
    """Require a usable ``path`` column shared by rows of one series."""
    paths = []
    for row in rows:
        raw = row.get("path", "")
        text = "" if raw is None else str(raw).strip()
        if text and text.lower() not in ("nan", "none", "null"):
            paths.append(text)
    if not paths:
        raise FileNotFoundError(
            "Annotation rows for %s/%s/%s have empty path"
            % case_key
        )
    unique = sorted(set(paths))
    if len(unique) > 1:
        raise ValueError(
            "Conflicting path values for %s/%s/%s: %s"
            % (case_key[0], case_key[1], case_key[2], unique[:3])
        )
    path = Path(unique[0])
    if not path.is_file():
        raise FileNotFoundError("Missing NPZ from annotation path: %s" % path)
    return path


class OrganizedNPZCTDatasetYibao(CTBatchDataset):
    """Competition organized NPZ dataset with optional GT boost sampling."""

    def __init__(
        self,
        data_root,
        set_name,
        annotation_csv,
        cfg,
        mode="train",
        batch_size=8,
        target_spacing_zyx=(1.0, 0.6, 0.6),
        window_min=-1200.0,
        window_max=600.0,
        split_entries=None,
        train_neg_pos_ratio=1.0,
        gt_boost=False,
        gt_boost_factor=DEFAULT_GT_BOOST_FACTOR,
    ):
        if mode not in ("train", "val"):
            raise ValueError(
                "OrganizedNPZCTDatasetYibao supports train/val, got %s" % mode
            )

        self.mode = mode
        self.cfg = cfg
        self.batch_size = int(batch_size)
        self.data_root = Path(data_root) if data_root is not None else None
        self.data_dir = os.fspath(self.data_root) if self.data_root is not None else ""
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
        self.seed = int(cfg.get("organized_npz_seed", cfg.get("seed", 35202)))
        self.epoch = 0
        self.gt_boost = bool(gt_boost) and mode == "train"
        self.gt_boost_factor = float(gt_boost_factor)
        if self.gt_boost_factor <= 0:
            raise ValueError("gt_boost_factor must be > 0")

        self.split_entries = (
            list(split_entries) if split_entries is not None
            else _read_split_entries(self.set_name)
        )
        blacklist = set(cfg.get("blacklist", []))
        self.filenames = [entry for entry in self.split_entries if entry not in blacklist]
        self.filename_to_index = {
            filename: index for index, filename in enumerate(self.filenames)
        }
        self.case_keys = [_entry_key(filename) for filename in self.filenames]

        dtype = {column: str for column in TRIPLE_COLUMNS}
        for column in ("lesion_id", "nodule_id", "path"):
            dtype[column] = str
        annos = pd.read_csv(
            self.annotation_csv,
            dtype=dtype,
            keep_default_na=False,
        )
        required = set(TRIPLE_COLUMNS).union(BBOX_COLUMNS)
        required.add("path")
        missing = sorted(required.difference(annos.columns))
        if missing:
            raise ValueError("Annotation CSV missing columns: %s" % missing)
        if self.gt_boost:
            boost_missing = sorted(set(BOOST_ATTR_COLUMNS).difference(annos.columns))
            if boost_missing:
                raise ValueError(
                    "gt_boost requires annotation columns: %s" % boost_missing
                )

        annotation_by_key = {}
        for row in annos.to_dict("records"):
            key = tuple(str(row[column]) for column in TRIPLE_COLUMNS)
            annotation_by_key.setdefault(key, []).append(row)

        self.resample_scales = []
        self.output_shapes = []
        self.sample_bboxes = []
        self.case_series_uids = []
        self.case_metadata = []
        self.case_npz_paths = []
        self.zero_annotation_series = []
        self.positive_weights_by_ct = []
        self.boost_study_keys = set()

        for filename, case_key in zip(self.filenames, self.case_keys):
            patient_id, study_uid, series_uid = case_key
            self.case_series_uids.append(series_uid)
            rows = annotation_by_key.get(case_key, [])
            npz_path = _series_path_from_rows(rows, case_key) if rows else None
            if npz_path is None:
                # Split lists series with no annotation rows: still require path
                # via a later error only if used as positive; allow empty GT.
                raise FileNotFoundError(
                    "No annotation rows (hence no path) for split entry %s"
                    % filename
                )
            self.case_npz_paths.append(npz_path)

            with np.load(npz_path, allow_pickle=True) as data:
                if "image_shape" in data.files:
                    image_shape = tuple(
                        np.asarray(data["image_shape"][0], dtype=np.int64).tolist()
                    )
                else:
                    image_shape = tuple(np.asarray(data["image_original"]).shape)
                spacing_zyx = np.asarray(data["spacing"], dtype=np.float32).reshape(3)
                stored_uid = self._scalar_string(data["seriesUID"]) if "seriesUID" in data.files else series_uid
            if stored_uid != series_uid:
                raise ValueError(
                    "NPZ seriesUID mismatch: path=%s stored=%s path_series=%s"
                    % (npz_path, stored_uid, series_uid)
                )
            if len(image_shape) != 3 or any(int(value) <= 0 for value in image_shape):
                raise ValueError("Invalid image shape %s in %s" % (image_shape, npz_path))

            output_shape = np.maximum(
                1,
                np.rint(
                    np.asarray(image_shape, dtype=np.float32)
                    * spacing_zyx
                    / self.target_spacing_zyx
                ).astype(np.int32),
            )
            actual_scale = output_shape.astype(np.float32) / np.asarray(
                image_shape, dtype=np.float32
            )
            self.output_shapes.append(output_shape)
            self.resample_scales.append(actual_scale)

            boxes = []
            weights = []
            study_has_boost = False
            for row in rows:
                minimum = np.asarray(
                    [float(row[column]) for column in BBOX_COLUMNS[:3]],
                    dtype=np.float32,
                )
                maximum = np.asarray(
                    [float(row[column]) for column in BBOX_COLUMNS[3:]],
                    dtype=np.float32,
                )
                if np.any(minimum > maximum):
                    raise ValueError("Invalid bbox for %s" % (case_key,))
                if np.any(minimum < 0) or np.any(
                    maximum >= np.asarray(image_shape, dtype=np.float32)
                ):
                    raise ValueError(
                        "BBox outside image_original for %s: %s..%s shape=%s"
                        % (case_key, minimum.tolist(), maximum.tolist(), image_shape)
                    )

                resampled_min = minimum * actual_scale
                resampled_max = (maximum + 1.0) * actual_scale - 1.0
                boxes.append(
                    np.asarray(
                        [
                            resampled_min[0], resampled_max[0],
                            resampled_min[1], resampled_max[1],
                            resampled_min[2], resampled_max[2],
                        ],
                        dtype=np.float32,
                    )
                )
                weight = 1.0
                if self.gt_boost:
                    weight = float(
                        annotation_gt_boost_weight(
                            pd.Series(row),
                            boost=self.gt_boost_factor,
                        )
                    )
                    if weight > 1.0 + 1e-6:
                        study_has_boost = True
                weights.append(weight)

            if study_has_boost:
                self.boost_study_keys.add((patient_id, study_uid))

            case_boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 6)
            self.sample_bboxes.append(case_boxes)
            self.positive_weights_by_ct.append(
                np.asarray(weights, dtype=np.float64).reshape(-1)
            )
            if not len(case_boxes):
                self.zero_annotation_series.append(filename)
            self.case_metadata.append(
                {
                    "filename": filename,
                    "patient_id": patient_id,
                    "studyInstanceUID": study_uid,
                    "seriesInstanceUID": series_uid,
                    "image_shape": tuple(int(value) for value in image_shape),
                    "npz_path": str(npz_path),
                }
            )

        self.bboxes = [
            np.concatenate(([case_index], box)).astype(np.float32)
            for case_index, boxes in enumerate(self.sample_bboxes)
            for box in boxes
        ]
        self.bboxes = (
            np.asarray(self.bboxes, dtype=np.float32).reshape(-1, 7)
            if self.bboxes
            else np.zeros((0, 7), dtype=np.float32)
        )
        self.positives_by_ct = {
            index: [box.copy() for box in boxes]
            for index, boxes in enumerate(self.sample_bboxes)
        }
        self.pos_ct_indices = [
            index for index, boxes in self.positives_by_ct.items() if len(boxes) > 0
        ]

        self.positive_ct_indices_by_patient = {}
        for ct_index in self.pos_ct_indices:
            patient_id = self.case_metadata[ct_index]["patient_id"]
            self.positive_ct_indices_by_patient.setdefault(patient_id, []).append(ct_index)
        self.positive_patient_ids = sorted(self.positive_ct_indices_by_patient)
        self.selected_train_ct_indices = []
        self.selected_train_lesions = 0
        self._max_hu_cache = {}

        self.train_neg_pos_ratio = float(train_neg_pos_ratio)
        self.n_pos, self.n_neg = split_pos_neg_counts(
            self.batch_size, self.train_neg_pos_ratio
        )
        if self.mode == "val":
            self.n_neg = 0
        self.hard_fps_by_ct = {index: [] for index in range(len(self.filenames))}
        self.hard_fps = np.zeros((0, 6), dtype=np.float32)
        self.num_positive_samples = len(self.bboxes)
        self.num_negative_samples = 0
        self.num_random_samples = 0
        self.crop = self._make_crop()
        self._train_items = []
        self._val_items = []
        self._rebuild_items()

        if self.gt_boost:
            n_boost = sum(
                1
                for weights in self.positive_weights_by_ct
                for weight in weights
                if weight > 1.0 + 1e-6
            )
            n_lesion = sum(len(weights) for weights in self.positive_weights_by_ct)
            print(
                "[gt_boost] enabled factor=%.3f boost_studies=%d positive_patients=%d"
                % (
                    self.gt_boost_factor,
                    len(self.boost_study_keys),
                    len(self.positive_patient_ids),
                )
            )
            print(
                "[gt_boost] lesions=%d boosted=%d (weight>1)"
                % (n_lesion, n_boost)
            )

    @staticmethod
    def _scalar_string(value):
        array = np.asarray(value)
        return str(array.item()) if array.shape == () else str(value)

    def _resolve_npz_path(self, relative_case):
        # Kept for CTBatchDataset compatibility; paths come from annotation CSV.
        case_index = self.filename_to_index[relative_case]
        return Path(self.case_npz_paths[case_index])

    def _make_crop(self):
        from dataset.bbox_reader import Crop
        return Crop(self.cfg)

    def _max_hu(self, ct_index):
        if ct_index in self._max_hu_cache:
            return self._max_hu_cache[ct_index]
        npz_path = self.case_npz_paths[ct_index]
        with np.load(npz_path, allow_pickle=True) as data:
            image = np.asarray(data["image_original"])
            max_hu = float(np.max(image)) if image.size else float("-inf")
        self._max_hu_cache[ct_index] = max_hu
        return max_hu

    def _select_one_positive_series_per_patient(self, rng):
        """Legacy: exactly one positive series for every positive patient."""
        selected = []
        for patient_id in self.positive_patient_ids:
            candidates = self.positive_ct_indices_by_patient[patient_id]
            if len(candidates) == 1:
                selected.append(int(candidates[0]))
            else:
                selected.append(int(candidates[int(rng.integers(len(candidates)))]))
        return selected

    def _select_series_with_gt_boost(self, rng):
        """Boost-aware series selection (train only)."""
        selected = []
        for patient_id in self.positive_patient_ids:
            candidates = self.positive_ct_indices_by_patient[patient_id]
            boost_series = [
                ct_index
                for ct_index in candidates
                if (
                    self.case_metadata[ct_index]["patient_id"],
                    self.case_metadata[ct_index]["studyInstanceUID"],
                ) in self.boost_study_keys
            ]
            if boost_series:
                selected.extend(int(ct_index) for ct_index in boost_series)
                continue

            if len(candidates) == 1:
                selected.append(int(candidates[0]))
                continue

            lung_candidates = [
                ct_index
                for ct_index in candidates
                if self._max_hu(ct_index) > LUNG_WINDOW_MAX_HU
            ]
            pool = lung_candidates if lung_candidates else list(candidates)
            selected.append(int(pool[int(rng.integers(len(pool)))]))
        return selected

    def _sample_lesion_group(self, ct_index, rng):
        lesion_indices = np.arange(
            len(self.positives_by_ct[ct_index]), dtype=np.int64
        )
        if len(lesion_indices) == 0:
            return []

        weights = np.asarray(
            self.positive_weights_by_ct[ct_index], dtype=np.float64
        ).reshape(-1)
        use_weighted = (
            self.gt_boost
            and weights.shape[0] == lesion_indices.shape[0]
            and np.all(np.isfinite(weights))
            and np.any(weights > 0)
        )

        if len(lesion_indices) >= self.n_pos:
            if use_weighted:
                probs = weights / weights.sum()
                chosen = rng.choice(
                    lesion_indices,
                    size=self.n_pos,
                    replace=False,
                    p=probs,
                )
            else:
                chosen = rng.choice(lesion_indices, size=self.n_pos, replace=False)
            return [(int(lesion_index), False) for lesion_index in chosen]

        shuffled = lesion_indices.copy()
        rng.shuffle(shuffled)
        selected = [(int(lesion_index), False) for lesion_index in shuffled]
        missing = self.n_pos - len(selected)
        if missing:
            if use_weighted:
                probs = weights / weights.sum()
                fillers = rng.choice(
                    lesion_indices,
                    size=missing,
                    replace=True,
                    p=probs,
                )
            else:
                fillers = rng.choice(lesion_indices, size=missing, replace=True)
            selected.extend((int(lesion_index), True) for lesion_index in fillers)
        return selected

    def _rebuild_items(self):
        if self.mode == "train":
            rng = np.random.default_rng(self.seed + self.epoch)
            self._train_items = []

            if self.gt_boost:
                self.selected_train_ct_indices = self._select_series_with_gt_boost(rng)
            else:
                self.selected_train_ct_indices = (
                    self._select_one_positive_series_per_patient(rng)
                )
            self.selected_train_lesions = 0

            for ct_index in self.selected_train_ct_indices:
                selected = self._sample_lesion_group(ct_index, rng)
                self.selected_train_lesions += len(selected)
                self._train_items.append((ct_index, selected))
        else:
            self._val_items = []
            for ct_index in self.pos_ct_indices:
                lesion_indices = range(len(self.positives_by_ct[ct_index]))
                for start in range(0, len(self.positives_by_ct[ct_index]), self.batch_size):
                    selected = [
                        (int(lesion_index), False)
                        for lesion_index in list(lesion_indices)[start:start + self.batch_size]
                    ]
                    self._val_items.append((ct_index, selected))

    def set_epoch(self, epoch):
        """Select series/patient and reshuffle lesion groups for this epoch."""
        self.epoch = int(epoch)
        if self.mode == "train":
            self._rebuild_items()
            print(
                "[organized yibao train] epoch %d: positive_patients=%d "
                "selected_series=%d selected_lesions=%d train_batches=%d gt_boost=%s"
                % (
                    self.epoch,
                    len(self.positive_patient_ids),
                    len(self.selected_train_ct_indices),
                    self.selected_train_lesions,
                    len(self._train_items),
                    self.gt_boost,
                )
            )

    def _resample_and_window(self, image, output_shape):
        image = np.asarray(image, dtype=np.float32)
        zoom_factors = np.asarray(output_shape, dtype=np.float32) / np.asarray(
            image.shape, dtype=np.float32
        )
        image = zoom(
            image,
            zoom_factors.tolist(),
            order=1,
            mode="nearest",
            prefilter=False,
        )
        if tuple(image.shape) != tuple(output_shape):
            fixed = np.full(tuple(output_shape), self.window_min, dtype=np.float32)
            slices = tuple(
                slice(0, min(image.shape[i], output_shape[i])) for i in range(3)
            )
            fixed[slices] = image[slices]
            image = fixed
        image = np.clip(image, self.window_min, self.window_max)
        image = (image - self.window_min) * (255.0 / (self.window_max - self.window_min))
        return image.astype(np.float32, copy=False)

    def load_image(self, filename):
        case_index = self.filename_to_index[filename]
        npz_path = self.case_npz_paths[case_index]
        with np.load(npz_path, allow_pickle=True) as data:
            image = np.asarray(data["image_original"])
            spacing_zyx = np.asarray(data["spacing"], dtype=np.float32).reshape(3)
        output_shape = np.maximum(
            1,
            np.rint(
                np.asarray(image.shape, dtype=np.float32)
                * spacing_zyx
                / self.target_spacing_zyx
            ).astype(np.int32),
        )
        image = self._resample_and_window(image, output_shape)
        return image[np.newaxis, ...]

    def _apply_full_aug(self, sample, target, bboxes, is_random_crop):
        if self.mode != "train":
            return sample, target, bboxes
        return super()._apply_full_aug(sample, target, bboxes, is_random_crop)

    def _get_item_samples(self, item):
        ct_index, selected = item
        image = self.load_image(self.filenames[ct_index])
        samples = []
        for lesion_index, force_simple_aug in selected:
            lesion_box = self.positives_by_ct[ct_index][lesion_index]
            sample, target, bboxes, is_random_crop = self._crop_positive(
                image, ct_index, lesion_box
            )
            samples.append(
                self._assemble_from_crop(
                    sample,
                    target,
                    bboxes,
                    is_random_crop=is_random_crop,
                    force_simple_aug=force_simple_aug,
                )
            )

        if self.mode == "train" and self.n_neg:
            hard_fps = self.hard_fps_by_ct.get(ct_index, [])
            if len(hard_fps) >= self.n_neg:
                for fp_index, _force_simple in sample_pool_indices(
                    len(hard_fps), self.n_neg
                ):
                    crop = self._crop_hard_fp(image, ct_index, hard_fps[fp_index])
                    samples.append(
                        self._assemble_from_crop(
                            *crop,
                            force_simple_aug=False,
                        )
                    )
            else:
                for center in hard_fps:
                    crop = self._crop_hard_fp(image, ct_index, center)
                    samples.append(
                        self._assemble_from_crop(
                            *crop,
                            force_simple_aug=False,
                        )
                    )
                for _ in range(self.n_neg - len(hard_fps)):
                    crop = self._crop_random_bg(image, ct_index)
                    samples.append(
                        self._assemble_from_crop(
                            *crop,
                            force_simple_aug=False,
                        )
                    )
        if self.mode == "train":
            random.shuffle(samples)
        return samples

    def __getitem__(self, index):
        items = self._train_items if self.mode == "train" else self._val_items
        return self._get_item_samples(items[int(index)])

    def __len__(self):
        return len(self._train_items if self.mode == "train" else self._val_items)

    @property
    def summary(self):
        summary = {
            "patients": len({metadata["patient_id"] for metadata in self.case_metadata}),
            "studies": len({
                (metadata["patient_id"], metadata["studyInstanceUID"])
                for metadata in self.case_metadata
            }),
            "series": len(self.case_metadata),
            "zero_annotation_series": len(self.zero_annotation_series),
            "lesions": int(sum(len(boxes) for boxes in self.sample_bboxes)),
            "%s_batches" % self.mode: len(self),
            "gt_boost": bool(self.gt_boost),
        }
        if self.mode == "train":
            summary.update({
                "positive_patients": len(self.positive_patient_ids),
                "selected_series": len(self.selected_train_ct_indices),
                "selected_lesions": self.selected_train_lesions,
                "boost_studies": len(self.boost_study_keys),
            })
        return summary


def _print_split_summary(name, dataset):
    summary = dataset.summary
    print("%s:" % name)
    for key in ("patients", "studies", "series", "zero_annotation_series", "lesions"):
        print("  %s: %d" % (key, summary[key]))
    if name == "train" and "selected_series" in summary:
        print("  positive_patients: %d" % summary["positive_patients"])
        print("  boost_studies: %d" % summary.get("boost_studies", 0))
        print("  gt_boost: %s" % summary.get("gt_boost", False))
        print("  selected_series(epoch=%d): %d" % (dataset.epoch, summary["selected_series"]))
        print("  selected_lesions(epoch=%d): %d" % (dataset.epoch, summary["selected_lesions"]))
    print("  %s_batches: %d" % (name, summary["%s_batches" % name]))


def build_organized_npz_datasets(
    data_root,
    annotation_csv,
    train_list,
    val_list,
    cfg,
    batch_size=8,
    target_spacing_zyx=(1.0, 0.6, 0.6),
    window_min=-1200.0,
    window_max=600.0,
    train_neg_pos_ratio=1.0,
    gt_boost=False,
    gt_boost_factor=DEFAULT_GT_BOOST_FACTOR,
):
    """Build train/val CT-level datasets for competition organized NPZ layout."""
    train_entries = _read_split_entries(train_list)
    val_entries = _read_split_entries(val_list)
    _check_split_isolation(train_entries, val_entries, train_list, val_list)

    train_dataset = OrganizedNPZCTDatasetYibao(
        data_root,
        train_list,
        annotation_csv,
        cfg,
        mode="train",
        batch_size=batch_size,
        target_spacing_zyx=target_spacing_zyx,
        window_min=window_min,
        window_max=window_max,
        split_entries=train_entries,
        train_neg_pos_ratio=train_neg_pos_ratio,
        gt_boost=gt_boost,
        gt_boost_factor=gt_boost_factor,
    )
    val_dataset = OrganizedNPZCTDatasetYibao(
        data_root,
        val_list,
        annotation_csv,
        cfg,
        mode="val",
        batch_size=batch_size,
        target_spacing_zyx=target_spacing_zyx,
        window_min=window_min,
        window_max=window_max,
        split_entries=val_entries,
        train_neg_pos_ratio=train_neg_pos_ratio,
        gt_boost=False,
        gt_boost_factor=gt_boost_factor,
    )
    _print_split_summary("train", train_dataset)
    _print_split_summary("val", val_dataset)
    return train_dataset, val_dataset
