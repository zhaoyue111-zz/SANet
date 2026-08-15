#!/usr/bin/env python3
"""CT-centric train batch dataset for I/O speed experiments.

One dataset item = one positive CT = one full batch, with a single np.load.
Enabled via train_local.py --sample-by-ct (train only). Val/eval/test keep BboxReader.
"""

from __future__ import annotations

import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from dataset.bbox_reader import (
    Crop,
    _load_hard_fns,
    _load_hard_fps,
    _pid_keys,
    augment,
    augment_intensity,
    build_patch_truth_boxes,
    corner_form_to_center_form,
)


def _clamp_neg_pos_ratio(ratio):
    ratio = float(ratio)
    return min(1.0, max(1.0 / 3.0, ratio))


def split_pos_neg_counts(batch_size, neg_pos_ratio):
    """Return (n_pos, n_neg) with n_pos + n_neg == batch_size."""
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ratio = _clamp_neg_pos_ratio(neg_pos_ratio)
    n_neg = int(round(batch_size * ratio / (1.0 + ratio)))
    n_pos = batch_size - n_neg
    if batch_size % 2 == 1 and n_pos <= n_neg:
        n_pos = n_neg + 1
        n_neg = batch_size - n_pos
    if n_pos <= 0:
        n_pos = 1
        n_neg = batch_size - 1
    if n_neg < 0:
        n_neg = 0
        n_pos = batch_size
    return n_pos, n_neg


def sample_pool_indices(pool_size, n):
    """Sample n indices: without replacement if enough; else all once + extras with simple-aug."""
    pool_size = int(pool_size)
    n = int(n)
    if pool_size <= 0 or n <= 0:
        return []
    if pool_size >= n:
        chosen = np.random.choice(pool_size, size=n, replace=False)
        return [(int(j), False) for j in chosen]
    order = np.random.permutation(pool_size)
    picks = [(int(j), False) for j in order]
    for _ in range(n - pool_size):
        picks.append((int(np.random.randint(pool_size)), True))
    return picks


class CTBatchDataset(Dataset):
    """Train-only: ``__len__`` = #positive CTs; each item loads one CT and returns a full batch."""

    def __init__(self, data_dir, set_name, cfg, mode='train', batch_size=8):
        if mode != 'train':
            raise ValueError(
                "CTBatchDataset is train-only (val must use BboxReader); got mode=%s" % mode
            )
        self.mode = mode
        self.cfg = cfg
        self.batch_size = int(batch_size)
        self.data_dir = data_dir
        self.set_name = set_name
        self.dataset_name = cfg.get('dataset', '')
        self.augtype = cfg['augtype']
        self.pad_value = cfg['pad_value']
        self.partial_gt_positive_threshold = float(cfg.get('partial_gt_positive_threshold', 0.5))
        self.neg_pos_ratio = _clamp_neg_pos_ratio(cfg.get('train_neg_pos_ratio', 1.0))
        self.crop = Crop(cfg)

        with open(self.set_name, "r") as f:
            self.filenames = [line.strip() for line in f if line.strip()]
        if not self.filenames:
            raise ValueError("Dataset list is empty: %s" % self.set_name)

        blacklist = cfg.get('blacklist', [])
        self.filenames = [fn for fn in self.filenames if fn not in blacklist]

        missing_images = [
            fn for fn in self.filenames
            if not os.path.isfile(os.path.join(self.data_dir, '%s_zoom.npy' % fn))
        ]
        if missing_images and cfg.get('skip_missing', False):
            missing_set = set(missing_images)
            print(
                "[%s] Skipping %d missing preprocessed image(s) under %s"
                % (self.dataset_name or self.set_name, len(missing_images), self.data_dir)
            )
            self.filenames = [fn for fn in self.filenames if fn not in missing_set]
        elif missing_images:
            preview = ', '.join(missing_images[:5])
            raise FileNotFoundError(
                "Missing %d preprocessed image(s) under %s, including: %s"
                % (len(missing_images), self.data_dir, preview)
            )
        if not self.filenames:
            raise ValueError("No available preprocessed images for %s" % self.set_name)

        annos_all = pd.read_csv(cfg['train_anno'])
        filename_to_idx = {}
        for i, fn in enumerate(self.filenames):
            for key in _pid_keys(fn):
                filename_to_idx[key] = i

        labels = []
        for fn in self.filenames:
            annos = annos_all[annos_all['pid'] == int(fn)]
            temp_annos = []
            if len(annos) > 0:
                for index in range(len(annos)):
                    anno = annos.iloc[index]
                    temp_annos.append([
                        anno['zmin'], anno['zmax'], anno['ymin'], anno['ymax'],
                        anno['xmin'], anno['xmax'],
                    ])
            l = np.array(temp_annos, dtype=np.float32)
            if l.size == 0 or np.all(l == 0):
                l = np.zeros((0, 6), dtype=np.float32)
            labels.append(l)
        self.sample_bboxes = labels

        # Positive sources: GT + hard FN (same as BboxReader).
        self.bboxes = []
        for i, l in enumerate(labels):
            if len(l) > 0:
                for t in l:
                    self.bboxes.append([np.concatenate([[i], t])])
        hard_fns = _load_hard_fns(cfg.get('hard_fn_csv'), self.dataset_name, filename_to_idx)
        for sample in hard_fns:
            self.bboxes.append([sample])
        if not self.bboxes:
            raise ValueError(
                "No annotated boxes found for %s split %s"
                % (self.dataset_name or 'dataset', self.set_name)
            )
        self.bboxes = np.concatenate(self.bboxes, axis=0).astype(np.float32)

        self.hard_fps_by_ct = {i: [] for i in range(len(self.filenames))}
        hard_fps = _load_hard_fps(
            cfg.get('hard_fp_csv'),
            self.dataset_name,
            filename_to_idx,
            float(cfg.get('hard_fp_threshold', 0.9)),
        )
        for fp in hard_fps:
            ct_idx = int(fp[0])
            if 0 <= ct_idx < len(self.filenames):
                self.hard_fps_by_ct[ct_idx].append(np.asarray(fp[1:4], dtype=np.float32))

        self.positives_by_ct = {i: [] for i in range(len(self.filenames))}
        for row in self.bboxes:
            ct_idx = int(row[0])
            self.positives_by_ct[ct_idx].append(np.asarray(row[1:7], dtype=np.float32))

        self.pos_ct_indices = [i for i, boxes in self.positives_by_ct.items() if len(boxes) > 0]
        if not self.pos_ct_indices:
            raise ValueError("No CT with positive samples for %s" % self.set_name)

        self.n_pos, self.n_neg = split_pos_neg_counts(self.batch_size, self.neg_pos_ratio)

        print(
            "[%s] CTBatchDataset positive_cts=%d batch_size=%d n_pos=%d n_neg=%d"
            % (
                self.dataset_name or self.set_name,
                len(self.pos_ct_indices),
                self.batch_size,
                self.n_pos,
                self.n_neg,
            )
        )

    def __len__(self):
        return len(self.pos_ct_indices)

    def load_image(self, filename):
        path = os.path.join(self.data_dir, '%s_zoom.npy' % filename)
        last_exc = None
        for attempt in range(3):
            try:
                return np.load(path)
            except (EOFError, OSError, ValueError) as exc:
                last_exc = exc
                time.sleep(0.2 * (attempt + 1) + random.random() * 0.1)
        size = os.path.getsize(path) if os.path.exists(path) else -1
        raise RuntimeError(
            "Failed to load npy for dataset=%s, mode=%s, pid=%s, path=%s, size=%d"
            % (self.dataset_name or 'unknown', self.mode, filename, path, size)
        ) from last_exc

    def _enabled_simple_augs(self):
        choices = []
        if self.augtype.get('flip', False):
            choices.append('flip')
        if self.augtype.get('rotate', False):
            choices.append('rotate')
        if self.augtype.get('intensity', False):
            choices.append('intensity')
        if self.augtype.get('noise', False):
            choices.append('noise')
        return choices

    def _apply_simple_aug(self, sample, target, bboxes):
        choices = self._enabled_simple_augs()
        if not choices:
            return sample, target, bboxes
        choice = random.choice(choices)
        if choice == 'flip':
            sample, target, bboxes = augment(
                sample, target, bboxes,
                do_flip=True, do_rotate=False, do_swap=False,
                pad_value=self.pad_value,
                rotate_angle_range=self.cfg.get('rotate_angle_range', 10.0),
            )
        elif choice == 'rotate':
            sample, target, bboxes = augment(
                sample, target, bboxes,
                do_flip=False, do_rotate=True, do_swap=False,
                pad_value=self.pad_value,
                rotate_angle_range=self.cfg.get('rotate_angle_range', 10.0),
            )
        elif choice == 'intensity':
            sample = augment_intensity(
                sample, self.cfg.get('intensity_aug', {}),
                do_intensity=True, do_noise=False, do_blur=False,
            )
        else:
            sample = augment_intensity(
                sample, self.cfg.get('intensity_aug', {}),
                do_intensity=False, do_noise=True, do_blur=False,
            )
        return sample, target, bboxes

    def _apply_full_aug(self, sample, target, bboxes, is_random_crop):
        if not is_random_crop:
            sample, target, bboxes = augment(
                sample, target, bboxes,
                do_flip=self.augtype.get('flip', False),
                do_rotate=self.augtype.get('rotate', False),
                do_swap=self.augtype.get('swap', False),
                pad_value=self.pad_value,
                rotate_angle_range=self.cfg.get('rotate_angle_range', 10.0),
            )
        sample = augment_intensity(
            sample,
            self.cfg.get('intensity_aug', {}),
            do_intensity=self.augtype.get('intensity', False),
            do_noise=self.augtype.get('noise', False),
            do_blur=self.augtype.get('blur', False),
        )
        return sample, target, bboxes

    def _finalize_sample(self, sample, bboxes):
        sample = (sample.astype(np.float32) - 128) / 128
        bboxes = build_patch_truth_boxes(
            bboxes, sample.shape[1:], self.partial_gt_positive_threshold
        )
        bboxes = corner_form_to_center_form(bboxes, self.cfg['bbox_border'])
        bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 7)
        truth_labels = bboxes[:, -1].astype(np.int64)
        truth_bboxes = bboxes[:, :-1]
        return [torch.from_numpy(sample).float(), truth_bboxes, truth_labels]

    def _crop_positive(self, imgs, ct_idx, lesion_box):
        lesion_box = np.asarray(lesion_box, dtype=np.float32)
        all_boxes = np.asarray(self.sample_bboxes[ct_idx], dtype=np.float32).reshape(-1, 6)
        is_scale = self.augtype.get('scale', False)
        allow_large = bool(self.cfg.get('large_lesion_resize', True))
        sample, target, bboxes, _coord = self.crop(
            imgs, lesion_box, all_boxes,
            isScale=is_scale, isRand=False, allow_large_lesion_resize=allow_large,
        )
        return sample, target, bboxes, False

    def _crop_hard_fp(self, imgs, ct_idx, center_zyx):
        all_boxes = np.asarray(self.sample_bboxes[ct_idx], dtype=np.float32).reshape(-1, 6)
        sample, target, bboxes, _coord = self.crop.crop_at_center(imgs, center_zyx, all_boxes)
        return sample, target, bboxes, True

    def _crop_random_bg(self, imgs, ct_idx):
        all_boxes = np.asarray(self.sample_bboxes[ct_idx], dtype=np.float32).reshape(-1, 6)
        sample, target, bboxes, _coord = self.crop(
            imgs, [], all_boxes, isScale=False, isRand=True, allow_large_lesion_resize=False,
        )
        return sample, target, bboxes, True

    def _assemble_from_crop(self, sample, target, bboxes, is_random_crop, force_simple_aug):
        sample = np.array(sample, copy=True)
        target = np.array(target, copy=True)
        bboxes = np.array(bboxes, copy=True)
        if force_simple_aug:
            sample, target, bboxes = self._apply_simple_aug(sample, target, bboxes)
        sample, target, bboxes = self._apply_full_aug(sample, target, bboxes, is_random_crop)
        return self._finalize_sample(sample, bboxes)

    def __getitem__(self, index):
        ct_idx = int(self.pos_ct_indices[int(index)])
        imgs = self.load_image(self.filenames[ct_idx])

        pos_pool = self.positives_by_ct[ct_idx]
        hard_fps = self.hard_fps_by_ct.get(ct_idx, [])

        samples = []
        for pool_i, force_simple in sample_pool_indices(len(pos_pool), self.n_pos):
            crop = self._crop_positive(imgs, ct_idx, pos_pool[pool_i])
            samples.append(self._assemble_from_crop(*crop, force_simple_aug=force_simple))

        # Negatives: current-CT hard FP first, then same-CT random background.
        if self.n_neg > 0:
            if len(hard_fps) >= self.n_neg:
                for pool_i, _force in sample_pool_indices(len(hard_fps), self.n_neg):
                    crop = self._crop_hard_fp(imgs, ct_idx, hard_fps[pool_i])
                    samples.append(self._assemble_from_crop(*crop, force_simple_aug=False))
            else:
                for center in hard_fps:
                    crop = self._crop_hard_fp(imgs, ct_idx, center)
                    samples.append(self._assemble_from_crop(*crop, force_simple_aug=False))
                for _ in range(self.n_neg - len(hard_fps)):
                    crop = self._crop_random_bg(imgs, ct_idx)
                    samples.append(self._assemble_from_crop(*crop, force_simple_aug=False))

        random.shuffle(samples)
        return samples


def build_ct_batch_datasets(dataset_name, batch_size, hard_fp_csv=None, hard_fn_csv=None,
                            hard_fp_threshold=0.9, train_neg_pos_ratio=1.0):
    """Build train-only CTBatchDataset (or ConcatDataset for dataset=all)."""
    from config import dataset_configs

    cfgs = dataset_configs(dataset_name, skip_missing=(dataset_name == 'all'))
    datasets = []
    for dataset_cfg in cfgs:
        dataset_cfg = dict(dataset_cfg)
        dataset_cfg.update({
            'hard_fp_csv': hard_fp_csv,
            'hard_fn_csv': hard_fn_csv,
            'hard_fp_threshold': hard_fp_threshold,
            'train_neg_pos_ratio': train_neg_pos_ratio,
        })
        try:
            datasets.append(
                CTBatchDataset(
                    dataset_cfg['DATA_DIR'],
                    dataset_cfg['train_set_list'],
                    dataset_cfg,
                    mode='train',
                    batch_size=batch_size,
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            if dataset_name != 'all':
                raise
            print("[%s] Skip train split: %s" % (dataset_cfg['dataset'], exc))
    if not datasets:
        raise ValueError("No usable CTBatchDataset for train in dataset=%s" % dataset_name)
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)
