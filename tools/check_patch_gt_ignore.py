#!/usr/bin/env python3
"""Minimal checks for patch GT clipping and ignore-label target handling."""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch

from dataset.bbox_reader import Crop, build_patch_truth_boxes, corner_form_to_center_form
from net.layer.rpn_target import make_one_rpn_target


def assert_close(actual, expected, name):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape or not np.allclose(actual, expected):
        raise AssertionError("%s: got %s, expected %s" % (name, actual, expected))


def check_complete_gt():
    out = build_patch_truth_boxes([[10, 20, 10, 20, 10, 20]], [64, 64, 64], 0.5)
    assert out.shape == (1, 7)
    assert_close(out[0], [10, 20, 10, 20, 10, 20, 1], "complete GT")


def check_partial_positive_gt():
    out = build_patch_truth_boxes([[-5, 14, 10, 29, 10, 29]], [64, 64, 64], 0.5)
    assert out.shape == (1, 7)
    assert_close(out[0], [0, 14, 10, 29, 10, 29, 1], "partial positive GT")


def check_partial_ignore_gt():
    out = build_patch_truth_boxes([[-15, 4, 10, 29, 10, 29]], [64, 64, 64], 0.5)
    assert out.shape == (1, 7)
    assert_close(out[0], [0, 4, 10, 29, 10, 29, -1], "partial ignore GT")


def check_hard_fp_center_with_gt():
    cfg = {
        "crop_size": [64, 64, 64],
        "bound_size": 12,
        "stride": 4,
        "pad_value": 0,
    }
    cropper = Crop(cfg)
    image = np.zeros((1, 128, 128, 128), dtype=np.float32)
    original_gt = np.asarray([[50, 60, 50, 60, 50, 60]], dtype=np.float32)
    hard_fp_center = np.asarray([55, 55, 55], dtype=np.float32)
    _, _, shifted_gt, _ = cropper.crop_at_center(image, hard_fp_center, original_gt)
    out = build_patch_truth_boxes(shifted_gt, cfg["crop_size"], 0.5)
    assert out.shape == (1, 7)
    assert_close(out[0], [27, 37, 27, 37, 27, 37, 1], "hard FP centered patch GT")


def check_no_intersection():
    out = build_patch_truth_boxes([[70, 80, 10, 20, 10, 20]], [64, 64, 64], 0.5)
    if out.shape != (0, 7):
        raise AssertionError("non-intersecting GT should be ignored entirely, got %s" % out)


def check_low_visible_gt_not_negative_anchor():
    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *args, **kwargs: self

    corner = build_patch_truth_boxes([[-15, 4, 10, 29, 10, 29]], [64, 64, 64], 0.5)
    center = np.asarray(corner_form_to_center_form(corner, 0), dtype=np.float32)
    truth_boxes = center[:, :6]
    truth_labels = center[:, -1].astype(np.int64)
    window = np.concatenate([
        truth_boxes.copy(),
        np.asarray([[50, 50, 50, 5, 5, 5]], dtype=np.float32),
    ], axis=0)
    input_tensor = torch.zeros((1, 64, 64, 64), dtype=torch.float32)
    cfg = {
        "num_neg": 10,
        "rpn_train_bg_thresh_high": 0.02,
        "rpn_train_fg_thresh_low": 0.5,
        "box_reg_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }
    label, _, label_weight, _, target_weight = make_one_rpn_target(
        cfg,
        "train",
        input_tensor,
        window,
        truth_boxes,
        truth_labels,
    )
    if float(label_weight[0].detach().cpu()) != 0.0:
        raise AssertionError("low-visible GT anchor label_weight should be 0, got %s" % label_weight)
    if float(target_weight[0].detach().cpu()) != 0.0:
        raise AssertionError("low-visible GT anchor target_weight should be 0, got %s" % target_weight)
    if int(label[0].detach().cpu()) != 0:
        raise AssertionError("ignored anchor should not be labeled positive")
    if float(label_weight[1].detach().cpu()) <= 0.0:
        raise AssertionError("far background anchor should remain eligible as negative")


def main():
    check_complete_gt()
    check_partial_positive_gt()
    check_partial_ignore_gt()
    check_hard_fp_center_with_gt()
    check_no_intersection()
    check_low_visible_gt_not_negative_anchor()
    print("ok: patch GT clipping and ignore-anchor handling checks passed")


if __name__ == "__main__":
    main()
