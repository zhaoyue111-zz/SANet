#!/usr/bin/env python3
"""Checks for GT-size adaptive RPN foreground IoU thresholds."""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch

from net.layer.rpn_nms import rpn_encode
from net.layer.rpn_target import assign_rpn_anchors, make_one_rpn_target
from utils.pybox import torch_overlap


CFG = {
    "num_neg": 8,
    "rpn_train_bg_thresh_high": 0.02,
    "rpn_train_fg_thresh_low": 0.5,
    "rpn_train_adaptive_fg_thresh": True,
    "rpn_train_small_gt_max_side": 10.0,
    "rpn_train_small_gt_fg_thresh_low": 0.2,
    "bbox_border": 8.0,
    "box_reg_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}


def maybe_patch_cuda():
    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *args, **kwargs: self


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def iou(window, truth_box):
    return as_numpy(torch_overlap(np.asarray(window, np.float32), np.asarray(truth_box, np.float32)))


def check_small_gt_low_iou_anchor_is_positive():
    truth = np.asarray([[20, 20, 20, 16, 16, 16]], dtype=np.float32)
    labels = np.asarray([1], dtype=np.int64)
    window = np.asarray([
        [20, 20, 20, 16, 16, 16],
        [28, 20, 20, 16, 16, 16],
    ], dtype=np.float32)
    overlap = iou(window, truth)[:, 0]
    if not (0.2 <= overlap[1] < 0.5):
        raise AssertionError("test setup expected IoU in [0.2, 0.5), got %.6f" % overlap[1])
    label, assign, weight, _, _ = assign_rpn_anchors(CFG, window, truth, labels)
    if label[1] != 1 or weight[1] <= 0 or assign[1] != 0:
        raise AssertionError("small-GT anchor should be positive under adaptive threshold")


def check_raw_equal_10_uses_small_threshold():
    truth = np.asarray([[20, 20, 20, 18, 18, 18]], dtype=np.float32)
    labels = np.asarray([1], dtype=np.int64)
    window = np.asarray([
        [20, 20, 20, 18, 18, 18],
        [29, 20, 20, 18, 18, 18],
    ], dtype=np.float32)
    overlap = iou(window, truth)[:, 0]
    if not (0.2 <= overlap[1] < 0.5):
        raise AssertionError("test setup expected IoU in [0.2, 0.5), got %.6f" % overlap[1])
    label, _, weight, _, _ = assign_rpn_anchors(CFG, window, truth, labels)
    if label[1] != 1 or weight[1] <= 0:
        raise AssertionError("raw max_side=10 should use small threshold")


def check_large_gt_low_iou_anchor_not_adaptive_positive():
    truth = np.asarray([[20, 20, 20, 20, 20, 20]], dtype=np.float32)
    labels = np.asarray([1], dtype=np.int64)
    window = np.asarray([
        [20, 20, 20, 20, 20, 20],
        [30.5, 20, 20, 20, 20, 20],
    ], dtype=np.float32)
    overlap = iou(window, truth)[:, 0]
    if not (0.2 <= overlap[1] < 0.5):
        raise AssertionError("test setup expected IoU in [0.2, 0.5), got %.6f" % overlap[1])
    label, _, weight, _, _ = assign_rpn_anchors(CFG, window, truth, labels)
    if label[1] != 0 or weight[1] != 0:
        raise AssertionError("large-GT IoU<base threshold anchor should not become positive")


def check_mixed_gt_uses_each_gt_threshold():
    truth = np.asarray([
        [20, 20, 20, 16, 16, 16],
        [70, 70, 70, 20, 20, 20],
    ], dtype=np.float32)
    labels = np.asarray([1, 1], dtype=np.int64)
    window = np.asarray([
        [20, 20, 20, 16, 16, 16],
        [28, 20, 20, 16, 16, 16],
        [70, 70, 70, 20, 20, 20],
        [80.5, 70, 70, 20, 20, 20],
    ], dtype=np.float32)
    label, assign, weight, _, _ = assign_rpn_anchors(CFG, window, truth, labels)
    if label[1] != 1 or weight[1] <= 0 or assign[1] != 0:
        raise AssertionError("small GT in mixed patch should use small threshold")
    if label[3] != 0 or weight[3] != 0:
        raise AssertionError("large GT in mixed patch should keep base threshold")


def check_regression_target_matches_assigned_gt():
    maybe_patch_cuda()
    truth = np.asarray([
        [20, 20, 20, 16, 16, 16],
        [70, 70, 70, 20, 20, 20],
    ], dtype=np.float32)
    labels = np.asarray([1, 1], dtype=np.int64)
    window = np.asarray([
        [20, 20, 20, 16, 16, 16],
        [28, 20, 20, 16, 16, 16],
        [70, 70, 70, 20, 20, 20],
    ], dtype=np.float32)
    label, assign, weight, target, target_weight = make_one_rpn_target(
        CFG,
        "valid",
        torch.zeros((1, 96, 96, 96), dtype=torch.float32),
        window,
        truth,
        labels,
    )
    label = as_numpy(label)
    assign = as_numpy(assign)
    weight = as_numpy(weight)
    target = as_numpy(target)
    target_weight = as_numpy(target_weight)
    if label[1] != 1 or weight[1] <= 0 or target_weight[1] <= 0 or assign[1] != 0:
        raise AssertionError("adaptive positive anchor should be assigned to small GT")
    expected = rpn_encode(window[[1]], truth[[assign[1]]], CFG["box_reg_weight"])[0]
    if not np.allclose(target[1], expected):
        raise AssertionError("regression target does not match assigned GT: got %s expected %s" %
                             (target[1], expected))


def check_switch_off_uses_original_threshold():
    cfg = dict(CFG)
    cfg["rpn_train_adaptive_fg_thresh"] = False
    truth = np.asarray([[20, 20, 20, 16, 16, 16]], dtype=np.float32)
    labels = np.asarray([1], dtype=np.int64)
    window = np.asarray([
        [20, 20, 20, 16, 16, 16],
        [28, 20, 20, 16, 16, 16],
    ], dtype=np.float32)
    label, _, weight, _, _ = assign_rpn_anchors(cfg, window, truth, labels)
    if label[1] != 0 or weight[1] != 0:
        raise AssertionError("adaptive switch off should keep original fixed foreground threshold")


def main():
    check_small_gt_low_iou_anchor_is_positive()
    check_raw_equal_10_uses_small_threshold()
    check_large_gt_low_iou_anchor_not_adaptive_positive()
    check_mixed_gt_uses_each_gt_threshold()
    check_regression_target_matches_assigned_gt()
    check_switch_off_uses_original_threshold()
    print("ok: RPN adaptive foreground threshold checks passed")


if __name__ == "__main__":
    main()
