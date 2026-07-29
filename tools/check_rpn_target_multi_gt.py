#!/usr/bin/env python3
"""Minimal checks for multi-GT RPN anchor assignment."""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

import numpy as np
import random
import torch

from dataset.bbox_reader import Crop, build_patch_truth_boxes, corner_form_to_center_form
from net.layer.rpn_target import assign_rpn_anchors, make_one_rpn_target
from utils.pybox import torch_overlap


CFG = {
    "num_neg": 3,
    "rpn_train_bg_thresh_high": 0.02,
    "rpn_train_fg_thresh_low": 0.5,
    "box_reg_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}


def as_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def maybe_patch_cuda():
    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *args, **kwargs: self


def center_boxes(corner_boxes):
    center = np.asarray(corner_form_to_center_form(corner_boxes, 0), dtype=np.float32)
    return center[:, :6], center[:, -1].astype(np.int64)


def assign(window, truth_boxes, truth_labels):
    return assign_rpn_anchors(CFG, np.asarray(window, dtype=np.float32), truth_boxes, truth_labels)


def positive_assignments(label, label_assign, label_weight):
    index = np.where((label == 1) & (label_weight > 0))[0]
    return index, label_assign[index]


def assert_each_gt_assigned(label, label_assign, label_weight, gt_indices, name):
    _, assigns = positive_assignments(label, label_assign, label_weight)
    missing = sorted(set(gt_indices).difference(set(assigns.tolist())))
    if missing:
        raise AssertionError("%s: GT(s) %s did not receive a positive anchor; assigns=%s" %
                             (name, missing, assigns))


def assert_no_positive_for_gt(label, label_assign, label_weight, gt_idx, name):
    index, assigns = positive_assignments(label, label_assign, label_weight)
    if gt_idx in assigns.tolist():
        raise AssertionError("%s: ignore GT %d received positive anchors %s" %
                             (name, gt_idx, index[assigns == gt_idx]))


def check_single_complete_gt():
    truth_boxes = np.asarray([[16, 16, 16, 10, 10, 10]], dtype=np.float32)
    truth_labels = np.asarray([1], dtype=np.int64)
    window = np.asarray([[16, 16, 16, 10, 10, 10], [50, 50, 50, 8, 8, 8]], dtype=np.float32)
    label, label_assign, label_weight, _, _ = assign(window, truth_boxes, truth_labels)
    assert_each_gt_assigned(label, label_assign, label_weight, [0], "single complete GT")


def check_multi_complete_gt():
    truth_boxes = np.asarray([[16, 16, 16, 10, 10, 10], [42, 42, 42, 12, 12, 12]], dtype=np.float32)
    truth_labels = np.asarray([1, 1], dtype=np.int64)
    window = np.asarray([
        [16, 16, 16, 10, 10, 10],
        [42, 42, 42, 12, 12, 12],
        [52, 52, 52, 8, 8, 8],
    ], dtype=np.float32)
    label, label_assign, label_weight, _, _ = assign(window, truth_boxes, truth_labels)
    assert_each_gt_assigned(label, label_assign, label_weight, [0, 1], "multi complete GT")


def check_complete_and_partial_positive_gt():
    corner = build_patch_truth_boxes(
        [[10, 19, 10, 19, 10, 19], [-3, 14, 30, 47, 30, 47]],
        [64, 64, 64],
        0.5,
    )
    truth_boxes, truth_labels = center_boxes(corner)
    window = truth_boxes.copy()
    label, label_assign, label_weight, _, _ = assign(window, truth_boxes, truth_labels)
    assert_each_gt_assigned(label, label_assign, label_weight, [0, 1], "complete + partial positive GT")


def check_partial_ignore_gt_has_no_regression():
    maybe_patch_cuda()
    corner = build_patch_truth_boxes(
        [[10, 19, 10, 19, 10, 19], [-15, 4, 30, 47, 30, 47]],
        [64, 64, 64],
        0.5,
    )
    truth_boxes, truth_labels = center_boxes(corner)
    window = np.concatenate([truth_boxes.copy(), [[50, 50, 50, 8, 8, 8]]]).astype(np.float32)
    label, label_assign, label_weight, target, target_weight = make_one_rpn_target(
        CFG, "train", torch.zeros((1, 64, 64, 64), dtype=torch.float32), window, truth_boxes, truth_labels
    )
    label = as_numpy(label)
    label_assign = as_numpy(label_assign)
    label_weight = as_numpy(label_weight)
    target = as_numpy(target)
    target_weight = as_numpy(target_weight)
    assert_each_gt_assigned(label, label_assign, label_weight, [0], "complete + ignore GT")
    if label[1] != 0 or label_weight[1] != 0 or target_weight[1] != 0 or np.any(target[1] != 0):
        raise AssertionError("low-visible GT anchor should be ignored without regression target")
    assert_no_positive_for_gt(label, label_assign, label_weight, 1, "complete + ignore GT")


def check_threshold_positive_anchors_retained():
    truth_boxes = np.asarray([[20, 20, 20, 10, 10, 10]], dtype=np.float32)
    truth_labels = np.asarray([1], dtype=np.int64)
    window = np.asarray([
        [20, 20, 20, 10, 10, 10],
        [21, 20, 20, 10, 10, 10],
        [22, 20, 20, 10, 10, 10],
        [50, 50, 50, 8, 8, 8],
    ], dtype=np.float32)
    overlap = as_numpy(torch_overlap(window, truth_boxes))[:, 0]
    expected = set(np.where(overlap >= CFG["rpn_train_fg_thresh_low"])[0].tolist())
    label, _, label_weight, _, _ = assign(window, truth_boxes, truth_labels)
    actual = set(np.where((label == 1) & (label_weight > 0))[0].tolist())
    if not expected.issubset(actual):
        raise AssertionError("threshold positive anchors dropped: expected %s, actual %s" %
                             (sorted(expected), sorted(actual)))


def check_one_anchor_two_gt_matches_best_iou():
    truth_boxes = np.asarray([[20, 20, 20, 10, 10, 10], [22, 20, 20, 10, 10, 10]], dtype=np.float32)
    truth_labels = np.asarray([1, 1], dtype=np.int64)
    window = np.asarray([[20, 20, 20, 10, 10, 10]], dtype=np.float32)
    label, label_assign, label_weight, _, _ = assign(window, truth_boxes, truth_labels)
    if label[0] != 1 or label_weight[0] <= 0 or label_assign[0] != 0:
        raise AssertionError("single conflicting anchor should match the higher-IoU GT, got label=%s assign=%s weight=%s" %
                             (label[0], label_assign[0], label_weight[0]))


def check_ignore_does_not_clear_positive():
    truth_boxes = np.asarray([[20, 20, 20, 10, 10, 10], [20, 20, 20, 12, 12, 12]], dtype=np.float32)
    truth_labels = np.asarray([1, -1], dtype=np.int64)
    window = np.asarray([[20, 20, 20, 10, 10, 10], [25, 20, 20, 8, 8, 8]], dtype=np.float32)
    label, label_assign, label_weight, target_weight, _ = assign(window, truth_boxes, truth_labels)
    if label[0] != 1 or label_assign[0] != 0 or label_weight[0] <= 0:
        raise AssertionError("ignore GT cleared a valid positive anchor")
    if label[1] != 0 or label_weight[1] != 0 or target_weight[1] != 0:
        raise AssertionError("non-positive anchor overlapping ignore GT should be ignored")


def check_no_valid_gt_keeps_negative_sampling():
    maybe_patch_cuda()
    truth_boxes = np.zeros((0, 6), dtype=np.float32)
    truth_labels = np.zeros((0,), dtype=np.int64)
    window = np.asarray([[10, 10, 10, 8, 8, 8], [30, 30, 30, 8, 8, 8]], dtype=np.float32)
    label, _, label_weight, _, target_weight = make_one_rpn_target(
        CFG, "train", torch.zeros((1, 64, 64, 64), dtype=torch.float32), window, truth_boxes, truth_labels
    )
    label = as_numpy(label)
    label_weight = as_numpy(label_weight)
    target_weight = as_numpy(target_weight)
    if np.any(label == 1) or np.any(target_weight != 0) or not np.any(label_weight > 0):
        raise AssertionError("empty-GT patch should have no positives and keep sampled negatives")


def check_hard_fn_center_uses_original_gt_only():
    cfg = {"crop_size": [64, 64, 64], "bound_size": 12, "stride": 4, "pad_value": 0}
    cropper = Crop(cfg)
    image = np.zeros((1, 128, 128, 128), dtype=np.float32)
    original_gt = np.asarray([
        [50, 59, 50, 59, 50, 59],
        [70, 79, 70, 79, 70, 79],
    ], dtype=np.float32)
    hard_fn_pred_box = np.asarray([58, 66, 58, 66, 58, 66], dtype=np.float32)
    hard_fn_center = hard_fn_pred_box.reshape(3, 2).mean(axis=1)
    _, _, shifted_gt, _ = cropper.crop_at_center(image, hard_fn_center, original_gt)
    corner = build_patch_truth_boxes(shifted_gt, cfg["crop_size"], 0.5)
    truth_boxes, truth_labels = center_boxes(corner)
    if len(truth_boxes) != 2:
        raise AssertionError("hard-FN centered patch should include only the two original GTs, got %s" % truth_boxes)
    window = truth_boxes.copy()
    label, label_assign, label_weight, _, _ = assign(window, truth_boxes, truth_labels)
    assert_each_gt_assigned(label, label_assign, label_weight, [0, 1], "hard-FN centered multi-GT patch")


def main():
    np.random.seed(0)
    random.seed(0)
    check_single_complete_gt()
    check_multi_complete_gt()
    check_complete_and_partial_positive_gt()
    check_partial_ignore_gt_has_no_regression()
    check_threshold_positive_anchors_retained()
    check_one_anchor_two_gt_matches_best_iou()
    check_ignore_does_not_clear_positive()
    check_no_valid_gt_keeps_negative_sampling()
    check_hard_fn_center_uses_original_gt_only()
    print("ok: multi-GT RPN target assignment checks passed")


if __name__ == "__main__":
    main()
