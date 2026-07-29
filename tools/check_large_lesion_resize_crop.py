#!/usr/bin/env python3
"""Check large-lesion resize crop behavior."""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

import numpy as np

from dataset.bbox_reader import Crop, build_patch_truth_boxes, corner_form_to_center_form
from net.layer.rpn_target import assign_rpn_anchors


CFG = {
    "crop_size": [64, 64, 64],
    "bound_size": 0,
    "stride": 4,
    "pad_value": -1,
}


RPN_CFG = {
    "num_neg": 4,
    "rpn_train_bg_thresh_high": 0.02,
    "rpn_train_fg_thresh_low": 0.5,
    "box_reg_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}


def assert_close(actual, expected, name, atol=1e-4):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=atol):
        raise AssertionError("%s: got %s, expected %s" % (name, actual, expected))


def center_boxes(corner_boxes):
    center = np.asarray(corner_form_to_center_form(corner_boxes, 0), dtype=np.float32)
    return center[:, :6], center[:, -1].astype(np.int64)


def check_normal_lesion_does_not_trigger_resize():
    image = np.arange(1 * 96 * 96 * 96, dtype=np.float32).reshape(1, 96, 96, 96)
    target = np.asarray([20, 39, 20, 39, 20, 39], dtype=np.float32)
    boxes = np.asarray([target], dtype=np.float32)
    cropper = Crop(CFG)

    np.random.seed(7)
    sample_a, target_a, boxes_a, _ = cropper(
        image, target, boxes, isScale=False, isRand=False, allow_large_lesion_resize=False
    )
    np.random.seed(7)
    sample_b, target_b, boxes_b, _ = cropper(
        image, target, boxes, isScale=False, isRand=False, allow_large_lesion_resize=True
    )

    if sample_b.shape[1:] != tuple(CFG["crop_size"]):
        raise AssertionError("normal crop shape mismatch: %s" % (sample_b.shape,))
    assert_close(sample_b, sample_a, "normal lesion sample")
    assert_close(target_b, target_a, "normal lesion target")
    assert_close(boxes_b, boxes_a, "normal lesion boxes")


def check_large_lesion_single_axis_resize():
    image = np.zeros((1, 160, 96, 96), dtype=np.float32)
    target = np.asarray([40, 119, 20, 39, 20, 39], dtype=np.float32)
    boxes = np.asarray([target], dtype=np.float32)
    cropper = Crop(CFG)

    sample, _, patch_boxes, _ = cropper(
        image, target, boxes, isScale=False, isRand=False, allow_large_lesion_resize=True
    )
    if sample.shape[1:] != tuple(CFG["crop_size"]):
        raise AssertionError("large crop shape mismatch: %s" % (sample.shape,))
    patch_truth = build_patch_truth_boxes(patch_boxes, CFG["crop_size"], 0.5)
    if patch_truth.shape != (1, 7) or int(patch_truth[0, -1]) != 1:
        raise AssertionError("large target should remain a positive GT, got %s" % patch_truth)
    if patch_truth[0, 0] < 0 or patch_truth[0, 1] > CFG["crop_size"][0] - 1:
        raise AssertionError("large target should be fully inside output patch, got %s" % patch_truth[0])


def check_large_lesion_multi_axis_resize():
    image = np.zeros((1, 180, 180, 120), dtype=np.float32)
    target = np.asarray([40, 119, 30, 109, 20, 39], dtype=np.float32)
    boxes = np.asarray([target], dtype=np.float32)
    cropper = Crop(CFG)

    sample, _, patch_boxes, _ = cropper(
        image, target, boxes, isScale=False, isRand=False, allow_large_lesion_resize=True
    )
    if sample.shape[1:] != tuple(CFG["crop_size"]):
        raise AssertionError("multi-axis large crop shape mismatch: %s" % (sample.shape,))
    patch_truth = build_patch_truth_boxes(patch_boxes, CFG["crop_size"], 0.5)
    if patch_truth.shape != (1, 7) or int(patch_truth[0, -1]) != 1:
        raise AssertionError("multi-axis large target should remain positive, got %s" % patch_truth)


def check_hard_fn_like_large_target_can_resize_without_being_truth():
    image = np.zeros((1, 180, 120, 120), dtype=np.float32)
    hard_fn_target = np.asarray([39, 121, 19, 42, 19, 42], dtype=np.float32)
    original_gt = np.asarray([[40, 119, 20, 39, 20, 39]], dtype=np.float32)
    cropper = Crop(CFG)

    sample, _, patch_boxes, _ = cropper(
        image,
        hard_fn_target,
        original_gt,
        isScale=False,
        isRand=False,
        allow_large_lesion_resize=True,
    )
    if sample.shape[1:] != tuple(CFG["crop_size"]):
        raise AssertionError("hard-FN-like large crop shape mismatch: %s" % (sample.shape,))
    patch_truth = build_patch_truth_boxes(patch_boxes, CFG["crop_size"], 0.5)
    if patch_truth.shape != (1, 7) or int(patch_truth[0, -1]) != 1:
        raise AssertionError("original GT should provide positive supervision for hard-FN crop, got %s" % patch_truth)
    if np.any(np.all(np.isclose(patch_truth[:, :6], hard_fn_target), axis=1)):
        raise AssertionError("hard-FN target box itself should not be inserted as truth")


def check_large_lesion_boundary_padding():
    image = np.zeros((1, 100, 80, 80), dtype=np.float32)
    target = np.asarray([0, 79, 2, 31, 2, 31], dtype=np.float32)
    boxes = np.asarray([target], dtype=np.float32)
    cropper = Crop(CFG)

    sample, _, patch_boxes, _ = cropper(
        image, target, boxes, isScale=False, isRand=False, allow_large_lesion_resize=True
    )
    if sample.shape[1:] != tuple(CFG["crop_size"]):
        raise AssertionError("boundary large crop shape mismatch: %s" % (sample.shape,))
    patch_truth = build_patch_truth_boxes(patch_boxes, CFG["crop_size"], 0.5)
    if patch_truth.shape != (1, 7) or int(patch_truth[0, -1]) != 1:
        raise AssertionError("boundary large target should remain positive, got %s" % patch_truth)
    if patch_truth[0, 0] < -1e-4:
        raise AssertionError("boundary target should not be truncated at z-min, got %s" % patch_truth[0])


def check_other_gt_visible_rules_and_no_original_mutation():
    image = np.zeros((1, 180, 120, 120), dtype=np.float32)
    boxes = np.asarray([
        [50, 129, 20, 40, 20, 40],
        [60, 65, 30, 35, 30, 35],
        [110, 149, 30, 35, 30, 35],
        [120, 159, 30, 35, 30, 35],
        [150, 160, 30, 35, 30, 35],
    ], dtype=np.float32)
    original = boxes.copy()
    cropper = Crop(CFG)

    sample_1, _, patch_boxes_1, _ = cropper(
        image, boxes[0], boxes, isScale=False, isRand=False, allow_large_lesion_resize=True
    )
    sample_2, _, patch_boxes_2, _ = cropper(
        image, boxes[0], boxes, isScale=False, isRand=False, allow_large_lesion_resize=True
    )
    if sample_1.shape[1:] != tuple(CFG["crop_size"]):
        raise AssertionError("multi-GT large crop shape mismatch: %s" % (sample_1.shape,))
    assert_close(boxes, original, "original boxes after repeated large crop")
    assert_close(patch_boxes_1, patch_boxes_2, "repeated large crop boxes")

    patch_truth = build_patch_truth_boxes(patch_boxes_1, CFG["crop_size"], 0.5)
    labels = patch_truth[:, -1].astype(np.int64).tolist()
    if labels.count(1) < 3 or labels.count(-1) < 1:
        raise AssertionError("expected target/full/partial-positive GTs and one ignore GT, got %s" % patch_truth)
    if len(patch_truth) != 4:
        raise AssertionError("non-intersecting GT should be excluded, got %s" % patch_truth)

    truth_boxes, truth_labels = center_boxes(patch_truth)
    positive_gt = truth_boxes[truth_labels > 0]
    window = positive_gt.copy()
    label, label_assign, label_weight, _, _ = assign_rpn_anchors(RPN_CFG, window, truth_boxes, truth_labels)
    assigned = set(label_assign[(label == 1) & (label_weight > 0)].tolist())
    expected = set(range(len(positive_gt)))
    if assigned != expected:
        raise AssertionError("RPN should use transformed positive patch GTs, assigned=%s expected=%s" %
                             (assigned, expected))


def main():
    check_normal_lesion_does_not_trigger_resize()
    check_large_lesion_single_axis_resize()
    check_large_lesion_multi_axis_resize()
    check_hard_fn_like_large_target_can_resize_without_being_truth()
    check_large_lesion_boundary_padding()
    check_other_gt_visible_rules_and_no_original_mutation()
    print("ok: large-lesion resize crop checks passed")


if __name__ == "__main__":
    main()
