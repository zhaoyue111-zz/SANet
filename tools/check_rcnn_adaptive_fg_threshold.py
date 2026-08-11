#!/usr/bin/env python3
"""Checks for RCNN foreground assignment using RPN adaptive thresholds."""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch

import net.layer.rcnn_target as rcnn_target
from net.layer.rcnn_nms import rcnn_encode
from net.layer.rcnn_target import make_one_rcnn_target


CFG = {
    "num_class": 2,
    "rcnn_train_batch_size": 4,
    "rcnn_train_fg_fraction": 0.5,
    "rcnn_train_bg_thresh_high": 0.1,
    "rcnn_train_fg_thresh_low": 0.5,
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


def iou(boxes, truth):
    overlap = rcnn_target.torch_overlap(
        np.asarray(boxes, dtype=np.float32),
        np.asarray(truth, dtype=np.float32),
    )
    return as_numpy(overlap)


def proposal_row(box, score=1.0, batch=0):
    row = np.zeros((8,), dtype=np.float32)
    row[0] = batch
    row[1] = score
    row[2:8] = np.asarray(box, dtype=np.float32)
    return row


def make_target(proposals, truth, labels=None, ignore=None, cfg=None):
    maybe_patch_cuda()
    if labels is None:
        labels = np.ones((len(truth),), dtype=np.int64)
    if ignore is None:
        ignore = np.zeros((0, 6), dtype=np.float32)
    return make_one_rcnn_target(
        cfg or CFG,
        torch.zeros((1, 96, 96, 96), dtype=torch.float32),
        np.asarray(proposals, dtype=np.float32).reshape(-1, 8),
        np.asarray(truth, dtype=np.float32).reshape(-1, 6),
        np.asarray(labels, dtype=np.int64).reshape(-1),
        np.asarray(ignore, dtype=np.float32).reshape(-1, 6),
    )


def check_small_gt_iou_03_is_foreground_with_regression_target():
    truth = np.asarray([[20, 20, 20, 16, 16, 16]], dtype=np.float32)
    fg_box = np.asarray([28, 20, 20, 16, 16, 16], dtype=np.float32)
    bg_box = np.asarray([60, 60, 60, 16, 16, 16], dtype=np.float32)
    overlap = iou([fg_box], truth)[0, 0]
    if not (0.2 <= overlap < 0.5):
        raise AssertionError("test setup expected IoU in [0.2, 0.5), got %.6f" % overlap)

    proposals, labels, assigns, targets = make_target(
        [proposal_row(fg_box), proposal_row(bg_box)],
        truth,
    )
    labels = as_numpy(labels)
    assigns = np.asarray(assigns)
    proposals = as_numpy(proposals)
    targets = as_numpy(targets)

    fg = np.where(labels > 0)[0]
    if len(fg) != 1 or assigns[fg[0]] != 0:
        raise AssertionError("small GT IoU≈0.3 proposal should be RCNN foreground")
    if targets.shape[1] != 6:
        raise AssertionError("RCNN bbox regression target must have 6 columns")
    expected = rcnn_encode(proposals[fg, 2:8], truth[[0]], CFG["box_reg_weight"])
    if not np.allclose(targets, expected):
        raise AssertionError("foreground regression target mismatch")


def check_large_gt_iou_03_is_not_foreground():
    truth = np.asarray([[20, 20, 20, 20, 20, 20]], dtype=np.float32)
    candidate = np.asarray([30.5, 20, 20, 20, 20, 20], dtype=np.float32)
    bg_box = np.asarray([70, 70, 70, 20, 20, 20], dtype=np.float32)
    overlap = iou([candidate], truth)[0, 0]
    if not (0.2 <= overlap < 0.5):
        raise AssertionError("test setup expected IoU in [0.2, 0.5), got %.6f" % overlap)

    _, labels, _, _ = make_target([proposal_row(candidate), proposal_row(bg_box)], truth)
    if np.any(as_numpy(labels) > 0):
        raise AssertionError("large GT IoU≈0.3 proposal should not be RCNN foreground")


def check_small_gt_iou_below_01_is_background():
    truth = np.asarray([[20, 20, 20, 16, 16, 16]], dtype=np.float32)
    bg_box = np.asarray([60, 60, 60, 16, 16, 16], dtype=np.float32)
    overlap = iou([bg_box], truth)[0, 0]
    if overlap >= 0.1:
        raise AssertionError("test setup expected IoU < 0.1, got %.6f" % overlap)

    _, labels, assigns, targets = make_target([proposal_row(bg_box)], truth)
    labels = as_numpy(labels)
    if not len(labels) or np.any(labels != 0) or np.any(np.asarray(assigns) != -1):
        raise AssertionError("IoU<0.1 proposal should be RCNN background")
    if as_numpy(targets).shape != (0, 6):
        raise AssertionError("background-only sample should not create fg regression targets")


def check_large_gt_iou_05_is_foreground():
    truth = np.asarray([[20, 20, 20, 20, 20, 20]], dtype=np.float32)
    fg_box = np.asarray([23, 20, 20, 20, 20, 20], dtype=np.float32)
    bg_box = np.asarray([70, 70, 70, 20, 20, 20], dtype=np.float32)
    overlap = iou([fg_box], truth)[0, 0]
    if overlap < 0.5:
        raise AssertionError("test setup expected IoU >= 0.5, got %.6f" % overlap)

    _, labels, assigns, targets = make_target([proposal_row(fg_box), proposal_row(bg_box)], truth)
    labels = as_numpy(labels)
    fg = np.where(labels > 0)[0]
    if len(fg) != 1 or np.asarray(assigns)[fg[0]] != 0:
        raise AssertionError("large GT IoU>=0.5 proposal should be RCNN foreground")
    if as_numpy(targets).shape[1] != 6:
        raise AssertionError("foreground regression target must be 6D")


def check_no_gt_returns_background():
    bg_box = np.asarray([60, 60, 60, 16, 16, 16], dtype=np.float32)
    _, labels, assigns, targets = make_target([proposal_row(bg_box)], np.zeros((0, 6), np.float32))
    if not len(as_numpy(labels)) or np.any(as_numpy(labels) != 0):
        raise AssertionError("no-GT proposals should be sampled as background")
    if np.any(np.asarray(assigns) != -1) or as_numpy(targets).shape != (0, 6):
        raise AssertionError("no-GT background should have no GT assignment or fg target")


def check_ignore_gt_filters_background():
    ignore = np.asarray([[60, 60, 60, 16, 16, 16]], dtype=np.float32)
    _, labels, _, _ = make_target(
        [proposal_row(ignore[0])],
        np.zeros((0, 6), np.float32),
        ignore=ignore,
    )
    if len(as_numpy(labels)) != 0:
        raise AssertionError("proposal overlapping ignore GT should not be sampled as background")


def check_only_foreground_and_only_background_edges():
    truth = np.asarray([[20, 20, 20, 16, 16, 16]], dtype=np.float32)
    fg_box = np.asarray([28, 20, 20, 16, 16, 16], dtype=np.float32)
    proposals, labels, assigns, targets = make_target([proposal_row(fg_box)], truth)
    if len(as_numpy(labels)) != 1 or as_numpy(labels)[0] <= 0 or np.asarray(assigns)[0] != 0:
        raise AssertionError("only-foreground case should still sample foreground")
    if as_numpy(targets).shape != (1, 6):
        raise AssertionError("only-foreground case should produce one 6D regression target")

    bg_box = np.asarray([60, 60, 60, 16, 16, 16], dtype=np.float32)
    _, labels, assigns, targets = make_target([proposal_row(bg_box)], truth)
    labels = as_numpy(labels)
    assigns = np.asarray(assigns)
    if not len(labels) or np.any(labels != 0) or np.any(assigns != -1):
        raise AssertionError("only-background case should still sample background")
    if as_numpy(targets).shape != (0, 6):
        raise AssertionError("only-background case should not produce fg regression targets")


def check_adaptive_switch_off_uses_base_threshold():
    cfg = dict(CFG)
    cfg["rpn_train_adaptive_fg_thresh"] = False
    truth = np.asarray([[20, 20, 20, 16, 16, 16]], dtype=np.float32)
    candidate = np.asarray([28, 20, 20, 16, 16, 16], dtype=np.float32)
    bg_box = np.asarray([60, 60, 60, 16, 16, 16], dtype=np.float32)
    _, labels, _, _ = make_target([proposal_row(candidate), proposal_row(bg_box)], truth, cfg=cfg)
    if np.any(as_numpy(labels) > 0):
        raise AssertionError("adaptive switch off should keep base foreground threshold")


def main():
    check_small_gt_iou_03_is_foreground_with_regression_target()
    check_large_gt_iou_03_is_not_foreground()
    check_small_gt_iou_below_01_is_background()
    check_large_gt_iou_05_is_foreground()
    check_no_gt_returns_background()
    check_ignore_gt_filters_background()
    check_only_foreground_and_only_background_edges()
    check_adaptive_switch_off_uses_base_threshold()
    print("ok: RCNN adaptive foreground threshold checks passed")


if __name__ == "__main__":
    main()
