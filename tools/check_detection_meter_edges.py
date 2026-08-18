#!/usr/bin/env python3
"""Check validation detection-meter edge cases."""

import ast
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

import numpy as np


def load_detection_meter_functions():
    train_path = os.path.join(ROOT_DIR, "train.py")
    with open(train_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=train_path)
    wanted = {
        "init_detection_meter",
        "center_distance_match",
        "update_detection_meter",
        "summarize_detection_meter",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names.intersection({"VAL_FROC_THRESHOLDS", "VAL_SCORE_THRESHOLDS"}):
                body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, train_path, "exec"), namespace)
    return namespace


ns = load_detection_meter_functions()
init_detection_meter = ns["init_detection_meter"]
update_detection_meter = ns["update_detection_meter"]
summarize_detection_meter = ns["summarize_detection_meter"]


def check_candidates_without_gt_are_fp():
    meter = init_detection_meter()
    proposals = np.asarray([
        [0, 0.95, 10, 10, 10, 8, 8, 8],
        [0, 0.40, 20, 20, 20, 8, 8, 8],
    ], dtype=np.float32)
    update_detection_meter(meter, proposals, np.zeros((0, 6), dtype=np.float32))
    summary = summarize_detection_meter(meter)
    if meter["is_tp"] != [0, 0]:
        raise AssertionError("candidates without GT should be false positives, got %s" % meter["is_tp"])
    if summary["fp@0.10"] != 2 or summary["tp@0.10"] != 0 or summary["fn@0.10"] != 0:
        raise AssertionError("unexpected no-GT threshold stats: %s" % summary)


def check_rcnn_detection_nine_columns():
    meter = init_detection_meter()
    detections = np.asarray([
        [0, 0.95, 10, 10, 10, 8, 8, 8, 1],
        [0, 0.40, 40, 40, 40, 8, 8, 8, 1],
    ], dtype=np.float32)
    truth_boxes = np.asarray([[10, 10, 10, 8, 8, 8]], dtype=np.float32)
    update_detection_meter(meter, detections, truth_boxes)
    summary = summarize_detection_meter(meter)
    if meter["is_tp"] != [1, 0]:
        raise AssertionError("9-column RCNN detections should match using first 8 columns, got %s" % meter["is_tp"])
    if summary["tp@0.10"] != 1 or summary["fp@0.10"] != 1 or summary["fn@0.10"] != 0:
        raise AssertionError("unexpected 9-column detection stats: %s" % summary)


def main():
    check_candidates_without_gt_are_fp()
    check_rcnn_detection_nine_columns()
    print("ok: detection meter edge checks passed")


if __name__ == "__main__":
    main()
