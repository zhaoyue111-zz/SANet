#!/usr/bin/env python3
"""Check checkpoint selection uses FROC mean, including RCNN-stage best."""

import os
import sys
import ast

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)


def load_checkpoint_update_flags():
    train_path = os.path.join(ROOT_DIR, "train_local.py")
    with open(train_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=train_path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "checkpoint_update_flags":
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace = {}
            exec(compile(module, train_path, "exec"), namespace)
            return namespace["checkpoint_update_flags"]
    raise AssertionError("checkpoint_update_flags not found in train_local.py")


checkpoint_update_flags = load_checkpoint_update_flags()


def load_early_stop_improved():
    train_path = os.path.join(ROOT_DIR, "train_local.py")
    with open(train_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=train_path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "early_stop_improved":
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace = {}
            exec(compile(module, train_path, "exec"), namespace)
            return namespace["early_stop_improved"]
    raise AssertionError("early_stop_improved not found in train_local.py")


early_stop_improved = load_early_stop_improved()


def check_before_rcnn_updates_only_global_best():
    is_best, is_best_rcnn, best_froc, best_rcnn_froc = checkpoint_update_flags(
        epoch=39,
        epoch_rcnn=40,
        val_froc_mean=0.25,
        best_froc_mean=0.20,
        best_rcnn_froc_mean=-float("inf"),
    )
    if not is_best or is_best_rcnn or best_froc != 0.25 or best_rcnn_froc != -float("inf"):
        raise AssertionError("pre-RCNN epoch should update only global best by FROC")


def check_after_rcnn_updates_both_when_best():
    is_best, is_best_rcnn, best_froc, best_rcnn_froc = checkpoint_update_flags(
        epoch=40,
        epoch_rcnn=40,
        val_froc_mean=0.30,
        best_froc_mean=0.25,
        best_rcnn_froc_mean=0.10,
    )
    if not is_best or not is_best_rcnn or best_froc != 0.30 or best_rcnn_froc != 0.30:
        raise AssertionError("RCNN epoch should update both best checkpoints when FROC improves both")


def check_after_rcnn_updates_only_rcnn_best():
    is_best, is_best_rcnn, best_froc, best_rcnn_froc = checkpoint_update_flags(
        epoch=41,
        epoch_rcnn=40,
        val_froc_mean=0.28,
        best_froc_mean=0.35,
        best_rcnn_froc_mean=0.20,
    )
    if is_best or not is_best_rcnn or best_froc != 0.35 or best_rcnn_froc != 0.28:
        raise AssertionError("RCNN best should update independently from global best")


def check_equal_froc_does_not_update():
    is_best, is_best_rcnn, best_froc, best_rcnn_froc = checkpoint_update_flags(
        epoch=42,
        epoch_rcnn=40,
        val_froc_mean=0.28,
        best_froc_mean=0.28,
        best_rcnn_froc_mean=0.28,
    )
    if is_best or is_best_rcnn or best_froc != 0.28 or best_rcnn_froc != 0.28:
        raise AssertionError("equal FROC should not replace existing best checkpoints")


def check_early_stop_uses_rcnn_best_after_rcnn_start():
    if not early_stop_improved(epoch=39, epoch_rcnn=40, is_best=True, is_best_rcnn=False):
        raise AssertionError("pre-RCNN early stop should use global best improvement")
    if early_stop_improved(epoch=40, epoch_rcnn=40, is_best=True, is_best_rcnn=False):
        raise AssertionError("RCNN-stage early stop should not use global best improvement")
    if not early_stop_improved(epoch=40, epoch_rcnn=40, is_best=False, is_best_rcnn=True):
        raise AssertionError("RCNN-stage early stop should use RCNN best improvement")


def main():
    check_before_rcnn_updates_only_global_best()
    check_after_rcnn_updates_both_when_best()
    check_after_rcnn_updates_only_rcnn_best()
    check_equal_froc_does_not_update()
    check_early_stop_uses_rcnn_best_after_rcnn_start()
    print("ok: checkpoint FROC selection checks passed")


if __name__ == "__main__":
    main()
