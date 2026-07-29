#!/usr/bin/env python3
"""Check bbox augmentation keeps corner-format boxes aligned."""

import os
import sys
from unittest.mock import patch

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

import numpy as np

import dataset.bbox_reader as bbox_reader
from dataset.bbox_reader import augment


def nonzero_box(sample):
    coords = np.argwhere(sample[0] > 0.5)
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    return np.asarray([zmin, zmax, ymin, ymax, xmin, xmax], dtype=np.float32)


def make_sample(shape=(1, 32, 32, 32), box=(8, 12, 9, 14, 10, 16)):
    sample = np.zeros(shape, dtype=np.float32)
    z0, z1, y0, y1, x0, x1 = box
    sample[:, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = 1
    bboxes = np.asarray([box], dtype=np.float32)
    target = bboxes[0].copy()
    return sample, target, bboxes


def assert_close(name, actual, expected):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=1):
        raise AssertionError("%s: got %s, expected %s" % (name, actual, expected))


def check_flip_yx():
    sample, target, bboxes = make_sample()
    with patch.object(bbox_reader.np.random, "randint", side_effect=[0, 1]):
        out, target_out, boxes_out = augment(
            sample,
            target,
            bboxes,
            do_flip=True,
            do_rotate=False,
            do_swap=False,
        )
    expected = nonzero_box(out)
    if int(expected[2]) != 17 or int(expected[3]) != 22:
        raise AssertionError("flip should map y=[9,14] to y=[17,22], got %s" % expected)
    assert_close("flip box", boxes_out[0], expected)
    assert_close("flip target", target_out, expected)


def check_swap_axes():
    sample, target, bboxes = make_sample(shape=(1, 32, 32, 32), box=(4, 8, 9, 14, 20, 25))
    with patch.object(bbox_reader.np.random, "permutation", return_value=np.asarray([2, 1, 0])):
        out, target_out, boxes_out = augment(
            sample,
            target,
            bboxes,
            do_flip=False,
            do_rotate=False,
            do_swap=True,
        )
    expected = nonzero_box(out)
    assert_close("swap box", boxes_out[0], expected)
    assert_close("swap target", target_out, expected)


def check_rotate_90_degrees():
    sample, target, bboxes = make_sample(shape=(1, 32, 32, 32), box=(8, 12, 9, 14, 10, 16))
    with patch.object(bbox_reader.np.random, "rand", return_value=0.5):
        out, target_out, boxes_out = augment(
            sample,
            target,
            bboxes,
            do_flip=False,
            do_rotate=True,
            do_swap=False,
            pad_value=0,
        )
    expected = nonzero_box(out)
    assert_close("rotate box", boxes_out[0], expected)
    assert_close("rotate target", target_out, expected)


def check_rotate_uses_pad_value():
    sample, target, bboxes = make_sample(shape=(1, 32, 32, 32), box=(8, 12, 9, 14, 10, 16))
    seen = {}

    def fake_rotate(input_sample, angle, axes, reshape, order, mode, cval, prefilter):
        seen.update({
            "axes": axes,
            "reshape": reshape,
            "order": order,
            "mode": mode,
            "cval": cval,
            "prefilter": prefilter,
        })
        return input_sample

    with patch.object(bbox_reader.np.random, "rand", return_value=0.1), \
            patch.object(bbox_reader, "rotate", side_effect=fake_rotate):
        augment(
            sample,
            target,
            bboxes,
            do_flip=False,
            do_rotate=True,
            do_swap=False,
            pad_value=170,
        )

    expected = {
        "axes": (2, 3),
        "reshape": False,
        "order": 1,
        "mode": "constant",
        "cval": 170.0,
        "prefilter": False,
    }
    if seen != expected:
        raise AssertionError("rotate args got %s, expected %s" % (seen, expected))


def main():
    check_flip_yx()
    check_swap_axes()
    check_rotate_90_degrees()
    check_rotate_uses_pad_value()
    print("ok: corner-format bbox augmentation checks passed")


if __name__ == "__main__":
    main()
