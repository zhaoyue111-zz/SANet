#!/usr/bin/env python3
"""Check RPN classification loss stays finite for empty pos/neg edge cases."""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

import torch

from net.layer.rpn_loss import binary_cross_entropy_with_hard_negative_mining


def run_case(name, labels, weights):
    logits = torch.randn(len(labels), 1, requires_grad=True)
    labels = torch.tensor(labels, dtype=torch.long).view(-1, 1)
    weights = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
    loss, pos_correct, pos_total, neg_correct, neg_total = binary_cross_entropy_with_hard_negative_mining(
        logits,
        labels,
        weights,
        batch_size=1,
        num_hard=3,
    )
    if not torch.isfinite(loss):
        raise AssertionError("%s produced non-finite loss: %s" % (name, loss))
    loss.backward()
    if logits.grad is None:
        raise AssertionError("%s did not keep a gradient path to logits" % name)
    print(
        "%s ok: loss=%.6f pos_total=%s neg_total=%s"
        % (name, float(loss.detach()), int(pos_total), int(neg_total))
    )


def main():
    run_case("pos_and_neg", labels=[1, 0, 0], weights=[1, 1, 1])
    run_case("pos_only_no_neg", labels=[1, 0, 0], weights=[1, 0, 0])
    run_case("neg_only_no_pos", labels=[0, 0, 0], weights=[1, 1, 1])
    run_case("all_ignored", labels=[0, 0, 0], weights=[0, 0, 0])
    run_case("ignored_positive", labels=[1, 0, 0], weights=[0, 1, 1])
    print("ok: RPN cls loss is finite for empty pos/neg cases")


if __name__ == "__main__":
    main()
