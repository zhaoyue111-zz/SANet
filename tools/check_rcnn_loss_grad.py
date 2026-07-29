#!/usr/bin/env python3
"""Quick check for RCNN loss gradients in all-background batches."""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

import torch

from net.layer.rcnn_loss import rcnn_loss


def main():
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *args, **kwargs: self
        print("warning: CUDA is unavailable in this Python env; running the gradient check on CPU")

    num_samples = 8
    num_class = 2
    logits = torch.randn(num_samples, num_class, device=device, requires_grad=True)
    deltas = torch.randn(num_samples, num_class * 6, device=device, requires_grad=True)
    labels = torch.zeros(num_samples, dtype=torch.long, device=device)
    targets = torch.zeros(0, 6, device=device)

    cls_loss, reg_loss, _ = rcnn_loss(logits, deltas, labels, targets)
    loss = cls_loss + reg_loss
    loss.backward()

    if logits.grad is None:
        raise RuntimeError("logits.grad is None")
    if deltas.grad is None:
        raise RuntimeError("deltas.grad is None; DDP can hang on all-background RCNN batches")
    if not torch.allclose(deltas.grad, torch.zeros_like(deltas.grad)):
        raise RuntimeError("expected zero deltas.grad for all-background labels")

    print("ok: all-background RCNN batch keeps a zero gradient path for deltas")
    print("cls_loss=%.6f reg_loss=%.6f" % (float(cls_loss.detach()), float(reg_loss.detach())))


if __name__ == "__main__":
    main()
