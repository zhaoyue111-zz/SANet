import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


def multiclass_focal_loss(logits, labels, gamma=2.0, weight=None):
    """Softmax focal loss; optional per-class ``weight`` like cross_entropy."""
    if logits.numel() == 0:
        return logits.sum() * 0

    log_probs = F.log_softmax(logits, dim=1)
    probs = F.softmax(logits, dim=1)
    targets = labels.view(-1, 1).long()
    log_pt = log_probs.gather(1, targets).squeeze(1)
    pt = probs.gather(1, targets).squeeze(1).detach()
    loss = -(1.0 - pt).pow(gamma) * log_pt

    if weight is not None:
        loss = loss * weight[labels.long()]
    return loss.mean()


def rcnn_loss(logits, deltas, labels, targets, deltas_sigma=1.0, cfg=None):
    batch_size, num_class = logits.size(0), logits.size(1)
    gamma = 2.0 if cfg is None else float(cfg.get('focal_loss_gamma', 2.0))

    if batch_size == 0:
        zero = logits.sum() * 0 + deltas.sum() * 0
        confusion_matrix = np.zeros((num_class, num_class))
        return zero, zero, zero, [0, 0, 0, 0, 0, 0, confusion_matrix]

    # Weighted cross entropy for imbalance class distribution
    weight = torch.ones(num_class, device=logits.device, dtype=logits.dtype)
    total = len(labels)
    for i in range(num_class):
        num_pos = float((labels == i).sum())
        num_pos = max(num_pos, 1)
        weight[i] = total / num_pos

    weight = weight / weight.sum()
    rcnn_cls_loss = F.cross_entropy(logits, labels, weight=weight, reduction='mean')
    rcnn_focal_loss = multiclass_focal_loss(
        logits, labels, gamma=gamma, weight=weight
    )

    # If multi-class classification, compute the confusion metric to understand the mistakes
    confusion_matrix = np.zeros((num_class, num_class))
    probs = F.softmax(logits, dim=1)
    v, cat = torch.max(probs, dim=1)
    for i in labels.nonzero():
        i = i.item()
        confusion_matrix[labels.long().detach()[i].item()][cat[i].detach().item()] += 1

    num_pos = len(labels.nonzero())
    reg_losses = [0, 0, 0, 0, 0, 0]

    if num_pos > 0:
        # one hot encode
        select = Variable(torch.zeros((batch_size, num_class), device=logits.device))
        select.scatter_(1, labels.view(-1, 1), 1)
        select[:, 0] = 0
        select = select.view(batch_size, num_class, 1).expand((batch_size, num_class, 6)).contiguous().byte()
        select = select.bool()
        deltas = deltas.view(batch_size, num_class, 6)
        deltas = deltas[select].view(-1, 6)

        rcnn_reg_loss = 0
        reg_losses = []
        for i in range(6):
            l = F.smooth_l1_loss(deltas[:, i], targets[:, i])
            rcnn_reg_loss += l
            reg_losses.append(l.data.item())
    else:
        rcnn_reg_loss = deltas.sum() * 0

    return (
        rcnn_cls_loss,
        rcnn_reg_loss,
        rcnn_focal_loss,
        [
            reg_losses[0], reg_losses[1], reg_losses[2],
            reg_losses[3], reg_losses[4], reg_losses[5],
            confusion_matrix,
        ],
    )
