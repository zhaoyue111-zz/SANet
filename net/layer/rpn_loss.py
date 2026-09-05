import torch
import torch.nn.functional as F


def safe_binary_cross_entropy(prob, label):
    if prob.numel() == 0:
        return prob.sum() * 0
    return F.binary_cross_entropy(prob, label.float(), reduction='mean')


def weighted_focal_loss_for_cross_entropy(logits, labels, weights, gamma=2.):
    log_probs = F.log_softmax(logits, dim=1).gather(1, labels)
    probs = F.softmax(logits, dim=1).gather(1, labels)

    loss = - log_probs * (1 - probs) ** gamma
    loss = (weights * loss).sum() / (weights.sum() + 1e-12)

    return loss.sum()


def binary_cross_entropy_with_hard_negative_mining(logits, labels, weights, batch_size, num_hard=2):
    probs = torch.sigmoid(logits)[:, 0].view(-1, 1)
    pos_idcs = (labels[:, 0] == 1) & (weights[:, 0] != 0)

    pos_prob = probs[pos_idcs, 0]
    pos_labels = labels[pos_idcs, 0]

    # For those weights are zero, there are 2 cases,
    # 1. Because we first random sample num_neg negative boxes for OHEM
    # 2. Because those anchor boxes have some overlap with ground truth box,
    #    we want to maintain high sensitivity, so we do not count those as
    #    negative. It will not contribute to the loss
    neg_idcs = (labels[:, 0] == 0) & (weights[:, 0] != 0)
    neg_prob = probs[neg_idcs, 0]
    neg_labels = labels[neg_idcs, 0]
    if num_hard > 0 and len(pos_prob) > 0:
        neg_prob, neg_labels = OHEM(neg_prob, neg_labels, num_hard * len(pos_prob))

    pos_correct = 0
    pos_total = 0
    losses = []
    if pos_prob.numel() > 0:
        losses.append(safe_binary_cross_entropy(pos_prob, pos_labels))
        pos_correct = (pos_prob >= 0.5).sum()
        pos_total = len(pos_prob)
    if neg_prob.numel() > 0:
        losses.append(safe_binary_cross_entropy(neg_prob, neg_labels))

    if losses:
        cls_loss = sum(losses) / len(losses)
    else:
        cls_loss = logits.sum() * 0

    neg_correct = (neg_prob < 0.5).sum()
    neg_total = len(neg_prob)
    return cls_loss, pos_correct, pos_total, neg_correct, neg_total


def OHEM(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels


def weighted_focal_loss_with_logits(logits, labels, weights, gamma=2.):
    """Binary focal loss on logits; ignored where weights==0."""
    logits = logits.reshape(-1)
    labels = labels.reshape(-1).long()
    weights = weights.reshape(-1).float()

    valid = weights != 0
    if not valid.any():
        return logits.sum() * 0

    logits = logits[valid]
    labels = labels[valid]
    weights = weights[valid]

    probs = torch.sigmoid(logits)
    log_probs = F.logsigmoid(logits)

    pos = labels == 1
    neg = labels == 0

    loss = logits.sum() * 0
    if pos.any():
        pos_probs = probs[pos].detach()
        pos_loss = -log_probs[pos] * (1.0 - pos_probs) ** gamma
        loss = loss + (pos_loss * weights[pos]).sum()
    if neg.any():
        neg_probs = (1.0 - probs[neg]).detach()
        neg_logprobs = torch.log(torch.clamp(1.0 - probs[neg], min=1e-8))
        neg_loss = -neg_logprobs * (1.0 - neg_probs) ** gamma
        loss = loss + (neg_loss * weights[neg]).sum()

    return loss / (weights.sum() + 1e-12)


def rpn_loss(logits, deltas, labels, label_weights, targets, target_weights, cfg, mode='train', delta_sigma=3.0):
    batch_size, num_windows, num_classes = logits.size()
    batch_size_k = batch_size
    labels = labels.long()

    # Calculate classification score
    batch_size = batch_size * num_windows
    logits = logits.view(batch_size, num_classes)
    labels = labels.view(batch_size, 1)
    label_weights = label_weights.view(batch_size, 1)

    # Make sure OHEM is performed only in training mode
    if mode in ['train']:
        num_hard = cfg['num_hard']
    else:
        num_hard = 10000000

    rpn_cls_loss, pos_correct, pos_total, neg_correct, neg_total = \
        binary_cross_entropy_with_hard_negative_mining(
            logits, labels, label_weights, batch_size_k, num_hard
        )

    gamma = float(cfg.get('focal_loss_gamma', 2.0))
    rpn_focal_loss = weighted_focal_loss_with_logits(
        logits, labels, label_weights, gamma=gamma
    )

    # Calculate regression
    deltas = deltas.view(batch_size, 6)
    targets = targets.view(batch_size, 6)
    target_weights = target_weights.view(batch_size, 1)

    index = ((labels != 0) & (target_weights != 0)).nonzero()[:, 0]
    deltas = deltas[index]
    targets = targets[index]

    if len(index) == 0:
        rpn_reg_loss = deltas.sum() * 0
        reg_losses = [0.0] * 6
    else:
        rpn_reg_loss = 0
        reg_losses = []
        for i in range(6):
            l = F.smooth_l1_loss(deltas[:, i], targets[:, i])
            rpn_reg_loss += l
            reg_losses.append(l.data.item())

    return (
        rpn_cls_loss,
        rpn_reg_loss,
        rpn_focal_loss,
        [
            pos_correct, pos_total, neg_correct, neg_total,
            reg_losses[0], reg_losses[1], reg_losses[2],
            reg_losses[3], reg_losses[4], reg_losses[5],
        ],
    )
