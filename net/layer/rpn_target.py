import numpy as np
import torch
try:
    from utils.pybox import *
except ImportError:
    print('Warning: C++ module import failed! This should only happen in deployment')
    from utils.util import py_nms as torch_nms
    from utils.util import py_box_overlap as torch_overlap
import random
from net.layer.rpn_nms import rpn_encode
from torch.autograd import Variable


def _overlap_to_numpy(overlap):
    if hasattr(overlap, 'detach'):
        return overlap.detach().cpu().numpy()
    return overlap


def rpn_gt_fg_thresholds(cfg, pos_truth_box):
    base_thresh = float(cfg['rpn_train_fg_thresh_low'])
    pos_truth_box = np.asarray(pos_truth_box, dtype=np.float32).reshape(-1, 6)
    thresholds = np.full((len(pos_truth_box), ), base_thresh, np.float32)
    if not bool(cfg.get('rpn_train_adaptive_fg_thresh', False)) or len(pos_truth_box) == 0:
        return thresholds

    small_max_side = float(cfg.get('rpn_train_small_gt_max_side', 10.0))
    small_thresh = float(cfg.get('rpn_train_small_gt_fg_thresh_low', 0.2))
    max_side = np.max(pos_truth_box[:, 3:6], axis=1)
    thresholds[max_side < small_max_side] = small_thresh
    return thresholds


def assign_rpn_anchors(cfg, window, truth_box, truth_label):
    """
    Assign RPN anchors to valid GT boxes.

    Positive GTs have truth_label > 0. Ignore GTs have truth_label < 0 and only
    suppress anchors that were not already assigned to a positive GT.
    """
    num_window = len(window)
    label = np.zeros((num_window, ), np.float32)
    label_assign = np.zeros((num_window, ), np.int32) - 1
    label_weight = np.zeros((num_window, ), np.float32)
    target_weight = np.zeros((num_window, ), np.float32)

    truth_box = np.asarray(truth_box, dtype=np.float32).reshape(-1, 6)
    truth_label = np.asarray(truth_label, dtype=np.int64).reshape(-1)
    pos_index = np.where(truth_label > 0)[0]
    ignore_index = np.where(truth_label < 0)[0]
    pos_truth_box = truth_box[pos_index]
    ignore_truth_box = truth_box[ignore_index]

    num_truth_box = len(pos_truth_box)
    if num_truth_box:
        overlap = _overlap_to_numpy(torch_overlap(window, pos_truth_box))

        anchor_best_gt = np.argmax(overlap, 1)
        anchor_best_iou = overlap[np.arange(num_window), anchor_best_gt]
        gt_fg_thresholds = rpn_gt_fg_thresholds(cfg, pos_truth_box)
        anchor_fg_thresholds = gt_fg_thresholds[anchor_best_gt]

        bg_index = np.where(anchor_best_iou < cfg['rpn_train_bg_thresh_high'])[0]
        label[bg_index] = 0
        label_weight[bg_index] = 1

        fg_index = np.where(anchor_best_iou >= anchor_fg_thresholds)[0]
        label[fg_index] = 1
        label_weight[fg_index] = 1
        label_assign[fg_index] = anchor_best_gt[fg_index]

        gt_has_anchor = np.zeros((num_truth_box, ), np.bool_)
        assigned_fg = fg_index[label_assign[fg_index] >= 0]
        if len(assigned_fg):
            gt_has_anchor[np.unique(label_assign[assigned_fg])] = True

        gt_best_iou = overlap.max(axis=0)
        for gt_idx in range(num_truth_box):
            gt_best_anchor = np.where(overlap[:, gt_idx] == gt_best_iou[gt_idx])[0]
            label[gt_best_anchor] = 1
            label_weight[gt_best_anchor] = 1
            assign_to_gt = anchor_best_gt[gt_best_anchor] == gt_idx
            label_assign[gt_best_anchor[assign_to_gt]] = gt_idx
            other_best = gt_best_anchor[~assign_to_gt]
            if len(other_best):
                label_assign[other_best] = anchor_best_gt[other_best]

        assigned_fg = np.where((label != 0) & (label_assign >= 0))[0]
        if len(assigned_fg):
            gt_has_anchor[np.unique(label_assign[assigned_fg])] = True

        for gt_idx in np.where(~gt_has_anchor)[0]:
            own_best = np.where(anchor_best_gt == gt_idx)[0]
            own_best = own_best[label[own_best] == 0]
            if len(own_best):
                order = np.lexsort((own_best, -overlap[own_best, gt_idx]))
                anchor = own_best[order[0]]
            else:
                unused = np.where(label == 0)[0]
                if not len(unused):
                    continue
                order = np.lexsort((unused, -overlap[unused, gt_idx]))
                anchor = unused[order[0]]
            label[anchor] = 1
            label_weight[anchor] = 1
            label_assign[anchor] = gt_idx

        fg_index = np.where((label != 0) & (label_assign < 0))[0]
        if len(fg_index):
            label_assign[fg_index] = anchor_best_gt[fg_index]

    if not num_truth_box:
        label_weight[...] = 1

    if len(ignore_truth_box):
        ignore_overlap = _overlap_to_numpy(torch_overlap(window, ignore_truth_box))
        ignore_anchor_index = np.where((ignore_overlap.max(axis=1) > 0) & (label == 0))[0]
        label_weight[ignore_anchor_index] = 0
        target_weight[ignore_anchor_index] = 0

    return label, label_assign, label_weight, target_weight, pos_truth_box

'''
 给 anchor 分配正负标签和 bbox 回归目标
 '''
def make_one_rpn_target(cfg, mode, input, window, truth_box, truth_label):
    """
    Generate region proposal targets for one batch

    cfg: dict, for hyper-parameters
    mode: string, which phase/mode is used currently
    input: 5D torch tensor of [batch, channel, z, y, x], original input to the network
    window: list of anchor bounding boxes, [z, y, x, d, h, w]
    truth_box: list of ground truth bounding boxes, [z, y, x, d, h, w]
    truth_label: list of grount truth class label for each object in the correponding truth_box

    return torch tensors
    label: positive or negative (1 or 0) for each anchor box
    label_assign: index of the ground truth box, to which the anchor box is matched to
    label_weight: class weight for each sample, zero means current sample is protected,
                  and won't contribute to loss
    target: bounding box regression terms
    target_weight: weight for each regression term, by default it should all be ones
    """

    num_neg = cfg['num_neg']
    num_window = len(window)
    target = np.zeros((num_window, 6), np.float32)


    truth_box = np.asarray(truth_box, dtype=np.float32).reshape(-1, 6)
    truth_label = np.asarray(truth_label, dtype=np.int64).reshape(-1)
    label, label_assign, label_weight, target_weight, pos_truth_box = \
        assign_rpn_anchors(cfg, window, truth_box, truth_label)

    num_truth_box = len(pos_truth_box)
    if num_truth_box:
        # Prepare regression terms for each positive anchor
        fg_index = np.where(label != 0)[0]
        target_window = window[fg_index]
        target_truth_box = pos_truth_box[label_assign[fg_index]]
        target[fg_index] = rpn_encode(target_window, target_truth_box, cfg['box_reg_weight'])
        target_weight[fg_index] = 1

        if mode in ['train']:
            fg_index = np.where( (label_weight!=0) & (label!=0))[0]
            bg_index = np.where( (label_weight!=0) & (label==0))[0]

            # Random sample num_neg negative anchor boxes first
            # This is very strange, but it works well in practice
            # It makes the use of hard negative example mining loss, not 
            # actually hard negative example mining.
            # 随机采样最多num_neg个负样本
            label_weight[bg_index] = 0
            idx = random.sample(range(len(bg_index)), min(num_neg, len(bg_index)))
            bg_index = bg_index[idx]

            # Calculate weight for class balance
            num_fg = max(1, len(fg_index))
            num_bg = len(bg_index)
            if num_bg > 0:
                label_weight[bg_index] = float(num_fg) / num_bg

        fg_index = np.where(label != 0)[0]
        target_weight[fg_index] = label_weight[fg_index]
    else:
        # if there is no ground truth box in this batch

        if mode in ['train']:
            bg_index = np.where((label_weight!=0) & (label==0))[0]

            label_weight[bg_index] = 0
            idx = random.sample(range(len(bg_index)), min(num_neg, len(bg_index)))
            bg_index = bg_index[idx]
            if len(bg_index) > 0:
                label_weight[bg_index] = 1.0 / len(bg_index)


    label = Variable(torch.from_numpy(label)).cuda()
    label_assign = Variable(torch.from_numpy(label_assign)).cuda()
    label_weight = Variable(torch.from_numpy(label_weight)).cuda()
    target = Variable(torch.from_numpy(target)).cuda()
    target_weight = Variable(torch.from_numpy(target_weight)).cuda()
    return  label, label_assign, label_weight, target, target_weight


def make_rpn_target(cfg, mode, inputs, window, truth_boxes, truth_labels):
    rpn_labels = []
    rpn_label_assigns = []
    rpn_label_weights = []
    rpn_targets = []
    rpn_targets_weights = []

    batch_size = len(inputs)
    for b in range(batch_size):
        input = inputs[b]
        truth_box = np.asarray(truth_boxes[b], dtype=np.float32).reshape(-1, 6)
        truth_label = np.asarray(truth_labels[b], dtype=np.int64).reshape(-1)
        if len(truth_box):
            valid = np.isfinite(truth_box).all(axis=1) & (truth_box[:, 3:6] > 0).all(axis=1)
            truth_box = truth_box[valid]
            truth_label = truth_label[valid]

        rpn_label, rpn_label_assign, rpn_label_weight, rpn_target, rpn_targets_weight = \
            make_one_rpn_target(cfg, mode, input, window, truth_box, truth_label)

        rpn_labels.append(rpn_label.view(1, -1))
        rpn_label_assigns.append(rpn_label_assign.view(1, -1))
        rpn_label_weights.append(rpn_label_weight.view(1, -1))
        rpn_targets.append(rpn_target.view(1, -1, 6))
        rpn_targets_weights.append(rpn_targets_weight.view(1, -1))


    rpn_labels          = torch.cat(rpn_labels, 0)
    rpn_label_assigns   = torch.cat(rpn_label_assigns, 0)
    rpn_label_weights   = torch.cat(rpn_label_weights, 0)
    rpn_targets         = torch.cat(rpn_targets, 0)
    rpn_targets_weights = torch.cat(rpn_targets_weights, 0)

    return rpn_labels, rpn_label_assigns, rpn_label_weights, rpn_targets, rpn_targets_weights
