import copy
from net.layer.rcnn_nms import rcnn_encode
import time
import random
import numpy as np
import torch
from torch.autograd import Variable
from net.layer.rpn_target import rpn_gt_fg_thresholds
try:
    from utils.pybox import *
except ImportError:
    print('Warning: C++ module import failed! This should only happen in deployment')
    from utils.util import py_nms as torch_nms
    from utils.util import py_box_overlap as torch_overlap

score = 1


def _overlap_to_numpy(overlap):
    if hasattr(overlap, 'detach'):
        return overlap.detach().cpu().numpy()
    return overlap

'''
给 RPN proposals 分配背景/结节标签和 bbox 回归目标
'''
def add_truth_box_to_proposal(cfg, proposal, b, truth_box, truth_label, score=1):
    if len(truth_box) !=0:
        truth = np.zeros((len(truth_box), 8),np.float32)
        truth[:, 0] = b
        truth[:, 2:8] = truth_box
        truth[:, 1] = score #1  #
    else:
        truth = np.zeros((0, 8),np.float32)

    sampled_proposal = np.vstack([proposal,truth])
    return sampled_proposal


def make_one_rcnn_target(cfg, input, proposal, truth_box, truth_label, ignore_box=None):
    sampled_proposal = torch.zeros((0, 8)).float().cuda()
    sampled_label = torch.zeros((0,)).long().cuda()
    sampled_assign = np.zeros((0,), dtype=np.int32) - 1
    sampled_target = torch.zeros((0, 6)).float().cuda()
    ignore_box = np.asarray(ignore_box, dtype=np.float32).reshape(-1, 6)

    # Even if there is no ground truth box in this batch
    if len(proposal) == 0:
        return sampled_proposal, sampled_label, sampled_assign, sampled_target

    if len(truth_box) == 0:
        bg_index = np.arange(len(proposal))
        if len(ignore_box):
            ignore_overlap = _overlap_to_numpy(torch_overlap(proposal[:, 2:8], ignore_box))
            bg_index = bg_index[ignore_overlap.max(axis=1) <= 0]
        num_bg = min(len(bg_index), cfg['rcnn_train_batch_size'])
        bg_length = len(bg_index)
        if bg_length == 0:
            return sampled_proposal, sampled_label, sampled_assign, sampled_target
        bg_index = bg_index[np.random.choice(bg_length, size=num_bg, replace=bg_length<num_bg)]
        sampled_proposal = proposal[bg_index]
        sampled_proposal = torch.from_numpy(sampled_proposal).cuda()
        sampled_label = torch.zeros((num_bg)).long().cuda()
        sampled_assign = np.zeros((num_bg,), dtype=np.int32) - 1

        return sampled_proposal, sampled_label, sampled_assign, sampled_target 

    _, depth, height, width = input.size()
    num_proposal = len(proposal)
    box = proposal[:, 2:8]

    # Determine positive or negative purely based on threshold
    # Since the GT box has been added to proposal, it is gauranteed that
    # each ground truth would have one proposal
    overlap = _overlap_to_numpy(torch_overlap(box, truth_box))
    argmax_overlap = np.argmax(overlap,1)
    max_overlap = overlap[np.arange(num_proposal),argmax_overlap]

    # Keep RCNN foreground assignment consistent with RPN. Small GTs can use
    # the adaptive low IoU threshold configured for RPN instead of the legacy
    # fixed rcnn_train_fg_thresh_low value.
    gt_fg_thresholds = rpn_gt_fg_thresholds(cfg, truth_box)
    proposal_fg_thresholds = gt_fg_thresholds[argmax_overlap]
    fg_index = np.where(max_overlap >= proposal_fg_thresholds)[0]
    bg_index = np.where(max_overlap <  cfg['rcnn_train_bg_thresh_high'])[0]
    if len(ignore_box) and len(bg_index):
        ignore_overlap = _overlap_to_numpy(torch_overlap(box, ignore_box))
        bg_index = bg_index[ignore_overlap[bg_index].max(axis=1) <= 0]

    # sampling for class balance
    num_class = cfg['num_class']
    num = cfg['rcnn_train_batch_size']
    num_fg = int(np.round(cfg['rcnn_train_fg_fraction'] * cfg['rcnn_train_batch_size']))

    fg_length = len(fg_index)
    bg_length = len(bg_index)
    #print(fg_inds_length)

    sampled_assign = argmax_overlap[fg_index]

    # Need to consider four cases, corner cases
    if fg_length > 0 and bg_length > 0:
        idx = []
        idx = random.sample(range(len(fg_index)), min(num_fg, len(fg_index)))
        
        fg_index = fg_index[idx]
        num_fg = len(fg_index)

        num_bg  = num - num_fg
        bg_index = bg_index[np.random.choice(bg_length, size=num_bg, replace=bg_length<num_bg)]
    elif fg_length > 0:  #no bgs
        idx = []
        idx = random.sample(range(len(fg_index)), min(num_fg, len(fg_index)))
        
        fg_index = fg_index[idx]
        num_fg = len(fg_index)
        num = num_fg
        num_bg = 0

    elif bg_length > 0:  #no fgs
        print('[RCNN] No fgs')
        print(truth_box)
        print('---------------------------')
        print(proposal)

        num_fg = 0
        num_bg = num
        bg_index = bg_index[np.random.choice(bg_length, size=num_bg, replace=bg_length<num_bg)]
        num_fg_proposal = 0
    else:
        return sampled_proposal, sampled_label, sampled_assign, sampled_target

    assert ((num_fg+num_bg)== num)

    # selecting both fg and bg
    index = np.concatenate([fg_index, bg_index], 0)
    sampled_proposal = np.take(proposal, index, axis=0)

    # label
    sampled_assign = np.take(argmax_overlap, index)
    sampled_label = np.take(truth_label, sampled_assign)
    
    sampled_label[num_fg:] = 0   # Clamp labels for the background to 0
    sampled_assign[num_fg:] = -1 # Clamp label assignments for the background to -1

    # bounding box regression terms
    if num_fg>0:
        target_truth_box = truth_box[sampled_assign[:num_fg], :]
        if len(target_truth_box.shape) < 2: # one dimension lost after slicing
            target_truth_box = target_truth_box[np.newaxis, ...]
        target_box = sampled_proposal[:num_fg,:][:, 2:8]
        sampled_target = rcnn_encode(target_box, target_truth_box, cfg['box_reg_weight'])

    if not isinstance(sampled_target, np.ndarray):
        sampled_target = sampled_target.detach().cpu().numpy()
    sampled_target   = Variable(torch.from_numpy(sampled_target)).float().cuda()
    sampled_label    = Variable(torch.from_numpy(sampled_label).reshape(-1)).long().cuda()
    sampled_proposal = Variable(torch.from_numpy(sampled_proposal)).cuda()

    return sampled_proposal, sampled_label, sampled_assign, sampled_target


def make_rcnn_target(cfg, mode, inputs, proposals, truth_boxes, truth_labels):
    truth_boxes = copy.deepcopy(truth_boxes)
    truth_labels = copy.deepcopy(truth_labels)
    batch_size = len(inputs)
    for b in range(batch_size):
        truth_boxes[b] = np.asarray(truth_boxes[b], dtype=np.float32).reshape(-1, 6)
        truth_labels[b] = np.asarray(truth_labels[b], dtype=np.int64).reshape(-1)
        valid_box = np.isfinite(truth_boxes[b]).all(axis=1) & (truth_boxes[b][:, 3:6] > 0).all(axis=1)
        truth_boxes [b] = truth_boxes [b][valid_box]
        truth_labels[b] = truth_labels[b][valid_box]

    proposals = proposals.cpu().data.numpy()
    sampled_proposals = [] # 第 i 个给 RCNN 训练的候选框
    sampled_labels = []
    sampled_assigns = []  # 候选框匹配到的 truth_box 下标 一维数组 长度应该等于当前样本采样出来的 RCNN proposal 数量
    sampled_targets = []
    sampled_masks = []

    batch_size = len(truth_boxes)
    for b in range(batch_size):
        input = inputs[b]
        truth_box = truth_boxes[b]
        truth_label = truth_labels[b]
        pos_index = np.where(truth_label > 0)[0]
        ignore_index = np.where(truth_label < 0)[0]
        pos_truth_box = truth_box[pos_index]
        pos_truth_label = truth_label[pos_index]
        ignore_truth_box = truth_box[ignore_index]

        if len(proposals) == 0:
            proposal = np.zeros((0, 8),np.float32)
        else:
            proposal = proposals[proposals[:,0] == b]

        # Add ground truth box to proposal, so that even if the RPN branch fails to find something,
        # we can still get classification branch to work
        proposal = add_truth_box_to_proposal(cfg, proposal, b, pos_truth_box, pos_truth_label)

        sampled_proposal, sampled_label, sampled_assign, sampled_target = \
           make_one_rcnn_target(cfg, input, proposal, pos_truth_box, pos_truth_label, ignore_truth_box)

        sampled_proposals.append(sampled_proposal.view(-1, 8))
        sampled_labels.append(sampled_label.view(-1))
        sampled_assigns.append(np.asarray(sampled_assign, dtype=np.int32).reshape(-1))
        sampled_targets.append(sampled_target.view(-1, 6))

    sampled_proposals = torch.cat(sampled_proposals, 0)
    sampled_labels = torch.cat(sampled_labels, 0)
    sampled_targets = torch.cat(sampled_targets, 0)
    sampled_assigns = np.hstack(sampled_assigns)

    return sampled_proposals, sampled_labels, sampled_assigns, sampled_targets

 
