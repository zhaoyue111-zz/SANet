import torch
import numpy as np
try:
    from box import cpu_nms, cpu_overlap
except ImportError:
    cpu_nms = None
    cpu_overlap = None


def torch_nms(dets, thresh):
    """
    dets has to be a tensor
    """
    if isinstance(dets, np.ndarray):
        dets = torch.from_numpy(dets).float().contiguous()
        
    if not dets.is_cuda and cpu_nms is not None:
        z = dets[:, 1]
        y = dets[:, 2]
        x = dets[:, 3]
        d = dets[:, 4]
        h = dets[:, 5]
        w = dets[:, 6]
        scores = dets[:, 0]

        areas = d * h * w
        order = scores.sort(0, descending=True)[1]
        # order = torch.from_numpy(np.ascontiguousarray(scores.numpy().argsort()[::-1])).long()

        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)
        cpu_nms(keep, num_out, dets, order, areas, thresh)

        return dets[keep[:num_out[0]]], keep[:num_out[0]]
    if dets.is_cuda:
        dets_cpu = dets.detach().cpu()
    else:
        dets_cpu = dets

    dets_np = dets_cpu.numpy()
    z = dets_np[:, 1]
    y = dets_np[:, 2]
    x = dets_np[:, 3]
    d = dets_np[:, 4]
    h = dets_np[:, 5]
    w = dets_np[:, 6]
    scores = dets_np[:, 0]

    areas = d * h * w
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx0 = np.maximum(x[i] - w[i] / 2, x[order[1:]] - w[order[1:]] / 2)
        yy0 = np.maximum(y[i] - h[i] / 2, y[order[1:]] - h[order[1:]] / 2)
        zz0 = np.maximum(z[i] - d[i] / 2, z[order[1:]] - d[order[1:]] / 2)
        xx1 = np.minimum(x[i] + w[i] / 2, x[order[1:]] + w[order[1:]] / 2)
        yy1 = np.minimum(y[i] + h[i] / 2, y[order[1:]] + h[order[1:]] / 2)
        zz1 = np.minimum(z[i] + d[i] / 2, z[order[1:]] + d[order[1:]] / 2)

        inter_w = np.maximum(0.0, xx1 - xx0)
        inter_h = np.maximum(0.0, yy1 - yy0)
        inter_d = np.maximum(0.0, zz1 - zz0)
        intersect = inter_w * inter_h * inter_d
        overlap = intersect / np.maximum(areas[i] + areas[order[1:]] - intersect, 1e-12)

        inds = np.where(overlap <= thresh)[0]
        order = order[inds + 1]

    keep = torch.LongTensor(keep)
    output = dets_cpu[keep]
    if dets.is_cuda:
        output = output.cuda()
        keep = keep.cuda()
    return output, keep


def torch_overlap(boxes1, boxes2):
    """
    dets has to be a tensor
    """
    if isinstance(boxes1, np.ndarray):
        boxes1 = torch.from_numpy(boxes1).float().contiguous()
    if isinstance(boxes2, np.ndarray):
        boxes2 = torch.from_numpy(boxes2).float().contiguous()

    if not boxes1.is_cuda and not boxes2.is_cuda and cpu_overlap is not None:
        boxes1 = boxes1.float().contiguous()
        boxes2 = boxes2.float().contiguous()
        assert isinstance(boxes1, torch.FloatTensor) and isinstance(boxes2, torch.FloatTensor)
        overlap = torch.zeros([len(boxes1), len(boxes2)], dtype=torch.float32)
        cpu_overlap(boxes1, boxes2, overlap)

        return overlap

    boxes1_cpu = boxes1.detach().cpu().float()
    boxes2_cpu = boxes2.detach().cpu().float()
    if len(boxes1_cpu) == 0 or len(boxes2_cpu) == 0:
        overlap = torch.zeros([len(boxes1_cpu), len(boxes2_cpu)], dtype=torch.float32)
        return overlap.cuda() if boxes1.is_cuda or boxes2.is_cuda else overlap

    b1 = boxes1_cpu.numpy()
    b2 = boxes2_cpu.numpy()
    b1_min = b1[:, :3] - b1[:, 3:] / 2
    b1_max = b1[:, :3] + b1[:, 3:] / 2
    b2_min = b2[:, :3] - b2[:, 3:] / 2
    b2_max = b2[:, :3] + b2[:, 3:] / 2

    inter_min = np.maximum(b1_min[:, None, :], b2_min[None, :, :])
    inter_max = np.minimum(b1_max[:, None, :], b2_max[None, :, :])
    inter_size = np.maximum(inter_max - inter_min, 0)
    inter = inter_size[:, :, 0] * inter_size[:, :, 1] * inter_size[:, :, 2]

    vol1 = np.prod(b1[:, 3:], axis=1)
    vol2 = np.prod(b2[:, 3:], axis=1)
    union = np.maximum(vol1[:, None] + vol2[None, :] - inter, 1e-12)
    overlap = torch.from_numpy((inter / union).astype(np.float32))
    return overlap.cuda() if boxes1.is_cuda or boxes2.is_cuda else overlap
