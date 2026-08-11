#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
重新推理 SANet 单个病例，导出真实 RCNN 回归框及 RCNN NMS 的 3D IoU。

输出：
1) <pid>_rcnn_before_nms.csv
   RCNN bbox regression 后、RCNN NMS 前的所有前景候选。
   x,y,z 为中心坐标；dx,dy,dz 分别为 x/y/z 方向尺寸。
2) <pid>_rcnn_nms_iou_trace.csv
   RCNN NMS 贪心过程中实际需要比较的每一对框的 3D IoU，
   并标出是否因此被 suppress。
3) <pid>_rcnn_after_nms.csv
   SANet net.detections，也就是实际 RCNN NMS 后保留的框。

SANet 内部框格式：
    [z, y, x, d, h, w]
其中 d/h/w 分别对应 z/y/x 方向尺寸。
本脚本输出：
    x, y, z, dx=w, dy=h, dz=d
"""

import os
import sys
import gc
import argparse
import traceback

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ["SANET_DISABLE_INTERNAL_DATA_PARALLEL"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build", "box"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import config, dataset_configs
from dataset.bbox_reader import BboxReader
from net.sanet import SANet
from net.layer.rcnn_nms import rcnn_decode
from net.layer.util import clip_boxes
import utils.pybox as pybox


def pid_keys(value):
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    out = {s}
    if s.isdigit():
        out.add(str(int(s)))
        out.add(s.zfill(5))
        out.add(s.zfill(6))
    return out


def same_pid(a, b):
    return bool(pid_keys(a) & pid_keys(b))


def find_case(dataset_name, pid, test_set_name=None):
    cfgs = dataset_configs(dataset_name, skip_missing=(dataset_name == "all"))
    matches = []

    for cfg in cfgs:
        split_file = test_set_name or cfg["test_set_name"]
        if not os.path.isfile(split_file):
            continue

        with open(split_file, "r") as f:
            pids = [line.strip() for line in f if line.strip()]

        for p in pids:
            if same_pid(p, pid):
                matches.append((cfg, split_file, p))
                break

    if not matches:
        raise ValueError("PID %s not found in selected test split(s)." % pid)

    if len(matches) > 1 and dataset_name == "all":
        names = [m[0].get("dataset", "?") for m in matches]
        raise ValueError(
            "PID %s exists in multiple datasets: %s. Please set --dataset."
            % (pid, ", ".join(names))
        )

    return matches[0]


def load_case(cfg, split_file, pid):
    ds = BboxReader(cfg["DATA_DIR"], split_file, cfg, mode="eval")

    idx = None
    exact_pid = None
    for i, p in enumerate(ds.filenames):
        if same_pid(p, pid):
            idx = i
            exact_pid = p
            break

    if idx is None:
        raise ValueError("PID %s is not available in BboxReader." % pid)

    sample = ds[idx]
    inputs = sample[0].unsqueeze(0)
    truth_bboxes = np.asarray([sample[1]])
    truth_labels = np.asarray([sample[2]])

    return exact_pid, inputs, truth_bboxes, truth_labels


def iou_3d(box_a, box_b):
    """
    与 SANet NMS 相同的中心点-尺寸 3D IoU。
    box: [z,y,x,d,h,w]
    """
    za, ya, xa, da, ha, wa = [float(v) for v in box_a]
    zb, yb, xb, db, hb, wb = [float(v) for v in box_b]

    ax0, ax1 = xa - wa / 2.0, xa + wa / 2.0
    ay0, ay1 = ya - ha / 2.0, ya + ha / 2.0
    az0, az1 = za - da / 2.0, za + da / 2.0

    bx0, bx1 = xb - wb / 2.0, xb + wb / 2.0
    by0, by1 = yb - hb / 2.0, yb + hb / 2.0
    bz0, bz1 = zb - db / 2.0, zb + db / 2.0

    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    id_ = max(0.0, min(az1, bz1) - max(az0, bz0))
    inter = iw * ih * id_

    va = da * ha * wa
    vb = db * hb * wb
    return inter / max(va + vb - inter, 1e-12)


def box_record(box, prefix=""):
    z, y, x, d, h, w = [float(v) for v in box]
    return {
        prefix + "x": x,
        prefix + "y": y,
        prefix + "z": z,
        prefix + "dx": w,
        prefix + "dy": h,
        prefix + "dz": d,
        prefix + "xmin": x - w / 2.0,
        prefix + "xmax": x + w / 2.0,
        prefix + "ymin": y - h / 2.0,
        prefix + "ymax": y + h / 2.0,
        prefix + "zmin": z - d / 2.0,
        prefix + "zmax": z + d / 2.0,
    }


def build_rcnn_before_nms(net, inputs):
    """
    严格复现 rcnn_nms.py 中 NMS 之前的：
      softmax -> score filter -> rcnn_decode -> clip_boxes
    """
    proposals = net.rpn_proposals.detach().cpu().numpy()
    probs = F.softmax(net.rcnn_logits, dim=1).detach().cpu().numpy()

    num_class = int(config["num_class"])
    deltas = (
        net.rcnn_deltas.detach().cpu().numpy()
        .reshape(-1, num_class, 6)
    )

    score_th = float(config["rcnn_test_nms_pre_score_threshold"])
    nms_th = float(config["rcnn_test_nms_overlap_threshold"])

    candidates = []
    cid = 0

    for b in range(int(inputs.size(0))):
        batch_index = np.where(proposals[:, 0] == b)[0]
        if len(batch_index) == 0:
            continue

        prob_b = probs[batch_index]
        delta_b = deltas[batch_index]
        proposal_b = proposals[batch_index]

        for cls in range(1, num_class):
            local_idx = np.where(prob_b[:, cls] > score_th)[0]
            if len(local_idx) == 0:
                continue

            scores = prob_b[local_idx, cls]
            class_deltas = delta_b[local_idx, cls]

            boxes = rcnn_decode(
                proposal_b[local_idx, 2:8],
                class_deltas,
                config["box_reg_weight"],
            )
            boxes = clip_boxes(boxes, inputs.shape[2:])

            for k, li in enumerate(local_idx):
                proposal_index = int(batch_index[li])

                candidates.append({
                    "candidate_id": cid,
                    "batch_id": int(b),
                    "class_id": int(cls),
                    "proposal_index": proposal_index,
                    "rpn_score": float(proposals[proposal_index, 1]),
                    "rcnn_score": float(scores[k]),
                    "box": boxes[k].astype(np.float64),
                    "rpn_box": proposals[proposal_index, 2:8].astype(np.float64),
                })
                cid += 1

    return candidates, score_th, nms_th


def get_actual_keep_keys(net):
    """
    net.detections 与 net.keeps 在 rcnn_nms.py 中按相同顺序追加。
    """
    detections = net.detections.detach().cpu().numpy()
    keeps = list(net.keeps)

    if len(detections) != len(keeps):
        raise RuntimeError(
            "len(net.detections)=%d != len(net.keeps)=%d"
            % (len(detections), len(keeps))
        )

    keys = set()
    rows = []

    for det_id, (det, proposal_index) in enumerate(zip(detections, keeps)):
        # [batch, score, z,y,x,d,h,w,class]
        b = int(det[0])
        score = float(det[1])
        box = det[2:8]
        cls = int(det[8])
        proposal_index = int(proposal_index)

        keys.add((b, cls, proposal_index))

        row = {
            "detection_id": det_id,
            "batch_id": b,
            "class_id": cls,
            "proposal_index": proposal_index,
            "rcnn_score": score,
        }
        row.update(box_record(box))
        rows.append(row)

    return keys, rows


def trace_rcnn_nms(candidates, nms_th):
    """
    复现 utils.pybox.torch_nms 的贪心过程并保存每一次实际 IoU 比较。
    """
    compiled = pybox.cpu_nms is not None
    backend = "compiled_cpu_nms" if compiled else "python_fallback"

    groups = {}
    for c in candidates:
        groups.setdefault((c["batch_id"], c["class_id"]), []).append(c)

    trace = []
    reproduced_keep = set()
    suppressed_by = {}

    for (batch_id, class_id), group in groups.items():
        scores = np.asarray([g["rcnn_score"] for g in group], dtype=np.float32)

        # 与 utils.pybox.torch_nms 的排序方式一致
        if compiled:
            order = (
                torch.from_numpy(scores.copy())
                .sort(0, descending=True)[1]
                .cpu().numpy().tolist()
            )
        else:
            order = scores.argsort()[::-1].tolist()

        order = [int(v) for v in order]
        step = 0

        while order:
            keep_local = order[0]
            keeper = group[keep_local]
            keeper_key = (
                keeper["batch_id"],
                keeper["class_id"],
                keeper["proposal_index"],
            )
            reproduced_keep.add(keeper_key)

            next_order = []

            for compare_local in order[1:]:
                other = group[compare_local]
                val = float(iou_3d(keeper["box"], other["box"]))

                # build/box/nms.cpp 使用 >=
                # Python fallback 使用 >（因为 overlap <= threshold 会留下）
                suppress = (val >= nms_th) if compiled else (val > nms_th)

                row = {
                    "batch_id": batch_id,
                    "class_id": class_id,
                    "nms_step": step,
                    "nms_backend": backend,
                    "nms_iou_threshold": nms_th,
                    "keeper_candidate_id": keeper["candidate_id"],
                    "keeper_proposal_index": keeper["proposal_index"],
                    "keeper_rcnn_score": keeper["rcnn_score"],
                    "compared_candidate_id": other["candidate_id"],
                    "compared_proposal_index": other["proposal_index"],
                    "compared_rcnn_score": other["rcnn_score"],
                    "iou_3d": val,
                    "suppressed": int(suppress),
                }
                row.update(box_record(keeper["box"], "keeper_"))
                row.update(box_record(other["box"], "compared_"))
                trace.append(row)

                if suppress:
                    key = (
                        other["batch_id"],
                        other["class_id"],
                        other["proposal_index"],
                    )
                    suppressed_by[key] = keeper["candidate_id"]
                else:
                    next_order.append(compare_local)

            order = next_order
            step += 1

    return trace, reproduced_keep, suppressed_by, backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", required=True, type=str)
    parser.add_argument("--weight", required=True, type=str)
    parser.add_argument("--dataset", default=config["dataset"], type=str)
    parser.add_argument("--test-set-name", dest="test_set_name", default=None)
    parser.add_argument("--out-dir", dest="out_dir", default="test_instance_output")
    parser.add_argument(
        "--use-aspp",
        action="store_true",
        help="Checkpoint 若使用 ASPP，则加此参数。",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Current SANet inference path expects CUDA. Run this script in the same GPU environment as test.py."
        )

    cfg, split_file, split_pid = find_case(
        args.dataset, args.pid, args.test_set_name
    )

    exact_pid, inputs, truth_bboxes, truth_labels = load_case(
        cfg, split_file, split_pid
    )

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("dataset:", cfg.get("dataset", args.dataset))
    print("pid:", exact_pid)
    print("weight:", args.weight)
    print("split:", split_file)
    print("=" * 80)

    net = SANet(config, mode="eval", use_aspp=args.use_aspp).cuda()

    ckpt = torch.load(args.weight, weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    net.set_mode("eval")
    net.use_rcnn = True

    inputs = inputs.cuda()

    try:
        with torch.no_grad():
            net.forward(inputs, truth_bboxes, truth_labels)

        candidates, score_th, nms_th = build_rcnn_before_nms(net, inputs)
        actual_keep, after_rows = get_actual_keep_keys(net)

        trace, reproduced_keep, suppressed_by, backend = trace_rcnn_nms(
            candidates, nms_th
        )

        before_rows = []
        for c in candidates:
            key = (
                c["batch_id"],
                c["class_id"],
                c["proposal_index"],
            )

            row = {
                "candidate_id": c["candidate_id"],
                "batch_id": c["batch_id"],
                "class_id": c["class_id"],
                "proposal_index": c["proposal_index"],
                "rpn_score": c["rpn_score"],
                "rcnn_score": c["rcnn_score"],
                "actual_kept_by_rcnn_nms": int(key in actual_keep),
                "reproduced_kept_by_rcnn_nms": int(key in reproduced_keep),
                "suppressed_by_candidate_id": suppressed_by.get(key, ""),
            }
            row.update(box_record(c["box"]))
            row.update(box_record(c["rpn_box"], "rpn_"))
            before_rows.append(row)

        pid_safe = str(exact_pid).replace(os.sep, "_")
        p_before = os.path.join(
            args.out_dir, pid_safe + "_rcnn_before_nms.csv"
        )
        p_trace = os.path.join(
            args.out_dir, pid_safe + "_rcnn_nms_iou_trace.csv"
        )
        p_after = os.path.join(
            args.out_dir, pid_safe + "_rcnn_after_nms.csv"
        )

        pd.DataFrame(before_rows).to_csv(p_before, index=False)
        pd.DataFrame(trace).to_csv(p_trace, index=False)
        pd.DataFrame(after_rows).to_csv(p_after, index=False)

        print("")
        print("[summary]")
        print("RPN proposals:", len(net.rpn_proposals))
        print("RCNN score threshold:", score_th)
        print("RCNN candidates before NMS:", len(candidates))
        print("RCNN NMS IoU threshold:", nms_th)
        print("NMS backend:", backend)
        print("actual detections after RCNN NMS:", len(after_rows))
        print("reproduced keeps:", len(reproduced_keep))
        print("reproduced keeps == net.keeps:", reproduced_keep == actual_keep)

        if trace:
            max_iou = max(r["iou_3d"] for r in trace)
            suppressed_ious = [
                r["iou_3d"] for r in trace if r["suppressed"] == 1
            ]
            print("max compared IoU:", max_iou)
            if suppressed_ious:
                print("min suppressing IoU:", min(suppressed_ious))

        print("")
        print("saved:")
        print(" ", p_before)
        print(" ", p_trace)
        print(" ", p_after)

    finally:
        del inputs
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
