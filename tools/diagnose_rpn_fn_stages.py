#!/usr/bin/env python3
"""Diagnose where LUNA16/SANet false-negative nodules disappear.

This script does not change SANet inference. After each normal forward pass it
reconstructs the RPN/RCNN filtering stages from the tensors already exposed by
SANet and determines the first stage at which no candidate matches a GT nodule.

Matching follows this repository's LUNA16 evaluator: a candidate is regarded as
covering a nodule when its center lies within the GT nodule radius.
"""

import argparse
import gc
import os
import sys
import traceback
from collections import Counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("SANET_DISABLE_INTERNAL_DATA_PARALLEL", "1")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "build", "box"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import net.sanet as sanet_module
from config import config, dataset_config
from dataset.bbox_reader import BboxReader
from dataset.collate import train_collate
from net.layer.rcnn_nms import rcnn_decode
from net.layer.rpn_nms import rpn_decode, torch_nms
from net.layer.util import clip_boxes
from net.sanet import SANet


def enable_cpu_fallback():
    """Keep the same lightweight CPU fallback used by test.py."""
    if torch.cuda.is_available():
        return

    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    torch.cuda.empty_cache = lambda: None
    torch.cuda.device_count = lambda: 1
    torch.cuda.is_available = lambda: False

    def _cpu_data_parallel(module, *inputs, **kwargs):
        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            return module(*inputs[0], **kwargs)
        return module(*inputs, **kwargs)

    sanet_module.data_parallel = _cpu_data_parallel


def canonical_pid(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def pid_aliases(value):
    pid = canonical_pid(value)
    aliases = {pid}
    if pid.isdigit():
        aliases.add(str(int(pid)))
        aliases.add(pid.zfill(5))
        aliases.add(pid.zfill(6))
    return aliases


def load_annotations(path):
    ann = pd.read_csv(path)

    if "pid" not in ann.columns:
        if "seriesuid" in ann.columns:
            ann = ann.rename(columns={"seriesuid": "pid"})
        else:
            raise ValueError("Annotation CSV must contain 'pid' or 'seriesuid'.")

    center_cols = {"center_x", "center_y", "center_z"}
    if not center_cols.issubset(ann.columns):
        corner_cols = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
        missing = corner_cols.difference(ann.columns)
        if missing:
            raise ValueError(
                "Annotation CSV lacks center coordinates and corner columns: %s"
                % ", ".join(sorted(missing))
            )
        ann["center_x"] = (ann["xmin"] + ann["xmax"]) / 2.0
        ann["center_y"] = (ann["ymin"] + ann["ymax"]) / 2.0
        ann["center_z"] = (ann["zmin"] + ann["zmax"]) / 2.0

    if "diameter" not in ann.columns:
        corner_cols = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
        if not corner_cols.issubset(ann.columns):
            raise ValueError("Annotation CSV must contain diameter or all six corner columns.")
        sizes = np.stack(
            [
                ann["xmax"].to_numpy() - ann["xmin"].to_numpy() + 1.0,
                ann["ymax"].to_numpy() - ann["ymin"].to_numpy() + 1.0,
                ann["zmax"].to_numpy() - ann["zmin"].to_numpy() + 1.0,
            ],
            axis=1,
        )
        ann["diameter"] = sizes.max(axis=1)

    ann = ann.copy()
    ann["pid"] = ann["pid"].map(canonical_pid)
    ann["gt_index"] = np.arange(len(ann), dtype=np.int64)
    return ann


def load_fn_filter(path):
    """Read evaluator FNs.csv or a compatible FN list."""
    if path is None:
        return None

    fn = pd.read_csv(path)
    pid_col = "seriesuid" if "seriesuid" in fn.columns else "pid"
    if pid_col not in fn.columns:
        raise ValueError("FN CSV must contain 'seriesuid' or 'pid'.")

    rename = {pid_col: "pid"}
    coord_candidates = [
        ("coordX", "coordY", "coordZ"),
        ("center_x", "center_y", "center_z"),
    ]
    coords = None
    for cols in coord_candidates:
        if set(cols).issubset(fn.columns):
            coords = cols
            break
    if coords is None:
        raise ValueError(
            "FN CSV must contain coordX/coordY/coordZ or center_x/center_y/center_z."
        )

    rename.update({coords[0]: "center_x", coords[1]: "center_y", coords[2]: "center_z"})
    fn = fn.rename(columns=rename)[["pid", "center_x", "center_y", "center_z"]].copy()
    fn["pid"] = fn["pid"].map(canonical_pid)
    return fn


def select_fn_annotations(annotations, fn_rows):
    """Map every evaluator FN row to its nearest GT in the same scan."""
    if fn_rows is None:
        return annotations

    selected_indices = []
    for _, fn in fn_rows.iterrows():
        aliases = pid_aliases(fn["pid"])
        candidates = annotations[annotations["pid"].map(lambda x: bool(pid_aliases(x) & aliases))]
        if candidates.empty:
            print("[WARN] FN pid not found in annotations: %s" % fn["pid"])
            continue

        gt_xyz = candidates[["center_x", "center_y", "center_z"]].to_numpy(dtype=np.float32)
        fn_xyz = np.asarray([fn["center_x"], fn["center_y"], fn["center_z"]], dtype=np.float32)
        nearest = int(np.argmin(np.sum((gt_xyz - fn_xyz[None, :]) ** 2, axis=1)))
        selected_indices.append(int(candidates.iloc[nearest]["gt_index"]))

    selected_indices = sorted(set(selected_indices))
    return annotations[annotations["gt_index"].isin(selected_indices)].copy()


def annotations_by_pid(annotations):
    lookup = {}
    for _, row in annotations.iterrows():
        for key in pid_aliases(row["pid"]):
            lookup.setdefault(key, []).append(row)
    return lookup


def rows_for_pid(lookup, pid):
    for key in pid_aliases(pid):
        if key in lookup:
            return lookup[key]
    return []


def as_stage_array(value, source):
    """Normalize stages to [score, z, y, x, d, h, w]."""
    if value is None:
        return np.empty((0, 7), dtype=np.float32)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float32)
    if value.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    value = value.reshape(value.shape[0], -1)

    if source == "rpn":
        # [batch, score, z, y, x, d, h, w]
        return value[:, 1:8]
    if source == "rcnn":
        # [batch, score, z, y, x, d, h, w, class]
        return value[:, 1:8]
    if source == "stage":
        # already [score, z, y, x, d, h, w]
        return value[:, :7]
    raise ValueError("Unknown stage source: %s" % source)


def reconstruct_rpn_stages(net, inputs):
    scores = torch.sigmoid(net.rpn_logits_flat).detach().cpu().numpy()[0, :, 0]
    deltas = net.rpn_deltas_flat.detach().cpu().numpy()[0]
    windows = np.asarray(net.rpn_window)

    boxes = rpn_decode(windows, deltas, net.cfg["box_reg_weight"])
    boxes = clip_boxes(boxes, inputs.shape[2:])
    all_decoded = np.concatenate([scores[:, None], boxes], axis=1).astype(np.float32)

    score_threshold = float(net.cfg["rpn_test_nms_pre_score_threshold"])
    overlap_threshold = float(net.cfg["rpn_test_nms_overlap_threshold"])
    pre_nms_top_n = int(net.cfg.get("rpn_test_pre_nms_top_n", 6000))

    score_indices = np.where(scores > score_threshold)[0]
    after_score = all_decoded[score_indices]

    pre_indices = score_indices
    if pre_nms_top_n > 0 and len(pre_indices) > pre_nms_top_n:
        order = np.argsort(-scores[pre_indices])[:pre_nms_top_n]
        pre_indices = pre_indices[order]
    pre_nms = all_decoded[pre_indices]

    if len(pre_nms):
        nms_input = torch.from_numpy(pre_nms.copy()).float()
        post_nms_tensor, _ = torch_nms(nms_input, overlap_threshold)
        post_nms = post_nms_tensor.detach().cpu().numpy().astype(np.float32)
    else:
        post_nms = np.empty((0, 7), dtype=np.float32)

    rpn_final = as_stage_array(net.rpn_proposals, "rpn")
    return {
        "rpn_all_decoded": all_decoded,
        "rpn_after_score": after_score,
        "rpn_pre_nms": pre_nms,
        "rpn_post_nms": post_nms,
        "rpn_final": rpn_final,
    }


def reconstruct_rcnn_stages(net, inputs):
    empty = np.empty((0, 7), dtype=np.float32)
    if (
        getattr(net, "rcnn_logits", None) is None
        or getattr(net, "rcnn_deltas", None) is None
        or getattr(net, "rpn_proposals", None) is None
        or len(net.rpn_proposals) == 0
    ):
        return {
            "rcnn_all_decoded": empty,
            "rcnn_after_score": empty,
            "rcnn_final": as_stage_array(getattr(net, "detections", None), "rcnn"),
        }

    proposals = net.rpn_proposals.detach().cpu().numpy()
    probs = F.softmax(net.rcnn_logits, dim=1).detach().cpu().numpy()
    num_class = int(net.cfg["num_class"])
    deltas = net.rcnn_deltas.detach().cpu().numpy().reshape(-1, num_class, 6)

    # SANet has one foreground class (class index 1) for nodules.
    foreground_class = 1
    scores = probs[:, foreground_class]
    boxes = rcnn_decode(
        proposals[:, 2:8],
        deltas[:, foreground_class],
        net.cfg["box_reg_weight"],
    )
    boxes = clip_boxes(boxes, inputs.shape[2:])
    all_decoded = np.concatenate([scores[:, None], boxes], axis=1).astype(np.float32)

    threshold = float(net.cfg["rcnn_test_nms_pre_score_threshold"])
    after_score = all_decoded[scores > threshold]
    final = as_stage_array(net.detections, "rcnn")
    return {
        "rcnn_all_decoded": all_decoded,
        "rcnn_after_score": after_score,
        "rcnn_final": final,
    }


def match_stats(stage, gt):
    stage = as_stage_array(stage, "stage")
    if len(stage) == 0:
        return 0, np.nan, np.nan

    gt_x = float(gt["center_x"])
    gt_y = float(gt["center_y"])
    gt_z = float(gt["center_z"])
    radius = max(float(gt["diameter"]) / 2.0, 0.0)

    dz = stage[:, 1] - gt_z
    dy = stage[:, 2] - gt_y
    dx = stage[:, 3] - gt_x
    dist2 = dx * dx + dy * dy + dz * dz
    matched = dist2 < radius * radius

    if not np.any(matched):
        return 0, np.nan, float(np.sqrt(dist2.min()))
    return int(matched.sum()), float(stage[matched, 0].max()), float(np.sqrt(dist2[matched].min()))


def classify_stage(stats):
    # Final detection has priority. A supposedly FN row reaching this state usually
    # indicates that the FN CSV and this checkpoint/split do not correspond.
    if stats["rcnn_final_count"] > 0:
        return "detected_final"

    if stats["rpn_final_count"] > 0:
        if stats["rcnn_all_decoded_count"] == 0:
            return "lost_by_rcnn_bbox_regression"
        if stats["rcnn_after_score_count"] == 0:
            return "lost_by_rcnn_score_threshold"
        return "lost_by_rcnn_nms"

    if stats["rpn_post_nms_count"] > 0:
        return "lost_by_rpn_top_n"
    if stats["rpn_pre_nms_count"] > 0:
        return "lost_by_rpn_nms"
    if stats["rpn_after_score_count"] > 0:
        return "lost_by_pre_nms_topk"
    if stats["rpn_all_decoded_count"] > 0:
        return "lost_by_rpn_score_threshold"
    return "no_localized_rpn_box"


def diagnose_gt(pid, gt, stages):
    row = {
        "pid": canonical_pid(pid),
        "gt_index": int(gt["gt_index"]),
        "gt_center_x": float(gt["center_x"]),
        "gt_center_y": float(gt["center_y"]),
        "gt_center_z": float(gt["center_z"]),
        "gt_diameter": float(gt["diameter"]),
    }

    for name, stage in stages.items():
        count, best_score, nearest_distance = match_stats(stage, gt)
        row[name + "_count"] = count
        row[name + "_best_score"] = best_score
        row[name + "_nearest_center_distance"] = nearest_distance

    row["reason"] = classify_stage(row)
    return row


def save_case_stages(out_dir, pid, stages):
    stage_dir = os.path.join(out_dir, "stage_arrays")
    os.makedirs(stage_dir, exist_ok=True)
    safe_pid = canonical_pid(pid).replace(os.sep, "_")
    np.savez_compressed(
        os.path.join(stage_dir, safe_pid + ".npz"),
        rpn_pre_nms=stages["rpn_pre_nms"],
        rpn_post_nms=stages["rpn_post_nms"],
        rpn_final=stages["rpn_final"],
        rcnn_after_score=stages["rcnn_after_score"],
        rcnn_final=stages["rcnn_final"],
    )


def clear_inference_cache(net):
    cache_names = [
        "rpn_logits_flat",
        "rpn_deltas_flat",
        "rpn_window",
        "rpn_proposals",
        "raw_rpn_proposals",
        "detections",
        "ensemble_proposals",
        "rcnn_logits",
        "rcnn_deltas",
        "keeps",
        "mask_probs",
    ]
    for name in cache_names:
        if hasattr(net, name):
            setattr(net, name, None)
    gc.collect()
    torch.cuda.empty_cache()


def load_model(weight):
    net = SANet(config).cuda()
    checkpoint = torch.load(
        weight,
        map_location="cpu" if not torch.cuda.is_available() else None,
        weights_only=False,
    )
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
    net.load_state_dict(state_dict)
    net.set_mode("eval")
    net.use_rcnn = True
    return net


def parse_args():
    parser = argparse.ArgumentParser(
        description="Locate the filtering stage responsible for SANet/LUNA16 false negatives."
    )
    parser.add_argument("--dataset", default="LUNA16", help="Dataset directory name under DATA_ROOT.")
    parser.add_argument("--weight", required=True, help="SANet checkpoint path.")
    parser.add_argument("--out-dir", required=True, help="Directory for diagnosis CSV files.")
    parser.add_argument("--test-set-name", default=None, help="Optional test split override.")
    parser.add_argument("--anno", default=None, help="Optional annotation CSV override.")
    parser.add_argument(
        "--fn-csv",
        default=None,
        help="Optional evaluator FNs.csv. When supplied, only those GT nodules are diagnosed.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--save-stage-npy",
        action="store_true",
        help="Save compact per-scan pre/post-NMS arrays for manual inspection.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Debug limit; 0 means all cases.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    enable_cpu_fallback()
    os.makedirs(args.out_dir, exist_ok=True)

    dataset_cfg = dataset_config(args.dataset)
    split_path = args.test_set_name or dataset_cfg["test_set_name"]
    annotation_path = args.anno or dataset_cfg["test_anno"]

    annotations = load_annotations(annotation_path)
    fn_rows = load_fn_filter(args.fn_csv)
    selected_annotations = select_fn_annotations(annotations, fn_rows)
    selected_lookup = annotations_by_pid(selected_annotations)

    print("Dataset:", args.dataset)
    print("Split:", split_path)
    print("Annotations:", annotation_path)
    print("Selected GT nodules:", len(selected_annotations))

    dataset = BboxReader(dataset_cfg["DATA_DIR"], split_path, dataset_cfg, mode="eval")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=train_collate,
    )
    net = load_model(args.weight)

    rows = []
    processed_cases = 0
    for i, (input_tensor, truth_bboxes, truth_labels) in enumerate(loader):
        pid = dataset.filenames[i]
        gt_rows = rows_for_pid(selected_lookup, pid)
        if not gt_rows:
            continue
        if args.max_cases > 0 and processed_cases >= args.max_cases:
            break

        print("[%d/%d] %s: %d selected GT" % (i + 1, len(dataset), pid, len(gt_rows)))
        clear_inference_cache(net)
        try:
            input_tensor = Variable(input_tensor).cuda()
            with torch.no_grad():
                net.forward(input_tensor, np.asarray(truth_bboxes), np.asarray(truth_labels))

            stages = reconstruct_rpn_stages(net, input_tensor)
            stages.update(reconstruct_rcnn_stages(net, input_tensor))

            for gt in gt_rows:
                rows.append(diagnose_gt(pid, gt, stages))

            if args.save_stage_npy:
                save_case_stages(args.out_dir, pid, stages)
            processed_cases += 1
        except Exception:
            traceback.print_exc()
            raise
        finally:
            clear_inference_cache(net)

    result = pd.DataFrame(rows)
    result_path = os.path.join(args.out_dir, "fn_stage_diagnosis.csv")
    result.to_csv(result_path, index=False)

    if result.empty:
        summary = pd.DataFrame(columns=["reason", "count"])
    else:
        counts = Counter(result["reason"].tolist())
        summary = pd.DataFrame(
            sorted(counts.items(), key=lambda item: (-item[1], item[0])),
            columns=["reason", "count"],
        )
    summary_path = os.path.join(args.out_dir, "fn_stage_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nDiagnosis summary")
    print(summary.to_string(index=False))
    print("\nSaved:", result_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
