#!/usr/bin/env python3
"""Evaluate RPN/RCNN score ensembles under multiple weight ratios.

Runs inference once per case (use_rcnn=True), then builds weighted scores:
    score = (w_rpn * rpn_score + w_rcnn * rcnn_prob) / (w_rpn + w_rcnn)
using RPN proposal boxes. RCNN probabilities come from the original
get_probability() (eval threshold 0.0). FROC is computed with
froc_evaluation/noduleCADEvaluationLUNA16.py after pid normalization
(as in tools/prepare_froc_evaluation_inputs.py).

Does not save per-case *_detections.npy.
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ["SANET_DISABLE_INTERNAL_DATA_PARALLEL"] = "1"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "build/box"))
sys.path.insert(0, os.path.join(ROOT_DIR, "froc_evaluation"))

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import net.sanet as sanet_module
from config import config, dataset_configs, default_out_dir
from dataset.bbox_reader import BboxReader
from dataset.collate import train_collate
from net.layer.rcnn_nms import get_probability
from net.sanet import SANet

import froc_evaluation.noduleCADEvaluationLUNA16 as froc_eval


DEFAULT_RATIOS = (
    (1, 0),
    (2, 8),
    (4, 6),
    (1, 1),
    (6, 4),
    (8, 2),
    (0, 1),
)
DEFAULT_FP_POINTS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def enable_cpu_fallback():
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


def clear_inference_cache(net):
    cache_names = [
        "rpn_logits_flat",
        "rpn_deltas_flat",
        "rpn_window",
        "rpn_proposals",
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


def normalize_pid(pid):
    text = str(pid).strip()
    if not text:
        return ""
    try:
        return "%05d" % int(float(text))
    except ValueError:
        return text


def unique_pids(values):
    seen = set()
    out = []
    for value in values:
        pid = normalize_pid(value)
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def prepare_evaluation_annotations(annotation_path, eval_dir):
    annotations = pd.read_csv(annotation_path)
    center_columns = {"center_x", "center_y", "center_z", "diameter"}
    if not center_columns.issubset(annotations.columns):
        corner_columns = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
        missing = corner_columns.difference(annotations.columns)
        if missing:
            raise ValueError(
                "Annotation file %s is missing columns: %s"
                % (annotation_path, ", ".join(sorted(missing)))
            )
        annotations = annotations.copy()
        annotations["center_x"] = (annotations["xmin"] + annotations["xmax"]) / 2.0
        annotations["center_y"] = (annotations["ymin"] + annotations["ymax"]) / 2.0
        annotations["center_z"] = (annotations["zmin"] + annotations["zmax"]) / 2.0
        sizes = pd.DataFrame(
            {
                "x": annotations["xmax"] - annotations["xmin"] + 1,
                "y": annotations["ymax"] - annotations["ymin"] + 1,
                "z": annotations["zmax"] - annotations["zmin"] + 1,
            }
        )
        annotations["diameter"] = sizes.max(axis=1)

    annotations = annotations.copy()
    if "pid" in annotations.columns:
        annotations["pid"] = annotations["pid"].map(normalize_pid)

    os.makedirs(eval_dir, exist_ok=True)
    output_path = os.path.join(eval_dir, "annotations.csv")
    annotations.to_csv(output_path, index=False)
    return output_path


def rewrite_pid_column(src_path, dst_path):
    df = pd.read_csv(src_path)
    if "pid" not in df.columns:
        raise ValueError("missing pid column: %s" % src_path)
    df = df.copy()
    df["pid"] = df["pid"].map(normalize_pid)
    df.to_csv(dst_path, index=False)


def prepare_froc_eval_inputs(froc_dir, seriesuids):
    """Mirror tools/prepare_froc_evaluation_inputs.py naming/normalization."""
    froc_dir = Path(froc_dir)
    froc_dir.mkdir(parents=True, exist_ok=True)

    seriesuids_path = froc_dir / "seriesuids.csv"
    with seriesuids_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for pid in seriesuids:
            writer.writerow([pid])

    excluded_path = froc_dir / "annotations_excluded.csv"
    with excluded_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pid", "center_x", "center_y", "center_z", "diameter"])

    annotations_src = froc_dir / "annotations.csv"
    results_src = froc_dir / "results.csv"
    annotations_path = froc_dir / "annotations_froc_eval.csv"
    results_path = froc_dir / "results_froc_eval.csv"
    rewrite_pid_column(annotations_src, annotations_path)
    rewrite_pid_column(results_src, results_path)
    return (
        str(annotations_path),
        str(excluded_path),
        str(seriesuids_path),
        str(results_path),
    )


def proposals_to_rows(pid, proposals_zyxdhw, scores):
    """proposals: [N,6]=z,y,x,d,h,w ; scores: [N] -> CSV rows."""
    if len(proposals_zyxdhw) == 0:
        return []
    rows = []
    for box, score in zip(proposals_zyxdhw, scores):
        z, y, x, d, h, w = [float(v) for v in box]
        rows.append(
            {
                "pid": normalize_pid(pid),
                "center_x": x,
                "center_y": y,
                "center_z": z,
                "diameter": d,
                "probability": float(score),
            }
        )
    return rows


def extract_case_scores(net, inputs):
    """Return RPN boxes/scores and RCNN probs from original get_probability()."""
    proposals = net.rpn_proposals
    if proposals is None or (hasattr(proposals, "__len__") and len(proposals) == 0):
        return (
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    proposals_np = proposals.detach().cpu().numpy()
    # [b, score, z, y, x, d, h, w]
    rpn_scores = proposals_np[:, 1].astype(np.float32)
    boxes = proposals_np[:, 2:8].astype(np.float32)

    if (
        not hasattr(net, "rcnn_logits")
        or net.rcnn_logits is None
        or len(net.rcnn_logits) == 0
        or not hasattr(net, "rcnn_deltas")
        or net.rcnn_deltas is None
    ):
        rcnn_probs = np.zeros_like(rpn_scores)
        return boxes, rpn_scores, rcnn_probs

    # Same path as sanet.py ensemble: use get_probability score column.
    # With rcnn_test_nms_pre_score_threshold=0.0, returned rows match proposals.
    fpr_res = get_probability(
        net.cfg,
        net.mode,
        inputs,
        net.rpn_proposals,
        net.rcnn_logits,
        net.rcnn_deltas,
    )
    fpr_np = fpr_res.detach().cpu().numpy()
    if fpr_np.shape[0] != len(rpn_scores):
        raise RuntimeError(
            "get_probability rows (%d) != rpn_proposals (%d); "
            "check rcnn_test_nms_pre_score_threshold (expected 0.0 in eval)."
            % (fpr_np.shape[0], len(rpn_scores))
        )
    rcnn_probs = fpr_np[:, 0].astype(np.float32)
    return boxes, rpn_scores, rcnn_probs


def weighted_scores(rpn_scores, rcnn_probs, w_rpn, w_rcnn):
    total = float(w_rpn) + float(w_rcnn)
    if total <= 0:
        raise ValueError("weights must sum to > 0")
    return (float(w_rpn) * rpn_scores + float(w_rcnn) * rcnn_probs) / total


def read_froc_curve(froc_txt_path):
    fps, sens, thresholds = [], [], []
    with open(froc_txt_path, newline="") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
            if len(parts) < 2:
                continue
            try:
                fp = float(parts[0])
                se = float(parts[1])
                th = float(parts[2]) if len(parts) > 2 else float("nan")
            except ValueError:
                continue
            fps.append(fp)
            sens.append(se)
            thresholds.append(th)
    return np.asarray(fps, dtype=np.float64), np.asarray(sens, dtype=np.float64), np.asarray(thresholds, dtype=np.float64)


def interpolate_at_points(fps, sens, thresholds, points):
    if len(fps) == 0:
        return {p: (float("nan"), float("nan")) for p in points}
    order = np.argsort(fps)
    fps_s = fps[order]
    sens_s = sens[order]
    thr_s = thresholds[order]
    out = {}
    for p in points:
        se = float(np.interp(p, fps_s, sens_s, left=sens_s[0], right=sens_s[-1]))
        # threshold is not necessarily monotone with fps; nearest fps sample
        idx = int(np.argmin(np.abs(fps_s - p)))
        th = float(thr_s[idx]) if len(thr_s) else float("nan")
        out[p] = (se, th)
    return out


def run_froc(annotations_path, excluded_path, seriesuids_path, results_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Disable bootstrap for multi-ratio sweep speed.
    old_flag = froc_eval.bPerformBootstrapping
    froc_eval.bPerformBootstrapping = False
    try:
        froc_eval.noduleCADEvaluation(
            annotations_path,
            excluded_path,
            seriesuids_path,
            results_path,
            output_dir,
        )
    finally:
        froc_eval.bPerformBootstrapping = old_flag

    stem = os.path.splitext(os.path.basename(results_path))[0]
    froc_txt = os.path.join(output_dir, "froc_%s.txt" % stem)
    if not os.path.exists(froc_txt):
        # fallback: any froc_*.txt
        candidates = sorted(Path(output_dir).glob("froc_*.txt"))
        if not candidates:
            raise FileNotFoundError("FROC curve not found in %s" % output_dir)
        froc_txt = str(candidates[0])
    return froc_txt


def infer_dataset(net, dataset_cfg, test_set_name, num_workers):
    dataset = BboxReader(dataset_cfg["DATA_DIR"], test_set_name, dataset_cfg, mode="eval")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=train_collate,
    )

    net.set_mode("eval")
    net.use_rcnn = True

    case_cache = []  # list of dicts
    print("Total # of eval data %d" % len(loader))
    for i, (input_tensor, truth_bboxes, truth_labels) in enumerate(tqdm(loader, desc=dataset_cfg["dataset"])):
        try:
            clear_inference_cache(net)
            input_tensor = Variable(input_tensor).cuda()
            truth_bboxes = np.asarray(truth_bboxes)
            truth_labels = np.asarray(truth_labels)
            pid = dataset.filenames[i]

            with torch.no_grad():
                net.forward(input_tensor, truth_bboxes, truth_labels)

            boxes, rpn_scores, rcnn_probs = extract_case_scores(net, input_tensor)
            case_cache.append(
                {
                    "pid": pid,
                    "boxes": boxes,
                    "rpn_scores": rpn_scores,
                    "rcnn_probs": rcnn_probs,
                }
            )
        except Exception:
            traceback.print_exc()
            raise
        finally:
            clear_inference_cache(net)
            del input_tensor, truth_bboxes, truth_labels
            torch.cuda.empty_cache()

    return dataset, case_cache


def evaluate_ratio_on_dataset(
    dataset_cfg,
    dataset,
    case_cache,
    work_dir,
    w_rpn,
    w_rcnn,
    fp_points,
):
    ratio_name = "%d:%d" % (w_rpn, w_rcnn)
    froc_dir = Path(work_dir) / ("ratio_%s" % ratio_name.replace(":", "-"))
    froc_dir.mkdir(parents=True, exist_ok=True)

    # annotations once per dataset root; copy/link if already prepared at work_dir
    parent_anno = Path(work_dir) / "annotations.csv"
    if not parent_anno.exists():
        prepare_evaluation_annotations(dataset_cfg["test_anno"], str(work_dir))
    anno_dst = froc_dir / "annotations.csv"
    if not anno_dst.exists():
        pd.read_csv(parent_anno).to_csv(anno_dst, index=False)

    rows = []
    for case in case_cache:
        scores = weighted_scores(case["rpn_scores"], case["rcnn_probs"], w_rpn, w_rcnn)
        rows.extend(proposals_to_rows(case["pid"], case["boxes"], scores))

    results_df = pd.DataFrame(
        rows, columns=["pid", "center_x", "center_y", "center_z", "diameter", "probability"]
    )
    results_path = froc_dir / "results.csv"
    results_df.to_csv(results_path, index=False)

    seriesuids = unique_pids([c["pid"] for c in case_cache])
    if not seriesuids:
        # fallback to results/annotations pids
        seriesuids = unique_pids(results_df["pid"].tolist()) if len(results_df) else []
        if not seriesuids and anno_dst.exists():
            seriesuids = unique_pids(pd.read_csv(anno_dst)["pid"].tolist())

    annotations_path, excluded_path, seriesuids_path, results_eval_path = prepare_froc_eval_inputs(
        froc_dir, seriesuids
    )
    res_out = froc_dir / "res_froc_eval"
    froc_txt = run_froc(
        annotations_path,
        excluded_path,
        seriesuids_path,
        results_eval_path,
        str(res_out),
    )
    fps, sens, thresholds = read_froc_curve(froc_txt)
    point_vals = interpolate_at_points(fps, sens, thresholds, fp_points)
    recalls = [point_vals[p][0] for p in fp_points]
    froc_mean = float(np.nanmean(recalls)) if recalls else float("nan")

    row = {
        "dataset": dataset_cfg["dataset"],
        "ratio": ratio_name,
        "rpn_weight": w_rpn,
        "rcnn_weight": w_rcnn,
        "num_candidates": int(len(results_df)),
        "froc_mean": froc_mean,
        "froc_curve_file": froc_txt,
    }
    for p in fp_points:
        se, th = point_vals[p]
        key = ("%.3f" % p).rstrip("0").rstrip(".")
        row["sens@%s" % key] = se
        row["thr@%s" % key] = th
    return row


def parse_ratios(text):
    ratios = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("bad ratio %r, expected like 2:8" % item)
        a, b = item.split(":", 1)
        ratios.append((int(a), int(b)))
    if not ratios:
        raise ValueError("no ratios provided")
    return ratios


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep RPN:RCNN ensemble weights and report FROC.")
    parser.add_argument("--net", "-m", metavar="NET", default=config["net"], help="neural net")
    parser.add_argument("--weight", type=str, required=True, help="path to model weights")
    parser.add_argument("--out-dir", type=str, default="test_ensemble_output", help="output directory")
    parser.add_argument("--test-set-name", type=str, default=None, help="optional test list override")
    parser.add_argument("--dataset", default=config["dataset"], type=str, help="dataset name or 'all'")
    parser.add_argument(
        "--ratios",
        type=str,
        default=",".join("%d:%d" % r for r in DEFAULT_RATIOS),
        help="comma-separated RPN:RCNN weights, e.g. 1:0,2:8,1:1,0:1",
    )
    parser.add_argument(
        "--fp-points",
        type=str,
        default=",".join(str(x) for x in DEFAULT_FP_POINTS),
        help="FP/scan points for summary CSV",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    return parser.parse_args()


def main():
    logging.basicConfig(format="[%(levelname)s][%(asctime)s] %(message)s", level=logging.INFO)
    args = parse_args()
    enable_cpu_fallback()

    ratios = parse_ratios(args.ratios)
    fp_points = tuple(float(x) for x in args.fp_points.replace(";", ",").split(",") if x.strip())
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 0 if not torch.cuda.is_available() else 8

    cfgs = dataset_configs(args.dataset, skip_missing=(args.dataset == "all"))
    net = SANet(config)
    net = net.cuda()

    print("[Loading model from %s]" % args.weight)
    checkpoint = torch.load(
        args.weight,
        map_location="cpu" if not torch.cuda.is_available() else None,
        weights_only=False,
    )
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    net.load_state_dict(state)
    epoch = checkpoint.get("epoch", "na") if isinstance(checkpoint, dict) else "na"

    out_root = Path(args.out_dir or default_out_dir(args.dataset))
    exp_dir = out_root / "res" / ("ensemble_ratio_sweep_epoch_%s" % epoch)
    exp_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for dataset_cfg in cfgs:
        test_set_name = args.test_set_name or dataset_cfg["test_set_name"]
        work_dir = exp_dir / dataset_cfg["dataset"]
        work_dir.mkdir(parents=True, exist_ok=True)
        print("dataset", dataset_cfg["dataset"])
        try:
            dataset, case_cache = infer_dataset(net, dataset_cfg, test_set_name, num_workers)
        except (FileNotFoundError, ValueError) as exc:
            if args.dataset != "all":
                raise
            print("[%s] Skip: %s" % (dataset_cfg["dataset"], exc))
            continue

        prepare_evaluation_annotations(dataset_cfg["test_anno"], str(work_dir))

        for w_rpn, w_rcnn in ratios:
            print("[%s] evaluating ratio %d:%d" % (dataset_cfg["dataset"], w_rpn, w_rcnn))
            row = evaluate_ratio_on_dataset(
                dataset_cfg,
                dataset,
                case_cache,
                str(work_dir),
                w_rpn,
                w_rcnn,
                fp_points,
            )
            summary_rows.append(row)
            print(
                "[%s] ratio %s froc_mean=%.6f candidates=%d"
                % (row["dataset"], row["ratio"], row["froc_mean"], row["num_candidates"])
            )

    if not summary_rows:
        print("No results to write")
        return

    summary_df = pd.DataFrame(summary_rows)
    # Stable column order
    base_cols = ["dataset", "ratio", "rpn_weight", "rcnn_weight", "num_candidates", "froc_mean"]
    sens_cols = []
    thr_cols = []
    for p in fp_points:
        key = ("%.3f" % p).rstrip("0").rstrip(".")
        sens_cols.append("sens@%s" % key)
        thr_cols.append("thr@%s" % key)
    cols = base_cols + sens_cols + thr_cols + ["froc_curve_file"]
    cols = [c for c in cols if c in summary_df.columns] + [c for c in summary_df.columns if c not in cols]
    summary_df = summary_df[cols]

    summary_path = exp_dir / "ensemble_ratio_froc_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("Saved summary: %s" % summary_path)


if __name__ == "__main__":
    main()

'''
python test_ensemble.py \
  --dataset all \
  --weight /mnt/afs2/code/SANet/train_pretrained_hardsamples_rcnn40_PN11_v5/model/best_rcnn.ckpt \
  --out-dir /mnt/afs2/code/SANet/test_ensemble_output \
  --ratios 1:0,2:8,4:6,1:1,6:4,8:2,0:1
'''
