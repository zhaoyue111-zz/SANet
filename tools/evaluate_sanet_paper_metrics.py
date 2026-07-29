#!/usr/bin/env python3
"""Evaluate SANet test outputs with paper-style metrics.

This script scans one result root and evaluates each dataset directory that
contains a ``FROC/results.csv`` and ``FROC/annotations.csv`` pair.

Metrics:
  - FROCIoU at IoU threshold 0.25
  - AP@0.25 and AP@0.35
  - APs / APm / APl (size-stratified AP, default IoU threshold 0.25)

The script assumes SANet-style CSV outputs from ``test.py``:
  - predictions: pid, center_x, center_y, center_z, diameter, probability
  - annotations: pid, center_x, center_y, center_z, diameter, ...

If annotations only contain corner boxes, they will be converted to
center/diameter format before evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


FROC_POINTS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
DEFAULT_IOU_THRESHOLDS = [0.25, 0.35]
SIZE_AP_IOU_THRESHOLD = 0.25


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def parse_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = sanitize_columns(df)
    if "probability" not in df.columns:
        if "score" in df.columns:
            df["probability"] = df["score"]
        else:
            raise ValueError("Prediction CSV must contain a probability/score column.")

    center_cols = {"center_x", "center_y", "center_z", "diameter"}
    if center_cols.issubset(df.columns):
        out = df[["pid", "center_x", "center_y", "center_z", "diameter", "probability"]].copy()
        return out

    corner_cols = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
    missing = corner_cols.difference(df.columns)
    if missing:
        raise ValueError(
            "Annotation/prediction file must contain either center_x/center_y/center_z/diameter "
            "or xmin/xmax/ymin/ymax/zmin/zmax. Missing: %s" % ", ".join(sorted(missing))
        )

    out = pd.DataFrame()
    out["pid"] = df["pid"]
    out["center_x"] = (df["xmin"] + df["xmax"]) / 2.0
    out["center_y"] = (df["ymin"] + df["ymax"]) / 2.0
    out["center_z"] = (df["zmin"] + df["zmax"]) / 2.0
    dx = df["xmax"] - df["xmin"] + 1.0
    dy = df["ymax"] - df["ymin"] + 1.0
    dz = df["zmax"] - df["zmin"] + 1.0
    out["diameter"] = np.maximum.reduce([dx.to_numpy(), dy.to_numpy(), dz.to_numpy()])
    out["probability"] = df["probability"]
    return out


def parse_annotations(df: pd.DataFrame) -> pd.DataFrame:
    df = sanitize_columns(df)
    center_cols = {"center_x", "center_y", "center_z", "diameter"}
    if center_cols.issubset(df.columns):
        return df[["pid", "center_x", "center_y", "center_z", "diameter"]].copy()

    corner_cols = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
    missing = corner_cols.difference(df.columns)
    if missing:
        raise ValueError(
            "Annotation CSV must contain either center_x/center_y/center_z/diameter "
            "or xmin/xmax/ymin/ymax/zmin/zmax. Missing: %s" % ", ".join(sorted(missing))
        )

    out = pd.DataFrame()
    out["pid"] = df["pid"]
    out["center_x"] = (df["xmin"] + df["xmax"]) / 2.0
    out["center_y"] = (df["ymin"] + df["ymax"]) / 2.0
    out["center_z"] = (df["zmin"] + df["zmax"]) / 2.0
    dx = df["xmax"] - df["xmin"] + 1.0
    dy = df["ymax"] - df["ymin"] + 1.0
    dz = df["zmax"] - df["zmin"] + 1.0
    out["diameter"] = np.maximum.reduce([dx.to_numpy(), dy.to_numpy(), dz.to_numpy()])
    return out


def cube_to_corners(center_x: float, center_y: float, center_z: float, diameter: float) -> Tuple[float, float, float, float, float, float]:
    half = float(diameter) / 2.0
    return (
        float(center_x) - half,
        float(center_x) + half,
        float(center_y) - half,
        float(center_y) + half,
        float(center_z) - half,
        float(center_z) + half,
    )


def iou_3d_cube(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ax2, ay1, ay2, az1, az2 = a
    bx1, bx2, by1, by2, bz1, bz2 = b

    ix1 = max(ax1, bx1)
    ix2 = min(ax2, bx2)
    iy1 = max(ay1, by1)
    iy2 = min(ay2, by2)
    iz1 = max(az1, bz1)
    iz2 = min(az2, bz2)

    inter_x = max(0.0, ix2 - ix1)
    inter_y = max(0.0, iy2 - iy1)
    inter_z = max(0.0, iz2 - iz1)
    inter = inter_x * inter_y * inter_z
    if inter <= 0:
        return 0.0

    va = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) * max(0.0, az2 - az1)
    vb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1) * max(0.0, bz2 - bz1)
    denom = va + vb - inter
    return float(inter / denom) if denom > 0 else 0.0


def ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if len(recalls) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def greedy_match_predictions(
    preds: List[Dict[str, float]],
    gts: List[Dict[str, float]],
    iou_thr: float,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Match predictions to GTs greedily by descending score.

    Returns:
      tp_flags, fp_flags, matched_ious
    """
    if not preds:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), []

    order = np.argsort([-p["score"] for p in preds], kind="mergesort")
    matched = np.zeros((len(gts),), dtype=bool)
    tp = np.zeros((len(preds),), dtype=np.int32)
    fp = np.zeros((len(preds),), dtype=np.int32)
    matched_ious: List[float] = []

    gt_boxes = [g["box"] for g in gts]
    for out_idx, pred_idx in enumerate(order):
        pred = preds[pred_idx]
        best_iou = 0.0
        best_gt = -1
        for gi, gt_box in enumerate(gt_boxes):
            if matched[gi]:
                continue
            iou = iou_3d_cube(pred["box"], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_gt >= 0 and best_iou >= iou_thr:
            matched[best_gt] = True
            tp[pred_idx] = 1
            matched_ious.append(best_iou)
        else:
            fp[pred_idx] = 1
    return tp, fp, matched_ious


def evaluate_detection_lists(
    preds: List[Dict[str, float]],
    gts: List[Dict[str, float]],
    iou_thr: float,
    n_scans: int,
) -> Dict[str, float]:
    if n_scans <= 0:
        n_scans = 1

    tp, fp, matched_ious = greedy_match_predictions(preds, gts, iou_thr)
    num_gt = len(gts)
    num_pred = len(preds)

    if num_gt == 0:
        return {
            "num_gt": 0,
            "num_pred": num_pred,
            "ap": 0.0,
            "froc_mean": 0.0,
            **{f"froc@{p:g}": 0.0 for p in FROC_POINTS},
        }

    if num_pred == 0:
        return {
            "num_gt": num_gt,
            "num_pred": 0,
            "ap": 0.0,
            "froc_mean": 0.0,
            **{f"froc@{p:g}": 0.0 for p in FROC_POINTS},
        }

    order = np.argsort([-p["score"] for p in preds], kind="mergesort")
    tp_sorted = tp[order]
    fp_sorted = fp[order]
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(fp_sorted)

    recalls = tp_cum / float(num_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    ap = ap_from_pr(recalls, precisions)

    sensitivity = recalls
    fp_per_scan = fp_cum / float(n_scans)
    froc_at_points = {}
    for target in FROC_POINTS:
        valid = np.where(fp_per_scan <= target)[0]
        froc_at_points[f"froc@{target:g}"] = float(np.max(sensitivity[valid])) if len(valid) else 0.0
    froc_mean = float(np.mean(list(froc_at_points.values())))

    result = {
        "num_gt": num_gt,
        "num_pred": num_pred,
        "ap": float(ap),
        "froc_mean": froc_mean,
        "matched": int(tp.sum()),
        "precision_final": float(tp.sum() / max(tp.sum() + fp.sum(), 1)),
        "recall_final": float(tp.sum() / max(num_gt, 1)),
        "mean_iou_of_matched": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }
    result.update(froc_at_points)
    return result


def size_bucket(diameter: float) -> str:
    volume = float(diameter) ** 3
    if volume < 512.0:
        return "s"
    if volume < 4096.0:
        return "m"
    return "l"


def group_by_pid(df: pd.DataFrame) -> Dict[str, List[Dict[str, float]]]:
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for _, row in df.iterrows():
        pid = str(row["pid"])
        pred = {
            "pid": pid,
            "score": float(row["probability"]),
            "box": cube_to_corners(row["center_x"], row["center_y"], row["center_z"], row["diameter"]),
            "diameter": float(row["diameter"]),
        }
        grouped.setdefault(pid, []).append(pred)
    return grouped


def group_gt_by_pid(df: pd.DataFrame) -> Dict[str, List[Dict[str, float]]]:
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for _, row in df.iterrows():
        pid = str(row["pid"])
        gt = {
            "pid": pid,
            "box": cube_to_corners(row["center_x"], row["center_y"], row["center_z"], row["diameter"]),
            "diameter": float(row["diameter"]),
            "bucket": size_bucket(float(row["diameter"])),
        }
        grouped.setdefault(pid, []).append(gt)
    return grouped


def evaluate_dataset(results_csv: Path, annotations_csv: Path) -> Dict[str, float]:
    preds_raw = pd.read_csv(results_csv)
    ann_raw = pd.read_csv(annotations_csv)

    if len(preds_raw) == 0:
        preds_df = pd.DataFrame(columns=["pid", "center_x", "center_y", "center_z", "diameter", "probability"])
    else:
        preds_df = parse_predictions(preds_raw)

    if len(ann_raw) == 0:
        ann_df = pd.DataFrame(columns=["pid", "center_x", "center_y", "center_z", "diameter"])
    else:
        ann_df = parse_annotations(ann_raw)

    preds_by_pid = group_by_pid(preds_df)
    gts_by_pid = group_gt_by_pid(ann_df)
    pids = sorted(set(preds_by_pid.keys()) | set(gts_by_pid.keys()), key=lambda x: (len(str(x)), str(x)))

    all_preds: List[Dict[str, float]] = []
    all_gts: List[Dict[str, float]] = []
    size_preds: Dict[str, List[Dict[str, float]]] = {"s": [], "m": [], "l": []}
    size_gts: Dict[str, List[Dict[str, float]]] = {"s": [], "m": [], "l": []}

    for pid in pids:
        preds = preds_by_pid.get(pid, [])
        gts = gts_by_pid.get(pid, [])
        all_preds.extend(preds)
        all_gts.extend(gts)
        for gt in gts:
            size_gts[gt["bucket"]].append(gt)
        for pred in preds:
            # Prediction size bucket is not used in the paper metrics, but we keep
            # predictions tied to the same GT bucket evaluation by reusing all preds.
            pass

    n_scans = len(pids)
    metrics: Dict[str, float] = {
        "num_scans": float(n_scans),
        "num_gt": float(len(all_gts)),
        "num_pred": float(len(all_preds)),
    }

    for thr in DEFAULT_IOU_THRESHOLDS:
        det = evaluate_detection_lists(all_preds, all_gts, thr, n_scans)
        metrics[f"ap@{thr:g}"] = det["ap"]
        if math.isclose(thr, 0.25):
            metrics["froc_iou_mean@0.25"] = det["froc_mean"]
            for fp in FROC_POINTS:
                metrics[f"froc_iou@{fp:g}"] = det[f"froc@{fp:g}"]

        size_det_s = evaluate_detection_lists(all_preds, size_gts["s"], thr, n_scans)
        size_det_m = evaluate_detection_lists(all_preds, size_gts["m"], thr, n_scans)
        size_det_l = evaluate_detection_lists(all_preds, size_gts["l"], thr, n_scans)
        metrics[f"aps@{thr:g}"] = size_det_s["ap"]
        metrics[f"apm@{thr:g}"] = size_det_m["ap"]
        metrics[f"apl@{thr:g}"] = size_det_l["ap"]

    # Paper table usually reports APs/APm/APl once; default to IoU 0.25.
    metrics["AP@0.25"] = metrics["ap@0.25"]
    metrics["AP@0.35"] = metrics["ap@0.35"]
    metrics["APs"] = metrics["aps@0.25"]
    metrics["APm"] = metrics["apm@0.25"]
    metrics["APl"] = metrics["apl@0.25"]
    metrics["FROCIoU"] = metrics["froc_iou_mean@0.25"]
    return metrics


def discover_dataset_dirs(root: Path) -> List[Tuple[str, Path]]:
    root = root.resolve()
    found: List[Tuple[str, Path]] = []

    # Case 1: root itself is a dataset directory with FROC subdir.
    if (root / "FROC" / "results.csv").is_file():
        found.append((root.name, root))
        return found

    # Case 2: root contains dataset subdirectories.
    for path in sorted(root.glob("*/FROC/results.csv")):
        dataset_dir = path.parent.parent
        found.append((dataset_dir.name, dataset_dir))

    # Case 3: nested search as fallback.
    if not found:
        for path in sorted(root.rglob("FROC/results.csv")):
            dataset_dir = path.parent.parent
            found.append((dataset_dir.name, dataset_dir))

    # Deduplicate by dataset directory path.
    unique: Dict[Path, Tuple[str, Path]] = {}
    for name, d in found:
        unique[d.resolve()] = (name, d.resolve())
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SANet paper metrics from test.py outputs.")
    parser.add_argument(
        "--res-root",
        type=Path,
        default=Path("test_output/res/14"),
        help="Root directory containing dataset FROC outputs.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path to save the summary table.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path to save the summary table.",
    )
    args = parser.parse_args()

    dataset_dirs = discover_dataset_dirs(args.res_root)
    if not dataset_dirs:
        raise FileNotFoundError(
            f"No dataset directories with FROC/results.csv found under {args.res_root}"
        )

    rows = []
    for dataset_name, dataset_dir in dataset_dirs:
        results_csv = dataset_dir / "FROC" / "results.csv"
        annotations_csv = dataset_dir / "FROC" / "annotations.csv"
        if not results_csv.is_file():
            raise FileNotFoundError(results_csv)
        if not annotations_csv.is_file():
            raise FileNotFoundError(annotations_csv)

        metrics = evaluate_dataset(results_csv, annotations_csv)
        metrics["dataset"] = dataset_name
        metrics["dataset_dir"] = str(dataset_dir)
        rows.append(metrics)

    df = pd.DataFrame(rows)
    cols = [
        "dataset",
        "num_scans",
        "num_gt",
        "num_pred",
        "FROCIoU",
        "froc_iou_mean@0.25",
        "AP@0.25",
        "AP@0.35",
        "APs",
        "APm",
        "APl",
        "aps@0.35",
        "apm@0.35",
        "apl@0.35",
    ]
    cols = [c for c in cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in cols and c not in {"dataset_dir"}]
    df = df[cols + other_cols + ["dataset_dir"]]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print(df.to_string(index=False))

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
