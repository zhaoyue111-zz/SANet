# NOTE:
# The provided screenshots begin at source line 35. Lines 1-34 are not visible.
# `import os` is required by the visible code below and is added here so this
# extracted file is runnable. Please provide lines 1-34 if exact restoration is needed.

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter


# FROC evaluation parameters
FROC_minX = 0.125
FROC_maxX = 32
bLogPlot = True

# Validation list path
VALIDATION_LIST_PATH = "/2026aicompetition/workspace/tools/split/val.txt"


def load_val_list(val_list_path):
    """Load validation series list from file.

    Format: one line per series: patient_id/studyInstanceUID/seriesInstanceUID
    Returns: set of series IDs (seriesInstanceUID)
    """
    valid_series = set()
    with open(val_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("/")
            if len(parts) == 3:
                patient_id, study_uid, series_uid = parts
                valid_series.add(series_uid)
    return valid_series


def find_npz_path(data_root, patient_id, study_id, series_id):
    """Find the npz file path for a given series."""
    return os.path.join(
        data_root,
        str(patient_id),
        str(study_id),
        str(series_id),
        f"{series_id}_buffer.npz",
    )


spacing_cache = {}


def load_spacing(data_root, row):
    """Load spacing from npz file with caching.

    row must have:
      - 'pid': seriesInstanceUID (which is used as filename)
      - 'patient_id': for directory path
      - 'study_id': for directory path
    """
    series_uid = str(row["pid"])

    # Use series_uid directly as cache key
    if series_uid in spacing_cache:
        return spacing_cache[series_uid]

    # Find npz file using series_uid
    npz_path = os.path.join(
        data_root,
        str(row["patient_id"]),
        str(row["study_id"]),
        series_uid,
        f"{series_uid}_buffer.npz",
    )

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Cannot find npz file: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        spacing = np.asarray(data["spacing"], dtype=np.float32)

    spacing_cache[series_uid] = spacing
    return spacing


def voxel_xyz_to_mm(xyz, spacing):
    """Convert voxel coordinates to millimeters.

    spacing = [z, y, x]
    """
    return np.array(
        [
            xyz[0] * spacing[2],  # x: use x spacing
            xyz[1] * spacing[1],  # y: use y spacing
            xyz[2] * spacing[0],  # z: use z spacing
        ],
        dtype=np.float32,
    )


def load_gt(path, data_root, val_list_path=None):
    """Load ground truth annotations."""
    df = pd.read_csv(path)

    required = [
        "pid",
        "patient_id",
        "studyInstanceUID",
        "seriesInstanceUID",
        "bbox_min_z",
        "bbox_min_y",
        "bbox_min_x",
        "bbox_max_z",
        "bbox_max_y",
        "bbox_max_x",
        "diameter",
    ]

    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing GT column: {c}")

    # Filter GT by validation list FIRST
    # (before renaming, using original column name)
    if val_list_path is not None:
        valid_series = load_val_list(val_list_path)
        df = df[df["seriesInstanceUID"].isin(valid_series)]
        print(
            f"GT: {len(df)} samples from "
            f"{df['seriesInstanceUID'].nunique()} series (validation set)"
        )
    else:
        print(
            f"GT: {len(df)} samples from "
            f"{df['seriesInstanceUID'].nunique()} series (all data)"
        )

    # Drop the original integer pid column, then rename
    df = df.drop(columns=["pid"])  # Remove integer pid
    df = df.rename(
        columns={
            "seriesInstanceUID": "pid",  # Use series UID as pid for matching
            "studyInstanceUID": "study_id",
        }
    )

    # Also keep patient_id for spacing lookup
    return df


def load_pred(path):
    """Load prediction results.

    Expected columns: pid, center_x, center_y, center_z, probability
    Other columns will be ignored.
    """
    df = pd.read_csv(path)

    required = [
        "pid",
        "center_x",
        "center_y",
        "center_z",
        "probability",
    ]

    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing prediction column: {c}")

    # Keep only required columns
    df = df[required]

    print(f"Predictions: {len(df)} candidates")
    return df


def match_case(gt_case, pred_case, spacing):
    """
    Match predictions to ground truth for a single patient.
    This is the CORE matching logic - UNCHANGED from original.

    Returns:
        gt_records: list of dicts with matching info for each GT
        fp_records: list of dicts for unmatched predictions
    """
    # Convert all predictions to mm coordinates
    pred_mm = []
    for _, p in pred_case.iterrows():
        pred_mm.append(
            voxel_xyz_to_mm(
                [p.center_x, p.center_y, p.center_z],
                spacing,
            )
        )
    pred_mm = np.asarray(pred_mm)

    used = set()
    gt_records = []
    gt_idx = 0  # 使用独立计数器，而非DataFrame索引

    # Match each GT to best prediction
    for _, gt in gt_case.iterrows():
        # Calculate GT center in voxel space
        center_voxel = [
            (gt.bbox_min_x + gt.bbox_max_x) / 2,
            (gt.bbox_min_y + gt.bbox_max_y) / 2,
            (gt.bbox_min_z + gt.bbox_max_z) / 2,
        ]

        # Convert to mm
        center_mm = voxel_xyz_to_mm(center_voxel, spacing)

        # Calculate threshold
        threshold = min(
            max(float(gt.diameter) * 0.6, 2.0),
            15.0,
        )

        # Find all predictions within threshold
        candidates = []
        for i, p in enumerate(pred_mm):
            if i in used:
                continue

            dist = np.linalg.norm(p - center_mm)

            if dist <= threshold:
                candidates.append(
                    (i, dist, float(pred_case.iloc[i].probability))
                )

        if candidates:
            # Sort by probability (descending) - keep highest only
            candidates.sort(key=lambda x: x[2], reverse=True)
            best = candidates[0]
            used.add(best[0])

            gt_records.append(
                {
                    "gt_id": gt_idx,  # 使用相对计数器，而非DataFrame索引
                    "diameter": gt.diameter,
                    "threshold_mm": threshold,
                    "matched_probability": best[2],
                    "distance_mm": best[1],
                    "status": "TP",
                }
            )
        else:
            # No matched prediction = False Negative
            gt_records.append(
                {
                    "gt_id": gt_idx,  # 使用独立计数器，而非DataFrame索引
                    "diameter": gt.diameter,
                    "threshold_mm": threshold,
                    "matched_probability": np.nan,
                    "distance_mm": np.nan,
                    "status": "FN",
                }
            )

        gt_idx += 1  # 递增计数器

    # Unmatched predictions = False Positives
    fp_records = []
    for i, p in pred_case.iterrows():
        if i not in used:
            fp_records.append(
                {
                    "pid": p.pid,
                    "center_x": p.center_x,
                    "center_y": p.center_y,
                    "center_z": p.center_z,
                    "probability": p.probability,
                }
            )

    return gt_records, fp_records


def precompute_matches(gt_df, pred_df, data_root):
    """
    One-time precompute all GT matching results.

    This is the optimization: we only call match_case ONCE per GT,
    not once per threshold like the original slow version.
    """
    print("Precomputing matches...")

    tp_scores = []  # probabilities of TP predictions
    fn_count = 0
    all_fp = []
    all_match_records = []

    for pid, gt_case in tqdm(gt_df.groupby("pid"), desc="Matching"):
        pred_case = pred_df[pred_df.pid == pid]

        if len(pred_case) == 0:
            continue

        spacing = load_spacing(data_root, gt_case.iloc[0])

        # Use original match_case logic - UNCHANGED
        m, f = match_case(gt_case, pred_case, spacing)

        for x in m:
            x["pid"] = pid  # Add pid for saving
            if x["status"] == "TP":
                tp_scores.append(x["matched_probability"])
            else:
                fn_count += 1
            all_match_records.append(x)

        all_fp.extend(f)

    total_scans = gt_df.pid.nunique()

    print(
        f"Precomputation done: {len(tp_scores)} TP, "
        f"{fn_count} FN, {len(all_fp)} FP"
    )

    return tp_scores, fn_count, all_fp, total_scans, all_match_records


def evaluate_froc_fast(
    tp_scores,
    fn_count,
    fp_records,
    pred_df,
    gt_df,
    total_scans,
    output_dir,
    curve_name="all",
    dataset_name="test_output",
):
    """Calculate FROC curve from precomputed matches."""
    print(f"Calculating FROC curve for {curve_name}...")

    # Get all unique thresholds (descending)
    thresholds = sorted(pred_df.probability.unique(), reverse=True)

    # Precompute probability arrays
    tp_probs = np.array(sorted(tp_scores, reverse=True))
    fp_probs = (
        np.array([float(x["probability"]) for x in fp_records])
        if fp_records
        else np.array([])
    )

    output = []
    total_gt = len(gt_df)
    total_candidates = len(pred_df)

    for threshold in tqdm(thresholds, desc="Computing"):
        # Count TPs and FPs at this threshold
        tp = np.sum(tp_probs >= threshold)
        fp_raw = np.sum(fp_probs >= threshold) if len(fp_probs) > 0 else 0

        # 关键修复：FP应该排除已经被匹配为TP的预测
        # 在threshold下的有效FP = fp_raw - tp（因为tp数量的预测已经被计为TP了）
        # 但不能为负数
        fp = max(0, fp_raw - tp)

        output.append(
            {
                "threshold": threshold,
                "FP_per_scan": fp / total_scans,
                "sensitivity": tp / total_gt,
            }
        )

    froc_df = pd.DataFrame(output)

    # Save FROC curve data
    froc_df.to_csv(
        os.path.join(output_dir, f"froc_{curve_name}.csv"),
        index=False,
    )

    # Save recall_table.csv (FROC values at specific thresholds)
    save_recall_table(
        froc_df,
        output_dir,
        curve_name,
        dataset_name,
    )

    # Plot FROC curve
    plot_froc_curve(froc_df, output_dir, curve_name)

    # Calculate FROC score (mean sensitivity at specific FP rates)
    calculate_froc_score(froc_df, output_dir, curve_name)

    return froc_df


def save_recall_table(
    froc_df,
    output_dir,
    curve_name,
    dataset_name="test_output",
):
    """
    保存 recall_table.csv，记录特定FP/scan阈值下的灵敏度值。
    格式与 LUNA16 的 recall_table.csv 一致。
    """
    # 定义要记录的 FP/scan 阈值
    fp_thresholds = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16]

    # 创建一个空的 recall_table
    recall_row = {"dataset": dataset_name}

    for fp_thresh in fp_thresholds:
        # 找到最接近该 FP/scan 阈值的行
        fp_col = froc_df["FP_per_scan"]
        closest_idx = (fp_col - fp_thresh).abs().idxmin()
        recall_row[f"recall@{fp_thresh}/scan"] = froc_df.loc[
            closest_idx, "sensitivity"
        ]

    # 添加 unlimited 行（最大灵敏度）
    recall_row["recall@unlimited"] = froc_df["sensitivity"].max()

    # 转换为 DataFrame
    recall_df = pd.DataFrame([recall_row])

    # 保存（使用 curve_name 作为后缀）
    recall_path = os.path.join(
        output_dir,
        f"recall_table_{curve_name}.csv",
    )
    recall_df.to_csv(recall_path, index=False)
    print(f"Saved recall table ({curve_name}): {recall_path}")


def plot_froc_curve(
    froc_df,
    output_dir,
    curve_name,
    bootstrapping_data=None,
):
    """Plot FROC curve similar to froc_evaluation."""
    plt.figure()

    fps = froc_df["FP_per_scan"].values
    sens = froc_df["sensitivity"].values

    # Interpolate for smooth curve
    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)
    sens_itp = np.interp(fps_itp, fps, sens)

    plt.plot(fps_itp, sens_itp, color="b", label="FROC", lw=2)

    if bootstrapping_data is not None:
        # Add bootstrap confidence intervals if available
        bs_mean, bs_lb, bs_up = bootstrapping_data
        plt.plot(bs_mean[:, 0], bs_mean[:, 1], color="b", ls="--")
        plt.plot(bs_mean[:, 0], bs_lb[:, 1], color="b", ls=":")
        plt.plot(bs_mean[:, 0], bs_up[:, 1], color="b", ls=":")
        plt.fill_between(
            bs_mean[:, 0],
            bs_lb[:, 1],
            bs_up[:, 1],
            facecolor="blue",
            alpha=0.05,
        )

    plt.xlim(FROC_minX, FROC_maxX)
    plt.ylim(0, 1)
    plt.xlabel("Average number of false positives per scan")
    plt.ylabel("Sensitivity")
    plt.legend(loc="lower right")
    plt.title(f"FROC performance ({curve_name})")

    if bLogPlot:
        plt.xscale("log", base=2)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(
            FixedFormatter([0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32])
        )
        ax.xaxis.set_ticks([0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32])

    plt.grid(visible=True, which="both")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"froc_{curve_name}_curve.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def calculate_froc_score(froc_df, output_dir, curve_name):
    """Calculate FROC score (mean sensitivity at specific FP rates)."""
    fps = froc_df["FP_per_scan"].values
    sens = froc_df["sensitivity"].values

    # Interpolate
    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)
    sens_itp = np.interp(fps_itp, fps, sens)

    # Key FP rates (same as froc_evaluation)
    key_pt = [1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    froc_score = 0.0

    for k in key_pt:
        val_idxes = np.where(np.abs(fps_itp - k) <= 1e-2)
        froc_score += np.mean(sens_itp[val_idxes])

    froc_score /= len(key_pt)

    print(f"  FROC score ({curve_name}): {froc_score:.9f}")

    return froc_score


def output_nodule_details_tocsv(
    tp_records,
    fp_records,
    fn_count,
    gt_df,
    output_dir,
):
    """Output TPs, FPs, FNs CSV files (same format as froc_evaluation)."""

    # TPs
    tp_list = []
    for rec in tp_records:
        # Find GT info
        gt_row = (
            gt_df.iloc[rec["gt_id"]]
            if isinstance(rec["gt_id"], int)
            else None
        )
        tp_list.append(
            {
                "seriesuid": rec["pid"],
                "coordX": (
                    (gt_row["bbox_min_x"] + gt_row["bbox_max_x"]) / 2
                    if gt_row is not None
                    else 0
                ),
                "coordY": (
                    (gt_row["bbox_min_y"] + gt_row["bbox_max_y"]) / 2
                    if gt_row is not None
                    else 0
                ),
                "coordZ": (
                    (gt_row["bbox_min_z"] + gt_row["bbox_max_z"]) / 2
                    if gt_row is not None
                    else 0
                ),
                "radius": (
                    gt_row["diameter"] / 2 if gt_row is not None else 0
                ),
                "probability": rec["matched_probability"],
            }
        )

    pd.DataFrame(tp_list).to_csv(
        os.path.join(output_dir, "TPs.csv"),
        index=False,
    )

    # FPs
    fp_df = pd.DataFrame(fp_records)
    if len(fp_df) > 0:
        fp_output = fp_df.rename(
            columns={
                "pid": "seriesuid",
                "center_x": "coordX",
                "center_y": "coordY",
                "center_z": "coordZ",
            }
        )
        fp_output = fp_output[
            ["seriesuid", "coordX", "coordY", "coordZ", "probability"]
        ]
        # Add radius column (not available for FP)
        fp_output["radius"] = 0
        fp_output = fp_output[
            [
                "seriesuid",
                "coordX",
                "coordY",
                "coordZ",
                "radius",
                "probability",
            ]
        ]
    else:
        fp_output = pd.DataFrame(
            columns=[
                "seriesuid",
                "coordX",
                "coordY",
                "coordZ",
                "radius",
                "probability",
            ]
        )
    fp_output.to_csv(
        os.path.join(output_dir, "FPs.csv"),
        index=False,
    )

    # FNs - get from gt_df
    fn_records = [r for r in tp_records if r["status"] == "FN"]
    # Actually we need to track FNs separately
    # For now, we'll re-identify them from gt_df
    fn_df = pd.DataFrame()  # Will be populated during matching

    return tp_output, fp_output, fn_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred",
        required=True,
        help="Path to prediction CSV",
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="Path to GT annotation CSV",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to data root with npz files",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--val-list",
        default=VALIDATION_LIST_PATH,
        help="Path to validation list file",
    )

    args = parser.parse_args()

    # Create output directory
    save_dir = os.path.join(args.out_dir, "FROC_new")
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    gt = load_gt(args.gt, args.data_root, args.val_list)
    pred = load_pred(args.pred)

    total_scans = gt.pid.nunique()

    # ===== Phase 1: Precompute matches for ALL nodules =====
    print("\n" + "=" * 60)
    print("Phase 1: Precomputing matches for all nodules")
    print("=" * 60)
    (
        tp_scores,
        fn_count,
        all_fp,
        computed_total_scans,
        all_match_records,
    ) = precompute_matches(
        gt,
        pred,
        args.data_root,
    )

    # Save matching results
    pd.DataFrame(all_match_records).to_csv(
        os.path.join(save_dir, "matching_results.csv"),
        index=False,
    )

    pd.DataFrame(all_fp).to_csv(
        os.path.join(save_dir, "fp_results.csv"),
        index=False,
    )

    # Output detailed TP/FP/FN files
    print("\nOutputting detailed TP/FP/FN files...")
    tp_records = [
        r for r in all_match_records if r["status"] == "TP"
    ]
    fp_records = all_fp
    fn_records = [
        r for r in all_match_records if r["status"] == "FN"
    ]

    # TPs.csv
    tp_output = []
    for rec in tp_records:
        gt_row = (
            gt.iloc[rec["gt_id"]]
            if isinstance(rec.get("gt_id"), int)
            and rec["gt_id"] < len(gt)
            else None
        )
        if gt_row is not None:
            tp_output.append(
                {
                    "seriesuid": rec["pid"],
                    "coordX": (
                        gt_row["bbox_min_x"]
                        + gt_row["bbox_max_x"]
                    )
                    / 2,
                    "coordY": (
                        gt_row["bbox_min_y"]
                        + gt_row["bbox_max_y"]
                    )
                    / 2,
                    "coordZ": (
                        gt_row["bbox_min_z"]
                        + gt_row["bbox_max_z"]
                    )
                    / 2,
                    "radius": gt_row["diameter"] / 2,
                    "probability": rec["matched_probability"],
                }
            )

    pd.DataFrame(tp_output).to_csv(
        os.path.join(save_dir, "TPs.csv"),
        index=False,
    )

    # FPs.csv
    if len(fp_records) > 0:
        fp_df = pd.DataFrame(fp_records)
        fp_output = fp_df.rename(
            columns={
                "pid": "seriesuid",
                "center_x": "coordX",
                "center_y": "coordY",
                "center_z": "coordZ",
            }
        )
        fp_output = fp_output[
            ["seriesuid", "coordX", "coordY", "coordZ", "probability"]
        ]
        fp_output.insert(4, "radius", 0)
        fp_output = fp_output[
            [
                "seriesuid",
                "coordX",
                "coordY",
                "coordZ",
                "radius",
                "probability",
            ]
        ]
    else:
        fp_output = pd.DataFrame(
            columns=[
                "seriesuid",
                "coordX",
                "coordY",
                "coordZ",
                "radius",
                "probability",
            ]
        )
    fp_output.to_csv(
        os.path.join(save_dir, "FPs.csv"),
        index=False,
    )

    # FNs.csv
    if len(fn_records) > 0:
        fn_output = []
        for rec in fn_records:
            gt_row = (
                gt.iloc[rec["gt_id"]]
                if isinstance(rec.get("gt_id"), int)
                and rec["gt_id"] < len(gt)
                else None
            )
            if gt_row is not None:
                fn_output.append(
                    {
                        "seriesuid": rec["pid"],
                        "coordX": (
                            gt_row["bbox_min_x"]
                            + gt_row["bbox_max_x"]
                        )
                        / 2,
                        "coordY": (
                            gt_row["bbox_min_y"]
                            + gt_row["bbox_max_y"]
                        )
                        / 2,
                        "coordZ": (
                            gt_row["bbox_min_z"]
                            + gt_row["bbox_max_z"]
                        )
                        / 2,
                        "radius": gt_row["diameter"] / 2,
                        "probability": "",
                    }
                )
        pd.DataFrame(fn_output).to_csv(
            os.path.join(save_dir, "FNs.csv"),
            index=False,
        )
    else:
        pd.DataFrame(
            columns=[
                "seriesuid",
                "coordX",
                "coordY",
                "coordZ",
                "radius",
                "probability",
            ]
        ).to_csv(
            os.path.join(save_dir, "FNs.csv"),
            index=False,
        )

    # ===== Phase 2: Calculate FROC for ALL nodules =====
    print("\n" + "=" * 60)
    print("Phase 2: Calculating FROC for all nodules")
    print("=" * 60)
    all_froc = evaluate_froc_fast(
        tp_scores,
        fn_count,
        all_fp,
        pred,
        gt,
        computed_total_scans,
        save_dir,
        curve_name="all",
    )

    all_froc.to_csv(
        os.path.join(save_dir, "all_froc.csv"),
        index=False,
    )

    # ===== Phase 3: Calculate FROC for SMALL nodules (<=10mm) =====
    print("\n" + "=" * 60)
    print("Phase 3: Precomputing matches for small nodules (<=10mm)")
    print("=" * 60)
    small_gt = gt[gt.diameter <= 10]

    (
        small_tp_scores,
        small_fn_count,
        small_fp_records,
        small_total_scans,
        small_match_records,
    ) = precompute_matches(
        small_gt,
        pred,
        args.data_root,
    )

    # ===== Phase 4: Calculate FROC for SMALL nodules =====
    print("\n" + "=" * 60)
    print("Phase 4: Calculating FROC for small nodules")
    print("=" * 60)
    small_froc = evaluate_froc_fast(
        small_tp_scores,
        small_fn_count,
        small_fp_records,
        pred,
        small_gt,
        small_total_scans,
        save_dir,
        curve_name="small",
        dataset_name="small_<=10mm",
    )

    small_froc.to_csv(
        os.path.join(save_dir, "small_froc.csv"),
        index=False,
    )

    # ===== Phase 5: Merge recall tables =====
    print("\n" + "=" * 60)
    print("Merging recall tables")
    print("=" * 60)

    # 读取两个 recall table
    all_recall = pd.read_csv(
        os.path.join(save_dir, "recall_table_all.csv")
    )
    small_recall = pd.read_csv(
        os.path.join(save_dir, "recall_table_small.csv")
    )

    # 合并到一行
    merged_recall = pd.DataFrame(
        [
            {
                "dataset": "test_output",
            }
        ]
    )

    # 添加所有列（去掉 dataset 列）
    for col in all_recall.columns:
        if col != "dataset":
            merged_recall[col] = [all_recall[col].iloc[0]]

    # 重命名 small 列
    for col in small_recall.columns:
        if col != "dataset":
            merged_recall[col] = [small_recall[col].iloc[0]]

    # 保存合并后的 recall table
    final_recall_path = os.path.join(
        save_dir,
        "recall_table.csv",
    )
    merged_recall.to_csv(final_recall_path, index=False)
    print(f"Saved merged recall table: {final_recall_path}")

    # ===== Phase 5: Write summary =====
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Calculate stats
    n_tp = len(
        [r for r in all_match_records if r["status"] == "TP"]
    )
    n_fp = len(all_fp)
    n_fn = len(
        [r for r in all_match_records if r["status"] == "FN"]
    )
    n_total_preds = len(pred)
    n_double = len(
        [r for r in all_match_records if r.get("double", False)]
    )

    sensitivity = n_tp / len(gt) if len(gt) > 0 else 0
    avg_preds_per_scan = (
        n_total_preds / total_scans if total_scans > 0 else 0
    )

    with open(
        os.path.join(save_dir, "froc_summary.txt"),
        "w",
    ) as f:
        f.write("*" * 80 + "\n")
        f.write("CAD Analysis: SANet Custom FROC\n")
        f.write("*" * 80 + "\n")
        f.write("Candidate detection results:\n")
        f.write(f"    True positives: {n_tp}\n")
        f.write(f"    False positives: {n_fp}\n")
        f.write(f"    False negatives: {n_fn}\n")
        f.write(
            f"    Total number of candidates: {n_total_preds}\n"
        )
        f.write(f"    Total number of nodules: {len(gt)}\n")
        f.write("    Ignored candidates on excluded nodules: 0\n")
        f.write(
            "    Ignored candidates which were double detections "
            f"on a nodule: {n_double}\n"
        )
        f.write(f"    Sensitivity: {sensitivity:.9f}\n")
        f.write(
            "    Average number of candidates per scan: "
            f"{avg_preds_per_scan:.9f}\n"
        )
        f.write(
            f"\n    Small nodules (<=10mm): {len(small_gt)}\n"
        )

    print(f"\nResults saved to: {save_dir}")
    print("Done!")


if __name__ == "__main__":
    main()


# python tools/compute_froc_custom_v2.py \
#   --pred /2026aicompetition/workspace/SANet_simple/test_output/FROC/results.csv \
#   --gt /2026aicompetition/workspace/tools/split/annotation.csv \
#   --data-root /2026aicompetition/workspace/data_ytt/lung_pipeline/outputs \
#   --out-dir /2026aicompetition/workspace/SANet_simple/test_output