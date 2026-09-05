#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
筛选 FNs.csv 中所有“恶性主病灶”的漏检，并做可视化。

逻辑：
1. 从 FNs.csv 读取 FN；
2. 根据 series_id 在 nodule.csv 中找到同序列下所有病灶；
3. 用 diameter = 2 * radius 匹配病灶（允许一定误差）；
4. 判断该病灶是否为：
      - main_lesion == yes
      - detectResult == 恶性
5. 若满足，则读取：
      root1/patient_id/study_id/series_id/series_id.npz
   使用其中的 image_original (Z, Y, X) 可视化；
6. 每个病灶输出 1 张 png，包含 z-2, z-1, z, z+1, z+2 共 5 个切片。

说明：
- FN 坐标 coordX/Y/Z 已假定为 image_original 坐标系（即裁剪后、未重采样）
- image_original 的轴顺序为 (Z, Y, X)
- radius 用于反推 diameter 以匹配 nodule.csv，不在图上画物理半径圈
- 如果 nodule.csv 中同一 series 下有多个 diameter 很接近的病灶：
    * 若 nodule.csv 里也有坐标列，会进一步按坐标最近匹配
    * 否则选 diameter 最接近的一个
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 可根据你的实际情况修改的候选列名
# =========================
FN_SERIES_CANDS = [
    "series_id", "seriesuid", "series_uid", "seriesid", "seriesUID"
]
FN_X_CANDS = ["coordX"]
FN_Y_CANDS = ["coordY"]
FN_Z_CANDS = ["coordZ"]
FN_RADIUS_CANDS = ["radius"]
FN_PROB_CANDS = ["probability"]

NOD_SERIES_CANDS = [
    "series_id", "seriesuid", "series_uid", "seriesid", "seriesUID"
]
NOD_DIAMETER_CANDS = [
    "diameter", "diameter_mm", "nodule_diameter", "lesion_diameter",
    "diam", "病灶直径", "结节直径"
]
NOD_MAIN_CANDS = [
    "main_lesion", "mainlesion", "is_main_lesion", "主病灶"
]
NOD_RESULT_CANDS = [
    "detectresult", "detect_result", "result", "malignancy",
    "是否恶性", "恶性良性", "病灶性质"
]
NOD_ID_CANDS = [
    "lesion_id", "nodule_id", "id", "病灶id", "结节id"
]

# 如果 nodule.csv 有坐标，可用于多候选时做更稳妥的匹配
NOD_X_CANDS = ["coordx", "x", "centerx", "coord_x"]
NOD_Y_CANDS = ["coordy", "y", "centery", "coord_y"]
NOD_Z_CANDS = ["coordz", "z", "centerz", "coord_z"]


# =========================
# 工具函数
# =========================
def canon_col(s: str) -> str:
    """列名规范化：小写，去掉非字母数字"""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s)
    return s


def find_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    """在 dataframe 中按候选名字寻找真实列名"""
    canon_map = {canon_col(c): c for c in df.columns}
    for cand in candidates:
        key = canon_col(cand)
        if key in canon_map:
            return canon_map[key]
    if required:
        raise KeyError(f"找不到列，候选列名：{candidates}，实际列名：{list(df.columns)}")
    return None


def normalize_series_id(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_yes_no(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"yes", "y", "true", "1", "是", "主病灶", "main"}


def normalize_malignant(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"恶性", "malignant", "yes", "true", "1", "highrisk", "high_risk"}


def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def build_npz_index(npz_root: Path) -> Dict[str, Path]:
    """
    建立 series_id -> npz_path 的索引。
    默认用 npz 文件名（stem）作为 series_id。
    """
    npz_index = {}
    duplicates = {}

    for p in npz_root.rglob("*.npz"):
        sid = p.stem.strip()
        if sid in npz_index:
            duplicates.setdefault(sid, []).append(p)
        else:
            npz_index[sid] = p

    if duplicates:
        print("[Warning] 发现重复的 series_id npz，默认保留第一次出现的路径：")
        for sid, paths in duplicates.items():
            print(f"  {sid}")
            print(f"    keep: {npz_index[sid]}")
            for q in paths:
                print(f"    dup : {q}")

    print(f"[Info] 已索引 npz 文件数：{len(npz_index)}")
    return npz_index


def hu_window(img: np.ndarray, center: float = -600, width: float = 1500) -> np.ndarray:
    """
    HU 窗宽窗位，输出 [0,1]
    """
    low = center - width / 2.0
    high = center + width / 2.0
    out = np.clip(img, low, high)
    out = (out - low) / (high - low + 1e-8)
    return out


def extract_patch_2d(
    slice2d: np.ndarray,
    center_x: int,
    center_y: int,
    patch_size: int
) -> Tuple[np.ndarray, int, int]:
    """
    从 2D 切片中截取以 (x,y) 为中心的 patch
    返回:
        patch
        local_x  # 中心点在 patch 内的 x
        local_y  # 中心点在 patch 内的 y
    """
    H, W = slice2d.shape
    half = patch_size // 2

    x1 = max(0, center_x - half)
    x2 = min(W, center_x + half)
    y1 = max(0, center_y - half)
    y2 = min(H, center_y + half)

    patch = slice2d[y1:y2, x1:x2]
    local_x = center_x - x1
    local_y = center_y - y1
    return patch, local_x, local_y


def choose_best_candidate(
    candidates: pd.DataFrame,
    fn_x: float,
    fn_y: float,
    fn_z: float,
    nod_x_col: Optional[str],
    nod_y_col: Optional[str],
    nod_z_col: Optional[str],
    nod_diam_col: str,
    target_diameter: float
) -> pd.Series:
    """
    多个候选病灶时的选择策略：
    1) 若 nodule.csv 有坐标列，则优先按 3D 欧氏距离最近
    2) 否则按 diameter 最接近
    """
    cands = candidates.copy()

    if nod_x_col and nod_y_col and nod_z_col:
        cands["_dist3d"] = np.sqrt(
            (cands[nod_x_col].astype(float) - fn_x) ** 2 +
            (cands[nod_y_col].astype(float) - fn_y) ** 2 +
            (cands[nod_z_col].astype(float) - fn_z) ** 2
        )
        cands = cands.sort_values(by=["_dist3d"])
        return cands.iloc[0]

    cands["_diam_diff"] = (cands[nod_diam_col].astype(float) - target_diameter).abs()
    cands = cands.sort_values(by=["_diam_diff"])
    return cands.iloc[0]


def visualize_one_fn(
    image_zyx: np.ndarray,
    center_x: int,
    center_y: int,
    center_z: int,
    out_png: Path,
    title_prefix: str = "",
    patch_size: int = 96,
    window_center: float = -600,
    window_width: float = 1500,
):
    """
    输出一张 1x5 的 png，显示 z-2~z+2
    """
    assert image_zyx.ndim == 3, f"期望 image_original 是 3D 数组，实际 shape={image_zyx.shape}"

    Z, Y, X = image_zyx.shape
    offsets = [-2, -1, 0, 1, 2]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=12)

    for ax, off in zip(axes, offsets):
        z = center_z + off
        if z < 0 or z >= Z:
            ax.axis("off")
            ax.set_title(f"z{off:+d} (OOB)")
            continue

        sl = image_zyx[z]  # (Y, X)
        sl = hu_window(sl, center=window_center, width=window_width)

        if patch_size and patch_size > 0:
            patch, lx, ly = extract_patch_2d(sl, center_x, center_y, patch_size)
            ax.imshow(patch, cmap="gray")
            ax.scatter([lx], [ly], c="r", s=18, marker="+")
        else:
            ax.imshow(sl, cmap="gray")
            ax.scatter([center_x], [center_y], c="r", s=18, marker="+")

        ax.set_title(f"z{off:+d}  (slice={z})")
        ax.axis("off")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =========================
# 主流程
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn_csv", type=str, required=True, help="FNs.csv 路径")
    parser.add_argument("--nodule_csv", type=str, required=True, help="nodule.csv 路径")
    parser.add_argument("--npz_root1",type=str,required=True,help="优先查找的 npz 根目录")
    parser.add_argument("--npz_root2",type=str,required=True,help="备用 npz 根目录，root1 找不到时再查这里")
    parser.add_argument("--out_dir", type=str, required=True, help="输出 png 的目录")
    parser.add_argument("--diameter_tol", type=float, default=1.0, help="用 diameter=2*radius 匹配病灶的容差（mm）")
    parser.add_argument("--patch_size", type=int, default=96, help="显示的局部 patch 大小；设为 0 则显示整张切片")
    parser.add_argument("--window_center", type=float, default=-600, help="肺窗窗位")
    parser.add_argument("--window_width", type=float, default=1500, help="肺窗窗宽")
    parser.add_argument("--save_summary_csv", action="store_true", help="是否保存处理汇总 csv")
    args = parser.parse_args()

    fn_csv = Path(args.fn_csv)
    nodule_csv = Path(args.nodule_csv)
    npz_root1 = Path(args.npz_root1)
    npz_root2 = Path(args.npz_root2)
    out_dir = Path(args.out_dir)

    print(f"[Info] 读取 FN: {fn_csv}")
    fn_df = pd.read_csv(fn_csv)
    print(f"[Info] 读取 nodule: {nodule_csv}")
    nod_df = pd.read_csv(nodule_csv)

    # -------- 解析 FN 列 --------
    fn_series_col = find_column(fn_df, FN_SERIES_CANDS, required=True)
    fn_x_col = find_column(fn_df, FN_X_CANDS, required=True)
    fn_y_col = find_column(fn_df, FN_Y_CANDS, required=True)
    fn_z_col = find_column(fn_df, FN_Z_CANDS, required=True)
    fn_radius_col = find_column(fn_df, FN_RADIUS_CANDS, required=True)
    fn_prob_col = find_column(fn_df, FN_PROB_CANDS, required=False)

    # -------- 解析 nodule 列 --------
    nod_series_col = find_column(nod_df, NOD_SERIES_CANDS, required=True)
    nod_diam_col = find_column(nod_df, NOD_DIAMETER_CANDS, required=True)
    nod_main_col = find_column(nod_df, NOD_MAIN_CANDS, required=True)
    nod_result_col = find_column(nod_df, NOD_RESULT_CANDS, required=True)
    nod_id_col = find_column(nod_df, NOD_ID_CANDS, required=False)

    nod_x_col = find_column(nod_df, NOD_X_CANDS, required=False)
    nod_y_col = find_column(nod_df, NOD_Y_CANDS, required=False)
    nod_z_col = find_column(nod_df, NOD_Z_CANDS, required=False)

    # -------- 规范化 series_id --------
    fn_df["_series_norm"] = fn_df[fn_series_col].map(normalize_series_id)
    nod_df["_series_norm"] = nod_df[nod_series_col].map(normalize_series_id)

    # -------- 规范化标签 --------
    nod_df["_is_main"] = nod_df[nod_main_col].map(normalize_yes_no)
    nod_df["_is_malignant"] = nod_df[nod_result_col].map(normalize_malignant)

    # diameter 转 float
    nod_df["_diameter"] = nod_df[nod_diam_col].map(safe_float)

    # npz 建索引
    print("[Info] 建立 root1 的 NPZ 索引...")
    npz_index1 = build_npz_index(npz_root1)

    print("[Info] 建立 root2 的 NPZ 索引...")
    npz_index2 = build_npz_index(npz_root2)

    summary_rows = []

    total = len(fn_df)
    visualized = 0
    malignant_main_count = 0
    skipped_no_match = 0
    skipped_not_malignant_main = 0
    skipped_no_npz = 0
    skipped_bad_coord = 0

    print(f"[Info] 开始处理 FN，总数: {total}")

    for i, row in fn_df.iterrows():
        series_id = normalize_series_id(row[fn_series_col])
        fn_x = safe_float(row[fn_x_col])
        fn_y = safe_float(row[fn_y_col])
        fn_z = safe_float(row[fn_z_col])
        fn_radius = safe_float(row[fn_radius_col])
        fn_prob = safe_float(row[fn_prob_col]) if fn_prob_col else None

        if any(v is None for v in [fn_x, fn_y, fn_z, fn_radius]) or series_id == "":
            summary_rows.append({
                "fn_index": i,
                "series_id": series_id,
                "status": "skip_bad_fn_row",
                "reason": "坐标或 radius 缺失/非法"
            })
            skipped_bad_coord += 1
            continue

        target_diameter = 2.0 * fn_radius

        # 同 series 的所有病灶
        sub = nod_df[nod_df["_series_norm"] == series_id].copy()
        if len(sub) == 0:
            summary_rows.append({
                "fn_index": i,
                "series_id": series_id,
                "status": "skip_no_match",
                "reason": "nodule.csv 中找不到同 series_id 的病灶"
            })
            skipped_no_match += 1
            continue

        # 先按 diameter 容差筛一遍
        sub = sub[sub["_diameter"].notna()].copy()
        sub["_diam_diff"] = (sub["_diameter"] - target_diameter).abs()
        cands = sub[sub["_diam_diff"] <= args.diameter_tol].copy()

        if len(cands) == 0:
            summary_rows.append({
                "fn_index": i,
                "series_id": series_id,
                "status": "skip_no_match",
                "reason": f"同 series 下无 diameter 匹配病灶，target={target_diameter:.3f}, tol={args.diameter_tol}"
            })
            skipped_no_match += 1
            continue

        # 多个候选时做进一步选择
        if len(cands) == 1:
            best = cands.iloc[0]
        else:
            best = choose_best_candidate(
                candidates=cands,
                fn_x=fn_x, fn_y=fn_y, fn_z=fn_z,
                nod_x_col=nod_x_col, nod_y_col=nod_y_col, nod_z_col=nod_z_col,
                nod_diam_col="_diameter",
                target_diameter=target_diameter
            )

        is_main = bool(best["_is_main"])
        is_malignant = bool(best["_is_malignant"])

        lesion_id = str(best[nod_id_col]) if nod_id_col and pd.notna(best[nod_id_col]) else ""

        if not (is_main and is_malignant):
            summary_rows.append({
                "fn_index": i,
                "series_id": series_id,
                "lesion_id": lesion_id,
                "status": "skip_not_malignant_main",
                "reason": f"is_main={is_main}, is_malignant={is_malignant}",
                "matched_diameter": best["_diameter"],
                "target_diameter": target_diameter
            })
            skipped_not_malignant_main += 1
            continue

        malignant_main_count += 1

        # =========================
        # 优先从 root1 查找，找不到再查 root2
        # =========================
        npz_path = npz_index1.get(series_id, None)
        npz_source = "root1"
        if npz_path is None:
            npz_path = npz_index2.get(series_id, None)
            npz_source = "root2"
        if npz_path is None:
            summary_rows.append({
                "fn_index": i,
                "series_id": series_id,
                "lesion_id": lesion_id,
                "status": "skip_no_npz",
                "reason": "root1 和 root2 中都找不到对应的 series_id.npz"
            })
            skipped_no_npz += 1
            continue

        # 读取 npz
        try:
            data = np.load(npz_path, allow_pickle=True)
            if "image_original" not in data:
                summary_rows.append({
                    "fn_index": i,
                    "series_id": series_id,
                    "lesion_id": lesion_id,
                    "status": "skip_no_image_original",
                    "reason": f"{npz_path} 中不存在 image_original"
                })
                continue

            image = data["image_original"]  # (Z, Y, X)
        except Exception as e:
            summary_rows.append({
                "fn_index": i,
                "series_id": series_id,
                "lesion_id": lesion_id,
                "status": "skip_npz_read_error",
                "reason": str(e)
            })
            continue

        if image.ndim != 3:
            summary_rows.append({
                "fn_index": i,
                "series_id": series_id,
                "lesion_id": lesion_id,
                "status": "skip_bad_image_shape",
                "reason": f"image_original shape 异常: {image.shape}"
            })
            continue

        Z, Y, X = image.shape
        cx = int(round(fn_x))
        cy = int(round(fn_y))
        cz = int(round(fn_z))

        if not (0 <= cx < X and 0 <= cy < Y and 0 <= cz < Z):
            summary_rows.append({
                "fn_index": i,
                "series_id": series_id,
                "lesion_id": lesion_id,
                "status": "skip_bad_coord",
                "reason": f"坐标越界: (x,y,z)=({cx},{cy},{cz}), image_shape={image.shape}"
            })
            skipped_bad_coord += 1
            continue

        prob_str = f"{fn_prob:.4f}" if fn_prob is not None else "NA"
        lesion_str = lesion_id if lesion_id != "" else "unknown"

        title = (
            f"FN#{i} | series={series_id} | lesion_id={lesion_str} | "
            f"diam={target_diameter:.2f}mm | prob={prob_str}"
        )

        safe_series = re.sub(r"[^\w\-\.]+", "_", series_id)
        safe_lesion = re.sub(r"[^\w\-\.]+", "_", lesion_str)
        out_png = out_dir / f"{i:05d}_{safe_series}_{safe_lesion}.png"

        try:
            visualize_one_fn(
                image_zyx=image,
                center_x=cx,
                center_y=cy,
                center_z=cz,
                out_png=out_png,
                title_prefix=title,
                patch_size=args.patch_size,
                window_center=args.window_center,
                window_width=args.window_width,
            )
            visualized += 1
            summary_rows.append({
                "fn_index": i,
                "series_id": series_id,
                "lesion_id": lesion_id,
                "status": "visualized",
                "npz_source": npz_source,
                "npz_path": str(npz_path),
                "out_png": str(out_png),
                "matched_diameter": best["_diameter"],
                "target_diameter": target_diameter,
                "is_main": is_main,
                "is_malignant": is_malignant,
                "probability": fn_prob
            })
            print(f"[{visualized}] saved: {out_png}")
        except Exception as e:
            summary_rows.append({
                "fn_index": i,
                "series_id": series_id,
                "lesion_id": lesion_id,
                "status": "skip_visualize_error",
                "reason": str(e)
            })

    print("\n========== Done ==========")
    print(f"FN 总数                     : {total}")
    print(f"匹配到恶性主病灶数          : {malignant_main_count}")
    print(f"成功可视化输出数            : {visualized}")
    print(f"跳过（无病灶匹配）          : {skipped_no_match}")
    print(f"跳过（不是恶性主病灶）      : {skipped_not_malignant_main}")
    print(f"跳过（找不到 npz）          : {skipped_no_npz}")
    print(f"跳过（坐标异常）            : {skipped_bad_coord}")

    if args.save_summary_csv:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = out_dir / "summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print(f"[Info] 已保存汇总: {summary_csv}")


if __name__ == "__main__":
    main()