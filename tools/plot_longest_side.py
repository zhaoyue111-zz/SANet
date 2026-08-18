import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BBOX_COLUMNS = [
    "zmin", "zmax",
    "ymin", "ymax",
    "xmin", "xmax",
]


def round_to_nearest_5(values):
    """常规四舍五入到最接近的5，避免np.round的银行家舍入。"""
    return np.floor(values / 5.0 + 0.5).astype(int) * 5


def collect_longest_sides(data_root):
    data_root = Path(data_root)

    # 递归查找所有数据集的 split/all_anno.csv
    csv_paths = sorted(data_root.glob("**/split/all_anno.csv"))

    if not csv_paths:
        raise FileNotFoundError(
            f"在 {data_root} 下没有找到 */split/all_anno.csv"
        )

    all_results = []

    for csv_path in csv_paths:
        # 数据集名称，通常是split目录的上一级目录名
        dataset_name = csv_path.parent.parent.name

        df = pd.read_csv(csv_path)

        missing_columns = [
            column for column in BBOX_COLUMNS
            if column not in df.columns
        ]
        if missing_columns:
            print(
                f"[跳过] {csv_path} 缺少列：{missing_columns}"
            )
            continue

        bbox = df[BBOX_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )

        # 删除坐标缺失或无法转换的标注
        valid_mask = bbox.notna().all(axis=1)
        invalid_number_count = int((~valid_mask).sum())
        bbox = bbox.loc[valid_mask].copy()

        # 计算三个方向的边长
        z_length = bbox["zmax"] - bbox["zmin"]
        y_length = bbox["ymax"] - bbox["ymin"]
        x_length = bbox["xmax"] - bbox["xmin"]

        # 排除边长小于等于0的无效框
        positive_mask = (
            (z_length > 0)
            & (y_length > 0)
            & (x_length > 0)
        )
        invalid_bbox_count = int((~positive_mask).sum())

        bbox = bbox.loc[positive_mask].copy()
        z_length = z_length.loc[positive_mask]
        y_length = y_length.loc[positive_mask]
        x_length = x_length.loc[positive_mask]

        if bbox.empty:
            print(f"[跳过] {dataset_name} 中没有有效标注框")
            continue

        longest_side = pd.concat(
            [z_length, y_length, x_length],
            axis=1,
        ).max(axis=1)

        result = pd.DataFrame({
            "dataset": dataset_name,
            "source_csv": str(csv_path),
            "z_length": z_length.to_numpy(),
            "y_length": y_length.to_numpy(),
            "x_length": x_length.to_numpy(),
            "longest_side": longest_side.to_numpy(),
            "rounded_longest_side": round_to_nearest_5(
                longest_side.to_numpy()
            ),
        })

        all_results.append(result)

        print(
            f"[读取] {dataset_name}: "
            f"有效框 {len(result)}, "
            f"数值无效 {invalid_number_count}, "
            f"坐标范围无效 {invalid_bbox_count}, "
            f"最长边范围 "
            f"{longest_side.min():.2f}–{longest_side.max():.2f}"
        )

    if not all_results:
        raise ValueError("没有从任何数据集中读取到有效标注框")

    return pd.concat(all_results, ignore_index=True)


def plot_distribution(data_root, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = collect_longest_sides(data_root)

    rounded = result["rounded_longest_side"]
    min_value = int(rounded.min())
    max_value = int(rounded.max())

    # 补齐最小值到最大值之间所有间隔为5的刻度
    value_range = np.arange(min_value, max_value + 5, 5)

    counts = (
        rounded.value_counts()
        .reindex(value_range, fill_value=0)
        .sort_index()
    )

    # 保存每个框的详细计算结果
    detail_path = output_dir / "longest_side_details.csv"
    result.to_csv(detail_path, index=False)

    # 保存总体分布
    count_path = output_dir / "longest_side_distribution.csv"
    pd.DataFrame({
        "rounded_longest_side": value_range,
        "count": counts.to_numpy(),
    }).to_csv(count_path, index=False)

    # 保存各数据集分别统计的分布
    per_dataset_counts = (
        result.groupby(
            ["dataset", "rounded_longest_side"]
        )
        .size()
        .rename("count")
        .reset_index()
    )
    per_dataset_path = (
        output_dir / "longest_side_distribution_by_dataset.csv"
    )
    per_dataset_counts.to_csv(per_dataset_path, index=False)

    print(f"\n数据集数量：{result['dataset'].nunique()}")
    print(f"有效标注框总数：{len(result)}")
    print(
        f"原始最长边范围："
        f"{result['longest_side'].min():.4f}–"
        f"{result['longest_side'].max():.4f}"
    )
    print(f"四舍五入后范围：{min_value}–{max_value}")

    print("\n最长边分布：")
    print(counts.to_string())

    # 绘制总体直方图
    figure_width = max(12, min(30, len(value_range) * 0.35))
    plt.figure(figsize=(figure_width, 7))

    bars = plt.bar(
        value_range,
        counts.to_numpy(),
        width=4.5,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.5,
    )

    # 在每个非零柱子上方标注数量
    plt.bar_label(
        bars,
        labels=[
            str(count) if count > 0 else ""
            for count in counts.to_numpy()
        ],
        padding=3,
        fontsize=8
    )

    plt.xlabel("Longest side (rounded to nearest 5)")
    plt.ylabel("Number of bounding boxes")
    plt.title("Distribution of Bounding Box Longest Side")
    plt.xticks(value_range, rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    figure_path = output_dir / "longest_side_histogram.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n直方图：{figure_path}")
    print(f"总体统计：{count_path}")
    print(f"分数据集统计：{per_dataset_path}")
    print(f"详细结果：{detail_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="统计所有数据集标注框最长边的分布"
    )
    parser.add_argument(
        "data_root",
        help="所有数据集所在的父目录，例如 /data",
    )
    parser.add_argument(
        "--output-dir",
        default="./longest_side_statistics",
        help="结果输出目录",
    )
    args = parser.parse_args()

    plot_distribution(
        data_root=args.data_root,
        output_dir=args.output_dir,
    )

'''
python tools/plot_longest_side.py /mnt/afs2/data
'''