"""Visualize SANet-ready volumes and 3D bounding boxes in three views."""
'''
可视化处理后数据集的工具，给定PID和nodule_id，找到对应的体积数据和标注，
在三个视图（轴向、冠状、矢状）中显示切片和标注框，并保存为PNG图片。
'''

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def normalize_pid(value: str, width: int = 5) -> str:
    return str(value).strip().zfill(width)


def load_volume(path: Path) -> np.ndarray:
    volume = np.load(path)
    if volume.ndim != 4 or volume.shape[0] != 1:
        raise ValueError(f"Expected [1,D,H,W], got {volume.shape}: {path}")
    return volume[0]


def load_annotations(sanet_dir: Path) -> pd.DataFrame:
    split_dir = sanet_dir / "split"
    all_path = split_dir / "all_anno.csv"
    if all_path.exists():
        return pd.read_csv(all_path)

    paths = [
        split_dir / name
        for name in ["train_anno.csv", "val_anno.csv", "test_anno.csv"]
        if (split_dir / name).exists()
    ]
    if not paths:
        raise FileNotFoundError(f"No annotation CSV found under {split_dir}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True).drop_duplicates()


def select_annotation(
    annotations: pd.DataFrame,
    pid: int,
    nodule_id: int | None,
) -> pd.Series:
    case_rows = annotations[annotations["pid"].astype(int) == int(pid)]
    if case_rows.empty:
        raise ValueError(f"PID {pid} has no annotations")
    if nodule_id is not None:
        case_rows = case_rows[case_rows["nodule_id"].astype(int) == nodule_id]
        if case_rows.empty:
            raise ValueError(f"PID {pid} has no nodule_id={nodule_id}")
    return case_rows.iloc[0]


def add_box(ax, xy, width, height, label):
    ax.add_patch(
        Rectangle(
            xy,
            width,
            height,
            fill=False,
            edgecolor="lime",
            linewidth=2,
        )
    )
    ax.text(
        xy[0],
        xy[1] - 3,
        label,
        color="lime",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
    )


def display_limits(image: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(image, [0.5, 99.5])
    if high <= low:
        low, high = float(image.min()), float(image.max())
    return float(low), float(high)


def visualize_sanet_output(
    sanet_dir: str | Path,
    case_pid: int,
    nodule_id: int | None = None,
    output: str | Path | None = None,
) -> Path:
    sanet_dir = Path(sanet_dir)
    annotations = load_annotations(sanet_dir)
    annotation = select_annotation(annotations, case_pid, nodule_id)

    candidate_paths = [
        sanet_dir / "full" / f"{normalize_pid(case_pid, width)}_zoom.npy"
        for width in [5, 6]
    ]
    volume_path = next((path for path in candidate_paths if path.exists()), None)
    if volume_path is None:
        raise FileNotFoundError(
            "Cannot find volume. Tried: " + ", ".join(map(str, candidate_paths))
        )
    volume = load_volume(volume_path)

    zmin, zmax = int(annotation.zmin), int(annotation.zmax)
    ymin, ymax = int(annotation.ymin), int(annotation.ymax)
    xmin, xmax = int(annotation.xmin), int(annotation.xmax)
    z = (zmin + zmax) // 2
    y = (ymin + ymax) // 2
    x = (xmin + xmax) // 2

    if not (0 <= z < volume.shape[0] and 0 <= y < volume.shape[1] and 0 <= x < volume.shape[2]):
        raise ValueError(
            f"Box center {(z, y, x)} is outside volume shape {volume.shape}"
        )

    views = [
        ("Axial", volume[z], (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1),
        ("Coronal", volume[:, y, :], (xmin, zmin), xmax - xmin + 1, zmax - zmin + 1),
        ("Sagittal", volume[:, :, x], (ymin, zmin), ymax - ymin + 1, zmax - zmin + 1),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    label = f"GT {int(annotation.nodule_id)}"
    for ax, (title, image, xy, width, height) in zip(axes, views):
        vmin, vmax = display_limits(image)
        ax.imshow(image, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
        add_box(ax, xy, width, height, label)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"PID={normalize_pid(case_pid)}, nodule={int(annotation.nodule_id)}, "
        f"center zyx=({z},{y},{x}), volume={volume.shape}"
    )
    fig.tight_layout()

    output_path = (
        Path(output)
        if output
        else sanet_dir
        / "visualizations"
        / f"{normalize_pid(case_pid)}_nodule_{int(annotation.nodule_id)}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Volume: {volume_path}")
    print(f"Shape: {volume.shape}, range: [{volume.min():.3f}, {volume.max():.3f}]")
    print(f"Annotation: z[{zmin},{zmax}] y[{ymin},{ymax}] x[{xmin},{xmax}]")
    print(f"Saved: {output_path}")
    return output_path


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Visualize a SANet-ready annotation in axial/coronal/sagittal views."
    )
    parser.add_argument(
        "--sanet-dir",
        default="/mnt/afs2/data/PN9",
    )
    parser.add_argument("--pid", type=int, default=9448)
    parser.add_argument("--nodule-id", type=int, default=None,help="pid对应的病例的结节id，在split/all_anno_3D.csv中可以查看；不指定默认选择第一个")
    parser.add_argument("--output", default=None,help="输出的PNG图片 path，默认在sanet_dir/visualizations目录下")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    visualize_sanet_output(
        args.sanet_dir,
        case_pid=args.pid,
        nodule_id=args.nodule_id,
        output=args.output,
    )
