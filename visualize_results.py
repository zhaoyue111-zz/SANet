'''
可视化test.py的输出文件
'''

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


DETECTION_COLUMNS = [
    "probability",
    "center_z",
    "center_y",
    "center_x",
    "depth",
    "height",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a SANet volume with predicted and ground-truth boxes."
    )
    parser.add_argument("--pid", default="00009", help="Case ID, for example 00021.")
    parser.add_argument(
        "--data-dir",
        default="/mnt/afs2/data/PN9/full",
        help="Directory containing {pid}_zoom.npy files.",
    )
    parser.add_argument(
        "--detections-dir",
        default="/mnt/afs2/code/SANet/test_output/res/14/PN9",
        help="Directory containing {pid}_detections.npy files.",
    )
    parser.add_argument(
        "--annotations",
        default="/mnt/afs2/code/SANet/test_output/res/14/PN9/FROC/annotations.csv",
        help="Ground-truth annotation CSV. Use an empty string to disable.",
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        help="Axial Z slice. Defaults to the highest-scoring prediction center.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Minimum prediction probability.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum predictions retained before selecting boxes on the slice.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to results/res/95/visualizations/.",
    )
    return parser.parse_args()


def normalize_pid(pid):
    return str(pid).strip().zfill(5)


def load_volume(path):
    volume = np.load(path)
    if volume.ndim == 4 and volume.shape[0] == 1:
        volume = volume[0]
    if volume.ndim != 3:
        raise ValueError(
            "Expected volume shape [1, D, H, W] or [D, H, W], got %s"
            % (volume.shape,)
        )
    return volume


def load_detections(path, threshold, top_k):
    if not path.exists():
        return np.empty((0, len(DETECTION_COLUMNS)), dtype=np.float32)

    detections = np.load(path)
    if detections.ndim != 2 or detections.shape[1] < 5:
        raise ValueError(
            "Expected detections with at least 5 columns, got %s"
            % (detections.shape,)
        )

    order = np.argsort(detections[:, 0])[::-1]
    detections = detections[order]
    detections = detections[detections[:, 0] >= threshold]
    return detections[:top_k]


def load_annotations(path, pid):
    if not path or not path.exists():
        return []

    target_pid = int(pid)
    annotations = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["pid"])) == target_pid:
                annotations.append(row)
    return annotations


def detection_size(detection):
    depth = float(detection[4])
    height = float(detection[5]) if len(detection) > 5 else depth
    width = float(detection[6]) if len(detection) > 6 else depth
    return depth, height, width


def detections_on_slice(detections, slice_index):
    selected = []
    for detection in detections:
        depth, _, _ = detection_size(detection)
        if abs(float(detection[1]) - slice_index) <= depth / 2.0:
            selected.append(detection)
    return selected


def annotations_on_slice(annotations, slice_index):
    return [
        annotation
        for annotation in annotations
        if float(annotation["zmin"]) <= slice_index <= float(annotation["zmax"])
    ]


def add_prediction_boxes(ax, detections):
    for index, detection in enumerate(detections, start=1):
        score, _, center_y, center_x = map(float, detection[:4])
        _, height, width = detection_size(detection)
        ax.add_patch(
            Rectangle(
                (center_x - width / 2.0, center_y - height / 2.0),
                width,
                height,
                fill=False,
                edgecolor="red",
                linewidth=1.5,
            )
        )
        ax.text(
            center_x - width / 2.0,
            center_y - height / 2.0 - 3,
            "P%d %.4f" % (index, score),
            color="red",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
        )


def add_ground_truth_boxes(ax, annotations):
    for annotation in annotations:
        xmin = float(annotation["xmin"])
        xmax = float(annotation["xmax"])
        ymin = float(annotation["ymin"])
        ymax = float(annotation["ymax"])
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin + 1,
                ymax - ymin + 1,
                fill=False,
                edgecolor="lime",
                linewidth=2,
            )
        )
        ax.text(
            xmin,
            ymin - 3,
            "GT %s" % annotation.get("nodule_id", ""),
            color="lime",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
        )


def zoom_limits(image_shape, detections, annotations, margin=30):
    points_x = []
    points_y = []

    for detection in detections:
        _, _, center_y, center_x = map(float, detection[:4])
        _, height, width = detection_size(detection)
        points_x.extend([center_x - width / 2.0, center_x + width / 2.0])
        points_y.extend([center_y - height / 2.0, center_y + height / 2.0])

    for annotation in annotations:
        points_x.extend([float(annotation["xmin"]), float(annotation["xmax"])])
        points_y.extend([float(annotation["ymin"]), float(annotation["ymax"])])

    height, width = image_shape
    if not points_x:
        return (0, width), (height, 0)

    xmin = max(0, min(points_x) - margin)
    xmax = min(width, max(points_x) + margin)
    ymin = max(0, min(points_y) - margin)
    ymax = min(height, max(points_y) + margin)
    return (xmin, xmax), (ymax, ymin)


def main():
    args = parse_args()
    pid = normalize_pid(args.pid)

    data_path = Path(args.data_dir) / ("%s_zoom.npy" % pid) # shape:[1, depth, height, width]
    detections_path = Path(args.detections_dir) / (
        "%s_detections.npy" % pid
    )
    annotations_path = Path(args.annotations) if args.annotations else None

    volume = load_volume(data_path)
    detections = load_detections(
        detections_path, args.threshold, args.top_k
    )
    annotations = load_annotations(annotations_path, pid)

    if args.slice is not None:
        slice_index = args.slice
    elif len(detections):
        slice_index = int(round(float(detections[0, 1])))
    elif annotations:
        slice_index = int(
            round(
                (
                    float(annotations[0]["zmin"])
                    + float(annotations[0]["zmax"])
                )
                / 2.0
            )
        )
    else:
        slice_index = volume.shape[0] // 2

    if not 0 <= slice_index < volume.shape[0]:
        raise ValueError(
            "Slice %d is outside valid range [0, %d]"
            % (slice_index, volume.shape[0] - 1)
        )

    slice_detections = detections_on_slice(detections, slice_index)
    slice_annotations = annotations_on_slice(annotations, slice_index)
    image = volume[slice_index]

    output_path = (
        Path(args.output)
        if args.output
        else Path(args.detections_dir)
        / "visualizations"
        / ("%s_z%04d.png" % (pid, slice_index))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax in axes:
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        add_prediction_boxes(ax, slice_detections)
        add_ground_truth_boxes(ax, slice_annotations)
        ax.axis("off")

    axes[0].set_title(
        "PID %s, axial Z=%d\nred=prediction, green=ground truth"
        % (pid, slice_index)
    )
    xlim, ylim = zoom_limits(
        image.shape, slice_detections, slice_annotations
    )
    axes[1].set_xlim(*xlim)
    axes[1].set_ylim(*ylim)
    axes[1].set_title("Region zoom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Volume:", data_path)
    print("Shape:", (1,) + volume.shape)
    print("Value range: [%.3f, %.3f]" % (volume.min(), volume.max()))
    print("Detections:", detections_path)
    print("Detection columns:", DETECTION_COLUMNS)
    print(
        "Kept %d detections (threshold %.4f, top-k %d); %d intersect Z=%d"
        % (
            len(detections),
            args.threshold,
            args.top_k,
            len(slice_detections),
            slice_index,
        )
    )
    for detection in detections:
        print(
            "  score=%.6f, zyx=(%.1f, %.1f, %.1f), size=%s"
            % (
                detection[0],
                detection[1],
                detection[2],
                detection[3],
                tuple(round(value, 1) for value in detection_size(detection)),
            )
        )
    print("Ground truths on slice:", len(slice_annotations))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
