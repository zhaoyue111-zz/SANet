import numpy as np
import torch
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# SANet contains optional internal data_parallel calls. Under torchrun each
# process must own exactly one GPU, so internal DataParallel must stay disabled.
os.environ["SANET_DISABLE_INTERNAL_DATA_PARALLEL"] = "1"

import gc
import traceback
import sys
import logging
import argparse
import pandas as pd
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build/box"))

from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable

from net.sanet import SANet
from dataset.collate import train_collate
from dataset.bbox_reader import BboxReader
from config import config, dataset_configs, default_out_dir
from evaluationScript.noduleCADEvaluationLUNA16 import noduleCADEvaluation
import net.sanet as sanet_module


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


def init_distributed():
    """
    Initialize torchrun environment.

    Important: inference does NOT wrap the model in DistributedDataParallel.
    There are no gradients to synchronize. Each process simply owns one GPU
    and a disjoint subset of cases; torch.distributed is used only for barriers.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
            device = torch.device("cuda", local_rank)
        else:
            backend = "gloo"
            device = torch.device("cpu")

        dist.init_process_group(backend=backend, init_method="env://")
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    return distributed, rank, local_rank, world_size, device


def is_main_process(rank):
    return rank == 0


def barrier(distributed):
    if distributed:
        dist.barrier()


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def strip_module_prefix(state_dict):
    """Accept checkpoints saved from DataParallel/DDP as well as single-GPU."""
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


this_module = sys.modules[__name__]

parser = argparse.ArgumentParser()
parser.add_argument("--net", "-m", metavar="NET", default=config["net"], help="neural net")
parser.add_argument("--mode", type=str, default="eval", help="you want to test or val")
parser.add_argument(
    "--weight",
    type=str,
    default="train_output/model/best.ckpt",
    help="path to model weights to be used",
)
parser.add_argument("--dicom-path", type=str, default=None, help="path to dicom files of patient")
parser.add_argument(
    "--out-dir",
    "--out_dir",
    dest="out_dir",
    type=str,
    default="test_output",
    help="path to save the results",
)
parser.add_argument(
    "--test-set-name",
    "--test_set_name",
    dest="test_set_name",
    type=str,
    default=None,
    help="path to the test case list",
)
parser.add_argument(
    "--dataset",
    default=config["dataset"],
    type=str,
    help="dataset name under data/, or 'all' for all SANet-ready datasets",
)
parser.add_argument("--use-aspp", action="store_true", help="Enable ASPP3D in the RPN head.")
parser.add_argument(
    "--workers",
    type=int,
    default=4,
    help="DataLoader workers PER torchrun process. 4 GPUs x 4 workers = 16 workers total.",
)


def main():
    logging.basicConfig(
        format="[%(levelname)s][%(asctime)s] %(message)s",
        level=logging.INFO,
    )
    args = parser.parse_args()

    enable_cpu_fallback()
    distributed, rank, local_rank, world_size, device = init_distributed()

    try:
        if args.mode != "eval":
            if is_main_process(rank):
                logging.error("Mode %s is not supported", args.mode)
            return

        initial_checkpoint = args.weight
        if not initial_checkpoint:
            if is_main_process(rank):
                print("No model weight file specified")
            return

        cfgs = dataset_configs(args.dataset, skip_missing=(args.dataset == "all"))

        # Every rank owns an independent model on its LOCAL_RANK GPU.
        net_cls = getattr(this_module, args.net)
        net = net_cls(config, use_aspp=args.use_aspp)
        net = net.to(device)

        if is_main_process(rank):
            print("[Loading model from %s]" % initial_checkpoint)

        checkpoint = torch.load(
            initial_checkpoint,
            map_location=device,
            weights_only=False,
        )
        epoch = checkpoint["epoch"]
        state_dict = strip_module_prefix(checkpoint["state_dict"])
        net.load_state_dict(state_dict)

        if distributed:
            print(
                "[rank %d/%d] local_rank=%d device=%s"
                % (rank, world_size, local_rank, str(device))
            )

        for dataset_cfg in cfgs:
            test_set_name = args.test_set_name or dataset_cfg["test_set_name"]
            out_dir = args.out_dir or default_out_dir(args.dataset)

            if args.dataset == "all":
                save_dir = os.path.join(
                    out_dir, "res", str(epoch), dataset_cfg["dataset"]
                )
            else:
                save_dir = os.path.join(
                    out_dir,
                    "res",
                    str(epoch) + "_pretrained_hardsamples_rcnn40_PN11_v5_ensemble",
                    args.dataset,
                )

            try:
                dataset = BboxReader(
                    dataset_cfg["DATA_DIR"],
                    test_set_name,
                    dataset_cfg,
                    mode="eval",
                )
            except (FileNotFoundError, ValueError) as exc:
                if args.dataset != "all":
                    raise
                if is_main_process(rank):
                    print("[%s] Skip test split: %s" % (dataset_cfg["dataset"], exc))
                continue

            if is_main_process(rank):
                print("dataset", dataset_cfg["dataset"])
                print("out_dir", out_dir)
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(os.path.join(save_dir, "FROC"), exist_ok=True)

            # Directory creation must finish before any rank starts np.save().
            barrier(distributed)

            # No-padding distributed evaluation:
            # rank0: 0, world_size, 2*world_size, ...
            # rank1: 1, world_size+1, ...
            # Unlike DistributedSampler, this NEVER pads/duplicates cases.
            if distributed:
                local_indices = list(range(rank, len(dataset), world_size))
            else:
                local_indices = list(range(len(dataset)))

            local_dataset = Subset(dataset, local_indices)
            num_workers = 0 if not torch.cuda.is_available() else max(0, args.workers)

            test_loader = DataLoader(
                local_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=train_collate,
            )

            run_inference(
                net=net,
                loader=test_loader,
                base_dataset=dataset,
                dataset_indices=local_indices,
                save_dir=save_dir,
                device=device,
                rank=rank,
                world_size=world_size,
            )

            # All per-case *_detections.npy files must exist before rank0 merges.
            barrier(distributed)

            if is_main_process(rank):
                res_path = generate_results_csv(dataset, save_dir)
                run_froc_evaluation(dataset, res_path, save_dir)

            # Non-zero ranks wait here while rank0 writes results.csv and FROC.
            # This is the key fix for the "results.csv 后进程冲突" problem.
            barrier(distributed)

    finally:
        cleanup_distributed(distributed)


def run_inference(
    net,
    loader,
    base_dataset,
    dataset_indices,
    save_dir,
    device,
    rank,
    world_size,
):
    net.set_mode("eval")
    net.use_rcnn = True

    print(
        "[rank %d/%d] eval cases: %d"
        % (rank, world_size, len(dataset_indices))
    )

    for local_i, (input, truth_bboxes, truth_labels) in enumerate(loader):
        # local_i is NOT a dataset index under distributed inference.
        dataset_index = dataset_indices[local_i]
        pid = base_dataset.filenames[dataset_index]

        try:
            clear_inference_cache(net)

            input = Variable(input).to(device, non_blocking=True)
            truth_bboxes = np.array(truth_bboxes)
            truth_labels = np.array(truth_labels)

            print(
                "[rank %d][%d/%d] Predicting %s"
                % (rank, local_i + 1, len(dataset_indices), pid)
            )

            with torch.no_grad():
                net.forward(input, truth_bboxes, truth_labels)

            # Keep the repository's current output choice.
            detections = net.ensemble_proposals.detach().cpu().numpy()

            detections_path = os.path.join(
                save_dir, "%s_detections.npy" % pid
            )

            if len(detections):
                # Original code keeps [probability, z, y, x, diameter]
                # after removing the batch column.
                detections = detections[:, 1:]
                np.save(detections_path, detections)
            elif os.path.exists(detections_path):
                # Prevent a stale result from an earlier run being reused.
                os.remove(detections_path)

            del input, truth_bboxes, truth_labels
            clear_inference_cache(net)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception:
            clear_inference_cache(net)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            traceback.print_exc()
            raise


def generate_results_csv(dataset, save_dir):
    """
    Rank-0 only.

    Merge the disjoint per-case outputs produced by all ranks into ONE results.csv.
    """
    res = []

    for pid in dataset.filenames:
        detections_path = os.path.join(
            save_dir, "%s_detections.npy" % pid
        )
        if not os.path.exists(detections_path):
            continue

        detections = np.load(detections_path)

        # Stored format is [probability, z, y, x, diameter].
        # results.csv format remains:
        # pid, center_x, center_y, center_z, diameter, probability
        detections = detections[:, [3, 2, 1, 4, 0]]
        names = np.array([[pid]] * len(detections), dtype=object)
        res.append(np.concatenate([names, detections], axis=1))

    col_names = [
        "pid",
        "center_x",
        "center_y",
        "center_z",
        "diameter",
        "probability",
    ]

    if res:
        res = np.concatenate(res, axis=0)
        df = pd.DataFrame(res, columns=col_names)
        for c in ["center_x", "center_y", "center_z", "diameter", "probability"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    else:
        df = pd.DataFrame(columns=col_names)

    eval_dir = os.path.join(save_dir, "FROC")
    os.makedirs(eval_dir, exist_ok=True)

    res_path = os.path.join(eval_dir, "results.csv")
    df.to_csv(res_path, index=False)

    print(
        "[rank 0] Saved results.csv: %s (%d candidates)"
        % (res_path, len(df))
    )
    return res_path


def run_froc_evaluation(dataset, res_path, save_dir):
    """Rank-0 only: keep the repository's existing FROC call."""
    eval_dir = os.path.join(save_dir, "FROC")
    froc_out = os.path.join(eval_dir, "res")
    os.makedirs(froc_out, exist_ok=True)

    annotations_filename = prepare_evaluation_annotations(
        dataset.cfg["test_anno"], eval_dir
    )
    val_path = dataset.set_name

    print("[rank 0] Starting FROC evaluation...")
    noduleCADEvaluation(
        annotations_filename,
        res_path,
        val_path,
        froc_out,
    )
    print("[rank 0] FROC evaluation finished.")


def clear_inference_cache(net):
    """Clear case-level tensors retained by SANet."""
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


def prepare_evaluation_annotations(annotation_path, eval_dir):
    annotations = pd.read_csv(annotation_path)

    center_columns = {
        "center_x",
        "center_y",
        "center_z",
        "diameter",
    }
    if not center_columns.issubset(annotations.columns):
        corner_columns = {
            "xmin",
            "xmax",
            "ymin",
            "ymax",
            "zmin",
            "zmax",
        }
        missing = corner_columns.difference(annotations.columns)
        if missing:
            raise ValueError(
                "Annotation file %s is missing columns: %s"
                % (annotation_path, ", ".join(sorted(missing)))
            )

        annotations["center_x"] = (
            annotations["xmin"] + annotations["xmax"]
        ) / 2.0
        annotations["center_y"] = (
            annotations["ymin"] + annotations["ymax"]
        ) / 2.0
        annotations["center_z"] = (
            annotations["zmin"] + annotations["zmax"]
        ) / 2.0

        sizes = pd.DataFrame(
            {
                "x": annotations["xmax"] - annotations["xmin"] + 1,
                "y": annotations["ymax"] - annotations["ymin"] + 1,
                "z": annotations["zmax"] - annotations["zmin"] + 1,
            }
        )
        annotations["diameter"] = sizes.max(axis=1)

    output_path = os.path.join(eval_dir, "annotations.csv")
    annotations.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    main()


"""
Single GPU:
python test.py \
  --dataset your_dataset \
  --weight /path/to/best.ckpt \
  --out-dir test_output

4-GPU distributed inference:
torchrun --standalone --nproc_per_node=4 test.py \
  --dataset your_dataset \
  --weight /path/to/best.ckpt \
  --out-dir test_output \
  --workers 4
"""
