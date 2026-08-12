import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')


def preset_cuda_visible_devices():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[i + 1]
            return
        if arg.startswith('--gpu='):
            os.environ['CUDA_VISIBLE_DEVICES'] = arg.split('=', 1)[1]
            return


preset_cuda_visible_devices()

from net.sanet import SANet
import time
from dataset.collate import train_collate, test_collate, eval_collate, ct_batch_collate
from dataset.bbox_reader import BboxReader
from dataset.ct_batch_reader import build_ct_batch_datasets
from utils.util import Logger
from config import (
    train_config, data_config, net_config, config,
    dataset_configs, default_out_dir,
)
import pprint
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import argparse
from tqdm import tqdm
import random
import traceback
from torch.utils.tensorboard import SummaryWriter

this_module = sys.modules[__name__]


def enable_cpu_fallback():
    if torch.cuda.is_available():
        return

    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    torch.cuda.empty_cache = lambda: None
    torch.cuda.device_count = lambda: 1

    import net.sanet as sanet_module

    def _cpu_data_parallel(module, *inputs, **kwargs):
        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            return module(*inputs[0], **kwargs)
        return module(*inputs, **kwargs)

    sanet_module.data_parallel = _cpu_data_parallel


def to_numpy_safe(value):
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        return [to_numpy_safe(v) for v in value]
    return value


VAL_FROC_THRESHOLDS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
VAL_SCORE_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]


def init_detection_meter():
    return {
        'num_scans': 0,
        'num_gt': 0,
        'num_candidates': 0,
        'scores': [],
        'is_tp': [],
        'best_score_per_gt': [],
        'score_thresholds': {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in VAL_SCORE_THRESHOLDS},
    }


def center_distance_match(candidates, truths):
    if len(candidates) == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32) - 1
    if len(truths) == 0:
        return np.zeros((len(candidates),), dtype=np.int32), np.zeros((len(candidates),), dtype=np.int32) - 1

    order = np.argsort(candidates[:, 0])[::-1]
    matched_truth = np.zeros((len(truths),), dtype=bool)
    is_tp = np.zeros((len(candidates),), dtype=np.int32)
    assign = np.zeros((len(candidates),), dtype=np.int32) - 1

    for ci in order:
        cand = candidates[ci]
        dz = cand[1] - truths[:, 0]
        dy = cand[2] - truths[:, 1]
        dx = cand[3] - truths[:, 2]
        dist2 = dz * dz + dy * dy + dx * dx
        radius = np.maximum.reduce([truths[:, 3], truths[:, 4], truths[:, 5]]) / 2.0
        valid = (~matched_truth) & (dist2 <= radius * radius)
        if np.any(valid):
            valid_idx = np.where(valid)[0]
            ti = valid_idx[np.argmin(dist2[valid_idx])]
            matched_truth[ti] = True
            is_tp[ci] = 1
            assign[ci] = ti
    return is_tp, assign


def update_detection_meter(meter, proposals, truth_boxes):
    proposals = np.asarray(proposals, dtype=np.float32)
    if proposals.size == 0:
        proposals = np.empty((0, 8), dtype=np.float32)
    elif proposals.ndim == 1:
        proposals = proposals.reshape(1, -1)
    if proposals.shape[1] < 8:
        raise ValueError("proposals must have at least 8 columns, got shape %s" % (proposals.shape,))
    truth_boxes = np.asarray(truth_boxes, dtype=np.float32).reshape(-1, 6)
    meter['num_scans'] += 1
    meter['num_gt'] += len(truth_boxes)
    meter['num_candidates'] += len(proposals)

    candidates = proposals[:, 1:8] if len(proposals) else np.empty((0, 7), dtype=np.float32)
    is_tp, assign = center_distance_match(candidates, truth_boxes)

    meter['scores'].extend(candidates[:, 0].tolist())
    meter['is_tp'].extend(is_tp.tolist())

    best_scores = np.zeros((len(truth_boxes),), dtype=np.float32) - np.inf
    for ci, ti in enumerate(assign):
        if ti >= 0:
            best_scores[ti] = max(best_scores[ti], candidates[ci, 0])
    meter['best_score_per_gt'].extend(best_scores.tolist())

    for threshold, stat in meter['score_thresholds'].items():
        kept = candidates[:, 0] >= threshold if len(candidates) else np.zeros((0,), dtype=bool)
        tp = int(np.sum(is_tp[kept]))
        fp = int(np.sum(kept) - tp)
        fn = int(len(truth_boxes) - tp)
        stat['tp'] += tp
        stat['fp'] += fp
        stat['fn'] += fn


def summarize_detection_meter(meter):
    num_scans = max(1, meter['num_scans'])
    num_gt = max(1, meter['num_gt'])
    scores = np.asarray(meter['scores'], dtype=np.float32)
    is_tp = np.asarray(meter['is_tp'], dtype=np.int32)
    best_scores = np.asarray(meter['best_score_per_gt'], dtype=np.float32)

    summary = {
        'num_scans': meter['num_scans'],
        'num_gt': meter['num_gt'],
        'num_candidates': meter['num_candidates'],
        'candidates_per_scan': float(meter['num_candidates']) / num_scans,
    }

    for threshold, stat in meter['score_thresholds'].items():
        summary['tp@%.2f' % threshold] = stat['tp']
        summary['fp@%.2f' % threshold] = stat['fp']
        summary['fn@%.2f' % threshold] = stat['fn']
        summary['sensitivity@%.2f' % threshold] = float(stat['tp']) / num_gt
        summary['fp_per_scan@%.2f' % threshold] = float(stat['fp']) / num_scans

    if len(scores) == 0:
        for fp_target in VAL_FROC_THRESHOLDS:
            summary['froc_sens@%.3g' % fp_target] = 0.0
        summary['froc_mean'] = 0.0
        return summary

    order = np.argsort(scores)[::-1]
    tp_cum = np.cumsum(is_tp[order])
    fp_cum = np.cumsum(1 - is_tp[order])
    sensitivity = tp_cum / float(num_gt)
    fp_per_scan = fp_cum / float(num_scans)

    froc_values = []
    for fp_target in VAL_FROC_THRESHOLDS:
        valid = np.where(fp_per_scan <= fp_target)[0]
        value = float(np.max(sensitivity[valid])) if len(valid) else 0.0
        summary['froc_sens@%.3g' % fp_target] = value
        froc_values.append(value)
    summary['froc_mean'] = float(np.mean(froc_values))
    summary['gt_detected_any_score'] = float(np.sum(np.isfinite(best_scores))) / num_gt if len(best_scores) else 0.0
    return summary


def merge_detection_meters(meters):
    merged = init_detection_meter()
    merged['num_scans'] = 0
    merged['num_gt'] = 0
    merged['num_candidates'] = 0
    for meter in meters:
        merged['num_scans'] += meter['num_scans']
        merged['num_gt'] += meter['num_gt']
        merged['num_candidates'] += meter['num_candidates']
        merged['scores'].extend(meter['scores'])
        merged['is_tp'].extend(meter['is_tp'])
        merged['best_score_per_gt'].extend(meter['best_score_per_gt'])
        for threshold in VAL_SCORE_THRESHOLDS:
            for key in ['tp', 'fp', 'fn']:
                merged['score_thresholds'][threshold][key] += meter['score_thresholds'][threshold][key]
    return merged


def distributed_merge_detection_meter(meter, args):
    if not getattr(args, 'distributed', False):
        return meter
    gathered = [None for _ in range(args.world_size)]
    dist.all_gather_object(gathered, meter)
    return merge_detection_meters(gathered)


parser = argparse.ArgumentParser(description='PyTorch Detector')
parser.add_argument('--net', '-m', metavar='NET', default=train_config['net'],
                    help='neural net')
parser.add_argument('--epochs', default=train_config['epochs'], type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=train_config['batch_size'], type=int, metavar='N',
                    help='batch size')
parser.add_argument('--epoch-rcnn', default=train_config['epoch_rcnn'], type=int, metavar='NR',
                    help='number of epochs before training rcnn')
parser.add_argument('--ckpt', default=None, type=str, metavar='CKPT',
                    help='checkpoint to use. Defaults to pretrained weights, or model/final.ckpt with --resume')
parser.add_argument('--optimizer', default=train_config['optimizer'], type=str, metavar='SPLIT',
                    help='which split set to use')
parser.add_argument('--init-lr', default=train_config['init_lr'], type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=train_config['momentum'], type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=train_config['weight_decay'], type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--epoch-save', default=train_config['epoch_save'], type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--early-stop-patience', default=train_config.get('early_stop_patience', 40), type=int,
                    help='Stop if the active FROC metric does not improve for this many epochs. <=0 disables.')
parser.add_argument('--out-dir', default="train_output", type=str, metavar='OUT',
                    help='directory to save results of this training')
parser.add_argument('--train-set', default=train_config['train_set_list'], nargs='+', type=str,
                    help='train set paths list')
parser.add_argument('--val-set', default=train_config['val_set_list'], nargs='+', type=str,
                    help='val set paths list')
parser.add_argument('--data-dir', default=train_config['DATA_DIR'], type=str, metavar='OUT',
                    help='path to load data')
parser.add_argument('--num-workers', default=train_config['num_workers'], type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--limit-train-samples', default=0, type=int, metavar='N',
                    help='use only first N training samples for debugging. 0 means all samples.')
parser.add_argument('--limit-val-samples', default=0, type=int, metavar='N',
                    help='use only first N validation samples for debugging. 0 means all samples.')
parser.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'), type=str,
                    help='CUDA_VISIBLE_DEVICES value. Ignored when CUDA is unavailable.')
parser.add_argument('--dataset', default=train_config['dataset'], type=str,
                    help="dataset name under data/, or 'all' for all SANet-ready datasets")
parser.add_argument('--resume', action='store_true',
                    help='resume epoch and optimizer from checkpoint instead of using it as pretrained weights')
parser.add_argument('--hard-fp-csv', nargs='+', type=str, default=None, help='One or more hard FP CSV files.')
parser.add_argument('--hard-fn-csv', nargs='+', type=str, default=None, help='One or more hard FN CSV files.')
parser.add_argument('--hard-fp-threshold', default=0.9, type=float,
                    help='Minimum FP probability used by --hard-fp-csv. Default: 0.9.')
parser.add_argument('--train-neg-pos-ratio', default=net_config.get('train_neg_pos_ratio', 1.0), type=float,
                    help='Negative/positive sample ratio per training epoch. Clamped to [1/3, 1].')
parser.add_argument('--sample-by-ct', action='store_true',
                    help='Train only: each positive CT is one item; load once and crop a full batch. '
                         'Val/eval/test keep BboxReader.')
parser.add_argument('--local-rank', '--local_rank', default=-1, type=int,
                    help='local rank passed by torchrun/torch.distributed.launch')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend for DDP training')
parser.add_argument('--use-aspp', action='store_true', help='Enable ASPP3D in the RPN head.')


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = -1

    args.distributed = args.world_size > 1
    if not args.distributed:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return

    if not torch.cuda.is_available():
        raise RuntimeError('DDP training requires CUDA. Launch single-process CPU training without torchrun.')

    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)
    dist.init_process_group(backend=args.dist_backend, init_method='env://')
    dist.barrier()
    os.environ['SANET_DISABLE_INTERNAL_DATA_PARALLEL'] = '1'


def cleanup_distributed(args):
    if getattr(args, 'distributed', False) and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args):
    return getattr(args, 'rank', 0) == 0


def unwrap_model(net):
    return net.module if isinstance(net, DistributedDataParallel) else net


def strip_module_prefix(state_dict):
    if not any(key.startswith('module.') for key in state_dict.keys()):
        return state_dict
    return {key[7:] if key.startswith('module.') else key: value for key, value in state_dict.items()}


def distributed_mean(value, args):
    if not getattr(args, 'distributed', False):
        return value
    tensor = torch.tensor([float(value)], dtype=torch.float32, device=args.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= args.world_size
    return tensor.item()


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def clear_model_batch_state(model):
    attrs = [
        'rpn_proposals',
        'raw_rpn_proposals',
        'detections',
        'ensemble_proposals',
        'rpn_labels',
        'rpn_label_assigns',
        'rpn_label_weights',
        'rpn_targets',
        'rpn_target_weights',
        'rpn_window',
        'rpn_logits_flat',
        'rpn_deltas_flat',
        'rcnn_proposals',
        'rcnn_labels',
        'rcnn_assigns',
        'rcnn_targets',
        'rcnn_logits',
        'rcnn_deltas',
        'keeps',
        'mask_probs',
        'total_loss',
        'rpn_cls_loss',
        'rpn_reg_loss',
        'rcnn_cls_loss',
        'rcnn_reg_loss',
    ]
    for attr in attrs:
        if hasattr(model, attr):
            delattr(model, attr)


def checkpoint_update_flags(epoch, epoch_rcnn, val_froc_mean, best_froc_mean, best_rcnn_froc_mean):
    is_best = val_froc_mean > best_froc_mean
    is_rcnn_epoch = epoch >= epoch_rcnn
    is_best_rcnn = is_rcnn_epoch and val_froc_mean > best_rcnn_froc_mean
    next_best_froc_mean = val_froc_mean if is_best else best_froc_mean
    next_best_rcnn_froc_mean = val_froc_mean if is_best_rcnn else best_rcnn_froc_mean
    return is_best, is_best_rcnn, next_best_froc_mean, next_best_rcnn_froc_mean


def early_stop_improved(epoch, epoch_rcnn, is_best, is_best_rcnn):
    return is_best_rcnn if epoch >= epoch_rcnn else is_best


def build_split_dataset(dataset_name, split, hard_fp_csv=None, hard_fn_csv=None,
                        hard_fp_threshold=0.9, train_neg_pos_ratio=1.0):
    cfgs = dataset_configs(dataset_name, skip_missing=(dataset_name == 'all'))
    datasets = []
    for dataset_cfg in cfgs:
        dataset_cfg = dict(dataset_cfg)
        if split == 'train':
            dataset_cfg.update({
                'hard_fp_csv': hard_fp_csv,
                'hard_fn_csv': hard_fn_csv,
                'hard_fp_threshold': hard_fp_threshold,
                'train_neg_pos_ratio': train_neg_pos_ratio,
            })
        if split == 'train':
            set_name = dataset_cfg['train_set_list']
            mode = 'train'
        elif split == 'val':
            set_name = dataset_cfg['val_set_list']
            mode = 'val'
        else:
            raise ValueError("Unsupported split: %s" % split)
        try:
            datasets.append(BboxReader(dataset_cfg['DATA_DIR'], set_name, dataset_cfg, mode=mode))
        except (FileNotFoundError, ValueError) as exc:
            if dataset_name != 'all':
                raise
            print("[%s] Skip %s split: %s" % (dataset_cfg['dataset'], split, exc))
    if not datasets:
        raise ValueError("No usable datasets for %s split in dataset=%s" % (split, dataset_name))
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def limit_dataset(dataset, limit):
    if limit is None or limit <= 0:
        return dataset
    return Subset(dataset, range(min(limit, len(dataset))))


def main():
    # Load training configuration
    args = parser.parse_args()
    init_distributed_mode(args)
    if args.gpu and not args.distributed:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if torch.cuda.is_available():
        pass
    else:
        enable_cpu_fallback()

    net = args.net
    out_dir = args.out_dir or default_out_dir(args.dataset)
    initial_checkpoint = args.ckpt
    if args.resume and initial_checkpoint is None:
        initial_checkpoint = os.path.join(out_dir, 'model', 'final.ckpt')
    elif initial_checkpoint is None:
        initial_checkpoint = train_config['initial_checkpoint']
    args.ckpt = initial_checkpoint
    weight_decay = args.weight_decay
    momentum = args.momentum
    optimizer = args.optimizer
    init_lr = args.init_lr
    epochs = args.epochs
    epoch_save = args.epoch_save
    epoch_rcnn = args.epoch_rcnn
    batch_size = args.batch_size
    lr_schdule = train_config['lr_schedule']
    if args.sample_by_ct:
        train_dataset = build_ct_batch_datasets(
            args.dataset,
            batch_size=batch_size,
            hard_fp_csv=args.hard_fp_csv,
            hard_fn_csv=args.hard_fn_csv,
            hard_fp_threshold=args.hard_fp_threshold,
            train_neg_pos_ratio=args.train_neg_pos_ratio,
        )
        # Val must stay on BboxReader so FROC / early-stop metrics stay comparable.
        val_dataset = build_split_dataset(args.dataset, 'val')
    else:
        train_dataset = build_split_dataset(
            args.dataset,
            'train',
            hard_fp_csv=args.hard_fp_csv,
            hard_fn_csv=args.hard_fn_csv,
            hard_fp_threshold=args.hard_fp_threshold,
            train_neg_pos_ratio=args.train_neg_pos_ratio,
        )
        val_dataset = build_split_dataset(args.dataset, 'val')
    train_dataset = limit_dataset(train_dataset, args.limit_train_samples)
    val_dataset = limit_dataset(val_dataset, args.limit_val_samples)

    pin_memory = torch.cuda.is_available()
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True) \
        if args.distributed else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False) \
        if args.distributed else None
    if args.sample_by_ct:
        # Each train item already contains ``batch_size`` patches from one main CT.
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=(train_sampler is None),
                                  sampler=train_sampler,
                                  num_workers=args.num_workers, pin_memory=pin_memory,
                                  collate_fn=ct_batch_collate)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                sampler=val_sampler,
                                num_workers=args.num_workers, pin_memory=pin_memory,
                                collate_fn=train_collate)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                                  sampler=train_sampler,
                                  num_workers=args.num_workers, pin_memory=pin_memory,
                                  collate_fn=train_collate)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                sampler=val_sampler,
                                num_workers=args.num_workers, pin_memory=pin_memory,
                                collate_fn=train_collate)

    # Initilize network
    net = getattr(this_module, net)(net_config, use_aspp=args.use_aspp)
    net = net.cuda()

    start_epoch = 0
    best_loss = np.inf
    best_froc_mean = -np.inf
    best_rcnn_froc_mean = -np.inf
    early_stop_no_improve = 0
    optimizer_state = None

    if initial_checkpoint:
        if args.resume and not os.path.isfile(initial_checkpoint):
            raise FileNotFoundError(
                "Resume checkpoint not found: %s. Run training first or pass --ckpt."
                % initial_checkpoint
            )
        print('[Loading model from %s]' % initial_checkpoint)
        checkpoint = torch.load(
            initial_checkpoint,
            map_location='cpu',
            weights_only=False,
        )
        checkpoint_state = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        checkpoint_state = strip_module_prefix(checkpoint_state)
        if args.resume:
            start_epoch = checkpoint.get('epoch', 0)
            best_loss = checkpoint.get('best_loss', np.inf)
            best_froc_mean = checkpoint.get('best_froc_mean', -np.inf)
            best_rcnn_froc_mean = checkpoint.get('best_rcnn_froc_mean', -np.inf)
            early_stop_no_improve = checkpoint.get('early_stop_no_improve', 0)
            optimizer_state = checkpoint.get('optimizer')

        state = net.state_dict()
        state.update(checkpoint_state)

        try:
            net.load_state_dict(state)
        except:
            print('Load something failed!')
            traceback.print_exc()

    if args.distributed:
        net = DistributedDataParallel(
            net,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    optimizer_cls = getattr(torch.optim, optimizer)
    model = unwrap_model(net)

    rcnn_params = list(model.rcnn_crop.parameters()) + list(model.rcnn_head.parameters())
    rcnn_param_ids = {id(p) for p in rcnn_params}
    base_params = [p for p in model.parameters() if id(p) not in rcnn_param_ids]

    optimizer = optimizer_cls([
        {'params': base_params, 'lr': init_lr, 'name': 'base'},
        {'params': rcnn_params, 'lr': 0.0, 'name': 'rcnn'},
    ], weight_decay=weight_decay, momentum=momentum)

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        optimizer_to_device(optimizer, args.device)

    start_epoch = start_epoch + 1

    model_out_dir = os.path.join(out_dir, 'model')
    tb_out_dir = os.path.join(out_dir, 'runs')
    if is_main_process(args) and not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    if args.distributed:
        dist.barrier()
    if is_main_process(args):
        logfile = os.path.join(out_dir, 'log_train')
        sys.stdout = Logger(logfile)

    print('[Training configuration]')
    for arg in vars(args):
        print(arg, getattr(args, arg))

    print('[Model configuration]')
    pprint.pprint(net_config)

    print('[start_epoch %d, out_dir %s]' % (start_epoch, out_dir))
    print('[length of train loader %d, length of valid loader %d]' % (len(train_loader), len(val_loader)))

    # Write graph to tensorboard for visualization
    if is_main_process(args):
        writer = SummaryWriter(tb_out_dir)
        train_writer = SummaryWriter(os.path.join(tb_out_dir, 'train'))
        val_writer = SummaryWriter(os.path.join(tb_out_dir, 'val'))
    else:
        writer = train_writer = val_writer = NullWriter()
    # writer.add_graph(net, (torch.zeros((16, 1, 128, 128, 128)).cuda(), [[]], [[]], [[]], [torch.zeros((16, 128, 128, 128))]), verbose=False)

    for i in tqdm(range(start_epoch, epochs + 1), desc='Total', disable=not is_main_process(args)):
        if train_sampler is not None:
            train_sampler.set_epoch(i)
        # learning rate schedule
        if isinstance(optimizer, torch.optim.SGD):
            base_lr = lr_schdule(i, init_lr=init_lr, total=epochs)

            rcnn_warmup_epochs = 10
            rcnn_lr_scale = 0.1
            rcnn_target_lr = base_lr * rcnn_lr_scale

            if i < epoch_rcnn:
                rcnn_lr = 0.0
            elif i < epoch_rcnn + rcnn_warmup_epochs:
                warmup_progress = float(i - epoch_rcnn + 1) / float(rcnn_warmup_epochs)
                rcnn_lr = rcnn_target_lr * warmup_progress
            else:
                rcnn_lr = rcnn_target_lr

            for param_group in optimizer.param_groups:
                if param_group.get('name') == 'rcnn':
                    param_group['lr'] = rcnn_lr
                else:
                    param_group['lr'] = base_lr
        else:
            base_lr = optimizer.param_groups[0]['lr']
            rcnn_lr = optimizer.param_groups[1]['lr']

        lr = base_lr

        model = unwrap_model(net)
        if i >= epoch_rcnn:
            model.use_rcnn = True
        else:
            model.use_rcnn = False

        if i < epoch_rcnn:
            model.loss_weights = (1.0, 1.0, 0.0, 0.0)
        elif i < epoch_rcnn + 10:
            model.loss_weights = (2.0, 1.0, 0.25, 1.0)
        else:
            model.loss_weights = (2.0, 1.0, 0.5, 1.0)

        print('[loss weights: rpn_cls %.2f, rpn_reg %.2f, rcnn_cls %.2f, rcnn_reg %.2f]' % model.loss_weights)

        print('[epoch %d, base_lr %.6f, rcnn_lr %.6f, use_rcnn: %r]' % (i, base_lr, rcnn_lr, model.use_rcnn))
        train(net, train_loader, optimizer, i, train_writer, args)
        val_summary = validate(net, val_loader, i, val_writer, args)
        val_loss = distributed_mean(val_summary['loss'], args)
        val_froc_mean = distributed_mean(val_summary['froc_mean'], args)

        is_best, is_best_rcnn, best_froc_mean, best_rcnn_froc_mean = checkpoint_update_flags(
            i,
            epoch_rcnn,
            val_froc_mean,
            best_froc_mean,
            best_rcnn_froc_mean,
        )
        early_stop_patience = int(args.early_stop_patience)
        early_stop_metric = 'best_rcnn_froc_mean' if i >= epoch_rcnn else 'best_froc_mean'
        should_stop = False
        if early_stop_patience > 0:
            if early_stop_improved(i, epoch_rcnn, is_best, is_best_rcnn):
                early_stop_no_improve = 0
            else:
                early_stop_no_improve += 1
            should_stop = early_stop_no_improve >= early_stop_patience

        if not is_main_process(args):
            if args.distributed:
                dist.barrier()
            if should_stop:
                break
            continue

        state_dict = unwrap_model(net).state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        if val_loss < best_loss:
            best_loss = val_loss

        checkpoint = {
            'epoch': i,
            'out_dir': out_dir,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
            'best_froc_mean': best_froc_mean,
            'best_rcnn_froc_mean': best_rcnn_froc_mean,
            'early_stop_no_improve': early_stop_no_improve,
            'val_loss': val_loss,
            'val_froc_mean': val_froc_mean,
            'val_det_summary': val_summary['det'],
            'val_rpn_det_summary': val_summary['rpn_det'],
        }
        # if i % epoch_save == 0:
        #     torch.save(checkpoint, os.path.join(model_out_dir, '%03d.ckpt' % i))

        torch.save(checkpoint, os.path.join(model_out_dir, 'final.ckpt'))
        if is_best:
            torch.save(checkpoint, os.path.join(model_out_dir, 'best.ckpt'))
            print('[best checkpoint updated: epoch %d, val_froc_mean %.6f, val_loss %.6f]' % (
                i, val_froc_mean, val_loss))
        if is_best_rcnn:
            torch.save(checkpoint, os.path.join(model_out_dir, 'best_rcnn.ckpt'))
            print('[best RCNN checkpoint updated: epoch %d, val_froc_mean %.6f, val_loss %.6f]' % (
                i, val_froc_mean, val_loss))

        if early_stop_patience > 0:
            print('[early stop: metric %s, no_improve %d/%d]' % (
                early_stop_metric, early_stop_no_improve, early_stop_patience))
            if should_stop:
                print('[early stop triggered at epoch %d using %s]' % (i, early_stop_metric))
                if args.distributed:
                    dist.barrier()
                break
        if args.distributed:
            dist.barrier()

    writer.close()
    train_writer.close()
    val_writer.close()
    cleanup_distributed(args)


def train(net, train_loader, optimizer, epoch, writer, args):
    model = unwrap_model(net)
    model.set_mode('train')
    s = time.time()
    rpn_cls_loss, rpn_reg_loss = [], []
    rcnn_cls_loss, rcnn_reg_loss = [], []
    total_loss = []
    rpn_stats = []
    rcnn_stats = []

    for j, (input, truth_box, truth_label) in tqdm(enumerate(train_loader), total=len(train_loader),
                                                   desc='Train %d' % epoch, disable=not is_main_process(args)):
        input = Variable(input).cuda()

        net(input, truth_box, truth_label)

        loss, rpn_stat, rcnn_stat = model.loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rpn_cls_loss.append(model.rpn_cls_loss.cpu().data.item())
        rpn_reg_loss.append(model.rpn_reg_loss.cpu().data.item())
        rcnn_cls_loss.append(model.rcnn_cls_loss.cpu().data.item())
        rcnn_reg_loss.append(model.rcnn_reg_loss.cpu().data.item())

        total_loss.append(loss.cpu().data.item())
        rpn_stats.append(to_numpy_safe(rpn_stat))
        rcnn_stats.append(to_numpy_safe(rcnn_stat))

        del input, truth_box, truth_label, loss, rpn_stat, rcnn_stat
        clear_model_batch_state(model)

    rpn_stats = np.asarray(rpn_stats, np.float32)

    print('Train Epoch %d, iter %d, total time %f, loss %f' % (epoch, j, time.time() - s, np.average(total_loss)))
    print('rpn_cls %f, rpn_reg %f, rcnn_cls %f, rcnn_reg %f' % \
          (np.average(rpn_cls_loss), np.average(rpn_reg_loss),
           np.average(rcnn_cls_loss), np.average(rcnn_reg_loss)
           ))
    print('rpn_stats: tpr %f, tnr %f, total pos %d, total neg %d, reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
        100.0 * np.sum(rpn_stats[:, 0]) / np.sum(rpn_stats[:, 1]),
        100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3]),
        np.sum(rpn_stats[:, 1]),
        np.sum(rpn_stats[:, 3]),
        np.mean(rpn_stats[:, 4]),
        np.mean(rpn_stats[:, 5]),
        np.mean(rpn_stats[:, 6]),
        np.mean(rpn_stats[:, 7]),
        np.mean(rpn_stats[:, 8]),
        np.mean(rpn_stats[:, 9])))

    # Write to tensorboard
    train_loss = np.average(total_loss)
    writer.add_scalar('loss', train_loss, epoch)
    writer.add_scalar('rpn_cls', np.average(rpn_cls_loss), epoch)
    writer.add_scalar('rpn_reg', np.average(rpn_reg_loss), epoch)
    writer.add_scalar('rcnn_cls', np.average(rcnn_cls_loss), epoch)
    writer.add_scalar('rcnn_reg', np.average(rcnn_reg_loss), epoch)

    writer.add_scalar('rpn_reg_z', np.mean(rpn_stats[:, 4]), epoch)
    writer.add_scalar('rpn_reg_y', np.mean(rpn_stats[:, 5]), epoch)
    writer.add_scalar('rpn_reg_x', np.mean(rpn_stats[:, 6]), epoch)
    writer.add_scalar('rpn_reg_d', np.mean(rpn_stats[:, 7]), epoch)
    writer.add_scalar('rpn_reg_h', np.mean(rpn_stats[:, 8]), epoch)
    writer.add_scalar('rpn_reg_w', np.mean(rpn_stats[:, 9]), epoch)

    if model.use_rcnn:
        confusion_matrix = np.asarray([stat[-1] for stat in rcnn_stats], np.int32)
        rcnn_stats = np.asarray([stat[:-1] for stat in rcnn_stats], np.float32)

        confusion_matrix = np.sum(confusion_matrix, 0)

        print('rcnn_stats: reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            np.mean(rcnn_stats[:, 0]),
            np.mean(rcnn_stats[:, 1]),
            np.mean(rcnn_stats[:, 2]),
            np.mean(rcnn_stats[:, 3]),
            np.mean(rcnn_stats[:, 4]),
            np.mean(rcnn_stats[:, 5])))
        # print_confusion_matrix(confusion_matrix)
        writer.add_scalar('rcnn_reg_z', np.mean(rcnn_stats[:, 0]), epoch)
        writer.add_scalar('rcnn_reg_y', np.mean(rcnn_stats[:, 1]), epoch)
        writer.add_scalar('rcnn_reg_x', np.mean(rcnn_stats[:, 2]), epoch)
        writer.add_scalar('rcnn_reg_d', np.mean(rcnn_stats[:, 3]), epoch)
        writer.add_scalar('rcnn_reg_h', np.mean(rcnn_stats[:, 4]), epoch)
        writer.add_scalar('rcnn_reg_w', np.mean(rcnn_stats[:, 5]), epoch)

    torch.cuda.empty_cache()

def validate(net, val_loader, epoch, writer, args):
    model = unwrap_model(net)
    model.set_mode('valid')
    rpn_cls_loss, rpn_reg_loss = [], []
    rcnn_cls_loss, rcnn_reg_loss = [], []
    total_loss = []
    rpn_stats = []
    rcnn_stats = []
    rpn_detection_meter = init_detection_meter()
    detection_meter = init_detection_meter()

    s = time.time()
    for j, (input, truth_box, truth_label) in tqdm(enumerate(val_loader), total=len(val_loader), desc='Val %d' % epoch,
                                                   disable=not is_main_process(args)):
        with torch.no_grad():
            input = Variable(input).cuda()

            net(input, truth_box, truth_label)
            loss, rpn_stat, rcnn_stat = model.loss()
            raw_proposals = getattr(model, 'raw_rpn_proposals', model.rpn_proposals)
            raw_proposals = raw_proposals.detach().cpu().numpy() if torch.is_tensor(raw_proposals) else np.empty((0, 8), dtype=np.float32)
            detections = model.detections.detach().cpu().numpy() if torch.is_tensor(model.detections) else np.empty((0, 8), dtype=np.float32)
            for b in range(len(truth_box)):
                update_detection_meter(
                    rpn_detection_meter,
                    raw_proposals[raw_proposals[:, 0] == b] if len(raw_proposals) else np.empty((0, 8), dtype=np.float32),
                    truth_box[b],
                )
                update_detection_meter(
                    detection_meter,
                    detections[detections[:, 0] == b] if len(detections) else np.empty((0, 8), dtype=np.float32),
                    truth_box[b],
                )

        rpn_cls_loss.append(model.rpn_cls_loss.cpu().data.item())
        rpn_reg_loss.append(model.rpn_reg_loss.cpu().data.item())
        rcnn_cls_loss.append(model.rcnn_cls_loss.cpu().data.item())
        rcnn_reg_loss.append(model.rcnn_reg_loss.cpu().data.item())

        total_loss.append(loss.cpu().data.item())
        rpn_stats.append(to_numpy_safe(rpn_stat))
        rcnn_stats.append(to_numpy_safe(rcnn_stat))
        del input, truth_box, truth_label, loss, rpn_stat, rcnn_stat, raw_proposals, detections
        clear_model_batch_state(model)

    rpn_stats = np.asarray(rpn_stats, np.float32)
    print('Val Epoch %d, iter %d, total time %f, loss %f' % (epoch, j, time.time() - s, np.average(total_loss)))
    print('rpn_cls %f, rpn_reg %f, rcnn_cls %f, rcnn_reg %f' % \
          (np.average(rpn_cls_loss), np.average(rpn_reg_loss),
           np.average(rcnn_cls_loss), np.average(rcnn_reg_loss)
           ))
    print('rpn_stats: tpr %f, tnr %f, total pos %d, total neg %d, reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
        100.0 * np.sum(rpn_stats[:, 0]) / np.sum(rpn_stats[:, 1]),
        100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3]),
        np.sum(rpn_stats[:, 1]),
        np.sum(rpn_stats[:, 3]),
        np.mean(rpn_stats[:, 4]),
        np.mean(rpn_stats[:, 5]),
        np.mean(rpn_stats[:, 6]),
        np.mean(rpn_stats[:, 7]),
        np.mean(rpn_stats[:, 8]),
        np.mean(rpn_stats[:, 9])))
    rpn_detection_meter = distributed_merge_detection_meter(rpn_detection_meter, args)
    detection_meter = distributed_merge_detection_meter(detection_meter, args)
    rpn_det_summary = summarize_detection_meter(rpn_detection_meter)
    det_summary = summarize_detection_meter(detection_meter)
    print('val_rpn_det: scans %d, gt %d, candidates %d, cand/scan %.4f, FROC_mean %.6f, gt_detected_any %.6f' % (
        rpn_det_summary['num_scans'],
        rpn_det_summary['num_gt'],
        rpn_det_summary['num_candidates'],
        rpn_det_summary['candidates_per_scan'],
        rpn_det_summary['froc_mean'],
        rpn_det_summary.get('gt_detected_any_score', 0.0),
    ))
    for threshold in VAL_SCORE_THRESHOLDS:
        print('val_rpn_det@%.2f: TP %d, FP %d, FN %d, sens %.6f, fp/scan %.6f' % (
            threshold,
            rpn_det_summary['tp@%.2f' % threshold],
            rpn_det_summary['fp@%.2f' % threshold],
            rpn_det_summary['fn@%.2f' % threshold],
            rpn_det_summary['sensitivity@%.2f' % threshold],
            rpn_det_summary['fp_per_scan@%.2f' % threshold],
        ))
    print('val_rpn_froc: ' + ', '.join(
        ['%.3gfp=%.6f' % (fp, rpn_det_summary['froc_sens@%.3g' % fp]) for fp in VAL_FROC_THRESHOLDS]
    ))
    print('val_det: scans %d, gt %d, candidates %d, cand/scan %.4f, FROC_mean %.6f, gt_detected_any %.6f' % (
        det_summary['num_scans'],
        det_summary['num_gt'],
        det_summary['num_candidates'],
        det_summary['candidates_per_scan'],
        det_summary['froc_mean'],
        det_summary.get('gt_detected_any_score', 0.0),
    ))
    for threshold in VAL_SCORE_THRESHOLDS:
        print('val_det@%.2f: TP %d, FP %d, FN %d, sens %.6f, fp/scan %.6f' % (
            threshold,
            det_summary['tp@%.2f' % threshold],
            det_summary['fp@%.2f' % threshold],
            det_summary['fn@%.2f' % threshold],
            det_summary['sensitivity@%.2f' % threshold],
            det_summary['fp_per_scan@%.2f' % threshold],
        ))
    print('val_det_froc: ' + ', '.join(
        ['%.3gfp=%.6f' % (fp, det_summary['froc_sens@%.3g' % fp]) for fp in VAL_FROC_THRESHOLDS]
    ))

    # Write to tensorboard
    val_loss = np.average(total_loss)
    writer.add_scalar('loss', val_loss, epoch)
    writer.add_scalar('rpn_cls', np.average(rpn_cls_loss), epoch)
    writer.add_scalar('rpn_reg', np.average(rpn_reg_loss), epoch)
    writer.add_scalar('rcnn_cls', np.average(rcnn_cls_loss), epoch)
    writer.add_scalar('rcnn_reg', np.average(rcnn_reg_loss), epoch)

    writer.add_scalar('rpn_reg_z', np.mean(rpn_stats[:, 4]), epoch)
    writer.add_scalar('rpn_reg_y', np.mean(rpn_stats[:, 5]), epoch)
    writer.add_scalar('rpn_reg_x', np.mean(rpn_stats[:, 6]), epoch)
    writer.add_scalar('rpn_reg_d', np.mean(rpn_stats[:, 7]), epoch)
    writer.add_scalar('rpn_reg_h', np.mean(rpn_stats[:, 8]), epoch)
    writer.add_scalar('rpn_reg_w', np.mean(rpn_stats[:, 9]), epoch)
    writer.add_scalar('det/rpn_candidates_per_scan', rpn_det_summary['candidates_per_scan'], epoch)
    writer.add_scalar('det/rpn_froc_mean', rpn_det_summary['froc_mean'], epoch)
    writer.add_scalar('det/rpn_gt_detected_any_score', rpn_det_summary.get('gt_detected_any_score', 0.0), epoch)
    writer.add_scalar('det/candidates_per_scan', det_summary['candidates_per_scan'], epoch)
    writer.add_scalar('det/froc_mean', det_summary['froc_mean'], epoch)
    writer.add_scalar('det/gt_detected_any_score', det_summary.get('gt_detected_any_score', 0.0), epoch)
    for threshold in VAL_SCORE_THRESHOLDS:
        writer.add_scalar('det/rpn_sensitivity@%.2f' % threshold, rpn_det_summary['sensitivity@%.2f' % threshold], epoch)
        writer.add_scalar('det/rpn_fp_per_scan@%.2f' % threshold, rpn_det_summary['fp_per_scan@%.2f' % threshold], epoch)
        writer.add_scalar('det/sensitivity@%.2f' % threshold, det_summary['sensitivity@%.2f' % threshold], epoch)
        writer.add_scalar('det/fp_per_scan@%.2f' % threshold, det_summary['fp_per_scan@%.2f' % threshold], epoch)
    for fp in VAL_FROC_THRESHOLDS:
        writer.add_scalar('det/rpn_froc_sens@%.3gfp' % fp, rpn_det_summary['froc_sens@%.3g' % fp], epoch)
        writer.add_scalar('det/froc_sens@%.3gfp' % fp, det_summary['froc_sens@%.3g' % fp], epoch)

    if model.use_rcnn:
        confusion_matrix = np.asarray([stat[-1] for stat in rcnn_stats], np.int32)
        rcnn_stats = np.asarray([stat[:-1] for stat in rcnn_stats], np.float32)

        confusion_matrix = np.sum(confusion_matrix, 0)
        print('rcnn_stats: reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            np.mean(rcnn_stats[:, 0]),
            np.mean(rcnn_stats[:, 1]),
            np.mean(rcnn_stats[:, 2]),
            np.mean(rcnn_stats[:, 3]),
            np.mean(rcnn_stats[:, 4]),
            np.mean(rcnn_stats[:, 5])))
        # print_confusion_matrix(confusion_matrix)
        writer.add_scalar('rcnn_reg_z', np.mean(rcnn_stats[:, 0]), epoch)
        writer.add_scalar('rcnn_reg_y', np.mean(rcnn_stats[:, 1]), epoch)
        writer.add_scalar('rcnn_reg_x', np.mean(rcnn_stats[:, 2]), epoch)
        writer.add_scalar('rcnn_reg_d', np.mean(rcnn_stats[:, 3]), epoch)
        writer.add_scalar('rcnn_reg_h', np.mean(rcnn_stats[:, 4]), epoch)
        writer.add_scalar('rcnn_reg_w', np.mean(rcnn_stats[:, 5]), epoch)

    torch.cuda.empty_cache()
    return {
        'loss': val_loss,
        'froc_mean': det_summary['froc_mean'],
        'det': det_summary,
        'rpn_det': rpn_det_summary,
    }


def print_confusion_matrix(confusion_matrix):
    line_new = '{:>4}  ' * (len(config['roi_names']) + 2)
    print(line_new.format('gt/p', *list(range(len(config['roi_names']) + 1))))

    for i in range(len(config['roi_names']) + 1):
        print(line_new.format(i, *list(confusion_matrix[i])))


if __name__ == '__main__':
    main()

'''
直接训练：
    python train.py --dataset histopathology --epochs 10 --out-dir output

恢复训练：
默认从输出目录的 model/final.ckpt 恢复：
  python train.py --dataset histopathology --resume --out-dir train_output
指定某个 checkpoint 恢复：
  python train.py --dataset histopathology --resume --ckpt train_output/model/final.ckpt
'''
