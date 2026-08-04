import argparse
import gc
import os
import sys
import traceback

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'build/box'))

'''
对原训练集推理,调用 evaluationScript/noduleCADEvaluationLUNA16.py,汇总输出：hard_examples/hard_fps.csv
hard_examples/hard_fns.csv
'''

def preset_cuda_visible_devices():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[i + 1]
            return
        if arg.startswith('--gpu='):
            os.environ['CUDA_VISIBLE_DEVICES'] = arg.split('=', 1)[1]
            return


preset_cuda_visible_devices()

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

sanet_module = None
config = None
dataset_configs = None
BboxReader = None
train_collate = None
noduleCADEvaluation = None
SANet = None


def load_project_modules():
    global sanet_module, config, dataset_configs, BboxReader, train_collate, noduleCADEvaluation, SANet
    if SANet is not None:
        return

    import net.sanet as _sanet_module
    from config import config as _config, dataset_configs as _dataset_configs
    from dataset.bbox_reader import BboxReader as _BboxReader
    from dataset.collate import train_collate as _train_collate
    from evaluationScript.noduleCADEvaluationLUNA16 import noduleCADEvaluation as _noduleCADEvaluation
    from net.sanet import SANet as _SANet

    sanet_module = _sanet_module
    config = _config
    dataset_configs = _dataset_configs
    BboxReader = _BboxReader
    train_collate = _train_collate
    noduleCADEvaluation = _noduleCADEvaluation
    SANet = _SANet


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
        'rpn_logits_flat', 'rpn_deltas_flat', 'rpn_window', 'rpn_proposals',
        'detections', 'ensemble_proposals', 'rcnn_logits', 'rcnn_deltas',
        'keeps', 'mask_probs',
    ]
    for name in cache_names:
        if hasattr(net, name):
            setattr(net, name, None)
    gc.collect()


def prepare_evaluation_annotations(annotation_path, eval_dir):
    annotations = pd.read_csv(annotation_path)
    center_columns = {'center_x', 'center_y', 'center_z', 'diameter'}
    if not center_columns.issubset(annotations.columns):
        corner_columns = {'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'}
        missing = corner_columns.difference(annotations.columns)
        if missing:
            raise ValueError(
                "Annotation file %s is missing columns: %s"
                % (annotation_path, ', '.join(sorted(missing)))
            )
        annotations['center_x'] = (annotations['xmin'] + annotations['xmax']) / 2.
        annotations['center_y'] = (annotations['ymin'] + annotations['ymax']) / 2.
        annotations['center_z'] = (annotations['zmin'] + annotations['zmax']) / 2.
        sizes = pd.DataFrame({
            'x': annotations['xmax'] - annotations['xmin'] + 1,
            'y': annotations['ymax'] - annotations['ymin'] + 1,
            'z': annotations['zmax'] - annotations['zmin'] + 1,
        })
        annotations['diameter'] = sizes.max(axis=1)

    output_path = os.path.join(eval_dir, 'annotations.csv')
    annotations.to_csv(output_path, index=False)
    return output_path


def pid_keys(value):
    pid = str(value).strip()
    if pid.endswith('.0'):
        pid = pid[:-2]
    keys = {pid}
    if pid.isdigit():
        keys.add(str(int(pid)))
        keys.add(pid.zfill(5))
        keys.add(pid.zfill(6))
    return keys


def annotation_lookup(annotation_path):
    annos = pd.read_csv(annotation_path)
    if not {'center_x', 'center_y', 'center_z', 'diameter'}.issubset(annos.columns):
        annos['center_x'] = (annos['xmin'] + annos['xmax']) / 2.
        annos['center_y'] = (annos['ymin'] + annos['ymax']) / 2.
        annos['center_z'] = (annos['zmin'] + annos['zmax']) / 2.
        sizes = pd.DataFrame({
            'x': annos['xmax'] - annos['xmin'] + 1,
            'y': annos['ymax'] - annos['ymin'] + 1,
            'z': annos['zmax'] - annos['zmin'] + 1,
        })
        annos['diameter'] = sizes.max(axis=1)

    lookup = {}
    for _, row in annos.iterrows():
        for key in pid_keys(row['pid']):
            lookup.setdefault(key, []).append(row)
    return lookup


def match_fn_to_gt(fn_row, annos_by_pid):
    matches = None
    for key in pid_keys(fn_row['seriesuid']):
        if key in annos_by_pid:
            matches = annos_by_pid[key]
            break
    if not matches:
        return None

    center = np.asarray([fn_row['coordX'], fn_row['coordY'], fn_row['coordZ']], dtype=np.float32)
    best_row = None
    best_dist = None
    for row in matches:
        gt_center = np.asarray([row['center_x'], row['center_y'], row['center_z']], dtype=np.float32)
        dist = float(np.sum((gt_center - center) ** 2))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_row = row
    return best_row


def read_fps(dataset_name, fp_csv, fp_threshold):
    df = pd.read_csv(fp_csv)
    out = []
    for _, row in df.iterrows():
        probability = float(row['probability']) if pd.notna(row['probability']) else 0.0
        if probability < fp_threshold:
            continue
        radius = float(row['radius']) if pd.notna(row['radius']) else 0.0
        out.append({
            'dataset': dataset_name,
            'pid': str(row['seriesuid']),
            'center_x': float(row['coordX']),
            'center_y': float(row['coordY']),
            'center_z': float(row['coordZ']),
            'diameter': radius * 2.0,
            'probability': probability,
        })
    return out


def read_fns(dataset_name, fn_csv, annotation_path):
    df = pd.read_csv(fn_csv)
    annos_by_pid = annotation_lookup(annotation_path)
    out = []
    for nodule_id, row in df.iterrows():
        gt = match_fn_to_gt(row, annos_by_pid)
        if gt is None:
            continue
        out.append({
            'dataset': dataset_name,
            'pid': str(gt['pid']),
            'zmin': float(gt['zmin']),
            'zmax': float(gt['zmax']),
            'ymin': float(gt['ymin']),
            'ymax': float(gt['ymax']),
            'xmin': float(gt['xmin']),
            'xmax': float(gt['xmax']),
            'nodule_id': int(nodule_id),
            'center_x': float(gt['center_x']),
            'center_y': float(gt['center_y']),
            'center_z': float(gt['center_z']),
            'diameter': float(gt['diameter']),
            'probability': '',
        })
    return out


def load_model(ckpt_path):
    net = SANet(config).cuda()
    print('[Loading model from %s]' % ckpt_path)
    checkpoint = torch.load(
        ckpt_path,
        map_location='cpu' if not torch.cuda.is_available() else None,
        weights_only=False,
    )
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key[7:] if key.startswith('module.') else key: value for key, value in state_dict.items()}
    net.load_state_dict(state_dict)
    net.set_mode('eval')
    # Hard-example mining uses the same RPN proposal CSV format as test.py
    # (pid, center_x, center_y, center_z, diameter, probability).  RCNN is not
    # needed here and can drop/mismatch proposals before we save rpn_proposals.
    net.use_rcnn = False
    return net


def infer_dataset(net, dataset_cfg, out_dir, num_workers):
    dataset_name = dataset_cfg['dataset']
    eval_dir = os.path.join(out_dir, 'per_dataset', dataset_name, 'FROC')
    det_dir = os.path.join(out_dir, 'per_dataset', dataset_name, 'detections')
    res_dir = os.path.join(eval_dir, 'res')
    os.makedirs(det_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    eval_cfg = dict(dataset_cfg)
    eval_cfg['test_anno'] = dataset_cfg['train_anno']
    eval_cfg['hard_fp_csv'] = None
    eval_cfg['hard_fn_csv'] = None
    dataset = BboxReader(eval_cfg['DATA_DIR'], eval_cfg['train_set_list'], eval_cfg, mode='eval')
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=train_collate,
    )

    res = []
    for i, (input_tensor, truth_bboxes, truth_labels) in enumerate(tqdm(loader, desc=dataset_name)):
        try:
            clear_inference_cache(net)
            input_tensor = Variable(input_tensor).cuda()
            truth_bboxes = np.asarray(truth_bboxes)
            truth_labels = np.asarray(truth_labels)
            pid = dataset.filenames[i]

            with torch.no_grad():
                net.forward(input_tensor, truth_bboxes, truth_labels)

            detections = net.rpn_proposals.detach().cpu().numpy()
            det_path = os.path.join(det_dir, '%s_detections.npy' % pid)
            if len(detections):
                detections = detections[:, 1:-1]
                np.save(det_path, detections)
                csv_dets = detections[:, [3, 2, 1, 4, 0]]
                names = np.asarray([[pid]] * len(csv_dets))
                res.append(np.concatenate([names, csv_dets], axis=1))
            elif os.path.exists(det_path):
                os.remove(det_path)
        except Exception:
            traceback.print_exc()
            raise
        finally:
            clear_inference_cache(net)
            torch.cuda.empty_cache()

    columns = ['pid', 'center_x', 'center_y', 'center_z', 'diameter', 'probability']
    if res:
        result_df = pd.DataFrame(np.concatenate(res, axis=0), columns=columns)
    else:
        result_df = pd.DataFrame(columns=columns)
    result_path = os.path.join(eval_dir, 'results.csv')
    result_df.to_csv(result_path, index=False)

    annotations_path = prepare_evaluation_annotations(dataset_cfg['train_anno'], eval_dir)
    noduleCADEvaluation(annotations_path, result_path, dataset_cfg['train_set_list'], res_dir)
    return annotations_path, os.path.join(res_dir, 'FPs.csv'), os.path.join(res_dir, 'FNs.csv')


def parse_args():
    parser = argparse.ArgumentParser(description='Mine hard FP/FN samples from the training set.')
    parser.add_argument('--dataset', default='all', help="dataset name under data/, or 'all'")
    parser.add_argument('--ckpt', default='/mnt/afs2/code/SANet/train_pretrained_hardsamples_rcnn40_PN11_v3_lt1002/model/best_rcnn.ckpt', help='checkpoint used for mining')
    parser.add_argument('--out-dir', default='hard_examples_train_v3', help='directory to save mined CSVs')
    parser.add_argument('--fp-threshold', type=float, default=0.9, help='minimum FP probability to keep')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--gpu', default=None, help='CUDA_VISIBLE_DEVICES value')
    return parser.parse_args()


def main():
    args = parse_args()
    load_project_modules()
    enable_cpu_fallback()
    os.makedirs(args.out_dir, exist_ok=True)

    cfgs = dataset_configs(args.dataset, skip_missing=(args.dataset == 'all'))
    net = load_model(args.ckpt)

    all_fps = []
    all_fns = []
    for dataset_cfg in cfgs:
        try:
            annotations_path, fp_csv, fn_csv = infer_dataset(net, dataset_cfg, args.out_dir, args.num_workers)
        except (FileNotFoundError, ValueError) as exc:
            if args.dataset != 'all':
                raise
            print('[%s] Skip mining: %s' % (dataset_cfg['dataset'], exc))
            continue
        all_fps.extend(read_fps(dataset_cfg['dataset'], fp_csv, args.fp_threshold))
        all_fns.extend(read_fns(dataset_cfg['dataset'], fn_csv, annotations_path))

    fp_out = os.path.join(args.out_dir, 'hard_fps.csv')
    fn_out = os.path.join(args.out_dir, 'hard_fns.csv')
    pd.DataFrame(
        all_fps,
        columns=['dataset', 'pid', 'center_x', 'center_y', 'center_z', 'diameter', 'probability'],
    ).to_csv(fp_out, index=False)
    pd.DataFrame(
        all_fns,
        columns=[
            'dataset', 'pid', 'zmin', 'zmax', 'ymin', 'ymax', 'xmin', 'xmax',
            'nodule_id', 'center_x', 'center_y', 'center_z', 'diameter', 'probability',
        ],
    ).to_csv(fn_out, index=False)
    print('Saved hard FP csv: %s (%d rows, threshold %.3f)' % (fp_out, len(all_fps), args.fp_threshold))
    print('Saved hard FN csv: %s (%d rows)' % (fn_out, len(all_fns)))


if __name__ == '__main__':
    main()
