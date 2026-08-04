import numpy as np
import torch
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ['SANET_DISABLE_INTERNAL_DATA_PARALLEL'] = '1'
import gc
import traceback
import time
import nrrd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build/box'))
import matplotlib.pyplot as plt
import logging
import argparse
import torch.nn.functional as F
from scipy.stats import norm
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import data_parallel
from scipy.ndimage import label
from scipy.ndimage import center_of_mass
from net.sanet import SANet
from dataset.collate import train_collate, test_collate, eval_collate
from dataset.bbox_reader import BboxReader
from config import config, dataset_configs, default_out_dir
import pandas as pd
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

this_module = sys.modules[__name__]
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--net', '-m', metavar='NET', default=config['net'],
                    help='neural net')
parser.add_argument("--mode", type=str, default = 'eval',
                    help="you want to test or val")
parser.add_argument("--weight", type=str, default='train_output/model/best.ckpt',
                    help="path to model weights to be used")
parser.add_argument("--dicom-path", type=str, default=None,
                    help="path to dicom files of patient")
parser.add_argument("--out-dir", "--out_dir", dest="out_dir", type=str, default="test_output",
                    help="path to save the results")
parser.add_argument("--test-set-name", "--test_set_name", dest="test_set_name", type=str,
                    default=None, help="path to the test case list")
parser.add_argument("--dataset", default=config['dataset'], type=str,
                    help="dataset name under data/, or 'all' for all SANet-ready datasets")


def main():
    logging.basicConfig(format='[%(levelname)s][%(asctime)s] %(message)s', level=logging.INFO)
    args = parser.parse_args()
    enable_cpu_fallback()

    if args.mode == 'eval':
        num_workers = 0 if not torch.cuda.is_available() else 16
        initial_checkpoint = args.weight
        net = args.net

        cfgs = dataset_configs(args.dataset, skip_missing=(args.dataset == 'all'))

        net = getattr(this_module, net)(config)
        net = net.cuda()

        if initial_checkpoint:
            print('[Loading model from %s]' % initial_checkpoint)
            checkpoint = torch.load(
                initial_checkpoint,
                map_location='cpu' if not torch.cuda.is_available() else None,
                weights_only=False,
            )
            epoch = checkpoint['epoch']

            net.load_state_dict(checkpoint['state_dict'])
        else:
            print('No model weight file specified')
            return

        for dataset_cfg in cfgs:
            test_set_name = args.test_set_name or dataset_cfg['test_set_name']
            out_dir = args.out_dir or default_out_dir(args.dataset)
            if args.dataset == 'all':
                save_dir = os.path.join(out_dir, 'res', str(epoch)+"_pretrained_hardsamples_rcnn40", dataset_cfg['dataset'])
            else:
                save_dir = os.path.join(out_dir, 'res', str(epoch)+"_pretrained_hardsamples_rcnn40",args.dataset)

            try:
                dataset = BboxReader(
                    dataset_cfg['DATA_DIR'], test_set_name, dataset_cfg, mode='eval'
                )
            except (FileNotFoundError, ValueError) as exc:
                if args.dataset != 'all':
                    raise
                print("[%s] Skip test split: %s" % (dataset_cfg['dataset'], exc))
                continue

            print('dataset', dataset_cfg['dataset'])
            print('out_dir', out_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if not os.path.exists(os.path.join(save_dir, 'FROC')):
                os.makedirs(os.path.join(save_dir, 'FROC'))

            test_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                     num_workers=num_workers, pin_memory=False, collate_fn=train_collate)
            eval(net, test_loader, save_dir)
    else:
        logging.error('Mode %s is not supported' % (args.mode))


def eval(net, dataset, save_dir=None):
    net.set_mode('eval')
    net.use_rcnn = True

    print('Total # of eval data %d' % (len(dataset)))
    for i, (input, truth_bboxes, truth_labels) in enumerate(dataset):
        try:
            clear_inference_cache(net)
            input = Variable(input).cuda()
            truth_bboxes = np.array(truth_bboxes)
            truth_labels = np.array(truth_labels)
            pid = dataset.dataset.filenames[i]

            print('[%d] Predicting %s' % (i, pid))

            with torch.no_grad():
                net.forward(input, truth_bboxes, truth_labels)

            # detections = net.rpn_proposals.cpu().numpy() # 进入RCNN的RPN最终 proposal
            detections = net.detections.cpu().numpy()  # RCNN分类、回归和NMS后的最终检测框

            print('detections', detections.shape)

            detections_path = os.path.join(save_dir, '%s_detections.npy' % pid)
            if len(detections):
                detections = detections[:, 1:-1]
                np.save(detections_path, detections)
            elif os.path.exists(detections_path):
                os.remove(detections_path)

            # Clear gpu memory
            del input, truth_bboxes, truth_labels
            clear_inference_cache(net)
            torch.cuda.empty_cache()

        except Exception:
            del input, truth_bboxes, truth_labels
            clear_inference_cache(net)
            torch.cuda.empty_cache()
            traceback.print_exc()
            raise
    
    # Generate prediction csv for the use of performning FROC analysis
    res = []
    for pid in dataset.dataset.filenames:
        if os.path.exists(os.path.join(save_dir, '%s_detections.npy' % (pid))):
            detections = np.load(os.path.join(save_dir, '%s_detections.npy' % (pid)))
            detections = detections[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(detections))
            res.append(np.concatenate([names, detections], axis=1))
    
    col_names = ['pid','center_x','center_y','center_z','diameter', 'probability']
    if res:
        res = np.concatenate(res, axis=0)
        df = pd.DataFrame(res, columns=col_names)
    else:
        df = pd.DataFrame(columns=col_names)

    eval_dir = os.path.join(save_dir, 'FROC')
    res_path = os.path.join(eval_dir, 'results.csv')
    df.to_csv(res_path, index=False)

    # Start evaluating
    if not os.path.exists(os.path.join(eval_dir, 'res')):
        os.makedirs(os.path.join(eval_dir, 'res'))

    annotations_filename = prepare_evaluation_annotations(
        dataset.dataset.cfg['test_anno'], eval_dir
    )
    val_path = dataset.dataset.set_name

    noduleCADEvaluation(annotations_filename, res_path, val_path, os.path.join(eval_dir, 'res'))

'''
清理病例间GPU缓存,解决病例间gpu溢出问题
'''
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


if __name__ == '__main__':
    main()

'''
python test.py --dataset all --weight /mnt/afs2/code/SANet/train_pretrained_rcnn20/model/best.ckpt --out-dir /mnt/afs2/code/SANet/test_output 
'''
