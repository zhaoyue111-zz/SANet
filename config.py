import os
import numpy as np
import torch
import random

# Set seed
SEED = 35202
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DATA_ROOT = '/mnt/afs2/data'
DEFAULT_DATASET = 'all'


def dataset_base(dataset_name):
    return os.path.join(DATA_ROOT, dataset_name) + os.sep


# Preprocessing using preserved HU in dilated part of mask
BASE = dataset_base(DEFAULT_DATASET)  # make sure you have the ending '/'
data_config = {
    # directory for putting all preprocessed results for training to this path
    'preprocessed_data_dir': BASE + 'full',

    'roi_names': ['nodule'],
    'crop_size': [128, 128, 128],  # 训练时裁剪的 3D patch 大小
    'bbox_border': 8,  # 给标注框额外扩大的边界
    'partial_gt_positive_threshold': 0.5,
    'pad_value': 170,  # 裁剪越界时填充值
    # 'jitter_range': [0, 0, 0],
}


def get_anchors(bases, aspect_ratios):
    anchors = []
    for b in bases:
        for asp in aspect_ratios:
            d, h, w = b * asp[0], b * asp[1], b * asp[2]
            anchors.append([d, h, w])

    return anchors


bases = [5, 10, 20, 30, 50]
aspect_ratios = [[1, 1, 1]]

net_config = {
    # Net configuration
    'anchors': get_anchors(bases, aspect_ratios),  # 5 种立方体 anchor
    'chanel': 1,  # 输入通道数，当前是 1，对应单通道 3D 图像
    'crop_size': data_config['crop_size'],
    'stride': 4,
    'max_stride': 16,
    'num_neg': 800,
    'th_neg': 0.02,  # 小于该 IoU 阈值的 anchor 作为负样本
    'th_pos_train': 0.5,  # 训练时大于该 IoU 阈值的 anchor 作为正样本
    'th_pos_val': 1,
    'num_hard': 3,  # hard negative mining 的负样本倍数
    'bound_size': 12,  # crop 时在结节框周围保留的边界范围
    'blacklist': [],  # 测试时不使用的病例

    'augtype': {
        'flip': True,
        'rotate': False,
        'scale': False,
        'swap': False,
        'intensity': True,
        'noise': True,
        'blur': False,
    },
    'r_rand_crop': 0.2,  # 随机背景 crop 比例，用于增加负样本覆盖
    'intensity_aug': {
        'contrast_range': [0.85, 1.15],
        'brightness_range': [-0.10, 0.10],
        'noise_std': 0.03,
        'blur_sigma': [0.5, 1.0],
    },
    'pad_value': 170,

    # region proposal network configuration
    'rpn_train_bg_thresh_high': 0.02,
    'rpn_train_fg_thresh_low': 0.5,

    'rpn_train_nms_num': 300,
    'rpn_train_pre_nms_top_n': 6000,
    'rpn_train_nms_pre_score_threshold': 0.5,
    'rpn_train_nms_overlap_threshold': 0.1,
    'rpn_test_nms_num': 300,
    'rpn_test_pre_nms_top_n': 6000,
    'rpn_test_nms_pre_score_threshold': 0.5, # RPN 分数低于该阈值的 anchor 不进入 RPN NMS，也就不会成为 proposal
    'rpn_test_nms_overlap_threshold': 0.1, # 控制 RPN NMS 去重强度，越小去重越狠，保留框越少。两个候选框如果重叠超过 0.1，就认为它们太像，只保留分数高的那个

    # false positive reduction network configuration
    'num_class': len(data_config['roi_names']) + 1,
    'rcnn_crop_size': (7, 7, 7),  # can be set smaller, should not affect much
    'rcnn_train_fg_thresh_low': 0.5,
    'rcnn_train_bg_thresh_high': 0.1,
    'rcnn_train_batch_size': 64,
    'rcnn_train_fg_fraction': 0.5,
    'rcnn_train_nms_pre_score_threshold': 0.5,
    'rcnn_train_nms_overlap_threshold': 0.1,
    'rcnn_test_nms_pre_score_threshold': 0.0,
    'rcnn_test_nms_overlap_threshold': 0.1,

    'box_reg_weight': [1., 1., 1., 1., 1., 1.]  # bbox 回归各维度权重，顺序通常是 z, y, x, d, h, w
}


def lr_shedule(epoch, init_lr=0.01, total=200):
    if epoch <= total * 0.5:
        lr = init_lr
    elif epoch <= total * 0.8:
        lr = 0.1 * init_lr
    else:
        lr = 0.01 * init_lr
    return lr


train_config = {
    'net': 'SANet',
    'batch_size': 8,

    'lr_schedule': lr_shedule,
    'optimizer': 'SGD',
    'momentum': 0.9,
    'weight_decay': 1e-4,

    'epochs': 200,
    'epoch_save': 20,
    'epoch_rcnn': 40,  # 从第几个 epoch 开始启用 RCNN/FPR 分支。当前 20，前 19 个 epoch只训练 RPN。

    'num_workers': 8,
    'dataset': DEFAULT_DATASET,

    'train_set_list': BASE + 'split/train.txt',
    'val_set_list': BASE + 'split/val.txt',
    'test_set_name': BASE + 'split/test.txt',
    'train_anno': BASE + 'split/train_anno.csv',
    'val_anno': BASE + 'split/val_anno.csv',
    'test_anno': BASE + 'split/test_anno.csv',
    'DATA_DIR': data_config['preprocessed_data_dir'],
    'ROOT_DIR': os.getcwd()
}

if train_config['optimizer'] == 'SGD':
    train_config['init_lr'] = 0.01
elif train_config['optimizer'] == 'Adam':
    train_config['init_lr'] = 0.001
elif train_config['optimizer'] == 'RMSprop':
    train_config['init_lr'] = 2e-3

train_config['RESULTS_DIR'] = os.path.join(train_config['ROOT_DIR'], train_config['dataset'], 'results')
train_config['out_dir'] = os.path.join(train_config['RESULTS_DIR'], train_config['dataset'], 'full01')
train_config['initial_checkpoint'] = "pretrained/model.ckpt"   #

config = dict(data_config, **net_config)
config = dict(config, **train_config)


def is_sanet_dataset(dataset_name):
    base = dataset_base(dataset_name)
    required_paths = [
        os.path.join(base, 'full'),
        os.path.join(base, 'split', 'train.txt'),
        os.path.join(base, 'split', 'val.txt'),
        os.path.join(base, 'split', 'test.txt'),
        os.path.join(base, 'split', 'train_anno.csv'),
        os.path.join(base, 'split', 'val_anno.csv'),
        os.path.join(base, 'split', 'test_anno.csv'),
    ]
    return all(os.path.exists(path) for path in required_paths)


def available_datasets():
    names = []
    for name in sorted(os.listdir(DATA_ROOT)):
        if os.path.isdir(os.path.join(DATA_ROOT, name)) and is_sanet_dataset(name):
            names.append(name)
    return names


def dataset_config(dataset_name, skip_missing=False):
    base = dataset_base(dataset_name)
    cfg = dict(config)
    cfg.update({
        'dataset': dataset_name,
        'preprocessed_data_dir': base + 'full',
        'train_set_list': base + 'split/train.txt',
        'val_set_list': base + 'split/val.txt',
        'test_set_name': base + 'split/test.txt',
        'train_anno': base + 'split/train_anno.csv',
        'val_anno': base + 'split/val_anno.csv',
        'test_anno': base + 'split/test_anno.csv',
        'DATA_DIR': base + 'full',
        'RESULTS_DIR': os.path.join(train_config['ROOT_DIR'], dataset_name, 'results'),
        'out_dir': os.path.join(train_config['ROOT_DIR'], dataset_name, 'results', dataset_name, 'full01'),
        'skip_missing': skip_missing,
    })
    return cfg


def dataset_configs(dataset_name, skip_missing=False):
    if dataset_name == 'all':
        return [dataset_config(name, skip_missing=skip_missing) for name in available_datasets()]
    if not is_sanet_dataset(dataset_name):
        raise ValueError(
            "Unknown or incomplete dataset '%s'. Available datasets: %s"
            % (dataset_name, ', '.join(available_datasets()))
        )
    return [dataset_config(dataset_name, skip_missing=skip_missing)]


def default_out_dir(dataset_name):
    return os.path.join(train_config['ROOT_DIR'], dataset_name, 'results', dataset_name, 'full01')
