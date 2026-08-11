import numpy as np
import torch
from torch.utils.data import Dataset
import os, imageio
from scipy.ndimage import gaussian_filter, zoom
import warnings
from scipy.ndimage import rotate
import math
import time
import pandas as pd
import random

try:
    import nrrd
except ImportError:
    nrrd = None

def _normalize_csv_paths(csv_paths):
    if csv_paths is None:
        return []
    if isinstance(csv_paths, (str, os.PathLike)):
        return [os.fspath(csv_paths)]
    return [os.fspath(path) for path in csv_paths if path]

def _pid_keys(value):
    pid = str(value).strip()
    if pid.endswith('.0'):
        pid = pid[:-2]
    keys = {pid}
    if pid.isdigit():
        keys.add(str(int(pid)))
        keys.add(pid.zfill(5))
        keys.add(pid.zfill(6))
    return keys


def _row_dataset_matches(row, dataset_name):
    if not dataset_name or 'dataset' not in row.index:
        return True
    return str(row['dataset']) == str(dataset_name)


def _load_hard_fns(csv_paths, dataset_name, filename_to_idx):
    csv_paths = _normalize_csv_paths(csv_paths)
    samples = []

    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError("Hard FN csv not found: %s" % csv_path)

        df = pd.read_csv(csv_path)
        required_box = {'pid', 'zmin', 'zmax', 'ymin', 'ymax', 'xmin', 'xmax'}
        required_center = {'pid', 'center_x', 'center_y', 'center_z', 'diameter'}
        has_box = required_box.issubset(df.columns)
        has_center = required_center.issubset(df.columns)

        if not has_box and not has_center:
            raise ValueError("Hard FN csv %s must contain either bbox columns %s or center columns %s" % (csv_path, sorted(required_box), sorted(required_center)))

        for _, row in df.iterrows():
            if not _row_dataset_matches(row, dataset_name):
                continue

            idx = None
            for key in _pid_keys(row['pid']):
                if key in filename_to_idx:
                    idx = filename_to_idx[key]
                    break

            if idx is None:
                continue

            if has_box:
                box = [row['zmin'], row['zmax'], row['ymin'], row['ymax'], row['xmin'], row['xmax']]
            else:
                radius = float(row['diameter']) / 2.0
                box = [float(row['center_z']) - radius, float(row['center_z']) + radius, float(row['center_y']) - radius, float(row['center_y']) + radius, float(row['center_x']) - radius, float(row['center_x']) + radius]

            box = np.asarray(box, dtype=np.float32)

            if not np.isfinite(box).all():
                continue
            if not (box[1] >= box[0] and box[3] > box[2] and box[5] > box[4]):
                continue

            samples.append(np.concatenate([[idx], box]).astype(np.float32))

    return samples


def _load_hard_fps(csv_paths, dataset_name, filename_to_idx, probability_threshold):
    csv_paths = _normalize_csv_paths(csv_paths)
    samples = []

    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError("Hard FP csv not found: %s" % csv_path)

        df = pd.read_csv(csv_path)
        required = {'pid', 'center_x', 'center_y', 'center_z', 'probability'}
        missing = required.difference(df.columns)

        if missing:
            raise ValueError("Hard FP csv %s is missing columns: %s" % (csv_path, sorted(missing)))

        for _, row in df.iterrows():
            if not _row_dataset_matches(row, dataset_name):
                continue

            prob = float(row['probability'])
            if not np.isfinite(prob) or prob < probability_threshold:
                continue

            idx = None
            for key in _pid_keys(row['pid']):
                if key in filename_to_idx:
                    idx = filename_to_idx[key]
                    break

            if idx is None:
                continue

            diameter = float(row['diameter']) if 'diameter' in row.index and pd.notna(row['diameter']) else 0.0
            sample = np.asarray([idx, float(row['center_z']), float(row['center_y']), float(row['center_x']), diameter, prob], dtype=np.float32)

            if np.isfinite(sample).all():
                samples.append(sample)

    return samples
class BboxReader(Dataset):
    def __init__(self, data_dir, set_name, cfg, mode='train'):
        self.mode = mode
        self.cfg = cfg
        self.r_rand = cfg['r_rand_crop']
        self.augtype = cfg['augtype']
        self.pad_value = cfg['pad_value']
        self.partial_gt_positive_threshold = float(cfg.get('partial_gt_positive_threshold', 0.5))
        self.data_dir = data_dir
        self.stride = cfg['stride']
        self.blacklist = cfg['blacklist']
        self.set_name = set_name
        self.dataset_name = cfg.get('dataset', '')

        sizelim = 0
        sizelim2 = 10
        sizelim3 = 20

        labels = []
        with open(self.set_name, "r") as f:
            self.filenames = [line.strip() for line in f if line.strip()]

        if not self.filenames:
            raise ValueError(
                "Dataset list is empty: %s. Add one case ID per line or use the "
                "configured split file." % self.set_name
            )

        if mode != 'test':
            self.filenames = [f for f in self.filenames if (f not in self.blacklist)]

        missing_images = [
            fn for fn in self.filenames
            if not os.path.isfile(os.path.join(self.data_dir, '%s_zoom.npy' % fn))
        ]
        if missing_images and cfg.get('skip_missing', False):
            missing_set = set(missing_images)
            print(
                "[%s] Skipping %d missing preprocessed image(s) under %s"
                % (self.dataset_name or self.set_name, len(missing_images), self.data_dir)
            )
            self.filenames = [fn for fn in self.filenames if fn not in missing_set]
            if not self.filenames:
                raise ValueError(
                    "No available preprocessed images remain for %s split %s"
                    % (self.dataset_name or 'dataset', self.set_name)
                )
        elif missing_images:
            preview = ', '.join(missing_images[:5])
            raise FileNotFoundError(
                "Missing %d preprocessed image(s) under %s, including: %s"
                % (len(missing_images), self.data_dir, preview)
            )

        if self.mode == 'train':
            csv_dir = cfg['train_anno']
        elif self.mode == 'val':
            csv_dir = cfg.get('val_anno', cfg['train_anno'])
        else:
            csv_dir = cfg['test_anno']

        annos_all = pd.read_csv(csv_dir)
        filename_to_idx = {}
        for i, fn in enumerate(self.filenames):
            for key in _pid_keys(fn):
                filename_to_idx[key] = i

        for fn in self.filenames:
            annos = annos_all[annos_all['pid'] == int(fn)] #
            temp_annos = []
            if len(annos) > 0:
                for index in range(len(annos)):
                    anno = annos.iloc[index]
                    temp_annos.append([anno['zmin'], anno['zmax'], anno['ymin'], anno['ymax'], anno['xmin'], anno['xmax']])
            l = np.array(temp_annos)
            if np.all(l==0):
                l=np.array([])
            labels.append(l)

        self.sample_bboxes = labels
        # 每个 GT box 作为一个训练样本
        if self.mode in ['train', 'val']:
            self.bboxes = []
            for i, l in enumerate(labels):
                if len(l) > 0 :
                    for t in l:
                        # diameter = max(t[5] - t[4], t[3] - t[2])
                        # if diameter>sizelim:
                        #     self.bboxes.append([np.concatenate([[i],t])])
                        # if diameter>sizelim2:
                        #     self.bboxes+=[[np.concatenate([[i],t])]]*2
                        # if diameter>sizelim3:
                        #     self.bboxes+=[[np.concatenate([[i],t])]]*4
                        self.bboxes.append([np.concatenate([[i],t])])
            if self.mode == 'train':  # 训练时读取 hard FN，加入正样本 patch 来源
                hard_fns = _load_hard_fns(
                    cfg.get('hard_fn_csv'),
                    self.dataset_name,
                    filename_to_idx,
                )
                for sample in hard_fns:
                    self.bboxes.append([sample])
            if not self.bboxes:
                raise ValueError(
                    "No annotated boxes found for %s split %s"
                    % (self.dataset_name or 'dataset', self.set_name)
                )
            self.bboxes = np.concatenate(self.bboxes, axis=0).astype(np.float32)
            self.num_positive_samples = len(self.bboxes)
            if self.mode == 'train':  # 训练时读取 hard FP，加入负样本 patch 来源
                self.hard_fps = _load_hard_fps(
                    cfg.get('hard_fp_csv'),
                    self.dataset_name,
                    filename_to_idx,
                    float(cfg.get('hard_fp_threshold', 0.9)),
                )
                self.hard_fps = np.asarray(self.hard_fps, dtype=np.float32).reshape(-1, 6)
                neg_pos_ratio = float(cfg.get('train_neg_pos_ratio', 1.0))
                neg_pos_ratio = min(1.0, max(1.0 / 3.0, neg_pos_ratio))
                self.num_negative_samples = int(round(self.num_positive_samples * neg_pos_ratio))
                self.num_random_samples = self.num_negative_samples
                print(
                    "[%s] hard mining samples: hard_fn=%d, hard_fp_pool=%d, sampled_neg=%d, pos:neg=%.2f:1"
                    % (
                        self.dataset_name or self.set_name,
                        len(hard_fns),
                        len(self.hard_fps),
                        self.num_negative_samples,
                        float(self.num_positive_samples) / max(1, self.num_negative_samples),
                    )
                )
            else:
                self.hard_fps = np.zeros((0, 6), dtype=np.float32)
                self.num_negative_samples = 0
                self.num_random_samples = 0
        self.crop = Crop(cfg)

    def __getitem__(self, idx):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time
        is_random_img  = False
        hard_fp = None
        if self.mode in ['train', 'val']:
            if self.mode == 'train' and idx >= self.num_positive_samples:
                if len(self.hard_fps) > 0:
                    hard_fp = self.hard_fps[np.random.randint(len(self.hard_fps))]
                    is_random_crop = False
                else:
                    is_random_crop = True
                    is_random_img = np.random.randint(2)  # 50%概率随机选图像
            elif idx >= len(self.bboxes): # val理论上不会走到这里
                is_random_crop = True
                idx = idx % len(self.bboxes)
                is_random_img = np.random.randint(2)
            else:
                is_random_crop = False
        else:
            is_random_crop = False

        if self.mode in ['train', 'val']:
            if hard_fp is not None:
                filename = self.filenames[int(hard_fp[0])]
                imgs = self.load_image(filename)
                bboxes = self.sample_bboxes[int(hard_fp[0])]
                center_zyx = hard_fp[1:4]
                sample, target, bboxes, coord = self.crop.crop_at_center(imgs, center_zyx, bboxes)
                if self.mode == 'train':
                    sample = augment_intensity(
                        sample,
                        self.cfg.get('intensity_aug', {}),
                        do_intensity=self.augtype.get('intensity', False),
                        do_noise=self.augtype.get('noise', False),
                        do_blur=self.augtype.get('blur', False),
                    )
            elif not is_random_img:
                bbox = self.bboxes[idx]
                filename = self.filenames[int(bbox[0])]
                imgs = self.load_image(filename)

                # lines = os.listdir(os.path.join(self.data_dir.replace('full', 'img'), filename))
                # lines = sorted(lines)
                # slice_files = [os.path.join(self.data_dir.replace('full', 'img'), filename, s) for s in lines]
                # slices = [imageio.imread(s) for s in slice_files]
                # imgs = np.array(slices)
                # imgs = imgs[np.newaxis, ...]

                bboxes = self.sample_bboxes[int(bbox[0])]
                isScale = self.augtype['scale'] and (self.mode=='train')
                allow_large_lesion_resize = (
                    self.mode == 'train'
                    and not is_random_crop
                    and bool(self.cfg.get('large_lesion_resize', True))
                )
                sample, target, bboxes, coord = self.crop(
                    imgs,
                    bbox[1:],
                    bboxes,
                    isScale,
                    is_random_crop,
                    allow_large_lesion_resize=allow_large_lesion_resize,
                )
                if self.mode == 'train' and not is_random_crop:
                     sample, target, bboxes = augment(
                         sample, target, bboxes,
                         do_flip=self.augtype['flip'],
                         do_rotate=self.augtype['rotate'],
                         do_swap=self.augtype['swap'],
                         pad_value=self.pad_value,
                         rotate_angle_range=self.cfg.get('rotate_angle_range', 10.0),
                     )
                if self.mode == 'train':
                    sample = augment_intensity(
                        sample,
                        self.cfg.get('intensity_aug', {}),
                        do_intensity=self.augtype.get('intensity', False),
                        do_noise=self.augtype.get('noise', False),
                        do_blur=self.augtype.get('blur', False),
                    )
            else: # 随机背景裁剪
                randimid = np.random.randint(len(self.filenames))
                filename = self.filenames[randimid]
                imgs = self.load_image(filename)

                # lines = os.listdir(os.path.join(self.data_dir.replace('full', 'img'), filename))
                # lines = sorted(lines)
                # slice_files = [os.path.join(self.data_dir.replace('full', 'img'), filename, s) for s in lines]
                # slices = [imageio.imread(s) for s in slice_files]
                # imgs = np.array(slices)
                # imgs = imgs[np.newaxis, ...]

                bboxes = self.sample_bboxes[randimid]
                isScale = self.augtype['scale'] and (self.mode=='train')
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes,isScale=False,isRand=True) # target传入空列表 []，不指定任何GT box作为裁剪中心
                if self.mode == 'train':
                    sample = augment_intensity(
                        sample,
                        self.cfg.get('intensity_aug', {}),
                        do_intensity=self.augtype.get('intensity', False),
                        do_noise=self.augtype.get('noise', False),
                        do_blur=self.augtype.get('blur', False),
                    )

            if sample.shape[1] != self.cfg['crop_size'][0] or sample.shape[2] != \
                self.cfg['crop_size'][1] or sample.shape[3] != self.cfg['crop_size'][2]:
                print(filename, sample.shape)

            sample = (sample.astype(np.float32)-128)/128
            bboxes = build_patch_truth_boxes(
                bboxes,
                self.cfg['crop_size'],
                self.partial_gt_positive_threshold,
            )
            bboxes = corner_form_to_center_form(bboxes, self.cfg['bbox_border'])
            bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 7)
            truth_labels = bboxes[:, -1].astype(np.int64)
            truth_bboxes = bboxes[:, :-1]

            return [torch.from_numpy(sample).float(), truth_bboxes, truth_labels]

        if self.mode in ['eval']:
            filename = self.filenames[idx]
            image = self.load_image(filename)

            # lines = os.listdir(os.path.join(self.data_dir.replace('full', 'img'), filename))
            # lines = sorted(lines)
            # slice_files = [os.path.join(self.data_dir.replace('full', 'img'), filename, s) for s in lines]
            # slices = [imageio.imread(s) for s in slice_files]
            # imgs = np.array(slices)
            # image = imgs[np.newaxis, ...]
            
            original_image = image[0]

            image = pad2factor(image[0])
            image = np.expand_dims(image, 0)
            bboxes = self.sample_bboxes[idx]

            bboxes = corner_form_to_center_form(bboxes, self.cfg['bbox_border'])
            bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 7)
            truth_labels = bboxes[:, -1].astype(np.int64)
            truth_bboxes = bboxes[:, :-1]

            input = (image.astype(np.float32) - 128.) / 128.

            return [torch.from_numpy(input).float(), truth_bboxes, truth_labels]

    def load_image(self, filename):
        path = os.path.join(self.data_dir, '%s_zoom.npy' % filename)
        last_exc = None
        for attempt in range(3):
            try:
                return np.load(path)
            except (EOFError, OSError, ValueError) as exc:
                last_exc = exc
                time.sleep(0.2 * (attempt + 1) + random.random() * 0.1)

        size = os.path.getsize(path) if os.path.exists(path) else -1
        raise RuntimeError(
            "Failed to load npy for dataset=%s, mode=%s, pid=%s, path=%s, size=%d bytes after retries. "
            "If integrity check passes, this is likely transient remote-filesystem I/O; reduce --num-workers "
            "or copy data to local SSD."
            % (self.dataset_name or 'unknown', self.mode, filename, path, size)
        ) from last_exc

    def __len__(self):
        if self.mode == 'train':
            return self.num_positive_samples + self.num_negative_samples
        elif self.mode =='val':
            return len(self.bboxes)
        else:
            return len(self.filenames)

def corner_form_to_center_form(boxes, border):
    '''
    将角点坐标 (zmin, zmax, ymin, ymax, xmin, xmax) 转换为中心坐标 (z_center, y_center, x_center, depth, height, width, label)
    '''
    bboxes = []
    for box in boxes:
        label = int(box[6]) if len(box) > 6 else 1
        bboxes.append([(box[0] + box[1]) / 2.,
                       (box[2] + box[3]) / 2.,
                       (box[4] + box[5]) / 2.,
                       box[1] - box[0] + 1 + border,
                       box[3] - box[2] + 1 + border,
                       box[5] - box[4] + 1 + border,
                       label])
    return bboxes


def build_patch_truth_boxes(bboxes, size, partial_positive_threshold=0.5):
    """
    Rebuild patch supervision from all original GT boxes after crop translation.

    Output columns are zmin,zmax,ymin,ymax,xmin,xmax,label in patch coordinates.
    label=1 means positive supervision; label=-1 means ignore.
    """
    size = np.asarray(size, dtype=np.float32)
    bboxes = np.asarray(bboxes, dtype=np.float32)
    if bboxes.size == 0:
        return np.zeros((0, 7), dtype=np.float32)
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape(1, -1)

    threshold = float(partial_positive_threshold)
    patch_min = np.zeros(3, dtype=np.float32)
    patch_max = size - 1.0
    out = []
    for box in bboxes:
        if len(box) < 6 or not np.isfinite(box[:6]).all():
            continue
        starts = box[[0, 2, 4]].astype(np.float32)
        ends = box[[1, 3, 5]].astype(np.float32)
        gt_sizes = ends - starts + 1.0
        if np.any(gt_sizes <= 0):
            continue

        inter_starts = np.maximum(starts, patch_min)
        inter_ends = np.minimum(ends, patch_max)
        inter_sizes = inter_ends - inter_starts + 1.0
        if np.any(inter_sizes <= 0):
            continue

        visible_ratio = float(np.prod(inter_sizes) / np.prod(gt_sizes))
        clipped = np.array([
            inter_starts[0], inter_ends[0],
            inter_starts[1], inter_ends[1],
            inter_starts[2], inter_ends[2],
        ], dtype=np.float32)
        if visible_ratio >= threshold:
            label = 1
        else:
            label = -1
        out.append(np.concatenate([clipped, np.array([label], dtype=np.float32)]))

    if not out:
        return np.zeros((0, 7), dtype=np.float32)
    return np.asarray(out, dtype=np.float32).reshape(-1, 7)

def pad2factor(image, factor=32, pad_value=0):
    depth, height, width = image.shape
    d = int(math.ceil(depth / float(factor))) * factor
    h = int(math.ceil(height / float(factor))) * factor
    w = int(math.ceil(width / float(factor))) * factor

    pad = []
    pad.append([0, d - depth])
    pad.append([0, h - height])
    pad.append([0, w - width])

    # pad = []
    # pad.append([0, w - width + height-depth])
    # pad.append([0, h - height])
    # pad.append([0, w - width])

    image = np.pad(image, pad, 'constant', constant_values=pad_value)

    return image



def fillter_box(bboxes, size):
    size = np.asarray(size, dtype=np.float32)
    res = []
    bboxes = np.asarray(bboxes, dtype=np.float32)
    if bboxes.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape(1, -1)
    num_cols = bboxes.shape[1]
    for box in bboxes:
        box = np.asarray(box, dtype=np.float32)
        if len(box) < 6 or not np.all(np.isfinite(box[:6])):
            continue
        starts = box[[0, 2, 4]]
        ends = box[[1, 3, 5]]
        if np.all(starts >= 0) and np.all(ends < size) and np.all(ends > starts):
            res.append(box)
    if not res:
        return np.zeros((0, num_cols), dtype=np.float32)
    return np.asarray(res, dtype=np.float32).reshape(-1, num_cols)

def _transform_corner_box_yx(box, rotmat, center_yx):
    corners = np.asarray([
        [box[2], box[4]],
        [box[2], box[5]],
        [box[3], box[4]],
        [box[3], box[5]],
    ], dtype=np.float32)
    rotated = np.dot(corners - center_yx, rotmat.T) + center_yx
    out = np.copy(box)
    out[2] = rotated[:, 0].min()
    out[3] = rotated[:, 0].max()
    out[4] = rotated[:, 1].min()
    out[5] = rotated[:, 1].max()
    return out


def _box_inside_shape(box, shape_zyx):
    starts = np.asarray(box[[0, 2, 4]], dtype=np.float32)
    ends = np.asarray(box[[1, 3, 5]], dtype=np.float32)
    shape_zyx = np.asarray(shape_zyx, dtype=np.float32)
    return np.all(starts >= 0) and np.all(ends < shape_zyx) and np.all(ends > starts)


def _has_valid_corner_box(box):
    box = np.asarray(box, dtype=np.float32).reshape(-1)
    if box.size < 6 or not np.isfinite(box[:6]).all():
        return False
    return bool(np.all(box[[1, 3, 5]] > box[[0, 2, 4]]))


def _filter_boxes_inside_shape(boxes, shape_zyx):
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 6)
    kept = []
    for box in boxes:
        if _has_valid_corner_box(box) and _box_inside_shape(box, shape_zyx):
            kept.append(box)
    if not kept:
        return np.zeros((0, boxes.shape[1]), dtype=np.float32)
    return np.asarray(kept, dtype=np.float32)


def _swap_corner_boxes(boxes, axisorder):
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes
    swapped = np.copy(boxes)
    pair_starts = boxes[:, [0, 2, 4]]
    pair_ends = boxes[:, [1, 3, 5]]
    swapped[:, [0, 2, 4]] = pair_starts[:, axisorder]
    swapped[:, [1, 3, 5]] = pair_ends[:, axisorder]
    return swapped


def augment(sample, target, bboxes, do_flip=True, do_rotate=True, do_swap=True,
            pad_value=170, rotate_angle_range=10.0):
    """Geometric augmentations with box sync.

    Rotate is restricted to a mild in-plane angle so the main lesion is less
    likely to leave the crop. Boxes that fall outside after rotation are dropped.
    """
    target = np.asarray(target, dtype=np.float32)
    bboxes = np.asarray(bboxes, dtype=np.float32)
    if bboxes.size == 0:
        bboxes = bboxes.reshape(0, 6)
    shape_zyx = sample.shape[1:4]
    max_angle = abs(float(rotate_angle_range))

    if do_rotate and max_angle > 0:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.uniform(-max_angle, max_angle)
            center_yx = (np.array(sample.shape[2:4]).astype(np.float32) - 1.0) / 2.0
            rotmat = np.array([
                [np.cos(angle1 / 180 * np.pi), -np.sin(angle1 / 180 * np.pi)],
                [np.sin(angle1 / 180 * np.pi), np.cos(angle1 / 180 * np.pi)],
            ], dtype=np.float32)

            if _has_valid_corner_box(newtarget):
                newtarget = _transform_corner_box_yx(newtarget, rotmat, center_yx)
                # 主病灶旋出 crop 则重试，避免框与 CT 错位训练
                if not _box_inside_shape(newtarget, shape_zyx):
                    counter += 1
                    if counter >= 3:
                        break
                    continue

            newbboxes = np.copy(bboxes)
            for i in range(len(newbboxes)):
                if _has_valid_corner_box(newbboxes[i]):
                    newbboxes[i] = _transform_corner_box_yx(newbboxes[i], rotmat, center_yx)
            newbboxes = _filter_boxes_inside_shape(newbboxes, shape_zyx)

            target = newtarget
            sample = rotate(
                sample,
                angle1,
                axes=(2, 3),
                reshape=False,
                order=1,
                mode='constant',
                cval=float(pad_value),
                prefilter=False,
            )
            bboxes = newbboxes
            validrot = True

    if do_swap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample, np.concatenate([[0], axisorder + 1]))
            if _has_valid_corner_box(target):
                target = _swap_corner_boxes(target.reshape(1, -1), axisorder).reshape(-1)
            bboxes = _swap_corner_boxes(bboxes, axisorder)

    if do_flip:
        # z 轴不翻转；仅随机翻转 y/x，并同步角点框
        flipid = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
        sample = np.ascontiguousarray(sample[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        for i in range(3):
            if flipid[i] == -1:
                tem = [0, 2, 4]
                ax = tem[i]
                dim = np.array(sample.shape[i + 1])
                if _has_valid_corner_box(target):
                    target_min = np.copy(target[ax])
                    target_max = np.copy(target[ax + 1])
                    target[ax] = dim - 1 - target_max
                    target[ax + 1] = dim - 1 - target_min
                if len(bboxes):
                    box_min = np.copy(bboxes[:, ax])
                    box_max = np.copy(bboxes[:, ax + 1])
                    bboxes[:, ax] = dim - 1 - box_max
                    bboxes[:, ax + 1] = dim - 1 - box_min
    return sample, target, bboxes


def augment_intensity(sample, cfg, do_intensity=True, do_noise=True, do_blur=False):
    sample = sample.astype(np.float32, copy=False)
    if do_intensity:
        contrast_range = cfg.get('contrast_range', [0.9, 1.1])
        brightness_range = cfg.get('brightness_range', [-0.05, 0.05])
        contrast = np.random.uniform(float(contrast_range[0]), float(contrast_range[1]))
        brightness = np.random.uniform(float(brightness_range[0]), float(brightness_range[1])) * 255.0
        sample = (sample - 128.0) * contrast + 128.0 + brightness

    if do_noise:
        noise_std = float(cfg.get('noise_std', 0.02)) * 255.0
        if noise_std > 0:
            sample = sample + np.random.normal(0.0, noise_std, size=sample.shape).astype(np.float32)

    if do_blur and np.random.rand() < 0.25:
        sigma_range = cfg.get('blur_sigma', [0.5, 1.0])
        sigma = np.random.uniform(float(sigma_range[0]), float(sigma_range[1]))
        sample = gaussian_filter(sample, sigma=(0, sigma, sigma, sigma))

    return np.clip(sample, 0.0, 255.0).astype(np.float32, copy=False)

class Crop(object):
    def __init__(self, config):
        self.crop_size = config['crop_size']
        self.bound_size = config['bound_size']
        self.stride = config['stride']
        self.pad_value = config['pad_value']

    def crop_at_center(self, imgs, center_zyx, bboxes):
        crop_size = np.asarray(self.crop_size, dtype=np.int32)
        center_zyx = np.asarray(center_zyx, dtype=np.float32)
        bboxes = np.copy(bboxes)
        start = np.round(center_zyx - crop_size / 2.0).astype(np.int32)

        normstart = start.astype('float32') / np.array(imgs.shape[1:]) - 0.5
        normsize = crop_size.astype('float32') / np.array(imgs.shape[1:])
        xx, yy, zz = np.meshgrid(
            np.linspace(normstart[0], normstart[0] + normsize[0], int(self.crop_size[0] / self.stride)),
            np.linspace(normstart[1], normstart[1] + normsize[1], int(self.crop_size[1] / self.stride)),
            np.linspace(normstart[2], normstart[2] + normsize[2], int(self.crop_size[2] / self.stride)),
            indexing='ij',
        )
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')

        pad = [[0, 0]]
        for i in range(3):
            leftpad = max(0, -int(start[i]))
            rightpad = max(0, int(start[i]) + int(crop_size[i]) - imgs.shape[i + 1])
            pad.append([leftpad, rightpad])

        crop = imgs[:,
            max(int(start[0]), 0):min(int(start[0] + crop_size[0]), imgs.shape[1]),
            max(int(start[1]), 0):min(int(start[1] + crop_size[1]), imgs.shape[2]),
            max(int(start[2]), 0):min(int(start[2] + crop_size[2]), imgs.shape[3])]
        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)

        for i in range(len(bboxes)):
            for j in range(3):
                tem = [0, 2, 4]
                k = tem[j]
                bboxes[i][k] = bboxes[i][k] - start[j]
                bboxes[i][k + 1] = bboxes[i][k + 1] - start[j]

        target = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
        return crop, target, bboxes, coord

    def _make_coord(self, start, crop_size, image_shape):
        normstart = np.asarray(start, dtype=np.float32) / np.asarray(image_shape, dtype=np.float32) - 0.5
        normsize = np.asarray(crop_size, dtype=np.float32) / np.asarray(image_shape, dtype=np.float32)
        xx, yy, zz = np.meshgrid(
            np.linspace(normstart[0], normstart[0] + normsize[0], int(self.crop_size[0] / self.stride)),
            np.linspace(normstart[1], normstart[1] + normsize[1], int(self.crop_size[1] / self.stride)),
            np.linspace(normstart[2], normstart[2] + normsize[2], int(self.crop_size[2] / self.stride)),
            indexing='ij',
        )
        return np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')

    def _fit_crop_start(self, target, crop_size, image_shape):
        start = []
        for axis, box_axis in enumerate([0, 2, 4]):
            crop_len = int(crop_size[axis])
            image_len = int(image_shape[axis])
            target_min = float(target[box_axis])
            target_max = float(target[box_axis + 1])
            center = (target_min + target_max) / 2.0
            axis_start = int(math.floor(center - (crop_len - 1) / 2.0))
            if axis_start > target_min:
                axis_start = int(math.floor(target_min))
            if axis_start + crop_len - 1 < target_max:
                axis_start = int(math.ceil(target_max - crop_len + 1))
            if crop_len <= image_len:
                axis_start = min(max(axis_start, 0), image_len - crop_len)
            start.append(axis_start)
        return np.asarray(start, dtype=np.int32)

    def _crop_with_padding(self, imgs, start, crop_size):
        pad = [[0, 0]]
        slices = []
        for axis in range(3):
            axis_start = int(start[axis])
            crop_len = int(crop_size[axis])
            image_len = int(imgs.shape[axis + 1])
            leftpad = max(0, -axis_start)
            rightpad = max(0, axis_start + crop_len - image_len)
            pad.append([leftpad, rightpad])
            slices.append(slice(max(axis_start, 0), min(axis_start + crop_len, image_len)))
        crop = imgs[:, slices[0], slices[1], slices[2]]
        return np.pad(crop, pad, 'constant', constant_values=self.pad_value)

    def _resize_to_crop_size(self, crop, source_crop_size):
        output_size = np.asarray(self.crop_size, dtype=np.int32)
        source_crop_size = np.asarray(source_crop_size, dtype=np.float32)
        resize_scale = output_size.astype(np.float32) / source_crop_size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            crop = zoom(
                crop,
                [1, resize_scale[0], resize_scale[1], resize_scale[2]],
                order=1,
                mode='grid-constant',
                cval=float(self.pad_value),
                prefilter=False,
                grid_mode=True,
            )
        pad = [[0, 0]]
        slices = [slice(None)]
        for axis in range(3):
            out_len = int(output_size[axis])
            cur_len = int(crop.shape[axis + 1])
            if cur_len >= out_len:
                slices.append(slice(0, out_len))
                pad.append([0, 0])
            else:
                slices.append(slice(0, cur_len))
                pad.append([0, out_len - cur_len])
        crop = crop[tuple(slices)]
        if any(p[1] > 0 for p in pad[1:]):
            crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)
        return crop, resize_scale

    def _large_lesion_resize_crop(self, imgs, target, bboxes):
        target = np.copy(np.asarray(target, dtype=np.float32))
        bboxes = np.copy(np.asarray(bboxes, dtype=np.float32))
        crop_size = np.asarray(self.crop_size, dtype=np.float32)
        target_size = target[[1, 3, 5]] - target[[0, 2, 4]] + 1.0
        scale = min(1.0, float(np.min(crop_size / target_size)))
        source_crop_size = np.ceil(crop_size / scale).astype(np.int32)
        source_crop_size = np.maximum(source_crop_size, 1)

        start = self._fit_crop_start(target, source_crop_size, imgs.shape[1:])
        coord = self._make_coord(start, source_crop_size, imgs.shape[1:])
        crop = self._crop_with_padding(imgs, start, source_crop_size)
        source_shape = np.asarray(crop.shape[1:], dtype=np.float32)
        crop, resize_scale = self._resize_to_crop_size(crop, source_shape)

        for axis, box_axis in enumerate([0, 2, 4]):
            target_min = target[box_axis] - start[axis]
            target_max_exclusive = target[box_axis + 1] - start[axis] + 1.0
            target[box_axis] = target_min * resize_scale[axis]
            target[box_axis + 1] = target_max_exclusive * resize_scale[axis] - 1.0
            for box_idx in range(len(bboxes)):
                box_min = bboxes[box_idx][box_axis] - start[axis]
                box_max_exclusive = bboxes[box_idx][box_axis + 1] - start[axis] + 1.0
                bboxes[box_idx][box_axis] = box_min * resize_scale[axis]
                bboxes[box_idx][box_axis + 1] = box_max_exclusive * resize_scale[axis] - 1.0

        return crop.astype(imgs.dtype, copy=False), target, bboxes, coord

    def __call__(self, imgs, target, bboxes, isScale=False, isRand=False, allow_large_lesion_resize=False):
        target = np.asarray(target, dtype=np.float32)
        if isRand:
            isScale = False
        crop_size_cfg = np.asarray(self.crop_size, dtype=np.float32)
        target_size = target[[1, 3, 5]] - target[[0, 2, 4]] + 1.0 if len(target) >= 6 else np.zeros((3,), dtype=np.float32)
        if (
            allow_large_lesion_resize
            and np.isfinite(target_size).all()
            and np.any(target_size > crop_size_cfg)
        ):
            return self._large_lesion_resize_crop(imgs, target, bboxes)

        if isScale:
            radiusLim = [8.,120.]
            scaleLim = [0.75,1.25]
            diameter = max(target[1] - target[0], target[3] - target[2], target[5] - target[4])
            if not np.isfinite(diameter) or diameter <= 0:
                scale = 1.0
                crop_size = np.asarray(self.crop_size, dtype=np.int32)
            else:
                scaleRange = [np.min([np.max([(radiusLim[0]/diameter),scaleLim[0]]),1])
                             ,np.max([np.min([(radiusLim[1]/diameter),scaleLim[1]]),1])]
                scaleRange = sorted(scaleRange)
                scale = np.random.rand()*(scaleRange[1]-scaleRange[0])+scaleRange[0]
                crop_size = (np.array(self.crop_size).astype('float')/scale).astype('int')
                crop_size = np.maximum(crop_size, 1)
        else:
            scale = 1.0
            crop_size = np.asarray(self.crop_size, dtype=np.int32)
        bound_size = self.bound_size
        target = np.copy(target)
        bboxes = np.copy(bboxes)

        start = []
        for i in range(3):
            image_size = int(imgs.shape[i + 1])
            crop_len = int(crop_size[i])
            if isRand:
                high = image_size - crop_len
                if high >= 0:
                    start.append(int(np.random.randint(0, high + 1)))
                else:
                    start.append(int(round((image_size - crop_len) / 2.0)))
                continue

            tem = [0,2,4]
            j = tem[i]
            target_min = float(target[j])
            target_max = float(target[j + 1])
            if not np.isfinite(target_min) or not np.isfinite(target_max):
                start.append(int(round((image_size - crop_len) / 2.0)))
                continue

            start_low = int(math.ceil(target_max + bound_size - crop_len))
            start_high = int(math.floor(target_min - bound_size))
            if start_low <= start_high:
                start.append(int(np.random.randint(start_low, start_high + 1)))
            else:
                center = (target_min + target_max) / 2.0
                jitter = np.random.randint(-bound_size // 2, bound_size // 2 + 1)
                start.append(int(round(center - crop_len / 2.0 + jitter)))

        if isRand:
            target = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32)


        normstart = np.array(start).astype('float32')/np.array(imgs.shape[1:])-0.5
        normsize = np.array(crop_size).astype('float32')/np.array(imgs.shape[1:])
        xx,yy,zz = np.meshgrid(np.linspace(normstart[0],normstart[0]+normsize[0],int(self.crop_size[0]/self.stride)),
                           np.linspace(normstart[1],normstart[1]+normsize[1],int(self.crop_size[1]/self.stride)),
                           np.linspace(normstart[2],normstart[2]+normsize[2],int(self.crop_size[2]/self.stride)),indexing ='ij')
        coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')

        pad = []
        pad.append([0,0])
        for i in range(3):
            leftpad = max(0,-start[i])
            rightpad = max(0,start[i]+crop_size[i]-imgs.shape[i+1])
            pad.append([leftpad,rightpad])
        crop = imgs[:,
            max(start[0],0):min(start[0] + crop_size[0],imgs.shape[1]),
            max(start[1],0):min(start[1] + crop_size[1],imgs.shape[2]),
            max(start[2],0):min(start[2] + crop_size[2],imgs.shape[3])]
        crop = np.pad(crop,pad,'constant',constant_values =self.pad_value)
        for i in range(3):
            tem = [0, 2, 4]
            j = tem[i]
            target[j] = target[j] - start[i]
            target[j + 1] = target[j + 1] - start[i]
            # target[i] = target[i] - start[i]
        for i in range(len(bboxes)):
            for j in range(3):
                tem = [0, 2, 4]
                k = tem[j]
                bboxes[i][k] = bboxes[i][k] - start[j]
                bboxes[i][k + 1] = bboxes[i][k + 1] - start[j]
                # bboxes[i][j] = bboxes[i][j] - start[j]

        if isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop,[1,scale,scale,scale],order=1)
            newpad = self.crop_size[0]-crop.shape[1:][0]
            if newpad<0:
                crop = crop[:,:-newpad,:-newpad,:-newpad]
            elif newpad>0:
                pad2 = [[0,0],[0,newpad],[0,newpad],[0,newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)
            for i in range(6):
                target[i] = target[i]*scale
            for i in range(len(bboxes)):
                for j in range(6):
                    bboxes[i][j] = bboxes[i][j]*scale
        return crop, target, bboxes, coord
