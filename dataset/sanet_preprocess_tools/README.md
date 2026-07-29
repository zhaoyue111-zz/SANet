# SANet / PN9-style preprocessing tools

This package converts external lung CT datasets to the same **processing stage** expected by SANet's downloaded PN9 `.npy` version:

```text
SANet_ready_DATASET/
  full/
    000001_zoom.npy        # float32, shape [1,D,H,W], spacing 1mm, intensity [0,255]
    000002_zoom.npy
  split/
    train.txt
    val.txt
    test.txt
    all_anno.csv
    train_anno.csv         # train + val boxes, matching SANet original reader behavior
    val_anno.csv
    test_anno.csv
  pid_map.csv              # 原始病例 ID 到 SANet 数字 ID 的映射
  meta.csv                 # 每个病例处理前后 shape、spacing、min/max、bbox 数量
  dataset_summary.json
  sanet_config_example.json
```

The conversion follows SANet's intensity convention:

```text
HU -> clip [-1200, 600] -> linear map to [0,255]
```

PN9 itself is already preprocessed by the SANet authors, so the PN9 script only normalizes the folder layout and copies/symlinks existing files.

## Install

```bash
conda create -n sanetprep python=3.9 -y
conda activate sanetprep
pip install -r requirements.txt
```

Run commands from the folder that contains `sanet_prep/`, e.g.:

```bash
cd sanet_preprocess_tools
```

---

## 1) PN9 downloaded npy

Use this after uncompressing the split volumes, e.g. `cat train.* > train.tar.gz && tar xvzf train.tar.gz`.

```bash
python -m sanet_prep.prepare_pn9 \
  --pn9-root /data/PN9/npy \
  --out-dir /data/SANet_ready/PN9 \
  --mode symlink
```

If your filesystem does not allow symlinks, use `--mode copy`.

---

## 2) Histopathology-based Dataset

Recommended input: **Version 1 `MHD_3D.zip` + `all_anno_3D.csv`**. BMP is CT slice image data, not mask; MHD is the better 3D input.

If you have an original split directory with `train.txt` and `test.txt`:

```bash
python -m sanet_prep.prepare_histopathology \
  --mhd-root /data/Histopathology/MHD_3D \
  --all-anno-3d /data/Histopathology/all_anno_3D.csv \
  --split-dir /data/Histopathology/ImageSets \
  --out-dir /data/SANet_ready/Histopathology
```

If no 3D split file is provided, the paper clearly mentions 4:1 train/test for the 2D ImageSets/classification folders. To use a deterministic 4:1 fallback at patient level:

```bash
python -m sanet_prep.prepare_histopathology \
  --mhd-root /data/Histopathology/MHD_3D \
  --all-anno-3d /data/Histopathology/all_anno_3D.csv \
  --out-dir /data/SANet_ready/Histopathology \
  --allow-fallback-split
```

SANet usually needs validation monitoring. If the original split has only train/test and you want to carve validation out of train:

```bash
  --val-frac-from-train 0.1
```

This is no longer a strictly original split; it is a deterministic additional validation split.

---

## 3) LNDb / LNDb v4

Recommended input: CT images plus segmentation masks if available. The script supports MHD and NIfTI.

Using masks:

```bash
python -m sanet_prep.prepare_lndb \
  --image-root /data/LNDb/images \
  --mask-root /data/LNDb/masks \
  --split-dir /data/LNDb/splits \
  --out-dir /data/SANet_ready/LNDb
```

Using annotation CSV instead of masks:

```bash
python -m sanet_prep.prepare_lndb \
  --image-root /data/LNDb/images \
  --anno-csv /data/LNDb/allNods.csv \
  --center-space physical \
  --out-dir /data/SANet_ready/LNDb \
  --split-dir /data/LNDb/splits
```

If no official/author split files are available and you want to reproduce the common SaTTCA-style 7:1:2 patient-level split:

```bash
  --allow-fallback-split
```

Again, this is a deterministic fallback, not a hidden random split.

---

## 4) NLSTseg

Recommended input: extracted folders containing `*_CT.nii.gz` and corresponding `*_tumor.nii.gz` masks.

```bash
python -m sanet_prep.prepare_nlstseg \
  --root /data/NLSTseg \
  --split-table /data/NLSTseg/1_Table/split.csv \
  --out-dir /data/SANet_ready/NLSTseg
```

If the downloaded table zip does not contain a split column but you have author split txt files:

```bash
python -m sanet_prep.prepare_nlstseg \
  --root /data/NLSTseg \
  --split-dir /data/NLSTseg/splits \
  --out-dir /data/SANet_ready/NLSTseg
```

If neither exists, you must either provide split files or explicitly enable a deterministic fallback:

```bash
  --allow-fallback-split --fallback-ratio 8 0 2
```

The script converts segmentation masks into connected-component 3D bounding boxes. That means the output task is lesion/tumor detection unless you separately filter lesions by type.

---

## Validate output

```bash
python -m sanet_prep.check_sanet_ready \
  --sanet-dir /data/SANet_ready/Histopathology \
  --max-npy-check 50
```

For final QA, increase `--max-npy-check` or remove the limit in the script.

---

## Notes for SANet config.py

For a converted dataset, patch `config.py` with paths from `sanet_config_example.json`:

```python
'preprocessed_data_dir': '/data/SANet_ready/Histopathology/full',
'train_set_list': '/data/SANet_ready/Histopathology/split/train.txt',
'val_set_list': '/data/SANet_ready/Histopathology/split/val.txt',
'test_set_name': '/data/SANet_ready/Histopathology/split/test.txt',
'annotation_dir': '/data/SANet_ready/Histopathology/split',
'train_anno': '/data/SANet_ready/Histopathology/split/train_anno.csv',
'test_anno': '/data/SANet_ready/Histopathology/split/test_anno.csv',
```

The generated ids are numeric, so SANet's original `int(fn)` logic should still work.
