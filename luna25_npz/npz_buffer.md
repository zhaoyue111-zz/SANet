# 肺实质预处理 NPZ（`*_buffer.npz`）说明

**用途**  
解码 `nodule_pipeline.py` 写出的肺实质预处理缓存，供下游算法或数据交接使用。

**生成链路**  
`DICOM` → `medai_lung_parenchyma_segment.cpp_preprocess_dicom_tobuffer` → `ExchangeImage` → `exchangeimage2npzdict` → `np.savez_compressed`（`nodule_pipeline.py` 约第 768 行）。

**读法**  
```python
d = np.load("xxx_buffer.npz", allow_pickle=True)
# 或 dict(np.load(...)) 以便多次随机访问
```

---

## 1. 文件位置

| 项 | 约定 |
|----|------|
| 目录 | `{output_dir}/npz/` |
| 文件名 | `{SeriesInstanceUID}_buffer.npz` |
| 格式 | `np.savez_compressed` |

历史批次可能写成 `{UID}_volume_buffer.npz`；`seriesUID` 字段也可能带 `_volume` 后缀。

---

## 2. 坐标约定（必读）

| 项 | 约定 |
|----|------|
| 数组轴序 | **ZYX**，`shape = (Z, Y, X)` |
| `origin` / `spacing` | 长度 3，**ZYX**，单位 mm |
| `direction` | 长度 9，**ZYX 打包**（相对 SimpleITK XYZ 方向做了 `[::-1]`） |
| 裁剪框 `mask_size*` | `(3, 2)`，行顺序 Z→Y→X，每行 `[start, end)` |

转 SimpleITK / NIfTI 时几何字段一律 `[::-1]`：

```python
img = sitk.GetImageFromArray(arr_zyx)
img.SetSpacing(d["spacing"][::-1].tolist())
img.SetOrigin(d["origin"][::-1].tolist())
img.SetDirection(d["direction"][::-1].tolist())
```

**重要（legacy origin）**  
预处理裁剪后，仍把**全图 origin** 写进 npz，而不是裁剪角点的物理原点。  
因此：

- 把 `image_original` **pad 回全图**后，可直接用 npz 里的 `origin/spacing/direction`；
- 若只拿裁剪体当 Sitk 图、却不改 origin，世界坐标会错。正确做法见第 5 节 `RegionOfInterest`。

---

## 3. NPZ 包含哪些键

当前批次完整 key 列表（14 个）：

```text
seriesUID
origin, spacing, direction
image_shape, image_shape_original
mask_size, mask_size_original
image_original, mask_original
image, mask, mask_lr
raw_lrmask
```

### 3.1 元信息

| Key | 类型 / shape | 含义 |
|-----|--------------|------|
| `seriesUID` | 字符串标量 | 序列 UID |
| `origin` | `float64 (3,)` | 全图原点，ZYX，mm（见上：裁剪后仍存全图 origin） |
| `spacing` | `float64 (3,)` | **原始 DICOM 间距**，ZYX，mm（对应 `image_original` / 全图） |
| `direction` | `float64 (9,)` | 方向余弦，ZYX 打包 |

### 3.2 形状与裁剪框

| Key | shape | 含义 |
|-----|-------|------|
| `image_shape_original` | `(2, 3)` | 行 0/1 均为**全图** ZYX shape（通常相同），如 `[194, 512, 512]` |
| `image_shape` | `(2, 3)` | 行 0 = 裁剪后原始分辨率 shape（=`image_original.shape`）；行 1 = 重采样工作体 shape（=`image.shape`） |
| `mask_size_original` | `(3, 2)` | 在**全图 ZYX** 上的肺框 `[z0,z1); [y0,y1); [x0,x1)` |
| `mask_size` | `(3, 2)` | 在**重采样体**上的有效框；常为满幅 `[[0,Z],[0,Y],[0,X]]` |

### 3.3 体数据

| Key | 典型 dtype / shape | 含义 |
|-----|-------------------|------|
| `image_original` | `int16 (Zc,Yc,Xc)` | 按肺框裁剪后的 **HU** 体 |
| `mask_original` | `uint8` 同 shape | 裁剪后肺实质：`0` 背景，`1` 肺 |
| `raw_lrmask` | `float32` 全图 shape | 未裁剪左右肺：`0` 背景，`1` 右肺，`2` 左肺 |
| `image` | `float32 (Z',Y',X')` | 裁剪区重采样后的归一化图（约 `[0,1]`，**不是 HU**） |
| `mask` | `uint8` 同 `image` | 重采样二值肺：`0/1` |
| `mask_lr` | `uint8` 同 `image` | 重采样左右肺：`0/1/2` |

标签约定：`1 = 右肺`，`2 = 左肺`。

---

## 4. 怎么裁剪的

流水线大致两步：

```text
全图 DICOM (image_shape_original)
    │  肺分割 → raw_lrmask
    │  按左右肺外接框（可带 margin）得到 mask_size_original
    ▼
裁剪体 image_original / mask_original
    │  重采样到约 1 mm 各向同性（物理尺寸 ≈ 原裁剪区 mm 尺寸）
    │  强度归一化 → image；mask → mask / mask_lr
    ▼
工作体 image / mask / mask_lr   （mask_size 多为满幅）
```

### 4.1 索引关系（实测成立）

```python
z0, z1 = d["mask_size_original"][0]
y0, y1 = d["mask_size_original"][1]
x0, x1 = d["mask_size_original"][2]

assert (z1 - z0, y1 - y0, x1 - x0) == d["image_original"].shape
assert d["raw_lrmask"][z0:z1, y0:y1, x0:x1].shape == d["mask_original"].shape
# 肺区域一致：
# (raw_lrmask[z0:z1,y0:y1,x0:x1] > 0) == (mask_original > 0)
```

样例：

```text
image_shape_original[0] = (194, 512, 512)     # 全图
mask_size_original      = [[1,194],[56,429],[9,512]]
image_original.shape    = (193, 373, 503)     # = end - start
image.shape             = (347, 240, 324)     # ≈ 裁剪区物理尺寸(mm)（~1mm 网格）
spacing (ZYX)           = (1.80, 0.645, 0.645)  # 原始 DICOM
```

物理尺寸核对：

\[
\texttt{image\_original.shape} \odot \texttt{spacing}
\;\approx\;
\texttt{image.shape} \odot (1,1,1)
\]

---

## 5. 怎么还原回原始空间

分三种常见需求。

### 5.1 索引空间：裁剪体 → 全图数组（最常用）

```python
import numpy as np

def restore_to_full_zyx(crop_zyx, d, pad_value=0):
    """把裁剪数组 pad 回全图 ZYX 网格。"""
    full_shape = tuple(np.asarray(d["image_shape_original"][0], dtype=int))
    box = np.asarray(d["mask_size_original"], dtype=int)  # (3,2)
    pad_start = box[:, 0]
    pad_end = np.asarray(full_shape) - box[:, 1]
    # pad 宽度：((z0, zend), (y0, yend), (x0, xend))
    return np.pad(crop_zyx, np.stack([pad_start, pad_end], axis=1),
                  constant_values=pad_value)

# HU 还原（肺外填空气）
full_hu = restore_to_full_zyx(d["image_original"], d, pad_value=-1024)
# mask 还原
full_mask = restore_to_full_zyx(d["mask_original"], d, pad_value=0)
# 左右肺也可直接用已有全图
full_lr = d["raw_lrmask"]  # 已是全图，无需 pad
```

等价写法（赋值）：

```python
full = np.full(tuple(d["image_shape_original"][0]), -1024, dtype=np.int16)
z0, z1 = d["mask_size_original"][0]
y0, y1 = d["mask_size_original"][1]
x0, x1 = d["mask_size_original"][2]
full[z0:z1, y0:y1, x0:x1] = d["image_original"]
```

Pad 回全图后，用 npz 几何即可写 NIfTI（与原始 DICOM 网格对齐）：

```python
import SimpleITK as sitk

img = sitk.GetImageFromArray(full_hu)
img.SetSpacing(d["spacing"][::-1].tolist())
img.SetOrigin(d["origin"][::-1].tolist())
img.SetDirection(d["direction"][::-1].tolist())
sitk.WriteImage(img, "restored_full.nii.gz")
```

### 5.2 只使用裁剪体，但要正确世界坐标（Sitk ROI）

检测等代码的做法：先建全图空参考，再 `RegionOfInterest`，把正确的 crop origin 带出来：

```python
import numpy as np
import SimpleITK as sitk

full_shape = tuple(np.asarray(d["image_shape_original"][0], dtype=int))
sref = sitk.GetImageFromArray(np.zeros(full_shape, np.int16))
sref.SetSpacing(d["spacing"][::-1].tolist())
sref.SetOrigin(d["origin"][::-1].tolist())
sref.SetDirection(d["direction"][::-1].tolist())

# Sitk Size/Index 是 XYZ
crop_start_xyz = d["mask_size_original"][:, 0][::-1]
crop_end_xyz = d["mask_size_original"][:, 1][::-1]
sref_crop = sitk.RegionOfInterest(
    sref,
    (crop_end_xyz - crop_start_xyz).tolist(),
    crop_start_xyz.tolist(),
)

crop_img = sitk.GetImageFromArray(d["image_original"])
crop_img.CopyInformation(sref_crop)  # 此时 origin 才是裁剪角的物理坐标
```

体素（裁剪索引）↔ 全图索引：

```text
full_zyx = crop_zyx + mask_size_original[:, 0]
```

体素 ↔ 世界坐标：对 `sref_crop` 或 pad 后的全图 Sitk 调用  
`TransformIndexToPhysicalPoint` / `TransformPhysicalPointToIndex`（注意 Sitk 索引为 XYZ）。

### 5.3 重采样工作体 `image` → 原始分辨率裁剪区 / 全图

`image` / `mask` / `mask_lr` 在约 **1 mm** 网格上，物理范围对应 `image_original` 那块肺框。

还原思路：

1. 用第 5.2 节得到 `sref_crop`（原始 spacing 的裁剪参考）；
2. 把 `image`（或 `mask`）建成 Sitk 图：spacing=`(1,1,1)`，origin/direction 与 `sref_crop` 对齐（或先用全图 origin + 再 ROI+resample，与检测代码一致）；
3. `sitk.Resample(..., referenceImage=sref_crop, interpolator=...)` 拉回原始裁剪分辨率；
4. 若要全图，再按 5.1 pad。

示意：

```python
# 工作体 → 原始裁剪分辨率（最近邻适合 mask）
work = sitk.GetImageFromArray(d["mask"].astype(np.float32))
work.SetSpacing((1.0, 1.0, 1.0))
work.CopyInformation(sref_crop)  # 若物理对齐；若不对齐需先按检测逻辑 ROI+resample

back = sitk.Resample(
    work, sref_crop,
    sitk.Transform(),
    sitk.sitkNearestNeighbor,
    0.0,
    sitk.sitkFloat32,
)
mask_on_original_crop = sitk.GetArrayFromImage(back)  # ZYX，shape ≈ image_original
full_mask = restore_to_full_zyx(mask_on_original_crop.astype(np.uint8), d, 0)
```

> 若 `CopyInformation` 后物理范围对不齐，以检测模块为准：对全图参考做 ROI 得到 `sref_crop`，再把 1 mm 体 resample 到该参考（见 `nodule_detect` 中 `_build_dist2chest_new` / `RegionOfInterest` + `resample_simg`）。

---

## 6. 三层数据对照图

```text
┌─────────────────────────────────────────────┐
│ raw_lrmask / 全图网格                         │
│ shape = image_shape_original[0]             │
│ 几何: origin, spacing, direction            │
│                                             │
│   ┌───────────────────────────────┐         │
│   │ image_original / mask_original│ ← mask_size_original 裁剪
│   │ shape = image_shape[0]        │         │
│   │ 仍挂全图 origin（legacy）      │         │
│   └───────────────────────────────┘         │
│                 │ 重采样 ~1mm + 归一化         │
│                 ▼                             │
│         image / mask / mask_lr                │
│         shape = image_shape[1]                │
│         spacing≈1mm；mask_size 常为满幅       │
└─────────────────────────────────────────────┘
```

---

## 7. 快速自检

```python
d = dict(np.load(path, allow_pickle=True))
assert set(d.keys()) >= {
    "seriesUID", "origin", "spacing", "direction",
    "image_shape", "image_shape_original",
    "mask_size", "mask_size_original",
    "image_original", "mask_original", "image", "mask", "mask_lr",
}
box = d["mask_size_original"]
assert tuple(box[:, 1] - box[:, 0]) == tuple(d["image_original"].shape)
assert tuple(d["image"].shape) == tuple(d["image_shape"][1])
assert tuple(d["raw_lrmask"].shape) == tuple(d["image_shape_original"][0])
```

---

## 8. 交接检查清单

1. 列出全部 key，确认有 `mask_size_original` + `image_shape_original`  
2. 分清三套体：`image_original`(HU 裁剪) / `image`(归一化重采样) / `raw_lrmask`(全图)  
3. 还原全图用 **pad / 切片赋值**，不要改 spacing  
4. 只用裁剪体做物理坐标时，必须走 **ROI** 或手动修正 origin  
5. `image` 不是 HU；左右肺标签 `1=右，2=左`  
6. 轴序全程 ZYX；写 Sitk 时几何 `[::-1]`

---

## 9. 参考

- 写出：`nodule_pipeline.py` → `process_simage2npz`  
- 还原实现参考：`nodule_seg_thick/project/src/util.py` → `restore_itk_from_npz`  
- ROI 用法：`nodule_detect/.../interface.py`（`mask_size_original` + `RegionOfInterest`）  
- 结果 CSV：[`nodule_detect_csv.md`](./nodule_detect_csv.md)  
