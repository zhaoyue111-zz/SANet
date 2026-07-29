# FROC Evaluation

基于 [LUNA16](https://luna16.grand-challenge.org/) 评估规则的肺结节 CAD 检测 FROC 计算工具。将模型推理结果（infer）与真值标注（GT）进行匹配，输出 FROC 曲线、综合得分及 TP/FP/FN 明细。

## 快速开始

### 1. 准备输入文件

将以下 4 个 CSV 放到 `annotations/` 目录（或任意路径，调用时传入即可）：

| 文件 | 说明 |
|------|------|
| GT 标注 | 参与评估的结节真值 |
| 排除标注 | 不参与计分、但会屏蔽 FP 的结节（可为空表） |
| 病例列表 | 参与评估的 CT `seriesuid` 列表 |
| 推理结果 | 模型输出的候选检测框 |

### 2. 修改并运行 `froc.sh`

编辑 `froc.sh` 中的 `input_csv`，指向你的推理结果文件名：

```bash
input_csv="your_model_predict.csv"
```

在 Git Bash / WSL / Linux 下执行：

```bash
bash froc.sh
```

脚本会创建带时间戳的输出目录（如 `froc-2026-07-10-17-30-00`），并将评估结果写入其中。

### 3. 直接调用 Python（推荐自定义路径时使用）

```bash
python noduleCADEvaluationLUNA16.py \
  <GT标注.csv> \
  <排除标注.csv> \
  <病例列表.csv> \
  <推理结果.csv> \
  <输出目录>
```

示例（使用仓库内默认数据）：

```bash
python noduleCADEvaluationLUNA16.py \
  ./annotations/lung_data_batch_99.csv \
  ./annotations/annotations_excluded.csv \
  ./annotations/batch_99_dirs.csv \
  ./annotations/your_model_predict.csv \
  ./output/my_froc_result
```

---

## 输入 CSV 格式

### GT 标注（annotations）

必须包含表头，至少有以下列：

```csv
seriesuid,coordX,coordY,coordZ,diameter_mm
00099e4f.de4e.45c8.a090.cd46cdd9abde,-49.21875,113.90625,514.4856567,8
```

| 列名 | 说明 |
|------|------|
| `seriesuid` | CT 序列唯一 ID |
| `coordX`, `coordY`, `coordZ` | 结节中心世界坐标（mm） |
| `diameter_mm` | 结节直径（mm） |

仓库示例：`annotations/lung_data_batch_99.csv`（可含额外列如 `nodule_type`，不影响评估）。

### 排除标注（annotations_excluded）

格式与 GT 相同。落在排除结节上的检测候选不会计为 FP，而是从评估中忽略。

若无排除结节，保留表头即可：

```csv
seriesuid,coordX,coordY,coordZ,diameter_mm
```

仓库示例：`annotations/annotations_excluded.csv`

### 病例列表（seriesuids）

每行一个 `seriesuid`，**无表头**：

```csv
00099e4f.de4e.45c8.a090.cd46cdd9abde
00158923.b8d4.45ae.b2a4.76187c03564c
```

仓库示例：`annotations/batch_99_dirs.csv`

### 推理结果（infer）

必须包含表头。坐标与 GT 使用同一坐标系（世界坐标，mm）。

```csv
seriesuid,coordX,coordY,coordZ,radius,probability
00099e4f.de4e.45c8.a090.cd46cdd9abde,-49.2,113.9,514.5,4.0,0.95
```

| 列名 | 说明 |
|------|------|
| `seriesuid` | 与 GT 对应的 CT 序列 ID |
| `coordX`, `coordY`, `coordZ` | 候选框中心坐标 |
| `probability` | 检测置信度（用于 FROC 阈值排序） |
| `radius` / `diameter_mm` / `diameter` | 候选尺寸（三选一，匹配逻辑不依赖此列，但建议保留） |

> 若模型输出的是直径，列名用 `diameter_mm` 或 `diameter` 均可；若输出半径，列名用 `radius`。

---

## 评估规则（LUNA16）

1. **命中判定**：候选框中心到 GT 结节中心的欧氏距离 < `diameter_mm × 2`（即落在以结节中心为圆心、直径 2 倍为半径的球体内）时，判定该结节被检出（TP）。
2. **重复检测**：同一结节被多个候选命中时，仅保留置信度最高的一个计入 FROC，其余忽略。
3. **漏检（FN）**：未被任何候选命中的 GT 结节。
4. **误检（FP）**：未命中任何 GT/排除结节的候选。
5. **候选数上限**：每个病例最多保留置信度最高的 200 个候选参与评估。
6. **FROC 横轴**：平均每例 CT 的 FP 数；**纵轴**：敏感度（Sensitivity）。
7. **Bootstrap**：默认 250 次重采样，输出 95% 置信区间。

---

## 输出说明

评估完成后，输出目录中包含：

| 文件 | 说明 |
|------|------|
| `CADAnalysis.txt` | 评估摘要：TP/FP/FN 数量、敏感度、FROC 综合得分等 |
| `froc_<模型名>.png` | FROC 曲线图（含 Bootstrap 置信带） |
| `froc_<模型名>.txt` | FROC 曲线原始数据（FP 率, 敏感度, 阈值） |
| `froc_<模型名>_bootstrapping.csv` | Bootstrap 均值及上下界 |
| `froc_gt_prob_vectors_<模型名>.csv` | 每个候选的 GT 标签与概率 |
| `nodulesWithoutCandidate_<模型名>.txt` | 未被检出的结节（FN）列表 |
| `TPs.csv` / `FPs.csv` / `FNs.csv` | 真阳性 / 假阳性 / 假阴性明细 |

**FROC 综合得分**：在 FP 率为 `[1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32]` 共 9 个点的敏感度取平均（见 `noduleCADEvaluationLUNA16.py` 中 `key_pt`）。

---

## 环境依赖

```bash
pip install numpy matplotlib scikit-learn tqdm pandas
```

脚本依赖 `tools.csvTools` 模块（LUNA16 官方评估工具中的 CSV 读写工具）。请确保项目根目录下存在 `tools/csvTools.py`，或将 LUNA16 评估包中的 `tools` 目录复制到本项目。

> 当前代码为 Python 2 语法（`print` 语句等）。建议使用 Python 2.7，或自行做 Python 3 兼容修改后运行。

---

## 其他脚本

| 脚本 | 用途 |
|------|------|
| `froc.sh` | 一键评估入口，创建输出目录并调用主评估脚本 |
| `noduleCADEvaluationLUNA16.py` | **主评估脚本**，计算 FROC 并输出结果 |
| `filter_nodules_detected.py` | 精简版 FROC 评估（不输出 TP/FP/FN CSV） |
| `process_nodules_detect.py` | 预处理：为每个候选标注是否命中 GT（`gt=0/1`） |
| `process_nodules_detect_val.py` | 同上，用于验证集路径 |
| `analyze.py` | 将自定义 JSON 文本标注转换为 CSV |
| `output_froc_details.py` | 从 CT 影像中裁剪结节切片（需 SimpleITK，可选） |
| `NoduleFinding.py` | 结节/候选数据结构定义 |

---

## 常见问题

**Q: 推理 CSV 的坐标系必须与 GT 一致吗？**  
A: 是。`coordX/Y/Z` 均须为世界坐标（mm），与 GT 标注使用同一坐标系。

**Q: 某病例没有预测结果怎么办？**  
A: 该病例所有 GT 结节均计为 FN，不影响其他病例评估。

**Q: `annotations_excluded.csv` 可以为空吗？**  
A: 可以，仅保留表头行即可。

**Q: 如何对比多个模型？**  
A: 分别修改 `input_csv` 或传入不同推理 CSV，多次运行；对比各次输出目录中的 `froc_*.png` 与 `CADAnalysis.txt` 即可。