#!/bin/bash

set -e

outdir="test_output/res/95_pretrained_hardsamples_rcnn40_PN11_v5_ensemble"

datasets=("histopathology"
"LNDB"
"LUNA25"
"LUNA16"
"NLSTSeg"
"PN9")

for dataset in "${datasets[@]}"; do

  echo "========================================"
  echo "Processing dataset: $dataset"
  echo "========================================"

  dataset_dir="$outdir/$dataset"

  # 1. 准备 FROC 输入文件
  python tools/prepare_froc_evaluation_inputs.py "$dataset_dir"

  # 2. FROC evaluation
  python froc_evaluation/noduleCADEvaluationLUNA16.py \
      "$dataset_dir/FROC/annotations_froc_eval.csv" \
      "$dataset_dir/FROC/annotations_excluded.csv" \
      "$dataset_dir/FROC/seriesuids.csv" \
      "$dataset_dir/FROC/results_froc_eval.csv" \
      "$dataset_dir/FROC/res_froc_eval_diamter05"

done

# 3. 一次性导出所有数据集的 recall table
python tools/export_forc_recall_table.py "$outdir" --result-dir res_froc_eval_diamter05