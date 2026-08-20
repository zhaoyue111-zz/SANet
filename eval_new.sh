#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# SANet organized *_buffer.npz DDP test + FROC evaluation
#
# test layout:
#   DATA_ROOT/{patient_id}/{studyInstanceUID}/{seriesInstanceUID}/
#       {seriesInstanceUID}_buffer.npz
#
# test.txt:
#   patient_id/studyInstanceUID/seriesInstanceUID
#
# --batch-size is PER GPU, but is a loading/scheduling batch only.
# Full CTs are forwarded one-by-one inside each loader batch.
# =============================================================================

# --------------------------- paths -------------------------------------------
DATA_ROOT="${DATA_ROOT:-/path/to/data_root}"
TEST_LIST="${TEST_LIST:-/path/to/test.txt}"
ANNOTATION="${ANNOTATION:-/path/to/annotation.csv}"
WEIGHT="${WEIGHT:-/path/to/best_rcnn.ckpt}"
OUT_DIR="${OUT_DIR:-test_output/luna25_organized_ddp_ensemble}"

# --------------------------- GPUs / loader -----------------------------------
# Visible physical GPUs. NUM_GPUS must match the number of IDs below.
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"

# Per-GPU loading/scheduling batch.
# Example: NUM_GPUS=4, BATCH_SIZE=2 -> each GPU loads 2 CTs per loader step,
# but forwards them sequentially with network forward batch=1.
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# --------------------------- ensemble ----------------------------------------
# Example 8:2 => final score = 0.8 * RPN + 0.2 * RCNN.
RPN_WEIGHT="${RPN_WEIGHT:-8}"
RCNN_WEIGHT="${RCNN_WEIGHT:-2}"

# --------------------------- online preprocessing ----------------------------
TARGET_SPACING_ZYX="${TARGET_SPACING_ZYX:-1.0,0.6,0.6}"
WINDOW_MIN="${WINDOW_MIN:--1200}"
WINDOW_MAX="${WINDOW_MAX:-600}"
PAD_FACTOR="${PAD_FACTOR:-32}"

# Set to 1 only if the checkpoint was trained with ASPP enabled.
USE_ASPP="${USE_ASPP:-0}"

# --------------------------- FROC --------------------------------------------
FROC_RESULT_DIR="${FROC_RESULT_DIR:-res_froc_eval_diameter05}"

# =============================================================================
# Validation
# =============================================================================
IFS=',' read -r -a _gpu_array <<< "${GPU_IDS}"
if [[ "${#_gpu_array[@]}" -ne "${NUM_GPUS}" ]]; then
  echo "[ERROR] GPU_IDS contains ${#_gpu_array[@]} GPU(s), but NUM_GPUS=${NUM_GPUS}"
  exit 2
fi

for required in "${TEST_LIST}" "${ANNOTATION}" "${WEIGHT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[ERROR] File not found: ${required}"
    exit 2
  fi
done

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT not found: ${DATA_ROOT}"
  exit 2
fi

mkdir -p "${OUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export SANET_DISABLE_INTERNAL_DATA_PARALLEL=1

echo "============================================================"
echo "SANet DDP test"
echo "DATA_ROOT      = ${DATA_ROOT}"
echo "TEST_LIST      = ${TEST_LIST}"
echo "ANNOTATION     = ${ANNOTATION}"
echo "WEIGHT         = ${WEIGHT}"
echo "OUT_DIR        = ${OUT_DIR}"
echo "GPU_IDS        = ${GPU_IDS}"
echo "NUM_GPUS       = ${NUM_GPUS}"
echo "BATCH_SIZE/GPU = ${BATCH_SIZE}"
echo "NUM_WORKERS/GPU= ${NUM_WORKERS}"
echo "RPN:RCNN       = ${RPN_WEIGHT}:${RCNN_WEIGHT}"
echo "============================================================"

TEST_ARGS=(
  --weight "${WEIGHT}"
  --data-root "${DATA_ROOT}"
  --test-list "${TEST_LIST}"
  --annotation "${ANNOTATION}"
  --out-dir "${OUT_DIR}"
  --num-gpus "${NUM_GPUS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --rpn-weight "${RPN_WEIGHT}"
  --rcnn-weight "${RCNN_WEIGHT}"
  --target-spacing-zyx "${TARGET_SPACING_ZYX}"
  --window-min "${WINDOW_MIN}"
  --window-max "${WINDOW_MAX}"
  --pad-factor "${PAD_FACTOR}"
)

if [[ "${USE_ASPP}" == "1" ]]; then
  TEST_ARGS+=(--use-aspp)
fi

# 1) Multi-GPU DDP inference.
#    test.py writes only CSVs; no *_detections.npy is created.
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  test_new.py "${TEST_ARGS[@]}"

# 2) Prepare FROC inputs.
#    The modified helper prefers FROC/seriesuids.csv written from test.txt,
#    so zero-GT/zero-prediction scans are still included in the denominator.
python tools/prepare_froc_evaluation_inputs_new.py \
  "${OUT_DIR}" \
  --prefer seriesuids

# 3) FROC evaluation.
python froc_evaluation/noduleCADEvaluationLUNA16.py \
  "${OUT_DIR}/FROC/annotations_froc_eval.csv" \
  "${OUT_DIR}/FROC/annotations_excluded.csv" \
  "${OUT_DIR}/FROC/seriesuids.csv" \
  "${OUT_DIR}/FROC/results_froc_eval.csv" \
  "${OUT_DIR}/FROC/${FROC_RESULT_DIR}"

# 4) Export recall@FP/scan table.
python tools/export_forc_recall_table.py \
  "${OUT_DIR}" \
  --result-dir "${FROC_RESULT_DIR}"

echo "============================================================"
echo "Finished."
echo "Predictions : ${OUT_DIR}/FROC/results.csv"
echo "GT          : ${OUT_DIR}/FROC/annotations.csv"
echo "FROC result : ${OUT_DIR}/FROC/${FROC_RESULT_DIR}"
echo "============================================================"
