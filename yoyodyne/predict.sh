#!/usr/bin/env bash
#

source ./train.conf

EXPERIMENT="morphemes-${LAN}"
OUT_FILE="${OUT_DIR}/${EXPERIMENT}.tsv"
RESULTS_FILE="${OUT_DIR}/${EXPERIMENT}.txt"

yoyodyne-predict \
    $1 \
    --model_dir "${MODEL_DIR}" \
    --experiment "${EXPERIMENT}" \
    --predict "${TEST_DATA}" \
    --output "${OUT_FILE}"

python "${EVALUATE_SCRIPT}" --gold "${TEST_DATA}" --guess "${OUT_FILE}" > "${RESULTS_FILE}"

cat "${RESULTS_FILE}"

