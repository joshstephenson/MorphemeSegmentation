#!/usr/bin/env bash
#

source ./train.conf

OUT_FILE="${OUT_DIR}/${EXPERIMENT}.tsv"
RESULTS_FILE="${OUT_DIR}/${EXPERIMENT}.txt"

echo "Generating predictions..."
yoyodyne-predict \
    --checkpoint "$1" \
    --model_dir "${MODEL_DIR}" \
    --experiment "${EXPERIMENT}" \
    --predict "${TEST_DATA}" \
    --output "${OUT_FILE}"

if [ $? -ne 0 ]; then
    echo "Prediction failed"
    exit 1
fi

echo "Evaluating model..."
python "${EVALUATE_SCRIPT}" --gold "${TEST_DATA}" --guess "${OUT_FILE}" > "${RESULTS_FILE}"

cat "${RESULTS_FILE}"

