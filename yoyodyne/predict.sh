#!/usr/bin/env bash
#

source ./train.conf

OUT_FILE="${OUT_DIR}/${EXPERIMENT}.tsv"
RESULTS_FILE="${OUT_DIR}/${EXPERIMENT}.txt"
GUESS_FILE="${OUT_DIR}/${EXPERIMENT}-guess.tsv"

if [ -f "${OUT_FILE}" ]; then
    echo "Prediction file already exists..."
else
    echo "Generating predictions..."
    yoyodyne-predict \
        --checkpoint "$1" \
        --model_dir "${MODEL_DIR}" \
        --experiment "${EXPERIMENT}" \
        --predict "${TEST_DATA}" \
        --output "${OUT_FILE}"
fi

if [ $? -ne 0 ]; then
    echo "Prediction failed"
    exit 1
fi

echo "Postprocessing..."

# Grab the first column of the test data (the words) and append the guesses as the second column
# default separator is <tab>
cut -f 1 "${TEST_DATA}" | paste - "${OUT_FILE}" > "${GUESS_FILE}"

echo "Evaluating model..."
python "${EVALUATE_SCRIPT}" --gold "${TEST_DATA}" --guess "${GUESS_FILE}" > "${RESULTS_FILE}"

cat "${RESULTS_FILE}"

