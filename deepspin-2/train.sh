#!/usr/bin/env bash
# Adapted from a script by Ben Peters which was adapted by a script by Kyle Gorman and Shijie Wu.
#
# Improved to make this a one-and-done script for preprocessing, training, decoding, postprocessing and evaluation
# Only required argument is the language code

# Load configuration variables from train.conf
source ./deepspin-2.sh
source ../deepspin-shared.sh


if [ ! -d "${PREPROCESSED_DIR}" ]; then
    mkdir -p "${PREPROCESSED_DIR}"
fi
if [ ! -d "${OUT_DIR}" ]; then
    mkdir -p "${OUT_DIR}"
fi
if [ ! -d "${MODEL_DIR}" ]; then
    mkdir -p "${MODEL_DIR}"
fi

###################
# Preprocessing
###################

# See ../deepspin-shared.sh

train() {
    get_checkpoint
    if [ -f "${CHECKPOINT}" ]; then
        log "Checkpount found: ${CHECKPOINT}\nProceeding..."
    else
        log "Training..."
        fairseq-train \
            "${PREPROCESSED_DIR}/" \
            --save-dir="${MODEL_DIR}" \
            --source-lang="${SOURCE}" \
            --target-lang="${TARGET}" \
            --disable-validation \
            --seed="${SEED}" \
            --arch=lstm \
            --encoder-bidirectional \
            --dropout="${DROPOUT}" \
            --encoder-embed-dim="${EED}" \
            --encoder-hidden-size="${EHS}" \
            --decoder-embed-dim="${DED}" \
            --decoder-out-embed-dim="${DED}" \
            --decoder-hidden-size="${DHS}" \
            --share-decoder-input-output-embed \
            --criterion="${CRITERION}" \
            --label-smoothing="${LABEL_SMOOTHING}" \
            --optimizer="${OPTIMIZER}" \
            --lr="${LR}" \
            --clip-norm="${CLIP_NORM}" \
            --batch-size="${BATCH}" \
            --max-update="${MAX_UPDATE}" \
            --save-interval="${SAVE_INTERVAL}" \
            --reset-optimizer # to make sure we get a checkpoint_best file
    fi
}

decode() {
    MODE="test"
    get_checkpoint
    # Fairseq insists on calling the dev-set "valid"; hack around this.
    FAIRSEQ_MODE="${MODE/dev/valid}"
    OUT_FILE="${OUT_DIR}/${MODE}-${BEAM}.out"
    # Bash won't overwrite the file
    # It's fine to regenerate
    if [ -f "${OUT_FILE}" ]; then
        log "${OUT_FILE} found. Proceeding..."
    else
        log "Decoding..."
        fairseq-generate \
            "${OUT_DIR}" \
            --source-lang="${SOURCE}" \
            --target-lang="${TARGET}" \
            --path="${CHECKPOINT}" \
            --gen-subset="${FAIRSEQ_MODE}" \
            --beam="${BEAM}" \
            > "${OUT_FILE}"
            #--alpha="${ENTMAX_ALPHA}" \
            #--batch-size 256 \
    fi
    if [ $? -ne 0 ]; then
        rm "${OUT_FILE}"
        echo "It appears decoding failed. Check output."
        exit 1
    fi
}
postprocess() {
    log "Postprocessing..."
    MODE="test"
    # Fairseq insists on calling the dev-set "valid"; hack around this.
    FAIRSEQ_MODE="${MODE/dev/valid}"
    OUT_FILE="${OUT_DIR}/${MODE}-${BEAM}.out"
    PRED="${OUT_DIR}/${MODE}-${BEAM}.pred"
    GUESS_FILE="${OUT_DIR}/${MODE}-${BEAM}.guess"

    # Extracts the predictions into a TSV file.
    # extract the 3rd to last column of the output from fairseq-generate only if the line starts with 'H-'
    echo "Writing predictions"
    cat "${OUT_FILE}" | grep -P '^H-' | \
        cut -c 3- | \
        # sort them numerically
        sort -n -k 1 | \
        # print only the last field
        awk -F "\t" '{print $NF}' | \
        # Remove whitespace and then replace underscores with whitespace (order here matters)
        sed -e 's/[[:space:]]*//g' -e 's/[▁_]/ /g' | \
        # swap out the internal morph separator to the external one
        sed -e 's/[[:space:]]*|/ @@/g' | \
        # Strip leading whitespace from the tokenization above and redirect to prediction file
        sed -e 's/^ //g' > $PRED

    echo "Generated prediction file: ${PRED}"
    # Now take the first column of the gold file (the input word) as the first column
    # and the generated prediction as the second column and output to .guess file
    cut -f 1 "${GOLD_FILE}" | paste - "${PRED}" > "${GUESS_FILE}"
    echo "Generated guess file: ${GUESS_FILE}"
}

evaluate() {
    log "Evaluating..."
    MODE="test"
    GUESS_FILE="${OUT_DIR}/${MODE}-${BEAM}.guess"
    # Applies the evaluation script to the generated TSV file.
    RESULTS_FILE="${RESULTS_DIR}/${MODE}-${BEAM}.results"
    python $EVALUATE_SCRIPT --gold "${GOLD_FILE}" --guess "${GUESS_FILE}" > "${RESULTS_FILE}"
    if [ $? -eq 0 ]; then
        cat "${RESULTS_FILE}"
    else
        echo "Failed to evaluate model."
        exit 1
    fi
}

preprocess "./data/${LAN}" $VOCAB
if [ $? -ne 0 ]; then
    echo "Preprocessing failed. Stopping here"
    exit 1
fi
echo "Preprocessing succeeded"
train
if [ $? -ne 0 ]; then
    echo "Training failed. Stopping here"
    exit 1
fi
echo "Training succeeded"
decode
if [ $? -ne 0 ]; then
    echo "It appears decoding failed. Check output."
    exit 1
fi
echo "Decoding succeeded"
postprocess
if [ $? -ne 0 ]; then
    echo "Postprocessing failed. Stopping here"
    exit 1
fi
echo "Postprocess succeeded"
evaluate
if [ $? -ne 0 ]; then
    echo "Evaluation failed. Stopping here"
    exit 1
fi
