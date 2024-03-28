#!/usr/bin/env bash
# Adapted from a script by Ben Peters which was adapted by a script by Kyle Gorman and Shijie Wu.
#
# Improved to make this a one-and-done script for preprocessing, training, decoding, postprocessing and evaluation
# Only required argument is the language code
LAN=$1 # Should match the name of the languages available in ../2022SegmentationST/data
if [ -z "${LAN}" ]; then
    echo "Please provide a language code. Options {ces|eng|fra|hun|ita|lat|mon|rus|spa}."
    exit 1
fi
readonly LAN=$(echo $LAN | tr '[:upper:]' '[:lower:]')
REPO_ROOT=$(git rev-parse --show-toplevel)
# Remote server doesn't have git command
if [ -z "${REPO_ROOT}" ]; then
    REPO_ROOT="$(pwd | cut -d / -f 1,2,3)/MorphemeSegmentation"
fi
readonly DATASET="${REPO_ROOT}/2022SegmentationST/data/${LAN}.word"
readonly IN_DIR="data/${LAN}/in"
readonly OUT_DIR="data/${LAN}/out"
readonly RESULTS_DIR="data/${LAN}"
readonly MODEL_DIR="data/${LAN}/models"
readonly EVALUATE_SCRIPT="${REPO_ROOT}/2022SegmentationST/evaluation/evaluate.py"
readonly DICT_FILE="${OUT_DIR}/dict.src.txt"

# Check if valid language
if [ ! -f "${DATASET}.train.tsv" ]; then
    echo "${LAN} is not a valid language option.  Options {ces|eng|fra|hun|ita|lat|mon|rus|spa}."
    exit 1
fi
if [ ! -d "${IN_DIR}" ]; then
    mkdir -p "${IN_DIR}"
fi
if [ ! -d "${OUT_DIR}" ]; then
    mkdir -p "${OUT_DIR}"
fi
if [ ! -d "${MODEL_DIR}" ]; then
    mkdir -p "${MODEL_DIR}"
fi
source ./train.conf

log() {
    printf '=%.0s' {1..80}
    printf "\n${1}\n"
    printf '=%.0s' {1..80}
    printf '\n'
}

preprocess() {
    log "Preprocessing..."
    if [ -f "${DICT_FILE}" ]; then
        echo "Files already preprocessed. Moving on."
        cd "${IN_DIR}"
        return 0
    fi
    for TASK in "train" "dev" "test.gold" ; do
        for FILE in "${DATASET}.${TASK}.tsv"; do
            SRC_FILE="${IN_DIR}/${TASK}.src"
            TGT_FILE="${IN_DIR}/${TASK}.tgt"
            # Segment the source strings by separating each grapheme (char)
            cut -f 1 "${FILE}" | sed 's/./& /g' > "${SRC_FILE}"
            # Segment the target strings by separating each morph
            # Change morph separator to single @ and spaces to underscores
            cut -f 2 "${FILE}" | sed 's/@@/|/g' > "${TGT_FILE}"
            echo "Created files: ${SRC_FILE} and ${TGT_FILE}"
        done
    done

    echo "Moving on to fairseq-preprocess"
    cd "${IN_DIR}"
    fairseq-preprocess \
        --source-lang="src" \
        --target-lang="tgt" \
        --trainpref=train \
        --validpref=dev \
        --testpref=test.gold \
        --tokenizer=space \
        --thresholdsrc=1 \
        --thresholdtgt=1 \
        --destdir="../../${LAN}/out"
}
train() {
    cd ../../..
    log "Training..."
    fairseq-train \
        "${OUT_DIR}/" \
        --save-dir="${MODEL_DIR}" \
        --source-lang="src" \
        --target-lang="tgt" \
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
        --save-interval="${SAVE_INTERVAL}"
}

decode() {
    log "Decoding..."
    local -r MODE="test"
    # Fairseq insists on calling the dev-set "valid"; hack around this.
    local -r FAIRSEQ_MODE="${MODE/dev/valid}"
    CHECKPOINT="${MODEL_DIR}/checkpoint_last.pt"
    OUT="${OUT_DIR}/${MODE}-${BEAM}.out"
    PRED="${OUT_DIR}/${MODE}-${BEAM}.pred"
    # Bash won't overwrite the file
    # It's fine to regenerate
    if [ -f "${OUT}" ]; then
        rm -f "${OUT}"
    fi
    fairseq-generate \
        "${OUT_DIR}" \
        --source-lang="src" \
        --target-lang="tgt" \
        --path="${CHECKPOINT}" \
        --gen-subset="${FAIRSEQ_MODE}" \
        --beam="${BEAM}" \
        #--alpha="${ENTMAX_ALPHA}" \
        #--batch-size 256 \
        --max-sentences=256 \
        > "${OUT}"
    # Extracts the predictions into a TSV file.
    if [ ! -f "${OUT}" ]; then
        echo "The last command failed to generate ${OUT}. Please check output for error."
        exit 1
    fi
    cat "${OUT}" | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | python detokenize.py > $PRED
    cut -f 1 $GOLD_PATH | paste - $PRED > "${OUT_DIR}/${MODE}-${BEAM}.guess"
}

evaluate() {
    log "Evaluating..."
    local -r MODE="test"
    # Applies the evaluation script to the generated TSV file.
    RESULTS_FILE="${RESULTS_DIR}/${MODE}-${BEAM}.results"
    python $EVALUATE_SCRIPT --gold $GOLD_PATH --guess "${OUT_DIR}/${MODE}-${BEAM}.guess" > "${RESULTS_FILE}"
    if [ $? -ne 0 ]; then
        cat "${RESULTS_FILE}"
    else
        echo "Failed to evaluate model."
        exit 1
    fi
}

preprocess
if [ $? -ne 0 ]; then
    echo "The previous command failed. Stopping here"
    exit 1
fi
train
if [ $? -ne 0 ]; then
    echo "The previous command failed. Stopping here"
    exit 1
fi
decode
if [ $? -ne 0 ]; then
    echo "It appears decoding failed. Check output."
    exit 1
fi
echo "Decoding succeeded!"
evaluate
