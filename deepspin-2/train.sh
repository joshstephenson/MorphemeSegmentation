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
HAS_GIT=$(which git)
if [ $? -eq 0]; then
    REPO_ROOT=$(git rev-parse --show-toplevel)
else # Remote server doesn't have git command in task runner environment
    # This assumes that MorphemeSegmentation is at third depth from root: /home/user/MorphemeSegmentation
    REPO_ROOT="$(pwd | cut -d / -f 1,2,3)/MorphemeSegmentation"
fi
readonly DATASET="${REPO_ROOT}/2022SegmentationST/data/${LAN}.word"
readonly IN_DIR="data/${LAN}/in"
readonly OUT_DIR="data/${LAN}/out"
readonly RESULTS_DIR="data/${LAN}"
readonly MODEL_DIR="data/${LAN}/models"
readonly EVALUATE_SCRIPT="${REPO_ROOT}/2022SegmentationST/evaluation/evaluate.py"
readonly DICT_FILE="${OUT_DIR}/dict.src.txt"
readonly CHECKPOINT_BEST="${MODEL_DIR}/checkpoint_best.pt"
readonly CHECKPOINT_LAST="${MODEL_DIR}/checkpoint_last.pt"

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

get_checkpoint() {
    if [ -f "${CHECKPOINT_BEST}" ]; then
        CHECKPOINT="${CHECKPOINT_BEST}"
    elif [ -f "${CHECKPOINT_LAST}" ]; then
        CHECKPOINT="${CHECKPOINT_LAST}"
    else
        CHECKPOINT=""
    fi
}
train() {
    cd ../../..
    get_checkpoint
    echo "CHECKPOINT: ${CHECKPOINT}"
    if [ -f "${CHECKPOINT}" ]; then
        log "Checkpount found: ${CHECKPOINT}\nNot retraining..."
    else
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
            --save-interval="${SAVE_INTERVAL}" \
            --reset-optimizer # to make sure we get a checkpoint_best file
    fi
}

decode() {
    log "Decoding..."
    local -r MODE="test"
    # Fairseq insists on calling the dev-set "valid"; hack around this.
    local -r FAIRSEQ_MODE="${MODE/dev/valid}"
    OUT="${OUT_DIR}/${MODE}-${BEAM}.out"
    PRED="${OUT_DIR}/${MODE}-${BEAM}.pred"
    GUESS="${OUT_DIR}/${MODE}-${BEAM}.guess"
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
        > "${OUT}"
        #--alpha="${ENTMAX_ALPHA}" \
        #--batch-size 256 \
    # Extracts the predictions into a TSV file.
    if [ ! -f "${OUT}" ]; then
        echo "The last command failed to generate ${OUT}. Please check output for error."
        exit 1
    fi
    #################################
    # Now do a little post-processing
    #################################

    # extract the 3rd to last column of the output from fairseq-generate only if the line starts with 'H-'
    cat "${OUT}" | grep -P '^H-' | \
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

    # Now take the first column of the gold file (the input word) as the first column
    # and the generated prediction as the second column and output to .guess file
    cut -f 1 $GOLD_PATH | paste - $PRED > $GUESS
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
echo "Preprocessing succeeded"
train
if [ $? -ne 0 ]; then
    echo "The previous command failed. Stopping here"
    exit 1
fi
echo "Training succeeded"
decode
if [ $? -ne 0 ]; then
    echo "It appears decoding failed. Check output."
    exit 1
fi
echo "Decoding succeeded"
evaluate
