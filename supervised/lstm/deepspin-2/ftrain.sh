#!/usr/bin/env bash
# Adapted from a script by Ben Peters which was adapted by a script by Kyle Gorman and Shijie Wu.
#
LAN=$1 # Should match the name of the languages available in ../2022SegmentationST/data
if [ -z "${LAN}" ]; then
    echo "Please provide a language name. Options {ces|eng|fra|hun|ita|lat|mon|rus|spa}."
    exit 1
fi
readonly LAN=$(echo $LAN | tr '[:upper:]' '[:lower:]')
readonly DATASET="../2022SegmentationST/data/${LAN}.word"
readonly IN_DIR="data/${LAN}/in"
readonly OUT_DIR="data/${LAN}/out"
readonly MODEL_DIR="data/${LAN}/models"

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
readonly SEED=1917
readonly CRITERION=label_smoothed_cross_entropy
readonly LABEL_SMOOTHING=.1
readonly OPTIMIZER=adam
readonly LR=1e-3
readonly CLIP_NORM=1.
readonly MAX_UPDATE=4000
readonly SAVE_INTERVAL=5
readonly EED=256
readonly EHS=256
readonly DED=256
readonly DHS=256

# Hyperparameters to be tuned.
readonly BATCH=256
readonly DROPOUT=.3
readonly ENTMAX_ALPHA=1.5

# Prediction options.
readonly BEAM=5

preprocess() {
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
    cd "${IN_DIR}" && \
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
    echo "Now in train"
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
    local -r CP="$1"; shift
    local -r MODE="$1"; shift
    # Fairseq insists on calling the dev-set "valid"; hack around this.
    local -r FAIRSEQ_MODE="${MODE/dev/valid}"
    echo "FAIRSEQ_MODE: ${FAIRSEQ_MODE}"
    CHECKPOINT="${CP}/checkpoint_last.pt"
    OUT="${CP}/${MODE}-${BEAM}.out"
    PRED="${CP}/${MODE}-${BEAM}.pred"
    echo "PRED: ${PRED}"
    echo "out: ${OUT}"
    # Makes raw predictions.
    fairseq-generate \
        "${OUT_DIR}" \
        --source-lang="src" \
        --target-lang="tgt" \
        --path="${CHECKPOINT}" \
        --gen-subset="${FAIRSEQ_MODE}" \
        --beam="${BEAM}" \
        #--alpha="${ENTMAX_ALPHA}" \
        #--batch-size 256 \
        > "${OUT}"
    # Extracts the predictions into a TSV file.
    if [ ! -f "${OUT}" ]; then
        echo "The last command failed to generate ${OUT}. Please check output for error."
        exit 1
    fi
    cat "${OUT}" | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | python detokenize.py > $PRED
    cut -f 1 $GOLD_PATH | paste - $PRED > "${CP}/${MODE}-${BEAM}.guess"
    # Applies the evaluation script to the TSV file.
    RESULTS_FILE="${CP}/${MODE}-${BEAM}.results"
    python 2022SegmentationST/evaluation/evaluate.py --gold $GOLD_PATH --guess "${CP}/${MODE}-${BEAM}.guess" > "${RESULTS_FILE}"
    if [ $? -ne 0 ]; then
        echo "Results of model evaluation written to: ${RESULTS_FILE}"
    else
        echo "Failed to evaluate model."
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
decode "${MODEL_DIR}" test
if [ $? -ne 0 ]; then
    echo "It appears decoding failed. Check output."
    exit 1
fi
echo "Decoding succeeded!"
