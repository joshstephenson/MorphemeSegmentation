#!/usr/bin/env bash
# Adapted from a script by Ben Peterns which was adapted by a script by Kyle Gorman and Shijie Wu.
#
LANG=$1 # Should match the name of the languages available in ../2022SegmentationST/data
if [ -z "${LANG}" ]; then
    echo "Please provide a language name. Options {ces|eng|fra|hun|ita|lat|mon|rus|spa}."
    exit 1
fi
readonly LANG=$(echo $LANG | tr '[:upper:]' '[:lower:]')
readonly DATASET="../2022SegmentationST/data/${LANG}.word"
readonly IN_DIR="data/${LANG}/in"
readonly OUT_DIR="data/${LANG}/out"
readonly MODEL_DIR="data/${LANG}/models"
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
    BASE=$(basename $DATASET)
    cd "${IN_DIR}" && \
    fairseq-preprocess \
        --source-lang="src" \
        --target-lang="tgt" \
        --trainpref=train \
        --validpref=dev \
        --tokenizer=space \
        --thresholdsrc=1 \
        --thresholdtgt=1 \
        --destdir="../../${LANG}/out"
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
