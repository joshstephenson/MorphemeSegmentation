#!/usr/bin/env bash
#

LAN=$1 # Should match the name of the languages available in ../2022SegmentationST/data
if [ -z "${LAN}" ]; then
    echo "Please provide a language code. Options {ces|eng|fra|hun|ita|lat|mon|rus|spa}."
    exit 1
fi
readonly LAN=$(echo $LAN | tr '[:upper:]' '[:lower:]')
HAS_GIT=$(which git)
if [ $? -eq 0 ]; then
    REPO_ROOT=$(git rev-parse --show-toplevel)
else # Remote server doesn't have git command in task runner environment
    # This assumes that MorphemeSegmentation is at third depth from root: /home/user/MorphemeSegmentation
    REPO_ROOT="$(pwd | cut -d / -f 1,2,3)/MorphemeSegmentation"
fi
readonly MODEL_DIR="./data/models"
readonly DATASET="${REPO_ROOT}/2022SegmentationST/data/${LAN}.word"
readonly TRAIN_DATA="${DATASET}.train.tsv"
readonly VAL_DATA="${DATASET}.dev.tsv"

mkdir -p $MODEL_DIR

yoyodyne-train \
    --model_dir "${MODEL_DIR}" \
    --experiment "morphemes" \
    --dropout 0.3 \
    --train "${TRAIN_DATA}" \
    --val "${VAL_DATA}" \
    --arch attentive_lstm \
    --max_epochs 10 \
    --accelerator gpu \
    --auto_select_gpus 1
