#!/usr/bin/env bash
#

LAN=$1
LAN=$1 # Should match the name of the languages available in ../2022SegmentationST/data
if [ -z "${LAN}" ]; then
    echo "Please provide a language name. Options {ces|eng|fra|hun|ita|lat|mon|rus|spa}."
    exit 1
fi
readonly LAN=$(echo $LAN | tr '[:upper:]' '[:lower:]')
readonly REPO_ROOT=$(git rev-parse --show-toplevel)
readonly IN_DIR="${REPO_ROOT}/2022SegmentationST/data"
readonly OUT_DIR="data/${LAN}"
if [ ! -d "${OUT_DIR}" ]; then
    mkdir -p "${OUT_DIR}"
fi
readonly TEST_FILE="${IN_DIR}/${LAN}.word.test.tsv"
readonly GUESS_FILE="${OUT_DIR}/predictions.guess"
readonly RESULTS_FILE="${OUT_DIR}/results.txt"
readonly GOLD_FILE="${IN_DIR}/${LAN}.word.test.gold.tsv"
readonly EVAL_SCRIPT="${REPO_ROOT}/2022SegmentationST/evaluation/evaluate.py"
readonly GENERATE_SCRIPT="./generate_morphs.py"

generate() {
    echo "Generating..."
    python $GENERATE_SCRIPT $TEST_FILE > $GUESS_FILE
}

evaluate(){
    echo "Evaluating..."
    python $EVAL_SCRIPT --gold $GOLD_FILE --guess $GUESS_FILE > $RESULTS_FILE
    cat $RESULTS_FILE
}

generate
evaluate
