#!/usr/bin/env bash
# Adapted from a script by Ben Peters

LAN=$1
LAN=$1 # Should match the name of the languages available in ../2022SegmentationST/data
if [ -z "${LAN}" ]; then
    echo "Please provide a language name. Options {ces|eng|fra|hun|ita|lat|mon|rus|spa}."
    exit 1
fi

# Lowercase the language name
readonly LAN=$(echo $LAN | tr '[:upper:]' '[:lower:]')
readonly REPO_ROOT=$(git rev-parse --show-toplevel)
readonly IN_DIR="${REPO_ROOT}/2022SegmentationST/data"
readonly OUT_DIR="data/${LAN}"
readonly MODEL_FILE="${OUT_DIR}/model.bin"
readonly OUT_FILE="${OUT_DIR}/predictions.out"
readonly GUESS_FILE="${OUT_DIR}/predictions.guess"
readonly RESULTS_FILE="${OUT_DIR}/results.txt"
readonly TRAIN_FILE="${OUT_DIR}/train.tmp"
readonly TEST_FILE="${OUT_DIR}/test.tmp"
readonly GOLD_FILE="${IN_DIR}/${LAN}.word.test.gold.tsv"
readonly EVAL_SCRIPT="${REPO_ROOT}/2022SegmentationST/evaluation/evaluate.py"

# Check if valid language
if [ ! -f "${IN_DIR}/${LAN}.word.train.tsv" ]; then
    echo "${LAN} is not a valid language option.  Options {ces|eng|fra|hun|ita|lat|mon|rus|spa}."
    exit 1
fi
if [ ! -d "${OUT_DIR}" ]; then
    mkdir -p "${OUT_DIR}"
fi

preprocess() {
    for dataset in train dev test; do
        IN_FILE="${IN_DIR}/${LAN}.word.${dataset}.tsv"
        TMP_FILE="${OUT_DIR}/${dataset}.tmp"

        # Grab the first column of data (just the word) and replace spaces with underscores
        cut -f 1 $IN_FILE | tr ' ' '_' > $TMP_FILE
    done
}

train() {
    # Check if model exists. If so, use it. If not, create it.
    if [ -f "${MODEL_FILE}" ]; then
        echo "${MODEL_FILE} found. Skipping training..."
    else
        echo "Training..."
        morfessor-train -s $MODEL_FILE $TRAIN_FILE
    fi
}

segment(){
    echo "Segmenting..."
    #morfessor-segment --output-format-separator ' @@' $TEST_FILE -l $MODEL_FILE | sed 's/_/ /g' > $OUT_FILE
    morfessor-segment $TEST_FILE -l $MODEL_FILE | sed 's/_/ /g' | sed 's/ / @@/g' > $OUT_FILE

    # Generate a guess file which basically inputs the first column of the original data
    cut -f 1 $TEST_FILE | paste - $OUT_FILE > $GUESS_FILE
}

evaluate(){
    echo "Evaluating..."
    python $EVAL_SCRIPT --gold $GOLD_FILE --guess $GUESS_FILE > $RESULTS_FILE
    cat $RESULTS_FILE
}

preprocess
train
segment
evaluate
