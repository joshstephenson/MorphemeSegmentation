#!/usr/bin/env bash
# ##########################################################################################################################
# Allows running of DeepSPIN-2 (LSTM with character level tokenization) or DeepSPIN-3 (Transformer with subword tokenization)
# or any other combination (LSTM with subwords)
#
# Invoke:
#   $ ./run.sh <language code> <architecture> <tokenization>
#   language code options: ces, eng, fra, hun, ita, lat, mon, rus, spa
#   architecture options:  lstm, transformer
#   tokenization options:  subword, char (default)
# ##########################################################################################################################

source ./deepspin.sh
if [ "$#" -gt 0 ]; then
    if [ "${TOKENIZATION}" == "subwords" ]; then
        VOCAB=$1; shift
        re='^[0-9]+$'
        if ! [[ $VOCAB =~ $re ]]; then
            echo "Vocab argument must be a number."
            exit 1
        fi
        echo "Using custom VOCAB size of ${VOCAB}"
    else
        echo "Vocab argument is meaningless without subwords."
        exit 1
    fi
fi

time preprocess
if [ $? -ne 0 ]; then
    echo "Preprocessing failed."
    exit 1
fi
time train
if [ $? -ne 0 ]; then
    echo "Training failed."
    exit 1
fi
time predict
if [ $? -ne 0 ]; then
    echo "Predictions failed."
    exit 1
fi
evaluate
if [ $? -ne 0 ]; then
    echo "Evaluation failed."
    exit 1
fi

#################################################
# Legacy stuff that should maybe get removed soon
#################################################

train_generate_evaluate() {
    local -r EXPERIMENT_DIR=$1
    local -r EMB=$2
    local -r HID=$3
    local -r LAYERS=$4
    local -r HEADS=$5
    local -r WARMUP=$6
    local -r DROPOUT=$7
    local -r BATCH=$8
    if [ -z "${EXPERIMENT_DIR}" ]
    then
        echo "train_generate_evaluate must receive a model directory argument"
        exit 1
    fi
    FILENAME="${EXPERIMENT_DIR}/dev-5.results"
    if [ -s "$FILENAME" ]
    then
        echo "Results file ${FILENAME} found. Not re-running."
        cat "${FILENAME}"
    else
        train $EXPERIMENT_DIR $EMB $HID $LAYERS $HEADS $WARMUP $DROPOUT $BATCH
        generate $EXPERIMENT_DIR test
        evaluate $EXPERIMENT_DIR test
        echo "Results written to: ${FILENAME}"
    fi
}

grid() {
    local -r EMB="$1"; shift
    local -r HID="$1"; shift
    local -r LAYERS="$1" ; shift
    local -r HEADS="$1" ; shift
    for WARMUP in 4000 8000 ; do
        for DROPOUT in 0.1 0.3 ; do
            for BATCH in $BATCHES ; do
                THIS_MODEL_DIR="${GRID_LOC}/${LAN}-entmax-minloss-${EMB}-${HID}-${LAYERS}-${HEADS}-${BATCH}-${ENTMAX_ALPHA}-${LR}-${WARMUP}-${DROPOUT}"
                mkdir -p "${THIS_MODEL_DIR}"
                train_generate_evaluate "${THIS_MODEL_DIR}" $EMB $HID $LAYERS $HEADS $WARMUP $DROPOUT $BATCH
            done
        done
    done
}

vocab_grid() {
    EMB=256
    HID=1024
    LAYERS=6
    HEADS=8
    WARMUP=4000
    DROPOUT=0.1
    for VOCAB in $(for i in $(seq 0 7); do echo $((93*(2**i))); done); do
        echo "Trying Vocab = ${VOCAB}"
        EXPERIMENT_DIR="./data/${LAN}/vocab-${VOCAB}"
        time preprocess "${EXPERIMENT_DIR}" $VOCAB
        train_generate_evaluate "${EXPERIMENT_DIR}" $EMB $HID $LAYERS $HEADS $WARMUP $DROPOUT $BATCHES
    done
    exit
}
simple_train() {
    EMB=256
    HID=1024
    LAYERS=6
    HEADS=8
    WARMUP=4000
    DROPOUT=0.1
    EXPERIMENT_DIR="./data/${LAN}"
    time preprocess "${EXPERIMENT_DIR}" $VOCAB
    train_generate_evaluate "${EXPERIMENT_DIR}" $EMB $HID $LAYERS $HEADS $WARMUP $DROPOUT $BATCHES
}

train_grid() {
    grid 256 1024 6 8
    grid 512 2048 6 8
}

#simple_train
#exit 0
#if [ "$(ls -A $OUT_DIR)" ]; then
#    echo "PROCESSED DATA found in ${OUT_DIR}. Proceeding to training..."
#    train_grid
#else
#    preprocess
#    if [ $? -ne 0 ]; then
#        echo "Data Processing failed. Do not proceed to training."
#        exit 1
#    else
#        echo "Data Processing succeeded. Proceeding to training."
#        train_grid
#    fi
#fi
