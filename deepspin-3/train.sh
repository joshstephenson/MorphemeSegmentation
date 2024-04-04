#!/usr/bin/env bash

# Load configuration variables from train.conf
source ./deepspin-3.sh
source ../deepspin-shared.sh

readonly GRID_LOC=$MODEL_DIR

# Mongolian and Czeck don't have enough data to support vocab size of 6000
if [ "${LAN}" == "mon" ] || [ "${LAN}" == "ces" ]; then
    VOCAB=2000
fi

echo "GOLD PATH: ${GOLD_FILE}"
echo "GRID LOC: ${GRID_LOC}"
echo "INPUT PATH: ${DATASET}"
echo "OUT DIR: ${OUT_DIR}"

VOCAB_DIR="./data/${LAN}"
PREPROCESSED_DIR="${VOCAB_DIR}/in"
OUT_DIR="${VOCAB_DIR}/out"
RESULTS_DIR="${VOCAB_DIR}"
MODEL_DIR="${VOCAB_DIR}/models"

###################
# Preprocessing
###################

#preprocess() {
#    EXPERIMENT_DIR=$1; shift
#    VOCAB=$1; shift
#    if [ -z $VOCAB ]; then
#        >&2 echo "preprocess function needs 2 arguments: experiment directory and vocab size"
#        exit 1
#    fi
#    RESULTS_DIR="${EXPERIMENT_DIR}"
#    PREPROCESSED_DIR="${EXPERIMENT_DIR}/in"
#    OUT_DIR="${EXPERIMENT_DIR}/out"
#    MODEL_DIR="${EXPERIMENT_DIR}/models"
#    SOURCE_VOCAB="${OUT_DIR}/${SOURCE}.fairseq.vocab"
#    TARGET_VOCAB="${OUT_DIR}/${TARGET}.fairseq.vocab"
#    if [ -f "${SOURCE_VOCAB}" ] && [ -f "${TARGET_VOCAB}" ]; then
#        echo "Looks like this configuration has already been preprocessed. Moving on..."
#        return
#    fi
#    mkdir -p "${PREPROCESSED_DIR}"
#    mkdir -p "${OUT_DIR}"
#    mkdir -p "${MODEL_DIR}"
#    log "Preprocessing..."
#    echo "EXPERIMENT DIR: ${EXPERIMENT_DIR}"
#    echo "VOCAB: $VOCAB"
#    # Hack for Mongolian
#    # Mongolian training set is in a sub-folder because it was a "surprise" language
#    if [ "${LAN}" == "mon" ]; then
#        DATASET2="${IN_DIR}/surprise/${LAN}.word"
#        TRAIN_FILE="${DATASET2}.train.tsv"
#        DEV_FILE="${DATASET2}.dev.tsv"
#    fi
#
#    # Run sentencepiece tokenizer script
#    # This is a key part of why the transformer works well
#    echo "${TRAIN_FILE}"
#    echo "${OUT_DIR}"
#    echo "${VOCAB}"
#    python ../tokenize.py "${TRAIN_FILE}" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --out-dir "${OUT_DIR}" --split train $@
#    if [ $? -ne 0 ]; then echo "Tokenizing train failed" && exit 1 ; fi
#    python ../tokenize.py "${DEV_FILE}" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --existing-src-spm "${OUT_DIR}/${SOURCE}" --existing-tgt-spm "${OUT_DIR}/${TARGET}" --out-dir "${OUT_DIR}" --split dev --shared-data
#    if [ $? -ne 0 ]; then echo "Tokenizing dev failed" && exit 1 ; fi
#    python ../tokenize.py "${GOLD_FILE}" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --existing-src-spm "${OUT_DIR}/${SOURCE}" --existing-tgt-spm "${OUT_DIR}/${TARGET}" --out-dir $OUT_DIR --split test --shared-data
#    if [ $? -ne 0 ]; then echo "Tokenizing test.gold failed" && exit 1 ; fi
#
#    tail -n +4 "${OUT_DIR}/${SOURCE}.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${SOURCE_VOCAB}"
#    tail -n +4 "${OUT_DIR}/${TARGET}.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${TARGET_VOCAB}"
#    #cd "${PREPROCESSED_DIR}"
#    fairseq-preprocess \
#        --source-lang="${SOURCE}" \
#        --target-lang="${TARGET}" \
#        --trainpref="${OUT_DIR}/train" \
#        --validpref="${OUT_DIR}/dev" \
#        --testpref="${OUT_DIR}/test" \
#        --tokenizer=space \
#        --thresholdsrc=1 \
#        --thresholdtgt=1 \
#        --srcdict "${SOURCE_VOCAB}" \
#        --tgtdict "${TARGET_VOCAB}" \
#        --destdir="${OUT_DIR}"
#}

###################
# Training
###################

train() {
    # PARAMS: $THIS_MODEL_DIR $EMB $HID $LAYERS $HEADS $WARMUP $DROPOUT $BATCH
    local -r EXPERIMENT_DIR=$1; shift
    local -r EMB=$1; shift
    local -r HID=$1; shift
    local -r LAYERS=$1; shift
    local -r HEADS=$1; shift
    local -r WARMUP=$1; shift
    local -r DROPOUT=$1; shift
    local -r BATCH=$1; shift
    OUT_DIR="${EXPERIMENT_DIR}/out"
    THIS_MODEL_DIR="${EXPERIMENT_DIR}/models"
    MODEL_FILE="${THIS_MODEL_DIR}/checkpoint_best.pt"
    if [ -f "${MODEL_FILE}" ]; then
        log "Looks like model exists: ${MODEL_FILE}"
        echo "Skipping Training..."
        return
    fi
    log "Training\n\t Embeddings: ${EMB}\n\t Hidden: ${HID}\n\t Layers: ${LAYERS}\n\t Heads: ${HEADS}\n\t Warmup: ${WARMUP}\n\t Dropout: ${DROPOUT}\n\t Batch: ${BATCH}"
    fairseq-train \
        "${OUT_DIR}" \
        --save-dir="${THIS_MODEL_DIR}" \
        --source-lang="${SOURCE}" \
        --target-lang="${TARGET}" \
        --seed="${SEED}" \
        --arch="${ARCH}" \
        --attention-dropout="${DROPOUT}" \
        --activation-dropout="${DROPOUT}" \
        --activation-fn="${ACTIVATION_FN}" \
        --encoder-embed-dim="${EMB}" \
        --encoder-ffn-embed-dim="${HID}" \
        --encoder-layers="${LAYERS}" \
        --encoder-attention-heads="${HEADS}" \
        --encoder-normalize-before \
        --decoder-embed-dim="${EMB}" \
        --decoder-ffn-embed-dim="${HID}" \
        --decoder-layers="${LAYERS}" \
        --decoder-attention-heads="${HEADS}" \
        --decoder-normalize-before \
        --share-decoder-input-output-embed \
        --criterion="${CRITERION}" \
        --loss-alpha="${ENTMAX_ALPHA}" \
        --optimizer="${OPTIMIZER}" \
        --lr="${LR}" \
        --lr-scheduler="${LR_SCHEDULER}" \
        --warmup-init-lr="${WARMUP_INIT_LR}" \
        --warmup-updates="${WARMUP}" \
        --clip-norm="${CLIP_NORM}" \
        --max-tokens="${BATCH}" \
        --max-update="${MAX_UPDATE}" \
        --save-interval="${SAVE_INTERVAL}" \
        --patience="${PATIENCE}" \
        --no-epoch-checkpoints \
        "$@"   # In case we need more configuration control.
        #--amp \
        #--best-checkpoint-metric "lev_dist" \
        #--eval-levenshtein \
        #--eval-bleu-remove-bpe "sentencepiece" \
        #--eval-bleu-args '{"beam_size": 5, "alpha": 1.5}' \
    if [ $? -ne 0 ]; then
        echo "fairseq-train failed."
        exit 1
    fi
}

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
    if [ -f "$FILENAME" ]
    then
        echo "Results file ${FILENAME} found. Not re-running."
        cat "${FILENAME}"
    else
        train $EXPERIMENT_DIR $EMB $HID $LAYERS $HEADS $WARMUP $DROPOUT $BATCH
        generate $EXPERIMENT_DIR test
        evaluate $EXPERIMENT_DIR test
        echo "Trained data written to: ${FILENAME}"
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
        time train_generate_evaluate "${EXPERIMENT_DIR}" $EMB $HID $LAYERS $HEADS $WARMUP $DROPOUT $BATCHES
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
    VOCAB=500
    EXPERIMENT_DIR="./data/${LAN}"
    preprocess "${EXPERIMENT_DIR}" $VOCAB
    train_generate_evaluate "${EXPERIMENT_DIR}" $EMB $HID $LAYERS $HEADS $WARMUP $DROPOUT $BATCHES
}

train_grid() {
    grid 256 1024 6 8
    grid 512 2048 6 8
}

###################
# Evaluation
###################

generate() {
    local -r EXPERIMENT_DIR="$1"; shift
    local -r MODE="$1"; shift
    # Fairseq insists on calling the dev-set "valid"; hack around this.
    local -r FAIRSEQ_MODE="${MODE/dev/valid}"
    THIS_MODEL_DIR="${EXPERIMENT_DIR}/models"
    OUT_DIR="${EXPERIMENT_DIR}/out"
    CHECKPOINT="${THIS_MODEL_DIR}/checkpoint_best.pt"
    PREFIX="${OUT_DIR}/${MODE}-${BEAM}"
    OUT_FILE="${PREFIX}.out"
    PRED_FILE="${PREFIX}.pred"
    GUESS_FILE="${PREFIX}.guess"
    if [ ! -f "${OUT_FILE}" ]; then
        log "Generating predictions..."
        # Makes raw predictions.
        fairseq-generate \
            "${OUT_DIR}" \
            --source-lang="${SOURCE}" \
            --target-lang="${TARGET}" \
            --path="${CHECKPOINT}" \
            --gen-subset="${FAIRSEQ_MODE}" \
            --beam="${BEAM}" \
            --alpha="${ENTMAX_ALPHA}" \
            --batch-size 256 \
            > "${OUT_FILE}"
        if [ $? -ne 0 ]; then
            echo "fairseq-train failed."
            exit 1
        fi
    else
        log "Looks like this configuration has already been generated"
        echo "Skipping generation..."
    fi
    if [ ! -f "${GUESS_FILE}" ]; then
        log "Postprocessing..."
        postprocess "${OUT_FILE}" "${GOLD_FILE}" "${PRED_FILE}" "${GUESS_FILE}"
    fi
}

evaluate() {
    log "Evaluating..."
    local -r EXPERIMENT_DIR="$1"; shift
    local -r MODE="$1"; shift
    THIS_MODEL_DIR="${EXPERIMENT_DIR}/models"
    # Applies the evaluation script to the TSV file.
    OUT_DIR="${EXPERIMENT_DIR}/out"
    PREFIX="${OUT_DIR}/${MODE}-${BEAM}"
    GUESS_FILE="${PREFIX}.guess"
    RESULTS_FILE="${PREFIX}.results"
    python $EVALUATE_SCRIPT --gold "${GOLD_FILE}" --guess "${GUESS_FILE}" > "${RESULTS_FILE}"
    if [ $? -eq 0 ]; then
        cat "${RESULTS_FILE}"
    else
        echo "Failed to evaluate model."
    fi
}

simple_train
exit 0
if [ "$(ls -A $OUT_DIR)" ]; then
    echo "PROCESSED DATA found in ${OUT_DIR}. Proceeding to training..."
    train_grid
else
    preprocess
    if [ $? -ne 0 ]; then
        echo "Data Processing failed. Do not proceed to training."
        exit 1
    else
        echo "Data Processing succeeded. Proceeding to training."
        train_grid
    fi
fi
