#!/usr/bin/env bash

# Load configuration variables from train.conf
source ./train.conf

# Make sure the user provides a valid language
if [ -z "$1" ]; then
    echo "Please provide a language name {ces|eng|fra|hun|ita|lat|mon|rus|spa}."
    exit 1
fi
readonly LAN="$1"; shift
readonly PREPROCESSED_DIR="data/${LAN}/in"
readonly MODEL_DIR="data/${LAN}/models"
readonly OUT_DIR="data/${LAN}/out"
readonly REPO_ROOT=$(git rev-parse --show-toplevel)
readonly IN_DIR="${REPO_ROOT}/2022SegmentationST/data"
readonly INPUT_PATH="${IN_DIR}/${LAN}.word"
readonly GOLD_PATH="2022SegmentationST/data/${LAN}.word.test.gold.tsv"
readonly GRID_LOC=$MODEL_DIR
if [ "${LAN}" == "mon" ]; then
    VOCAB=2000
fi

# Create directories recursively if necessary
mkdir -p "$PREPROCESSED_DIR"
mkdir -p "$OUT_DIR"
mkdir -p "$GRID_LOC"

echo "GOLD PATH: ${GOLD_PATH}"
echo "GRID LOC: ${GRID_LOC}"
echo "INPUT PATH: ${INPUT_PATH}"
echo "OUT DIR: ${OUT_DIR}"
echo "VOCAB: ${VOCAB}"

###################
# Preprocessing
###################

bin() {
    tail -n +4 "${OUT_DIR}/src.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${OUT_DIR}/src.fairseq.vocab"
    tail -n +4 "${OUT_DIR}/tgt.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${OUT_DIR}/tgt.fairseq.vocab"
    #cd "${PREPROCESSED_DIR}"
    fairseq-preprocess \
        --source-lang="src" \
        --target-lang="tgt" \
        --trainpref="${OUT_DIR}/train" \
        --validpref="${OUT_DIR}/dev" \
        --testpref="${OUT_DIR}/test" \
        --tokenizer=space \
        --thresholdsrc=1 \
        --thresholdtgt=1 \
        --srcdict "${OUT_DIR}/src.fairseq.vocab" \
        --tgtdict "${OUT_DIR}/tgt.fairseq.vocab" \
        --destdir="${OUT_DIR}"
}

preprocess() {
    TRAIN_FILE="${INPUT_PATH}.train.tsv"
    DEV_FILE="${INPUT_PATH}.dev.tsv"
    GOLD_FILE="${INPUT_PATH}.test.gold.tsv"
    # Mongolian training set is in a sub-folder because it was a "surprise" language
    if [ "${LAN}" == "mon" ]; then
        INPUT_PATH2="${IN_DIR}/surprise/${LAN}.word"
        TRAIN_FILE="${INPUT_PATH2}.train.tsv"
        DEV_FILE="${INPUT_PATH2}.dev.tsv"
    fi
    python tokenize.py "${TRAIN_FILE}" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --out-dir $OUT_DIR --split train $@
    if [ $? -ne 0 ]; then echo "Tokenizing train failed" && exit 1 ; fi
    python tokenize.py "${DEV_FILE}" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --existing-src-spm "${OUT_DIR}/src" --existing-tgt-spm "${OUT_DIR}/tgt" --out-dir $OUT_DIR --split dev --shared-data
    if [ $? -ne 0 ]; then echo "Tokenizing dev failed" && exit 1 ; fi
    python tokenize.py "${GOLD_FILE}" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --existing-src-spm "${OUT_DIR}/src" --existing-tgt-spm "${OUT_DIR}/tgt" --out-dir $OUT_DIR --split test --shared-data
    if [ $? -ne 0 ]; then echo "Tokenizing test.gold failed" && exit 1 ; fi
    bin
}

###################
# Training
###################

train() {
    # $THIS_MODEL_DIR $EMB $HID $LAYERS $HEADS $WARMUP $DROPOUT $BATCH
    local -r THIS_MODEL_DIR=$1; shift
    local -r EMB=$1; shift
    local -r HID=$1; shift
    local -r LAYERS=$1; shift
    local -r HEADS=$1; shift
    local -r WARMUP=$1; shift
    local -r DROPOUT=$1; shift
    local -r BATCH=$1; shift
    fairseq-train \
        "${OUT_DIR}" \
        --save-dir="${THIS_MODEL_DIR}" \
        --source-lang="src" \
        --target-lang="tgt" \
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
        #--amp \
        #--best-checkpoint-metric "lev_dist" \
        #--eval-levenshtein \
        #--eval-bleu-remove-bpe "sentencepiece" \
        #--eval-bleu-args '{"beam_size": 5, "alpha": 1.5}' \
        "$@"   # In case we need more configuration control.
    if [ $? -ne 0 ]; then
        echo "fairseq-train failed."
        exit 1
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
                FILENAME="${THIS_MODEL_DIR}/dev-5.results"
                if [ ! -f "$FILENAME" ]
                then
                    train $THIS_MODEL_DIR $EMB $HID $LAYERS $HEADS $WARMUP $DROPOUT $BATCH
                    generate $THIS_MODEL_DIR test
                    evaluate $THIS_MODEL_DIR test
                    echo "Trained data written to: ${FILENAME}"
                else
                    echo "Results file ${FILENAME} found. Not re-running."
                    cat "${FILENAME}"
                fi
            done
        done
    done
}

train_grid() {
    grid 256 1024 6 8
    grid 512 2048 6 8
}

###################
# Evaluation
###################

generate() {
    local -r THIS_MODEL_DIR="$1"; shift
    local -r MODE="$1"; shift
    # Fairseq insists on calling the dev-set "valid"; hack around this.
    local -r FAIRSEQ_MODE="${MODE/dev/valid}"
    CHECKPOINT="${THIS_MODEL_DIR}/checkpoint_best.pt"
    OUT="${THIS_MODEL_DIR}/${MODE}-${BEAM}.subwords.out"
    PRED="${THIS_MODEL_DIR}/${MODE}-${BEAM}.subwords.pred"
    # Makes raw predictions.
    fairseq-generate \
        "${OUT_DIR}" \
        --source-lang="src" \
        --target-lang="tgt" \
        --path="${CHECKPOINT}" \
        --gen-subset="${FAIRSEQ_MODE}" \
        --beam="${BEAM}" \
        --alpha="${ENTMAX_ALPHA}" \
        --batch-size 256 \
        > "${OUT}"
    if [ $? -ne 0 ]; then
        echo "fairseq-train failed."
        exit 1
    fi
    # Extracts the predictions into a TSV file.
    cat "${OUT}" | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' > $PRED
    cut -f 1 $GOLD_PATH | paste - $PRED > "${THIS_MODEL_DIR}/${MODE}-${BEAM}.guess"
}

evaluate() {
    local -r THIS_MODEL_DIR="$1"; shift
    local -r MODE="$1"; shift
    # Applies the evaluation script to the TSV file.
    PREFIX="${THIS_MODEL_DIR}/${MODE}-${BEAM}"
    RESULTS_FILE="${PREFIX}.guess"
    RESULTS_FILE="${PREFIX}.results"
    python 2022SegmentationST/evaluation/evaluate.py --gold "${GOLD_PATH}" --guess "${GUESS_FILE}" > "${RESULTS_FILE}"
    if [ $? -ne 0 ]; then
        cat "${RESULTS_FILE}"
    else
        echo "Failed to evaluate model."
    fi
}

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
