#!/usr/bin/env bash
# Allows running of DeepSPIN-2 (LSTM with character level tokenization) or DeepSPIN-3 (Transformer with subword tokenization)
# or any other combination (LSTM with subwords)
#
# Invoke:
#   $ ./train.sh <language code> <architecture> <tokenization>
#   language code options: ces, eng, fra, hun, ita, lat, mon, rus, spa
#   architecture options:  lstm, transformer
#   tokenization options:  subword, char (default)
LAN=$1
ARCH=$2
TOKENIZATION=$3
MODE=test

if [ "$#" -lt 3 ]; then
    echo "Must provide language, architecture (lstm|transformer) and subword tokenization(chars|subwords) arguments"
    exit 0
fi

while { test "$TOKENIZATION" != "chars"  && test "$TOKENIZATION" != "subwords"; }; do
    echo "Tokenization argument must be one of 'chars' or 'subwords'"
    exit 0
done

# sanitize the inputs
readonly LAN=$(echo $LAN | tr '[:upper:]' '[:lower:]')
readonly ARCH=$(echo $ARCH | tr '[:upper:]' '[:lower:]')
readonly TOKENIZATION=$(echo $TOKENIZATION | tr '[:upper:]' '[:lower:]')
echo "LANGUAGE: ${LAN}"
echo "ARCH: ${ARCH}"
echo "TOKENIZATION: ${TOKENIZATION}"

if [ -z "${LAN}" ]; then
    echo "Please provide a language code. Options {ces|eng|fra|hun|ita|lat|mon|rus|spa}."
    exit 1
fi

# Check for existence of git so we can find the root of the repository
which git > /dev/null 2>&1
if [ $? -eq 0 ]; then
    REPO_ROOT=$(git rev-parse --show-toplevel)
else # Remote server doesn't have git command in task runner environment
    # This assumes that MorphemeSegmentation is at third depth from root: /home/user/MorphemeSegmentation
    REPO_ROOT="$(pwd | cut -d / -f 1,2,3)/MorphemeSegmentation"
fi

readonly EVALUATE_SCRIPT="${REPO_ROOT}/2022SegmentationST/evaluation/evaluate.py"
readonly IN_DIR="${REPO_ROOT}/2022SegmentationST/data"
readonly DATASET="${IN_DIR}/${LAN}.word"

# Allowing these to be overwritten for vocab experiments
BASE_DIR="./data/${LAN}-${ARCH}-${TOKENIZATION}"
PREPROCESSED_DIR="${BASE_DIR}/in"
OUT_DIR="${BASE_DIR}/out"
MODEL_DIR="${BASE_DIR}/models"
CHECKPOINT_BEST="${MODEL_DIR}/checkpoint_best.pt"
CHECKPOINT_LAST="${MODEL_DIR}/checkpoint_last.pt"
GOLD_FILE="${DATASET}.test.gold.tsv"

readonly TRAIN_FILE="${DATASET}.train.tsv"
readonly DEV_FILE="${DATASET}.dev.tsv"
readonly SOURCE="src"
readonly TARGET="tgt"
readonly SEED=1917
readonly BEAM=5
readonly OPTIMIZER=adam
readonly LR=0.001
readonly CLIP_NORM=1.
readonly EMBED_DIM=256

# FOR TRANSFORMER ONLY

if [ "${ARCH}" == "lstm" ]; then
    readonly MAX_UPDATE=4000
    readonly SAVE_INTERVAL=5
    readonly BATCH=256
    readonly LABEL_SMOOTHING=.1
    readonly DROPOUT=.3
    readonly HIDDEN_DIM=256
else # Transformer
    readonly SAVE_INTERVAL=1
    readonly MAX_UPDATE=400000
    readonly LR_SCHEDULER="inverse_sqrt"
    readonly BATCH=8192
    readonly ACTIVATION_FN=relu
    readonly PATIENCE=5
    readonly WARMUP_INIT_LR=1e-7
    readonly WARMUP=4000
    readonly ENTMAX_ALPHA=1.5
    readonly LAYERS=6
    readonly HEADS=8
    readonly DROPOUT=.1
    readonly HIDDEN_DIM=1024
fi
DICT_FILE="${OUT_DIR}/dict.${SOURCE}.txt"
PREFIX="${OUT_DIR}/${MODE}-${BEAM}"
OUT_FILE="${PREFIX}.out"
PRED_FILE="${PREFIX}.pred"
GUESS_FILE="${PREFIX}.guess"
RESULTS_FILE="${PREFIX}.results"

VOCAB=6000 # No longer unexplored. Experiment for Hungarian showed that 200 is plenty.

mkdir -p "$PREPROCESSED_DIR"
mkdir -p "$OUT_DIR"
mkdir -p "$MODEL_DIR"

get_checkpoint() {
    MODEL_DIR=$1
    CHECKPOINT_BEST="${MODEL_DIR}/checkpoint_best.pt"
    CHECKPOINT_LAST="${MODEL_DIR}/checkpoint_last.pt"
    if [ -f "${CHECKPOINT_BEST}" ]; then
        CHECKPOINT="${CHECKPOINT_BEST}"
    elif [ -f "${CHECKPOINT_LAST}" ]; then
        CHECKPOINT="${CHECKPOINT_LAST}"
    else
        CHECKPOINT=""
    fi
    echo $CHECKPOINT
}

log() {
    printf '=%.0s' {1..80}
    printf "\n${1}\n"
}

tokenize_subwords() {
    SOURCE_VOCAB="${PREPROCESSED_DIR}/${SOURCE}.fairseq.vocab"
    TARGET_VOCAB="${PREPROCESSED_DIR}/${TARGET}.fairseq.vocab"
    if [ -f "${SOURCE_VOCAB}" ] && [ -f "${TARGET_VOCAB}" ]; then
        echo "Looks like this configuration has already been preprocessed. Moving on..."
        return
    fi
    log "Preprocessing..."

    # Run sentencepiece tokenizer script
    # This is a key part of why the transformer works well
    python ../tokenize.py "${TRAIN_FILE}" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --out-dir "${PREPROCESSED_DIR}" --split train $@
    if [ $? -ne 0 ]; then echo "Tokenizing train failed" && exit 1 ; fi
    python ../tokenize.py "${DEV_FILE}" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --existing-src-spm "${PREPROCESSED_DIR}/${SOURCE}" --existing-tgt-spm "${PREPROCESSED_DIR}/${TARGET}" --out-dir "${PREPROCESSED_DIR}" --split dev --shared-data
    if [ $? -ne 0 ]; then echo "Tokenizing dev failed" && exit 1 ; fi
    python ../tokenize.py "${GOLD_FILE}" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --existing-src-spm "${PREPROCESSED_DIR}/${SOURCE}" --existing-tgt-spm "${PREPROCESSED_DIR}/${TARGET}" --out-dir $PREPROCESSED_DIR --split test --shared-data
    if [ $? -ne 0 ]; then echo "Tokenizing test.gold failed" && exit 1 ; fi

    tail -n +4 "${PREPROCESSED_DIR}/${SOURCE}.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${SOURCE_VOCAB}"
    tail -n +4 "${PREPROCESSED_DIR}/${TARGET}.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${TARGET_VOCAB}"
    log "Creating ${SOURCE_VOCAB} and ${TARGET_VOCAB} via sentencepiece"
}
tokenize_chars() {
    log "Tokenizing..."
    for TASK in "train" "dev" "test" ; do
        for FILE in "${DATASET}.${TASK}.tsv"; do
            SOURCE_FILE="${PREPROCESSED_DIR}/${TASK}.${SOURCE}"
            TARGET_FILE="${PREPROCESSED_DIR}/${TASK}.${TARGET}"
            if [ -s "${SOURCE_FILE}" ] && [ -s "${TARGET_FILE}" ]; then
                log "Files already tokenized. Moving on..."
                return 0
            fi
            # Segment the source strings by separating each grapheme (char)
            cut -f 1 "${FILE}" | sed 's/./& /g' > "${SOURCE_FILE}"
            # Segment the target strings by separating each morph
            # Change morph separator to single @ and spaces to underscores
            cut -f 2 "${FILE}" | sed 's/@@/|/g' > "${TARGET_FILE}"
            echo "Created files: ${SOURCE_FILE} and ${TARGET_FILE}"
        done
    done
}
preprocess() {
    if [ "$SUBWORDS" == "subwords" ] # Allow user to set this in env
    then
        tokenize_subwords
        SOURCE_VOCAB="${PREPROCESSED_DIR}/${SOURCE}.fairseq.vocab"
        TARGET_VOCAB="${PREPROCESSED_DIR}/${TARGET}.fairseq.vocab"
        EXTRA_ARGS="--srcdict=${SOURCE_VOCAB} --tgtdict=${TARGET_VOCAB} "
    else
        tokenize_chars
    fi

    echo "Moving on to fairseq-preprocess"

    DICT_SOURCE="${PREPROCESSED_DIR}/dict.${SOURCE}.txt"
    DICT_TARGET="${PREPROCESSED_DIR}/dict.${TARGET}.txt"
    if [ -f "${DICT_SOURCE}" ] || [ -f "${DICT_TARGET}" ] ; then
        echo "Previously preprocessed files found: ${DICT_SOURCE} ${DICT_TARGET}"
        echo "Remove them if you wish to re-preprocess. Skipping for now..."
        return 0
    fi

    fairseq-preprocess \
        --source-lang="${SOURCE}" \
        --target-lang="${TARGET}" \
        --trainpref="${PREPROCESSED_DIR}/train" \
        --validpref="${PREPROCESSED_DIR}/dev" \
        --testpref="${PREPROCESSED_DIR}/test" \
        --tokenizer=space \
        --thresholdsrc=1 \
        --thresholdtgt=1 \
        $EXTRA_ARGS \
        --destdir="${PREPROCESSED_DIR}"
    if [ $? -ne 0 ]; then
        echo "fairseq-preprocess failed."
        exit 1
    fi
}
train_lstm() {
    log "LSTM Training..."
    time fairseq-train \
        "${PREPROCESSED_DIR}/" \
        --save-dir="${MODEL_DIR}" \
        --source-lang="${SOURCE}" \
        --target-lang="${TARGET}" \
        --disable-validation \
        --seed="${SEED}" \
        --arch=lstm \
        --encoder-bidirectional \
        --dropout="${DROPOUT}" \
        --encoder-embed-dim="${EMBED_DIM}" \
        --encoder-hidden-size="${HIDDEN_DIM}" \
        --decoder-embed-dim="${EMBED_DIM}" \
        --decoder-out-embed-dim="${EMBED_DIM}" \
        --decoder-hidden-size="${HIDDEN_DIM}" \
        --share-decoder-input-output-embed \
        --criterion=label_smoothed_cross_entropy \
        --label-smoothing="${LABEL_SMOOTHING}" \
        --optimizer="${OPTIMIZER}" \
        --lr="${LR}" \
        --clip-norm="${CLIP_NORM}" \
        --batch-size="${BATCH}" \
        --max-update="${MAX_UPDATE}" \
        --save-interval="${SAVE_INTERVAL}" \
        --reset-optimizer # to make sure we get a checkpoint_best file
    if [ $? -ne 0 ]; then
        echo "fairseq-train failed."
        exit 1
    fi
}
train_transformer() {
    log "Transformer Training..."
    time fairseq-train \
    "${PREPROCESSED_DIR}/" \
    --save-dir="${MODEL_DIR}" \
    --source-lang="${SOURCE}" \
    --target-lang="${TARGET}" \
    --seed="${SEED}" \
    --arch=transformer \
    --attention-dropout="${DROPOUT}" \
    --activation-dropout="${DROPOUT}" \
    --activation-fn="${ACTIVATION_FN}" \
    --encoder-embed-dim="${EMBED_DIM}" \
    --encoder-ffn-embed-dim="${HIDDEN_DIM}" \
    --encoder-layers="${LAYERS}" \
    --encoder-attention-heads="${HEADS}" \
    --encoder-normalize-before \
    --decoder-embed-dim="${EMBED_DIM}" \
    --decoder-ffn-embed-dim="${HIDDEN_DIM}" \
    --decoder-layers="${LAYERS}" \
    --decoder-attention-heads="${HEADS}" \
    --decoder-normalize-before \
    --share-decoder-input-output-embed \
    --criterion=entmax_loss \
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
    --no-epoch-checkpoints
    if [ $? -ne 0 ]; then
        echo "fairseq-train failed."
        exit 1
    fi
}
train() {
    CHECKPOINT=$(get_checkpoint "${MODEL_DIR}")
    if [ -s "${CHECKPOINT}" ]; then
        log "Checkpount found: ${CHECKPOINT}\nSkipping training..."
    else
        if [ "${ARCH}" == "lstm" ]; then
            train_lstm
        else
            train_transformer
        fi
    fi
}
predict() {
    CHECKPOINT=$(get_checkpoint "${MODEL_DIR}")
    EXTRA_ARGS=""
    if [ "$ARCH" == "transformer" ]; then
        EXTRA_ARGS="--batch-size 256 --alpha=${ENTMAX_ALPHA} "
    fi
    if [ -s "${OUT_FILE}" ]; then
        log "Looks like this configuration has already been generated: ${OUT_FILE}"
        echo "Skipping generation..."
    else
        log "Generating predictions..."
        REAL_PATH=$(realpath "${PREPROCESSED_DIR}")
        time fairseq-generate \
            "${REAL_PATH}" \
            --source-lang="${SOURCE}" \
            --target-lang="${TARGET}" \
            --path="${CHECKPOINT}" \
            --gen-subset="${MODE}" \
            --beam="${BEAM}" \
            --seed="${SEED}" \
            $EXTRA_ARGS \
            > "${OUT_FILE}"
    fi
    if [ $? -ne 0 ]; then
        echo "fairseq-generate failed."
        exit 1
    fi
    postprocess
}

postprocess() {
    # Extracts the predictions into a TSV file.
    # extract the 3rd to last column of the output from fairseq-generate only if the line starts with 'H-'
    cat "${OUT_FILE}" | grep -P '^H-' | \
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
        sed -e 's/^ //g' > "${PRED_FILE}"

    # Now take the first column of the gold file (the input word) as the first column
    # and the generated prediction as the second column and output to .guess file
    cut -f 1 "${GOLD_FILE}" | paste - "${PRED_FILE}" > "${GUESS_FILE}"
}
evaluate() {
    log "Evaluating..."
    python $EVALUATE_SCRIPT --gold "${GOLD_FILE}" --guess "${GUESS_FILE}" > "${RESULTS_FILE}"
    if [ $? -eq 0 ]; then
        cat "${RESULTS_FILE}"
    else
        echo "Failed to evaluate model."
    fi
}
