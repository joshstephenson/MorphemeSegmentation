
LAN=$1 # Should match the name of the languages available in ../2022SegmentationST/data
readonly LAN=$(echo $LAN | tr '[:upper:]' '[:lower:]')
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
PREPROCESSED_DIR="./data/${LAN}/in"
OUT_DIR="./data/${LAN}/out"
RESULTS_DIR="./data/${LAN}"
MODEL_DIR="./data/${LAN}/models"
CHECKPOINT_BEST="${MODEL_DIR}/checkpoint_best.pt"
CHECKPOINT_LAST="${MODEL_DIR}/checkpoint_last.pt"
GOLD_FILE="${DATASET}.test.gold.tsv"
DICT_FILE="${OUT_DIR}/dict.${SOURCE}.txt"

readonly TRAIN_FILE="${DATASET}.train.tsv"
readonly DEV_FILE="${DATASET}.dev.tsv"
readonly SOURCE="src"
readonly TARGET="tgt"
readonly SEED=1917
readonly BEAM=5
readonly ENTMAX_ALPHA=1.5
readonly OPTIMIZER=adam
readonly LR=0.001
readonly CLIP_NORM=1.
VOCAB=500 # No longer unexplored. Experiment for Hungarian showed that 200 is plenty.

mkdir -p "$PREPROCESSED_DIR"
mkdir -p "$OUT_DIR"

get_checkpoint() {
    if [ -f "${CHECKPOINT_BEST}" ]; then
        CHECKPOINT="${CHECKPOINT_BEST}"
    elif [ -f "${CHECKPOINT_LAST}" ]; then
        CHECKPOINT="${CHECKPOINT_LAST}"
    else
        CHECKPOINT=""
    fi
}

log() {
    printf '=%.0s' {1..80}
    printf "\n${1}\n"
}

postprocess() {
    OUT_FILE=$1; shift
    GOLD_FILE=$1; shift
    PRED_FILE=$1; shift
    GUESS_FILE=$1; shift

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
tokenize_subwords() {
    SOURCE_VOCAB="${PREPROCESSED_DIR}/${SOURCE}.fairseq.vocab"
    TARGET_VOCAB="${PREPROCESSED_DIR}/${TARGET}.fairseq.vocab"
    if [ -f "${SOURCE_VOCAB}" ] && [ -f "${TARGET_VOCAB}" ]; then
        echo "Looks like this configuration has already been preprocessed. Moving on..."
        return
    fi
    mkdir -p "${PREPROCESSED_DIR}"
    mkdir -p "${MODEL_DIR}"
    mkdir -p "${OUT_DIR}"
    log "Preprocessing..."
    echo "EXPERIMENT DIR: ${EXPERIMENT_DIR}"
    echo "VOCAB: $VOCAB"
    # Hack for Mongolian
    # Mongolian training set is in a sub-folder because it was a "surprise" language
    if [ "${LAN}" == "mon" ]; then
        DATASET2="${IN_DIR}/surprise/${LAN}.word"
        TRAIN_FILE="${DATASET2}.train.tsv"
        DEV_FILE="${DATASET2}.dev.tsv"
    fi

    # Run sentencepiece tokenizer script
    # This is a key part of why the transformer works well
    echo "${TRAIN_FILE}"
    echo "${PREPROCESSED_DIR}"
    echo "${VOCAB}"
    python ../tokenize.py "${TRAIN_FILE}" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --out-dir "${PREPROCESSED_DIR}" --split train $@
    if [ $? -ne 0 ]; then echo "Tokenizing train failed" && exit 1 ; fi
    python ../tokenize.py "${DEV_FILE}" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --existing-src-spm "${PREPROCESSED_DIR}/${SOURCE}" --existing-tgt-spm "${PREPROCESSED_DIR}/${TARGET}" --out-dir "${PREPROCESSED_DIR}" --split dev --shared-data
    if [ $? -ne 0 ]; then echo "Tokenizing dev failed" && exit 1 ; fi
    python ../tokenize.py "${GOLD_FILE}" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --existing-src-spm "${PREPROCESSED_DIR}/${SOURCE}" --existing-tgt-spm "${PREPROCESSED_DIR}/${TARGET}" --out-dir $PREPROCESSED_DIR --split test --shared-data
    if [ $? -ne 0 ]; then echo "Tokenizing test.gold failed" && exit 1 ; fi

    tail -n +4 "${PREPROCESSED_DIR}/${SOURCE}.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${SOURCE_VOCAB}"
    tail -n +4 "${PREPROCESSED_DIR}/${TARGET}.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${TARGET_VOCAB}"
    #cd "${PREPROCESSED_DIR}"
#    fairseq-preprocess \
#        --source-lang="${SOURCE}" \
#        --target-lang="${TARGET}" \
#        --trainpref="${PREPROCESSED_DIR}/train" \
#        --validpref="${PREPROCESSED_DIR}/dev" \
#        --testpref="${PREPROCESSED_DIR}/test" \
#        --tokenizer=space \
#        --thresholdsrc=1 \
#        --thresholdtgt=1 \
#        --srcdict "${SOURCE_VOCAB}" \
#        --tgtdict "${TARGET_VOCAB}" \
#        --destdir="${PREPROCESSED_DIR}"
}
tokenize_chars() {
    DICT_FILE="${OUT_DIR}/dict.${SOURCE}.txt"
    if [ -f "${DICT_FILE}" ]; then
        log "Files already preprocessed. Moving on..."
        cd "${PREPROCESSED_DIR}"
        return 0
    fi
    log "Preprocessing..."
    for TASK in "train" "dev" "test" ; do
        for FILE in "${DATASET}.${TASK}.tsv"; do
            SOURCE_FILE="${PREPROCESSED_DIR}/${TASK}.${SOURCE}"
            TARGET_FILE="${PREPROCESSED_DIR}/${TASK}.${TARGET}"
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
    EXPERIMENT_DIR=$1; shift
    VOCAB=$1; shift
    if [ -z $VOCAB ]; then
        >&2 echo "preprocess function needs 2 arguments: experiment directory and vocab size"
        exit 1
    fi
    RESULTS_DIR="${EXPERIMENT_DIR}"
    PREPROCESSED_DIR="${EXPERIMENT_DIR}/in"
    OUT_DIR="${EXPERIMENT_DIR}/out"
    MODEL_DIR="${EXPERIMENT_DIR}/models"
    if [ -n "${SUBWORDS}" ] # Allow user to set this in env
    then
        tokenize_subwords
        SOURCE_VOCAB="${PREPROCESSED_DIR}/${SOURCE}.fairseq.vocab"
        TARGET_VOCAB="${PREPROCESSED_DIR}/${TARGET}.fairseq.vocab"
        EXTRA_ARGS="--srcdict=${SOURCE_VOCAB} --tgtdict=${TARGET_VOCAB} "
    else
        tokenize_chars
    fi

    echo "Moving on to fairseq-preprocess"

    # REMOVE previously preprocessed files or fairseq will exit 1
    TXT_FILE1="${PREPROCESSED_DIR}/dict.${SOURCE}.txt"
    TXT_FILE2="${PREPROCESSED_DIR}/dict.${TARGET}.txt"
    if [ -f "${TXT_FILE1}" ] || [ -f "${TXT_FILE2}" ] ; then
        echo "Removing previously processed files"
        rm "${TXT_FILE1}" "${TXT_FILE2}" > /dev/null 2>&1
    fi

#    pushd . # save whatever directory we are currently in
#    cd "${PREPROCESSED_DIR}"
#    fairseq-preprocess \
#        --source-lang="${SOURCE}" \
#        --target-lang="${TARGET}" \
#        --trainpref=train \
#        --validpref=dev \
#        --testpref=test.gold \
#        --tokenizer=space \
#        --thresholdsrc=1 \
#        --thresholdtgt=1 \
#        --destdir="${OUT_DIR}"
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
#    popd # go back to whatever directory we saved above
}
