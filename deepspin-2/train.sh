#!/usr/bin/env bash
# Adapted from a script by Ben Peters which was adapted by a script by Kyle Gorman and Shijie Wu.
#
# Improved to make this a one-and-done script for preprocessing, training, decoding, postprocessing and evaluation
# Only required argument is the language code

# Load configuration variables from train.conf
source ./deepspin-2.conf
source ../deepspin-shared.conf


if [ ! -d "${PREPROCESSED_DIR}" ]; then
    mkdir -p "${PREPROCESSED_DIR}"
fi
if [ ! -d "${OUT_DIR}" ]; then
    mkdir -p "${OUT_DIR}"
fi
if [ ! -d "${MODEL_DIR}" ]; then
    mkdir -p "${MODEL_DIR}"
fi

###################
# Preprocessing
###################

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
    SOURCE_VOCAB="${OUT_DIR}/${SOURCE}.fairseq.vocab"
    TARGET_VOCAB="${OUT_DIR}/${TARGET}.fairseq.vocab"
    if [ -f "${SOURCE_VOCAB}" ] && [ -f "${TARGET_VOCAB}" ]; then
        echo "Looks like this configuration has already been preprocessed. Moving on..."
        return
    fi
    mkdir -p "${PREPROCESSED_DIR}"
    mkdir -p "${OUT_DIR}"
    mkdir -p "${MODEL_DIR}"
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
    echo "${OUT_DIR}"
    echo "${VOCAB}"
    python ../tokenize.py "${TRAIN_FILE}" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --out-dir "${OUT_DIR}" --split train $@
    if [ $? -ne 0 ]; then echo "Tokenizing train failed" && exit 1 ; fi
    python ../tokenize.py "${DEV_FILE}" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --existing-src-spm "${OUT_DIR}/${SOURCE}" --existing-tgt-spm "${OUT_DIR}/${TARGET}" --out-dir "${OUT_DIR}" --split dev --shared-data
    if [ $? -ne 0 ]; then echo "Tokenizing dev failed" && exit 1 ; fi
    python ../tokenize.py "${GOLD_FILE}" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --existing-src-spm "${OUT_DIR}/${SOURCE}" --existing-tgt-spm "${OUT_DIR}/${TARGET}" --out-dir $OUT_DIR --split test --shared-data
    if [ $? -ne 0 ]; then echo "Tokenizing test.gold failed" && exit 1 ; fi

    tail -n +4 "${OUT_DIR}/${SOURCE}.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${SOURCE_VOCAB}"
    tail -n +4 "${OUT_DIR}/${TARGET}.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${TARGET_VOCAB}"
    #cd "${PREPROCESSED_DIR}"
    fairseq-preprocess \
        --source-lang="${SOURCE}" \
        --target-lang="${TARGET}" \
        --trainpref="${OUT_DIR}/train" \
        --validpref="${OUT_DIR}/dev" \
        --testpref="${OUT_DIR}/test" \
        --tokenizer=space \
        --thresholdsrc=1 \
        --thresholdtgt=1 \
        --srcdict "${SOURCE_VOCAB}" \
        --tgtdict "${TARGET_VOCAB}" \
        --destdir="${OUT_DIR}"
}
#preprocess() {
#    log "Preprocessing..."
#    if [ -f "${DICT_FILE}" ]; then
#        echo "Files already preprocessed. Moving on."
#        cd "${PREPROCESSED_DIR}"
#        return 0
#    fi
#    for TASK in "train" "dev" "test.gold" ; do
#        for FILE in "${DATASET}.${TASK}.tsv"; do
#            SOURCE_FILE="${PREPROCESSED_DIR}/${TASK}.${SOURCE}"
#            TARGET_FILE="${PREPROCESSED_DIR}/${TASK}.${TARGET}"
#            # Segment the source strings by separating each grapheme (char)
#            cut -f 1 "${FILE}" | sed 's/./& /g' > "${SOURCE_FILE}"
#            # Segment the target strings by separating each morph
#            # Change morph separator to single @ and spaces to underscores
#            cut -f 2 "${FILE}" | sed 's/@@/|/g' > "${TARGET_FILE}"
#            echo "Created files: ${SOURCE_FILE} and ${TARGET_FILE}"
#        done
#    done
#
#    echo "Moving on to fairseq-preprocess"
#
#    # REMOVE previously preprocessed files or fairseq will exit 1
#    TXT_FILE1="${OUT_DIR}/dict.${SOURCE}.txt"
#    TXT_FILE2="${OUT_DIR}/dict.${TARGET}.txt"
#    if [ -f "${TXT_FILE1}" ] || [ -f "${TXT_FILE2}" ] ; then
#        echo "Removing previously processed files"
#        rm "${TXT_FILE1}" "${TXT_FILE2}" > /dev/null 2>&1
#    fi
#
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
#        --destdir="../../${LAN}/out"
#}

train() {
    #cd ../../..
    get_checkpoint
    if [ -f "${CHECKPOINT}" ]; then
        log "Checkpount found: ${CHECKPOINT}\nProceeding..."
    else
        log "Training..."
        fairseq-train \
            "${OUT_DIR}/" \
            --save-dir="${MODEL_DIR}" \
            --source-lang="${SOURCE}" \
            --target-lang="${TARGET}" \
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
            --save-interval="${SAVE_INTERVAL}" \
            --reset-optimizer # to make sure we get a checkpoint_best file
    fi
}

decode() {
    MODE="test"
    get_checkpoint
    # Fairseq insists on calling the dev-set "valid"; hack around this.
    FAIRSEQ_MODE="${MODE/dev/valid}"
    OUT_FILE="${OUT_DIR}/${MODE}-${BEAM}.out"
    # Bash won't overwrite the file
    # It's fine to regenerate
    if [ -f "${OUT_FILE}" ]; then
        log "${OUT_FILE} found. Proceeding..."
    else
        log "Decoding..."
        fairseq-generate \
            "${OUT_DIR}" \
            --source-lang="${SOURCE}" \
            --target-lang="${TARGET}" \
            --path="${CHECKPOINT}" \
            --gen-subset="${FAIRSEQ_MODE}" \
            --beam="${BEAM}" \
            > "${OUT_FILE}"
            #--alpha="${ENTMAX_ALPHA}" \
            #--batch-size 256 \
    fi
    if [ $? -ne 0 ]; then
        rm "${OUT_FILE}"
        echo "It appears decoding failed. Check output."
        exit 1
    fi
}
postprocess() {
    log "Postprocessing..."
    MODE="test"
    # Fairseq insists on calling the dev-set "valid"; hack around this.
    FAIRSEQ_MODE="${MODE/dev/valid}"
    OUT_FILE="${OUT_DIR}/${MODE}-${BEAM}.out"
    PRED="${OUT_DIR}/${MODE}-${BEAM}.pred"
    GUESS_FILE="${OUT_DIR}/${MODE}-${BEAM}.guess"

    # Extracts the predictions into a TSV file.
    # extract the 3rd to last column of the output from fairseq-generate only if the line starts with 'H-'
    echo "Writing predictions"
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
        sed -e 's/^ //g' > $PRED

    echo "Generated prediction file: ${PRED}"
    # Now take the first column of the gold file (the input word) as the first column
    # and the generated prediction as the second column and output to .guess file
    cut -f 1 "${GOLD_FILE}" | paste - "${PRED}" > "${GUESS_FILE}"
    echo "Generated guess file: ${GUESS_FILE}"
}

evaluate() {
    log "Evaluating..."
    MODE="test"
    GUESS_FILE="${OUT_DIR}/${MODE}-${BEAM}.guess"
    # Applies the evaluation script to the generated TSV file.
    RESULTS_FILE="${RESULTS_DIR}/${MODE}-${BEAM}.results"
    python $EVALUATE_SCRIPT --gold "${GOLD_FILE}" --guess "${GUESS_FILE}" > "${RESULTS_FILE}"
    if [ $? -eq 0 ]; then
        cat "${RESULTS_FILE}"
    else
        echo "Failed to evaluate model."
        exit 1
    fi
}

preprocess "./data/${LAN}" 6000
if [ $? -ne 0 ]; then
    echo "Preprocessing failed. Stopping here"
    exit 1
fi
echo "Preprocessing succeeded"
train
if [ $? -ne 0 ]; then
    echo "Training failed. Stopping here"
    exit 1
fi
echo "Training succeeded"
decode
if [ $? -ne 0 ]; then
    echo "It appears decoding failed. Check output."
    exit 1
fi
echo "Decoding succeeded"
postprocess
if [ $? -ne 0 ]; then
    echo "Postprocessing failed. Stopping here"
    exit 1
fi
echo "Postprocess succeeded"
evaluate
if [ $? -ne 0 ]; then
    echo "Evaluation failed. Stopping here"
    exit 1
fi
