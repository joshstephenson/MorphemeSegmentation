#!/usr/bin/env bash

#if [ -z "$1" ] || [ -z "$2" ]; then
#    echo "Please provide a language name {ces|eng|fra|hun|ita|lat|mon|rus|spa} and model dir path (data/models/hun/hun-entmax-minloss-256-1024-6-8-8192-1.5-0.001-4000-0.1/)"
#    exit 1
#fi
readonly DATA_BIN=$1; shift
readonly MODEL_PATH=$1; shift
#readonly DATA_BIN=$1
#readonly MODEL_PATH=$2
readonly ENTMAX_ALPHA=$1; shift
readonly BEAM=$1; shift
readonly GOLD_PATH=$1; shift
echo "FROM fairseq_segment.sh:
DATA_BIN: ${DATA_BIN}
MODEL_PATH: ${MODEL_PATH}
ENTMAX_ALPHA: ${ENTMAX_ALPHA}
BEAM: ${BEAM}
GOLD_PATH: ${GOLD_PATH}
====================================================
"

decode() {
    if [ ! -d "$DATA_BIN" ]; then
        echo "Creating $DATA_BIN"
        mkdir -p "$DATA_BIN"
    fi
    local -r CP="$1"; shift
    local -r MODE="$1"; shift
    # Fairseq insists on calling the dev-set "valid"; hack around this.
    local -r FAIRSEQ_MODE="${MODE/dev/valid}"
    echo "FAIRSEQ_MODE: ${FAIRSEQ_MODE}"
    CHECKPOINT="${CP}/checkpoint_best.pt"
    OUT="${CP}/${MODE}-${BEAM}.out"
    PRED="${CP}/${MODE}-${BEAM}.pred"
    echo "PRED: ${PRED}"
    # Makes raw predictions.
    fairseq-generate \
        "${DATA_BIN}" \
        --source-lang="src" \
        --target-lang="tgt" \
        --path="${CHECKPOINT}" \
        --gen-subset="${FAIRSEQ_MODE}" \
        --beam="${BEAM}" \
        --alpha="${ENTMAX_ALPHA}" \
	--batch-size 256 \
        > "${OUT}"
    # Extracts the predictions into a TSV file.
    if [ ! -f "${OUT}" ]; then
        echo "The last command failed to generate ${OUT}. Please check output for error."
        exit 1
    fi
    cat "${OUT}" | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | python detokenize.py > $PRED
    cut -f 1 $GOLD_PATH | paste - $PRED > "${CP}/${MODE}-${BEAM}.guess"
    # Applies the evaluation script to the TSV file.
    RESULTS_FILE="${CP}/${MODE}-${BEAM}.results"
    python 2022SegmentationST/evaluation/evaluate.py --gold $GOLD_PATH --guess "${CP}/${MODE}-${BEAM}.guess" > "${RESULTS_FILE}"
    if [ $? -ne 0 ]; then
        echo "Results of model evaluation written to: ${RESULTS_FILE}"
    else
        echo "Failed to evaluate model."
    fi
}

decode $MODEL_PATH test
