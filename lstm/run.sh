#!/usr/bin/env bash
#

SKIP_TRAIN=$1
if [ "${SKIP_TRAIN}" != '--skip' ]; then
    python train.py
fi
if [ $? -eq 0 ]; then
    LAN=$(python config.py -l)
    FILE=$(python config.py)
    #echo "${FILE}"
    if [ ! -f "${FILE}" ]; then
        echo "File doesn't exist: ${FILE}"
        exit 1
    fi
    python generate.py "${FILE}"
    GUESS_FILE=$(echo $FILE | sed 's/pt/tsv/g')
    RESULTS_FILE=$(echo $FILE | sed 's/pt/txt/g')
    #echo $GUESS_FILE
    #echo $RESULTS_FILE
    python ../2022SegmentationST/evaluation/evaluate.py --gold "../2022SegmentationST/data/${LAN}.word.test.gold.tsv" --guess "${GUESS_FILE}" > "${RESULTS_FILE}"
    cat "${RESULTS_FILE}"
fi
