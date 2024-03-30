#!/usr/bin/env bash
#

FILE=$1; shift # Should match the name of the languages available in ../2022SegmentationST/data
if [ -z "$FILE" ]; then
    echo "Running train.py ..."
    python train.py
else
    echo "Running generate.py ..."
    if [ -f "${FILE}" ]; then
        LAN=$(echo $FILE | awk -F '/' '{print $2}')
        python generate.py $FILE $@
        GUESS_FILE=$(echo $FILE | sed 's/pt/tsv/g')
        RESULTS_FILE=$(echo $FILE | sed 's/pt/txt/g')
        python ../2022SegmentationST/evaluation/evaluate.py --gold "../2022SegmentationST/data/${LAN}.word.test.gold.tsv" --guess "${GUESS_FILE}" > "${RESULTS_FILE}"
        cat "${RESULTS_FILE}"
    else
        echo "No FILE found: $FILE"
        exit 1
    fi
    if [ $? -ne 0 ]; then
        echo "generate for the previous model failed"
        exit 1
    fi
fi
