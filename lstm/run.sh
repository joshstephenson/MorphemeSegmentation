#!/usr/bin/env bash
#

SHOULD_TRAIN=true
ARGS=''
for arg in $@
do
    if [ "${arg}" == '--skip-train' ]; then
        SHOULD_TRAIN=false
        echo "Skipping training"
    elif [ "${arg}" == '--disable-tqdm' ]; then
        ARGS=arg
        echo "Disabling progress bar"
    fi
done

if [ "$SHOULD_TRAIN" == true ]; then
    python train.py $ARGS
fi

if [ $? -eq 0 ]; then
    LAN=$(python config.py -l)
    FILE=$(python config.py)
    #echo "${FILE}"
    if [ ! -f "${FILE}" ]; then
        echo "File doesn't exist: ${FILE}"
        exit 1
    fi
    python predict.py --file "${FILE}"
    GUESS_FILE=$(echo $FILE | sed 's/pt/tsv/g')
    RESULTS_FILE=$(echo $FILE | sed 's/pt/txt/g')
    #echo $GUESS_FILE
    #echo $RESULTS_FILE
    python ../2022SegmentationST/evaluation/evaluate.py --gold "../2022SegmentationST/data/${LAN}.word.test.gold.tsv" --guess "${GUESS_FILE}" > "${RESULTS_FILE}"
    cat "${RESULTS_FILE}"
fi
