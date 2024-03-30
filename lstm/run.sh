#!/usr/bin/env bash
#

if [ -z "$1" ]; then
    python train.py
fi

ROOT="./data/$1"
for dir in $(ls "${ROOT}/"); do
    file=$(ls "${ROOT}/${dir}/$1-by_char.pt")
    echo "Found model: $file"
    if [ -f "${file}" ]; then
        python generate.py $file
    else
        echo "No file: $file"
        exit 1
    fi
    if [ $? -ne 0 ]; then
        echo "generate for the previous model failed"
        exit 1
    fi
done


