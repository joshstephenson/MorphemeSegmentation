#!/usr/bin/env bash
#

LAN=$1; shift # Should match the name of the languages available in ../2022SegmentationST/data
if [ -n "${LAN}" ] && [ ! -d "./data/${LAN}" ]; then
    echo "${LAN} does not appear to be a valid language. Check ./data directory"
    exit
fi
if [ -z "$LAN" ]; then
    echo "Running train.py ..."
    python train.py
else
    ROOT="./data/$LAN"
    for dir in $(ls "${ROOT}/"); do
        file=$(ls "${ROOT}/${dir}/$LAN-by_char.pt")
        echo "Found model: $file"
        if [ -f "${file}" ]; then
            python generate.py $file $@
        else
            echo "No file: $file"
            exit 1
        fi
        if [ $? -ne 0 ]; then
            echo "generate for the previous model failed"
            exit 1
        fi
    done
    echo "Running generate.py ..."
fi



