#!/usr/bin/env bash
#

echo "Running train.py ..."
python train.py
if [ $? -eq 0 ]; then
    echo "Running generate.py ..."
    python generate.py
fi
