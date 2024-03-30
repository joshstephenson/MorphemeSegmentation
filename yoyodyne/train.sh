#!/usr/bin/env bash
#

source ./train.conf

yoyodyne-train \
    --model_dir "${MODEL_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dropout 0.3 \
    --train "${TRAIN_DATA}" \
    --val "${VAL_DATA}" \
    --arch attentive_lstm \
    --max_epochs "${EPOCHS}" \
    --accelerator gpu \
    --auto_select_gpus 1 \
    --seed "${SEED}"
