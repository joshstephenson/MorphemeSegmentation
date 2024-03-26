#!/usr/bin/env bash

readonly PROCESSED_DIR="data/processed"
readonly MODEL_DIR="data/models"
readonly ENTMAX_ALPHA=1.5
readonly LR=0.001
readonly BEAM=5
readonly BATCHES=8192
readonly VOCAB=8000 #(underexplored)

if [ -z "$1" ]; then
    echo "Please provide a language name {ces|eng|fra|hun|ita|lat|mon|rus|spa}."
    exit 1
else
    readonly LANG="$1"; shift
    readonly INPUT_PATH="2022SegmentationST/data/${LANG}.word"
    readonly OUT_PATH="${PROCESSED_DIR}/${LANG}"

    GOLD_PATH="2022SegmentationST/data/${LANG}.word.test.gold.tsv"
    GRID_LOC="data/models/${LANG}"

    if [ ! -d "$OUT_PATH" ]; then
        mkdir -p "$OUT_PATH"
    fi

    if [ ! -d "$GRID_LOC" ]; then
        mkdir -p "$GRID_LOC"
    fi

    echo "GOLD PATH: ${GOLD_PATH}"
    echo "GRID_LOC: ${GRID_LOC}"
    echo "INPUT_PATH: ${INPUT_PATH}"
    echo "OUT_PATH: ${OUT_PATH}"
    echo "VOCAB: ${VOCAB}"
fi

# For preprocessing
bin() {
    tail -n +4 "${OUT_PATH}/src.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${OUT_PATH}/src.fairseq.vocab"
    tail -n +4 "${OUT_PATH}/tgt.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${OUT_PATH}/tgt.fairseq.vocab"
    fairseq-preprocess \
        --source-lang="src" \
        --target-lang="tgt" \
        --trainpref="${OUT_PATH}/train" \
        --validpref="${OUT_PATH}/dev" \
        --testpref="${OUT_PATH}/test" \
        --tokenizer=space \
        --thresholdsrc=1 \
        --thresholdtgt=1 \
        --srcdict "${OUT_PATH}/src.fairseq.vocab" \
        --tgtdict "${OUT_PATH}/tgt.fairseq.vocab" \
        --destdir="${OUT_PATH}"
}

# For training
grid() {
    local -r EMB="$1"; shift
    local -r HID="$1"; shift
    local -r LAYERS="$1" ; shift
    local -r HEADS="$1" ; shift
    for WARMUP in 4000 8000 ; do
        for DROPOUT in 0.1 0.3 ; do
            for BATCH in $BATCHES ; do
                THIS_MODEL_DIR="${GRID_LOC}/${LANG}-entmax-minloss-${EMB}-${HID}-${LAYERS}-${HEADS}-${BATCH}-${ENTMAX_ALPHA}-${LR}-${WARMUP}-${DROPOUT}"
                FILENAME="${THIS_MODEL_DIR}/dev-5.results"
                if [ ! -f "$FILENAME" ]
                then
                    bash fairseq_train_entmax_transformer.sh $OUT_PATH $LANG $EMB $HID $LAYERS $HEADS $BATCH $ENTMAX_ALPHA $LR $WARMUP $DROPOUT $GRID_LOC
                    if [ $? -ne 0 ]; then
                        exit 1
                    fi
                    bash fairseq_segment.sh $OUT_PATH $THIS_MODEL_DIR $ENTMAX_ALPHA $BEAM $GOLD_PATH
                    if [ $? -ne 0 ]; then
                        echo "fairseq_segment.sh failed."
                        exit 1
                    else
                        echo "Trained data written to: ${FILENAME}"
                    fi
                else
                    echo "${FILENAME} found. Not re-running."
                fi
            done
        done
    done
}

preprocess() {
    python scripts/tokenize.py "${INPUT_PATH}.train.tsv" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --out-dir $OUT_PATH --split train $@
    if [ $? -ne 0 ]; then echo "Tokenizing train failed" && exit 1 ; fi
    python scripts/tokenize.py "${INPUT_PATH}.dev.tsv" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --existing-src-spm "${OUT_PATH}/src" --existing-tgt-spm "${OUT_PATH}/tgt" --out-dir $OUT_PATH --split dev --shared-data
    if [ $? -ne 0 ]; then echo "Tokenizing dev failed" && exit 1 ; fi
    python scripts/tokenize.py "${INPUT_PATH}.test.gold.tsv" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --existing-src-spm "${OUT_PATH}/src" --existing-tgt-spm "${OUT_PATH}/tgt" --out-dir $OUT_PATH --split test --shared-data
    if [ $? -ne 0 ]; then echo "Tokenizing test.gold failed" && exit 1 ; fi
    bin
}

train_model() {
    grid 256 1024 6 8
    grid 512 2048 6 8
}

decode() {
    local -r CP="$1"; shift
    local -r MODE="$1"; shift
    # Fairseq insists on calling the dev-set "valid"; hack around this.
    local -r FAIRSEQ_MODE="${MODE/dev/valid}"
    CHECKPOINT="${CP}/checkpoint_best.pt"
    OUT="${CP}/${MODE}-${BEAM}.subwords.out"
    PRED="${CP}/${MODE}-${BEAM}.subwords.pred"
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
    cat "${OUT}" | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' > $PRED
}


if [ "$(ls -A $OUT_PATH)" ]; then
    echo "PROCESSED DATA found in ${OUT_PATH}. Proceeding to training..."
    train_model
else
    preprocess
    if [ $? -ne 0 ]; then
        echo "Data Processing failed. Do not proceed to training."
        exit 1
    else
        echo "Data Processing succeeded. Proceeding to training."
        train_model
    fi
fi
