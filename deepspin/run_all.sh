#!/usr/bin/env bash

for lang in eng fra lat
do
    for tokens in chars subwords
    do
        for arch in lstm transformer
        do
            echo "LANG: ${lang}, ARCH: ${arch}, TOKENS: ${tokens}"
            ./run.sh "${lang}" "${arch}" "${tokens}" 2>&1 | tee "${lang}-${arch}-${tokens}.log"
        done
    done
done

