#!/usr/bin/env bash

for lang in ces eng fra hun ita lat mon rus spa
do
    echo $lang && ./run.sh $lang | grep f_measure && echo "" | tee results.txt
done
