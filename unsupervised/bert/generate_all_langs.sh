#!/usr/bin/env bash

for lang in ces eng fra hun ita lat mon rus spa; do echo $lang && ./btrain.sh $lang | grep f_measure && echo ""; done
