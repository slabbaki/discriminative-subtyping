#!/usr/bin/env bash

PROBLEM=tcga

CUDA_VISIBLE_DEVICES=0 \
python -u -m cen.run \
  experiment=train \
  problem=${PROBLEM} \
  encoder=${PROBLEM}/inception \
  model=${PROBLEM}/baseline \
  optimizer=rmsprop
$SHELL
