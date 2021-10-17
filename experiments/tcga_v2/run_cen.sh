#!/usr/bin/env bash

PROBLEM=tcga_v2

CUDA_VISIBLE_DEVICES=0 \
python -u -m cen.run \
  experiment=train \
  problem=${PROBLEM} \
  encoder=${PROBLEM}/inception \
  model=${PROBLEM}/cen_convex \
  optimizer=rmsprop
$SHELL
