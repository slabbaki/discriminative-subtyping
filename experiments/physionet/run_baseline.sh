#!/usr/bin/env bash

PROBLEM=physionet

CUDA_VISIBLE_DEVICES=0 \
python -u -m cen.run \
  experiment=train_eval \
  problem=${PROBLEM} \
  encoder=${PROBLEM}/lstm \
  model=${PROBLEM}/baseline \
  optimizer=adam
