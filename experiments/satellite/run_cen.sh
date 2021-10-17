#!/usr/bin/env bash

PROBLEM=satellite

CUDA_VISIBLE_DEVICES="" \
python -u -m cen.run \
  experiment=crossval \
  problem=${PROBLEM} \
  encoder=${PROBLEM}/mlp \
  model=${PROBLEM}/cen_convex \
  optimizer=adam
