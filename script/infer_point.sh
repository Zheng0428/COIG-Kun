#!/bin/bash


POINT_MODEL_PATH=$1
DATA_PATH=$2
OUTPUT_PATH=$3

python ../evaluation/vllm_inference.py \
  --model-path "$ANSWER_MODEL_PATH" \
  --model-id  "Yi-point" \
  --max_token 1024 \
  --data-path "$DATA_PATH" \
  --output-path "$OUTPUT_PATH" \
