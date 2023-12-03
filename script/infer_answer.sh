#!/bin/bash

CARD_ID=$1
ANSWER_MODEL_PATH=$2
DATA_PATH=$3
OUTPUT_PATH=$4

python ../evaluation/vllm_inference.py \
  --model-path "$ANSWER_MODEL_PATH" \
  --model-id  "Yi-answer" \
  --max_token 1024 \
  --data-path "$DATA_PATH" \
  --output-path "$OUTPUT_PATH" \
  --card-id $CARD_ID
