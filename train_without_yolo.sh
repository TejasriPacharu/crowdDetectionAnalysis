#!/bin/bash

# Set TRAIN_YOLO to false to skip YOLO training
TRAIN_YOLO=false
DATASET_DIR="UCF-QNRF_ECCV18"
OUTPUT_DIR="output"
BATCH_SIZE=8
EPOCHS=10  # Reduced from 30 to 10
LEARNING_RATE=1e-5
DEVICE="cpu"
NUM_WORKERS=4
EXPORT_MODELS=true

# Execute training command
python crowd_counting/train.py \
    --data-dir $DATASET_DIR \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --device $DEVICE \
    --num-workers $NUM_WORKERS \
    --export-models