#!/bin/bash

# Hybrid Crowd Counting Training Script
# This script sets up the environment and trains the hybrid crowd counting model

# Configuration
DATASET_DIR="UCF-QNRF_ECCV18"
OUTPUT_DIR="output"
BATCH_SIZE=8
EPOCHS=100  # Number of training epochs
LEARNING_RATE=1e-5
DEVICE="cuda:0"  # Use "cpu" if no GPU is available
NUM_WORKERS=4
TRAIN_YOLO=true  # Set to false to skip YOLOv8 training
YOLO_MODEL_SIZE="n"  # Options: n, s, m, l, x
YOLO_EPOCHS=50
EXPORT_MODELS=true  # Set to false to skip model export

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python -m venv venv
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/requirements_installed" ]; then
    echo -e "${BLUE}Installing requirements...${NC}"
    pip install -r requirements.txt
    touch venv/requirements_installed
fi

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo -e "${YELLOW}Warning: Dataset directory $DATASET_DIR not found!${NC}"
    echo -e "${YELLOW}Please make sure the UCF-QNRF dataset is placed in the project root directory.${NC}"
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Print training configuration
echo -e "${GREEN}=== Hybrid Crowd Counting Training ===${NC}"
echo -e "${GREEN}Dataset:${NC} $DATASET_DIR"
echo -e "${GREEN}Output Directory:${NC} $OUTPUT_DIR"
echo -e "${GREEN}Batch Size:${NC} $BATCH_SIZE"
echo -e "${GREEN}Epochs:${NC} $EPOCHS"
echo -e "${GREEN}Learning Rate:${NC} $LEARNING_RATE"
echo -e "${GREEN}Device:${NC} $DEVICE"
echo -e "${GREEN}Train YOLOv8:${NC} $TRAIN_YOLO"
if [ "$TRAIN_YOLO" = true ]; then
    echo -e "${GREEN}YOLOv8 Model Size:${NC} $YOLO_MODEL_SIZE"
    echo -e "${GREEN}YOLOv8 Epochs:${NC} $YOLO_EPOCHS"
fi
echo -e "${GREEN}Export Models:${NC} $EXPORT_MODELS"
echo -e "${GREEN}===============================${NC}"

# Start training
echo -e "${BLUE}Starting training...${NC}"

# Build training command
TRAIN_CMD="python crowd_counting/train.py \
    --data-dir $DATASET_DIR \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --device $DEVICE \
    --num-workers $NUM_WORKERS"

# Add YOLOv8 parameters if training YOLOv8
if [ "$TRAIN_YOLO" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --train-yolo --yolo-model-size $YOLO_MODEL_SIZE --yolo-epochs $YOLO_EPOCHS"
fi

# Add export flag if exporting models
if [ "$EXPORT_MODELS" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --export-models"
fi

# Execute training command
echo -e "${YELLOW}Executing: $TRAIN_CMD${NC}"
eval $TRAIN_CMD

# Check if training was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    
    # Optimize models for edge deployment
    echo -e "${BLUE}Optimizing models for edge deployment...${NC}"
    python crowd_counting/utils/optimize_for_edge.py \
        --model-dir $OUTPUT_DIR \
        --output-dir optimized_models \
        --target-device arm
    
    echo -e "${GREEN}All done! Optimized models are available in the 'optimized_models' directory.${NC}"
else
    echo -e "${YELLOW}Training failed with exit code $?${NC}"
fi

# Deactivate virtual environment
deactivate
